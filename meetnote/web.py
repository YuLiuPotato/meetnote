"""Internal web service: colleagues upload a recording in the browser and get the notes back.

One worker thread processes jobs in order, so the GPU only ever runs one
transcription at a time. Each job lives in <data>/jobs/<id>/ with a status.json;
in-memory state is the source of truth and is persisted on every change, so a
restart resumes unfinished jobs.
"""

from __future__ import annotations

import hmac
import io
import json
import os
import queue
import re
import shutil
import socket
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote

# Module-level (not inside create_app): FastAPI resolves the string annotations
# left by `from __future__ import annotations` against module globals.
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response

from .i18n import FILE_NAMES, find_file, tr
from .pipeline import rename_speakers, summarize_dir, transcribe_to_dir
from .summarize import SummarizeError, resolve_backend
from .transcribe import LANGUAGE_NAMES, log

STATIC_DIR = Path(__file__).parent / "static"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
AUDIO_EXTS = {
    ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr",
    ".mp4", ".mov", ".mkv", ".webm", ".m4v",
}
MAX_UPLOAD_MB = 2048

# Files the page can show or download, in tab order. Each job's files are named in the
# server's interface language; find_file() accepts either language.
FILES = ("highlights", "synopsis", "notes", "checks", "transcript", "subtitles")
assert set(FILES) <= set(FILE_NAMES)
ACTIVE = ("queued", "transcribing", "summarizing")


class JobStore:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.pending: list[str] = []
        self.queue: queue.Queue[str] = queue.Queue()
        for status in self.root.glob("*/status.json"):
            try:
                job = json.loads(status.read_text(encoding="utf-8"))
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                log(tr(f"跳过损坏的任务：{status.parent}", f"Skipping a corrupt job: {status.parent}"))

    def dir(self, jid: str) -> Path:
        return self.root / jid

    def _persist(self, job: dict) -> None:
        path = self.dir(job["id"]) / "status.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def create(self, job: dict) -> None:
        with self.lock:
            self.jobs[job["id"]] = job
            self._persist(job)

    def get(self, jid: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(jid)
            return dict(job) if job else None

    def update(self, jid: str, **fields) -> None:
        with self.lock:
            job = self.jobs[jid]
            job.update(fields)
            self._persist(job)

    def enqueue(self, jid: str) -> None:
        with self.lock:
            if jid not in self.pending:
                self.pending.append(jid)
                self.queue.put(jid)

    def done_with(self, jid: str) -> None:
        with self.lock:
            if jid in self.pending:
                self.pending.remove(jid)

    def jobs_ahead(self, jid: str) -> int:
        with self.lock:
            return self.pending.index(jid) if jid in self.pending else 0

    def resume_unfinished(self) -> None:
        """Requeue jobs that were waiting or running when the server stopped."""
        for job in sorted(self.jobs.values(), key=lambda j: j["created_at"]):
            if job["status"] not in ACTIVE:
                continue
            has_transcript = find_file(self.dir(job["id"]), "transcript") is not None
            if job["status"] == "summarizing" and has_transcript:
                self.update(job["id"], status="queued", mode="summarize", message="服务重启，重新排队")
            else:
                self.update(job["id"], status="queued", progress=0, message="服务重启，重新排队")
            self.enqueue(job["id"])

    def expire(self, keep_days: int) -> None:
        cutoff = time.time() - keep_days * 86400
        with self.lock:
            stale = [j["id"] for j in self.jobs.values() if j["created_at"] < cutoff and j["status"] not in ACTIVE]
            for jid in stale:
                self.jobs.pop(jid)
        for jid in stale:
            shutil.rmtree(self.dir(jid), ignore_errors=True)
            log(tr(f"已删除过期任务 {jid}", f"Deleted expired job {jid}"))


class Worker:
    def __init__(self, store: JobStore, cfg: dict):
        self.store, self.cfg = store, cfg

    def run_forever(self) -> None:
        while True:
            jid = self.store.queue.get()
            try:
                self.process(jid)
            except Exception:
                log(traceback.format_exc())
                self.store.update(jid, status="failed", error="处理时发生内部错误，请联系管理员查看服务日志")
            finally:
                self.store.done_with(jid)

    def process(self, jid: str) -> None:
        store, cfg = self.store, self.cfg
        job = store.get(jid)
        job_dir = store.dir(jid)

        if job.get("mode", "full") == "full":
            store.update(jid, status="transcribing", progress=0, message="正在转写", error=None)
            last = [0.0]

            def on_progress(pct: float) -> None:
                if pct - last[0] >= 1 or pct >= 100:
                    last[0] = pct
                    store.update(jid, progress=round(pct, 1))

            audio = job_dir / job["audio_file"]
            try:
                language = job.get("language", "auto")
                meta = transcribe_to_dir(
                    audio, job_dir, whisper_model=cfg["whisper_model"], device=cfg["device"],
                    language=None if language == "auto" else language,
                    terms=job.get("terms") or None, source_name=job["filename"], on_progress=on_progress,
                    num_speakers=job.get("num_speakers") or None,
                )
            except Exception as e:
                log(traceback.format_exc())
                store.update(jid, status="failed", error=f"转写失败：{e}（请确认上传的是有效的音频文件）")
                return
            if not cfg["keep_audio"]:
                audio.unlink(missing_ok=True)
            store.update(jid, duration=meta["duration"], detected_language=meta["language"],
                         speakers=meta.get("speakers") or [])

        store.update(jid, status="summarizing", progress=100, message="正在生成总结", error=None)
        try:
            summarize_dir(
                job_dir, backend=cfg["backend"], model=cfg["model"], context=job.get("context") or None,
                summary_lang=job.get("summary_lang", "auto"),
                on_step=lambda step: store.update(jid, message=f"正在{step}"),
            )
        except SummarizeError as e:
            store.update(jid, status="failed", error=f"总结失败：{e}", message="转写稿已生成，可稍后重新生成总结")
            return
        store.update(jid, status="done", message="完成", finished_at=time.time())


def create_app(store: JobStore, access_code: str | None):
    app = FastAPI(title="meetnote", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def check_access(request: Request, call_next):
        path = request.url.path
        if access_code and path.startswith("/api/") and path != "/api/config":
            given = request.headers.get("x-access-code") or request.cookies.get("mn_code") or ""
            if not hmac.compare_digest(given.encode(), access_code.encode()):
                return JSONResponse({"detail": "访问码错误"}, status_code=401)
        return await call_next(request)

    def public(job: dict) -> dict:
        job_dir = store.dir(job["id"])
        out = {k: job.get(k) for k in (
            "id", "filename", "created_at", "finished_at", "status", "progress",
            "message", "error", "context", "terms", "duration",
            "language", "detected_language", "summary_lang", "num_speakers", "speakers",
        )}
        out["files"] = [key for key in FILES if find_file(job_dir, key)]
        out["jobs_ahead"] = store.jobs_ahead(job["id"]) if job["status"] == "queued" else 0
        return out

    def require_job(jid: str) -> dict:
        job = store.get(jid) if JOB_ID.match(jid) else None
        if not job:
            raise HTTPException(404, "任务不存在或已过期")
        return job

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config():
        return {"access_code_required": bool(access_code), "max_upload_mb": MAX_UPLOAD_MB,
                "audio_exts": sorted(AUDIO_EXTS), "languages": LANGUAGE_NAMES}

    @app.post("/api/jobs")
    def create_job(file: UploadFile = File(...), context: str = Form(""), terms: str = Form(""),
                   language: str = Form("auto"), summary_lang: str = Form("auto"),
                   num_speakers: int = Form(0)):
        filename = Path(file.filename or "录音").name
        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise HTTPException(400, f"不支持的文件类型 {ext or '（无扩展名）'}，请上传音频或视频文件")
        for value in (language, summary_lang):
            if value != "auto" and value not in LANGUAGE_NAMES:
                raise HTTPException(400, f"不支持的语言：{value}")
        if not 0 <= num_speakers <= 20:
            raise HTTPException(400, "参会人数应在 1–20 之间")
        jid = uuid.uuid4().hex
        job_dir = store.dir(jid)
        job_dir.mkdir(parents=True)
        dest = job_dir / f"audio{ext}"
        size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with dest.open("wb") as f:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    f.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(413, f"文件超过 {MAX_UPLOAD_MB} MB")
                f.write(chunk)
        job = {
            "id": jid, "filename": filename, "audio_file": dest.name, "size": size,
            "context": context.strip()[:2000], "terms": terms.strip()[:500],
            "language": language, "summary_lang": summary_lang, "num_speakers": num_speakers,
            "created_at": time.time(), "status": "queued", "mode": "full",
            "progress": 0, "message": "排队中", "error": None,
        }
        store.create(job)
        store.enqueue(jid)
        log(tr(f"新任务 {jid}：{filename}（{size / 1e6:.1f} MB）", f"New job {jid}: {filename} ({size / 1e6:.1f} MB)"))
        return public(store.get(jid))

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return public(require_job(jid))

    @app.post("/api/jobs/{jid}/resummarize")
    def resummarize(jid: str):
        job = require_job(jid)
        if job["status"] in ACTIVE:
            raise HTTPException(409, "任务还在处理中")
        if not find_file(store.dir(jid), "transcript"):
            raise HTTPException(409, "还没有转写稿，无法生成总结")
        store.update(jid, status="queued", mode="summarize", message="排队中", error=None)
        store.enqueue(jid)
        return public(store.get(jid))

    @app.post("/api/jobs/{jid}/speakers")
    def set_speakers(jid: str, names: dict[str, str] = Body(..., embed=True)):
        """Rename speaker labels (e.g. S1 → 张三) in the transcript, then rewrite the notes."""
        job = require_job(jid)
        if job["status"] in ACTIVE:
            raise HTTPException(409, "任务还在处理中")
        try:
            labels = rename_speakers(store.dir(jid), names)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        store.update(jid, speakers=labels, status="queued", mode="summarize", message="排队中", error=None)
        store.enqueue(jid)
        return public(store.get(jid))

    def download_name(job: dict, file_name: str) -> str:
        return f"{Path(job['filename']).stem}_{file_name}"

    @app.get("/api/jobs/{jid}/files/{key}")
    def get_file(jid: str, key: str, inline: bool = False):
        job = require_job(jid)
        if key not in FILES or not (path := find_file(store.dir(jid), key)):
            raise HTTPException(404, "文件不存在")
        if inline:
            return PlainTextResponse(path.read_text(encoding="utf-8"))
        return FileResponse(path, filename=download_name(job, path.name), media_type="text/plain; charset=utf-8")

    @app.get("/api/jobs/{jid}/zip")
    def get_zip(jid: str):
        job = require_job(jid)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for key in FILES:
                if path := find_file(store.dir(jid), key):
                    z.write(path, download_name(job, path.name))
        zip_name = quote(f"{Path(job['filename']).stem}{tr('_纪要', '_notes')}.zip")
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{zip_name}"})

    return app


def _addresses(host: str, port: int) -> list[str]:
    urls = [f"http://localhost:{port}"]
    if host not in ("0.0.0.0", "::"):
        return urls
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    return urls


def serve(host: str, port: int, data_dir: Path, keep_days: int, keep_audio: bool,
          whisper_model: str, device: str, backend: str, model: str | None) -> None:
    import uvicorn

    try:
        log(tr(f"总结后端：{resolve_backend(backend)}", f"Summary backend: {resolve_backend(backend)}"))
    except SummarizeError as e:
        log(tr(f"警告：{e}\n（服务照常启动，但总结会失败，直到配置好为止）",
               f"Warning: {e}\n(The service starts anyway, but summaries will fail until this is configured.)"))

    store = JobStore(data_dir)
    store.expire(keep_days)
    cfg = {"whisper_model": whisper_model, "device": device, "keep_audio": keep_audio,
           "backend": backend, "model": model}
    threading.Thread(target=Worker(store, cfg).run_forever, daemon=True, name="worker").start()

    def expire_loop():
        while True:
            time.sleep(3600)
            store.expire(keep_days)

    threading.Thread(target=expire_loop, daemon=True, name="expire").start()
    store.resume_unfinished()

    access_code = os.environ.get("MEETNOTE_ACCESS_CODE") or None
    log(tr(f"数据目录：{data_dir}", f"Data folder: {data_dir}"))
    log(tr("访问码：" + ("已启用" if access_code else "未启用（在 .env 里设 MEETNOTE_ACCESS_CODE 可开启）"),
           "Access code: " + ("on" if access_code else "off (set MEETNOTE_ACCESS_CODE in .env to turn it on)")))
    log(tr("访问地址：", "Open at:") + "\n  " + "\n  ".join(_addresses(host, port)))
    if host in ("127.0.0.1", "localhost"):
        log(tr(f"（只监听本机；同事通过 `tailscale serve --bg {port}` 访问）",
               f"(Listening on this machine only; colleagues connect via `tailscale serve --bg {port}`)"))
    uvicorn.run(create_app(store, access_code), host=host, port=port, log_level="warning")
