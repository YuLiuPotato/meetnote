"""meetnote CLI: meeting recording → transcript + synopsis + two-page highlights."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .i18n import default_out_dir, file_path, tr
from .merge import collect_inputs, earliest_date, merge_to_mp3
from .pipeline import summarize_dir, transcribe_to_dir
from .summarize import BACKENDS, PIPELINES, SummarizeError
from .transcribe import log


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 18765


def load_env() -> Path | None:
    """Load settings from a .env file instead of global environment variables.

    Looks at $MEETNOTE_ENV_FILE, then ./.env, then the .env next to pyproject.toml.
    Variables already set in the environment win over the file.
    """
    from dotenv import load_dotenv

    override = os.environ.get("MEETNOTE_ENV_FILE")
    for path in ([Path(override)] if override else [Path.cwd() / ".env", PROJECT_DIR / ".env"]):
        if path.is_file():
            load_dotenv(path, override=False, encoding="utf-8")
            return path
    return None


def _inputs(args) -> tuple[list[Path], str]:
    """Audio files from the command line, and the name for outputs: the folder's name when a
    folder was given, else the first file's name."""
    paths = [Path(a).resolve() for a in args.audio]
    try:
        files = collect_inputs(paths)
    except FileNotFoundError as e:
        sys.exit(str(e))
    name = paths[0].name if paths[0].is_dir() else files[0].stem
    return files, name


def do_merge(args) -> None:
    files, name = _inputs(args)
    if args.out:
        out = Path(args.out).resolve()
    else:
        from_folder = len(args.audio) == 1 and Path(args.audio[0]).is_dir()
        out = Path.cwd() / (f"{name}.mp3" if from_folder else f"{name}_merged.mp3")
    if out.suffix.lower() != ".mp3":
        out = out.with_suffix(".mp3")
    log(tr(f"合并 {len(files)} 个文件 → {out}", f"Merging {len(files)} file(s) → {out}",
                                           f"Füge {len(files)} Datei(en) zusammen → {out}"))
    try:
        merge_to_mp3(files, out, bitrate=args.bitrate * 1000)
    except (RuntimeError, ValueError) as e:
        sys.exit(str(e))
    log(tr("完成。", "Done.", "Fertig."))


def do_transcribe(args) -> Path:
    files, name = _inputs(args)
    out_dir = Path(args.out).resolve() if args.out else default_out_dir(Path.cwd(), name)
    txt = file_path(out_dir, "transcript")

    if txt.exists() and not args.force:
        log(tr(f"已有转写稿，跳过转写（用 --force 重新转写）：{txt}",
               f"Transcript exists, skipping transcription (use --force to redo it): {txt}",
               f"Transkript vorhanden, Transkription übersprungen (mit --force neu erstellen): {txt}"))
        return out_dir

    extra = None
    audio = files[0]
    if len(files) > 1:
        # Several pieces of one meeting: join them, then transcribe the result as one recording
        audio = out_dir / f"{name}.mp3"
        log(tr(f"合并 {len(files)} 个文件 → {audio}", f"Merging {len(files)} file(s) → {audio}",
                                                 f"Füge {len(files)} Datei(en) zusammen → {audio}"))
        try:
            parts = merge_to_mp3(files, audio)
        except (RuntimeError, ValueError) as e:
            sys.exit(str(e))
        extra = {"parts": parts, "file_date": earliest_date(files)}

    transcribe_to_dir(
        audio, out_dir, whisper_model=args.whisper_model, language=args.language,
        device=args.device, terms=args.terms, source_name=str(audio),
        diarization=not args.no_diarize, num_speakers=args.speakers or None, extra_meta=extra,
    )
    log(f"  → {file_path(out_dir, 'transcript')}")
    return out_dir


def do_summarize(args, out_dir: Path | None = None) -> None:
    if out_dir is None:
        target = Path(args.target).resolve()
        out_dir = target if target.is_dir() else target.parent
        txt = file_path(target, "transcript") if target.is_dir() else target
    else:
        txt = file_path(out_dir, "transcript")
    if not txt.is_file():
        sys.exit(tr(f"找不到转写稿：{txt}", f"Transcript not found: {txt}",
                                     f"Transkript nicht gefunden: {txt}"))

    which = ("synopsis", "highlights") if args.only == "both" else (args.only,)
    try:
        summarize_dir(out_dir, txt, backend=args.backend, model=args.model, context=args.context,
                      which=which, summary_lang=args.summary_lang, pipeline=args.pipeline)
    except SummarizeError as e:
        sys.exit(tr(f"总结失败：{e}", f"Summarizing failed: {e}", f"Zusammenfassen fehlgeschlagen: {e}"))


def do_serve(args) -> None:
    from .web import serve

    serve(
        host=args.host, port=args.port, data_dir=Path(args.data).resolve(),
        keep_days=args.keep_days, keep_audio=args.keep_audio,
        whisper_model=args.whisper_model, device=args.device,
        backend=args.backend, model=args.model,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="meetnote",
        description=tr("会议录音 → 转写稿 + 全文梗概 + 两页精华",
                       "Meeting recording → transcript + full synopsis + two-page highlights",
                       "Besprechungsaufnahme → Transkript + ausführliche Zusammenfassung + Kernpunkte auf zwei Seiten"),
        epilog=tr("界面语言由 MEETNOTE_LANG=zh/en 设置（.env 或环境变量），默认跟随系统。",
                  "Interface language: MEETNOTE_LANG=zh/en (in .env or the environment); default follows the system.",
                  "Oberflächensprache: MEETNOTE_LANG=zh/en/de (in .env oder als Umgebungsvariable); standardmäßig wie "
                  "das System."),
    )
    p.add_argument("--version", action="version", version=f"meetnote {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_whisper_opts(sp):
        sp.add_argument("--whisper-model", default="large-v3",
                        help=tr("Whisper 模型（默认 large-v3；更快可用 medium）",
                                "Whisper model (default large-v3; medium is faster)",
                                "Whisper-Modell (Standard large-v3; medium ist schneller)"))
        sp.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    def add_transcribe_opts(sp):
        sp.add_argument("audio", nargs="+",
                        help=tr("音频/视频文件（m4a、mp3、wav、mp4 等）。给多个文件或一个文件夹时，先按顺序合并成一个 MP3 再转写"
                                "（文件夹内按文件名自然排序）",
                                "audio/video file (m4a, mp3, wav, mp4, …). Several files or a folder are first joined "
                                "into "
                                "one MP3, in the order given (a folder's files in natural name order), then transcribed",
                                "Audio-/Videodatei (m4a, mp3, wav, mp4, …). Mehrere Dateien oder ein Ordner werden "
                                "zuerst in der angegebenen Reihenfolge (Ordnerinhalt in natürlicher Namensreihenfolge) "
                                "zu einer MP3 zusammengefügt und dann transkribiert"))
        sp.add_argument("-o", "--out", help=tr("输出目录（默认：当前目录下的 <文件名>_纪要/）",
                                               "output folder (default: <file name>_notes/ in the current folder)",
                                               "Ausgabeordner (Standard: <Dateiname>_Protokoll/ im aktuellen Ordner)"))
        add_whisper_opts(sp)
        sp.add_argument("--language", default="auto",
                        help=tr("录音语言代码，如 zh、en、ja（默认 auto 自动检测；中英混杂的会议若检测不准可指定 zh）",
                                "recording language code such as zh, en, ja (default auto: detect; "
                                "force one if detection goes wrong)",
                                "Sprachcode der Aufnahme, z. B. de, en, zh (Standard auto: automatisch erkennen; festlegen, "
                                "falls die Erkennung danebenliegt)"))
        sp.add_argument("--terms", help=tr("术语/人名提示，逗号分隔，提升识别准确率，如 \"Kubernetes,OKR,张伟\"",
                                           "names and terms, comma-separated, to improve recognition, e.g. \"Kubernetes,OKR,Alice "
                                           "Chen\"",
                                           "Namen und Fachbegriffe, durch Kommas getrennt, für eine bessere Erkennung, "
                                           "z. B. \"Kubernetes,OKR,Anna Müller\""))
        sp.add_argument("--speakers", type=int, default=0,
                        help=tr("参会人数（知道就填，说话人分离会准很多；默认自动估计）",
                                "number of participants (give it if known: speaker separation gets much more reliable; "
                                "default: estimate)",
                                "Anzahl der Teilnehmenden (falls bekannt angeben: die Sprechertrennung wird deutlich "
                                "zuverlässiger; Standard: schätzen)"))
        sp.add_argument("--no-diarize", action="store_true", help=tr("不区分说话人", "don't separate speakers",
                                                                               "Sprecher nicht trennen"))
        sp.add_argument("--force", action="store_true", help=tr("已有转写稿时也重新转写", "transcribe again even if a transcript exists",
                                                                               "neu transkribieren, auch wenn ein Transkript "
                                                                               "existiert"))

    def add_backend_opts(sp):
        sp.add_argument("--backend", default="auto", choices=BACKENDS,
                        help=tr("deepseek（需 DEEPSEEK_API_KEY）；openai=任意 OpenAI 兼容服务（需 MEETNOTE_BASE_URL/MEETNOTE_API_KEY）；"
                                "claude-cli；claude-api（需 ANTHROPIC_API_KEY）。auto 按此顺序选第一个已配置的",
                                "deepseek (needs DEEPSEEK_API_KEY); openai = any OpenAI-compatible service "
                                "(needs MEETNOTE_BASE_URL/MEETNOTE_API_KEY); claude-cli; claude-api (needs ANTHROPIC_API_KEY). "
                                "auto picks the first one configured, in this order",
                                "deepseek (braucht DEEPSEEK_API_KEY); openai = beliebiger OpenAI-kompatibler Dienst "
                                "(braucht MEETNOTE_BASE_URL/MEETNOTE_API_KEY); claude-cli; claude-api (braucht ANTHROPIC_API_KEY). "
                                "auto nimmt den ersten konfigurierten in dieser Reihenfolge"))
        sp.add_argument("--model", help=tr("模型名（deepseek 默认 deepseek-chat，可换 deepseek-reasoner；claude-api 默认 claude-opus-5-5）",
                                           "model name (deepseek default deepseek-chat, or deepseek-reasoner; "
                                           "claude-api default claude-opus-5-5)",
                                           "Modellname (deepseek: Standard deepseek-chat, oder deepseek-reasoner; claude-api: "
                                           "Standard claude-opus-5-5)"))

    def add_summary_opts(sp):
        add_backend_opts(sp)
        sp.add_argument("--context", help=tr("会议背景，帮助纠正术语，如 \"产品团队的季度路线讨论\"",
                                             "background about the meeting, to fix terms, e.g. \"Product team quarterly "
                                             "roadmap review\"",
                                             "Hintergrund zur Besprechung, um Begriffe zu korrigieren, z. B. \"Quartalsplanung "
                                             "des Produktteams\""))
        sp.add_argument("--only", default="both", choices=["both", "synopsis", "highlights"],
                        help=tr("只生成全文梗概或两页精华", "generate only the synopsis or only the highlights",
                                                "nur die Zusammenfassung oder nur die Kernpunkte erstellen"))
        sp.add_argument("--summary-lang", default="auto",
                        help=tr("梗概和精华用什么语言写：auto=跟随录音语言（默认），或 zh、en 等语言代码",
                                "language of the notes: auto = same as the recording (default), or a code such as zh, "
                                "en",
                                "Sprache des Protokolls: auto = wie die Aufnahme (Standard), oder ein Code wie de, "
                                "en, zh"))
        sp.add_argument("--pipeline", default="grounded", choices=PIPELINES,
                        help=tr("grounded=逐段抽取→带出处撰写→核查修订（默认，更准，调用次数多）；simple=单次生成（快，易张冠李戴）",
                                "grounded = extract → write with citations → fact-check and revise (default, more accurate, "
                                "more calls); simple = single pass (fast, mixes things up more)",
                                "grounded = extrahieren → mit Belegstellen schreiben → Faktencheck und Überarbeitung "
                                "(Standard, genauer, mehr Aufrufe); simple = ein Durchgang (schnell, verwechselt mehr)"))

    sp_run = sub.add_parser("run", help=tr("完整流程：转写 + 全文梗概 + 两页精华", "full run: transcribe + synopsis + highlights",
                                                                    "kompletter Durchlauf: Transkription + Zusammenfassung "
                                                                    "+ Kernpunkte"))
    add_transcribe_opts(sp_run)
    add_summary_opts(sp_run)

    sp_tr = sub.add_parser("transcribe", help=tr("只转写", "transcribe only", "nur transkribieren"))
    add_transcribe_opts(sp_tr)

    sp_mrg = sub.add_parser("merge", help=tr("把多个录音（如录音笔分段的 WAV）按顺序合并成一个 MP3",
                                             "join several recordings (e.g. a recorder's WAV pieces) into one MP3, "
                                             "in order",
                                             "mehrere Aufnahmen (z. B. die WAV-Teile eines Diktiergeräts) der Reihe "
                                             "nach zu einer MP3 zusammenfügen"))
    sp_mrg.add_argument("audio", nargs="+", help=tr("音频文件，或包含它们的文件夹（按文件名自然排序）",
                                                    "audio files, or a folder of them (natural name order)",
                                                    "Audiodateien oder ein Ordner mit solchen (natürliche Namensreihenfolge)"))
    sp_mrg.add_argument("-o", "--out", help=tr("输出的 MP3 文件（默认：当前目录下的 <第一个文件名>_merged.mp3）",
                                               "output MP3 (default: <first file name>_merged.mp3 in the current folder)",
                                               "Ausgabe-MP3 (Standard: <erster Dateiname>_merged.mp3 im aktuellen Ordner)"))
    sp_mrg.add_argument("--bitrate", type=int, default=128,
                        help=tr("码率 kbps（默认 128；单声道自动减半）", "bitrate in kbps (default 128; halved for mono)",
                                                           "Bitrate in kbps (Standard 128; bei Mono halbiert)"))

    sp_sum = sub.add_parser("summarize", help=tr("对已有转写稿生成梗概和精华", "write the synopsis and highlights for an existing transcript",
                                                                  "Zusammenfassung und Kernpunkte zu einem vorhandenen "
                                                                  "Transkript schreiben"))
    sp_sum.add_argument("target", help=tr("输出目录，或转写稿 .txt 文件", "output folder, or a transcript .txt file",
                                                               "Ausgabeordner oder eine Transkript-.txt-Datei"))
    add_summary_opts(sp_sum)

    sp_chk = sub.add_parser("check", help=tr("评估纪要的忠实度：逐条对照转写稿，统计无依据、对象错误、归属错误等",
                                             "score how faithful the notes are: check each statement against the transcript "
                                             "for unsupported claims, wrong objects, wrong speakers, …",
                                             "bewerten, wie quellentreu das Protokoll ist: jede Aussage wird mit dem "
                                             "Transkript abgeglichen (unbelegte Aussagen, falscher Bezug, falscher "
                                             "Sprecher, …)"))
    sp_chk.add_argument("target", help=tr("纪要所在的输出目录", "output folder with the notes",
                                                       "Ausgabeordner mit dem Protokoll"))
    sp_chk.add_argument("--backend", default="auto", choices=BACKENDS)
    sp_chk.add_argument("--judge-model", help=tr("评审模型（deepseek 后端默认 deepseek-reasoner，与撰写模型不同）",
                                                 "judge model (deepseek backend default: deepseek-reasoner, "
                                                 "different from the writer)",
                                                 "Bewertungsmodell (beim deepseek-Backend standardmäßig deepseek-reasoner, "
                                                 "ein anderes als das schreibende)"))
    sp_chk.add_argument("--docs", nargs="+", default=["highlights", "synopsis"],
                        help=tr("要评估的文档：highlights、synopsis 或文件名", "documents to evaluate: highlights, synopsis, or file names",
                                                                   "zu bewertende Dokumente: highlights, synopsis oder "
                                                                   "Dateinamen"))

    sp_ren = sub.add_parser("rename", help=tr("把转写稿里的说话人标签换成人名，如 S1=张三 S2=李四；之后用 summarize 重新生成纪要",
                                              "replace speaker labels with names, e.g. S1=Alice S2=Bob; then run summarize "
                                              "again",
                                              "Sprecherkennungen durch Namen ersetzen, z. B. S1=Anna S2=Jonas; danach "
                                              "summarize erneut ausführen"))
    sp_ren.add_argument("target", help=tr("输出目录", "output folder", "Ausgabeordner"))
    sp_ren.add_argument("names", nargs="*", help=tr("标签=人名，如 S1=张三；不填则列出现有标签",
                                                    "label=name, e.g. S1=Alice; without any, list the current labels",
                                                    "Kennung=Name, z. B. S1=Anna; ohne Angabe werden die aktuellen "
                                                    "Kennungen aufgelistet"))

    sp_web = sub.add_parser("serve", help=tr("启动网页服务，同事在浏览器里上传录音",
                                             "start the web service: colleagues upload recordings in the browser",
                                             "Webdienst starten: Kolleginnen und Kollegen laden Aufnahmen im Browser "
                                             "hoch"))
    # `or`, not a get() default: an empty `MEETNOTE_PORT=` line in .env should fall back too
    sp_web.add_argument("--host", default=os.environ.get("MEETNOTE_HOST") or "127.0.0.1",
                        help=tr("监听地址（默认 127.0.0.1，配合 tailscale serve；局域网访问用 0.0.0.0。也可在 .env 里设 MEETNOTE_HOST）",
                                "listen address (default 127.0.0.1, for tailscale serve; 0.0.0.0 for LAN access; "
                                "or MEETNOTE_HOST in .env)",
                                "Adresse (Standard 127.0.0.1, für tailscale serve; 0.0.0.0 für Zugriff im lokalen Netz; "
                                "oder MEETNOTE_HOST in .env)"))
    sp_web.add_argument("--port", type=int, default=int(os.environ.get("MEETNOTE_PORT") or DEFAULT_PORT),
                        help=tr(f"端口（默认 {DEFAULT_PORT}，也可在 .env 里设 MEETNOTE_PORT）",
                                f"port (default {DEFAULT_PORT}, or MEETNOTE_PORT in .env)",
                                f"Port (Standard {DEFAULT_PORT}, oder MEETNOTE_PORT in .env)"))
    sp_web.add_argument("--data", default=os.environ.get("MEETNOTE_DATA_DIR") or str(Path.home() / "meetnote-data"),
                        help=tr("任务数据目录（默认 ~/meetnote-data，也可在 .env 里设 MEETNOTE_DATA_DIR）",
                                "job data folder (default ~/meetnote-data, or MEETNOTE_DATA_DIR in .env)",
                                "Datenordner für Aufträge (Standard ~/meetnote-data, oder MEETNOTE_DATA_DIR in .env)"))
    sp_web.add_argument("--keep-days", type=int, default=30,
                        help=tr("任务保留天数，过期自动删除（默认 30）", "days to keep jobs before deleting them (default 30)",
                                                        "Tage, die Aufträge aufbewahrt werden (Standard 30)"))
    sp_web.add_argument("--keep-audio", action="store_true",
                        help=tr("转写后保留原始录音（默认删除，只留文字结果）",
                                "keep the original recording after transcription (default: delete it, keep only text)",
                                "Originalaufnahme nach der Transkription behalten (Standard: löschen, nur Text behalten)"))
    add_whisper_opts(sp_web)
    add_backend_opts(sp_web)
    return p


def main(argv: list[str] | None = None) -> None:
    # Redirected output on Windows defaults to the ANSI code page, which can't encode Chinese
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # Before building the parser: serve's defaults and MEETNOTE_LANG come from .env
    env_file = load_env()
    args = build_parser().parse_args(argv)
    if getattr(args, "language", None) == "auto":
        args.language = None

    if args.cmd == "merge":
        do_merge(args)
    elif args.cmd == "transcribe":
        do_transcribe(args)
    elif args.cmd == "summarize":
        do_summarize(args)
    elif args.cmd == "check":
        from .evaluate import evaluate_dir

        out_dir = Path(args.target).resolve()
        try:
            path = evaluate_dir(out_dir, file_path(out_dir, "transcript"), args.backend, args.judge_model, args.docs)
        except SummarizeError as e:
            sys.exit(tr(f"评估失败：{e}", f"Evaluation failed: {e}", f"Bewertung fehlgeschlagen: {e}"))
        log(f"  → {path}")
    elif args.cmd == "rename":
        from .pipeline import rename_speakers

        out_dir = Path(args.target).resolve()
        names = {}
        for item in args.names:
            old, sep, new = item.partition("=")
            if not sep:
                sys.exit(tr(f"格式应为 标签=人名：{item}", f"Expected label=name: {item}",
                                                  f"Erwartet Kennung=Name: {item}"))
            names[old.strip()] = new.strip()
        try:
            labels = rename_speakers(out_dir, names)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(str(e))
        log(tr(f"说话人：{'、'.join(labels) or '无（转写时没有区分说话人）'}",
               f"Speakers: {', '.join(labels) or 'none (speakers were not separated)'}",
               f"Sprecher: {', '.join(labels) or 'keine (Sprecher wurden nicht getrennt)'}"))
        if names:
            log(tr("已更新转写稿。重新生成纪要：meetnote summarize ", "Transcript updated. Regenerate the notes: meetnote summarize ",
                                                        "Transkript aktualisiert. Protokoll neu erstellen: meetnote "
                                                        "summarize ")
                + str(out_dir))
    elif args.cmd == "serve":
        log(tr(f"配置文件：{env_file or '未找到 .env，只使用环境变量'}",
               f"Config file: {env_file or 'no .env found, using environment variables only'}",
               f"Konfigurationsdatei: {env_file or 'keine .env gefunden, nur Umgebungsvariablen'}"))
        do_serve(args)
    else:
        out_dir = do_transcribe(args)
        do_summarize(args, out_dir)
        log(tr(f"\n完成，结果在：{out_dir}", f"\nDone. Results are in: {out_dir}",
                                      f"\nFertig. Ergebnisse in: {out_dir}"))


if __name__ == "__main__":
    main()
