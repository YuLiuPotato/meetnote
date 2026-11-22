"""meetnote CLI: 会议录音 → 转写稿 + 全文梗概 + 两页精华."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .pipeline import TRANSCRIPT, summarize_dir, transcribe_to_dir
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


def default_out_dir(audio: Path) -> Path:
    return Path.cwd() / f"{audio.stem}_纪要"


def do_transcribe(args) -> Path:
    audio = Path(args.audio).resolve()
    if not audio.is_file():
        sys.exit(f"找不到音频文件：{audio}")
    out_dir = Path(args.out).resolve() if args.out else default_out_dir(audio)
    txt = out_dir / TRANSCRIPT

    if txt.exists() and not args.force:
        log(f"已有转写稿，跳过转写（用 --force 重新转写）：{txt}")
        return out_dir

    transcribe_to_dir(
        audio, out_dir, whisper_model=args.whisper_model, language=args.language,
        device=args.device, terms=args.terms, source_name=str(audio),
        diarization=not args.no_diarize, num_speakers=args.speakers or None,
    )
    log(f"  → {txt}")
    return out_dir


def do_summarize(args, out_dir: Path | None = None) -> None:
    if out_dir is None:
        target = Path(args.target).resolve()
        out_dir = target if target.is_dir() else target.parent
        txt = target / TRANSCRIPT if target.is_dir() else target
    else:
        txt = out_dir / TRANSCRIPT
    if not txt.is_file():
        sys.exit(f"找不到转写稿：{txt}")

    which = ("synopsis", "highlights") if args.only == "both" else (args.only,)
    try:
        summarize_dir(out_dir, txt, backend=args.backend, model=args.model, context=args.context,
                      which=which, summary_lang=args.summary_lang, pipeline=args.pipeline)
    except SummarizeError as e:
        sys.exit(f"总结失败：{e}")


def do_serve(args) -> None:
    from .web import serve

    serve(
        host=args.host, port=args.port, data_dir=Path(args.data).resolve(),
        keep_days=args.keep_days, keep_audio=args.keep_audio,
        whisper_model=args.whisper_model, device=args.device,
        backend=args.backend, model=args.model,
    )


def main(argv: list[str] | None = None) -> None:
    # Redirected output on Windows defaults to the ANSI code page, which can't encode Chinese
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # Before building the parser: serve's defaults come from MEETNOTE_* variables
    env_file = load_env()

    p = argparse.ArgumentParser(
        prog="meetnote",
        description="会议录音 → 转写稿 + 全文梗概 + 两页精华",
    )
    p.add_argument("--version", action="version", version=f"meetnote {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_whisper_opts(sp):
        sp.add_argument("--whisper-model", default="large-v3", help="Whisper 模型（默认 large-v3；更快可用 medium）")
        sp.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    def add_transcribe_opts(sp):
        sp.add_argument("audio", help="音频/视频文件（m4a、mp3、wav、mp4 等）")
        sp.add_argument("-o", "--out", help="输出目录（默认：当前目录下的 <文件名>_纪要/）")
        add_whisper_opts(sp)
        sp.add_argument("--language", default="auto",
                        help="录音语言代码，如 zh、en、ja（默认 auto 自动检测；中英混杂的会议若检测不准可指定 zh）")
        sp.add_argument("--terms", help="术语/人名提示，逗号分隔，提升识别准确率，如 \"Kubernetes,OKR,张伟\"")
        sp.add_argument("--speakers", type=int, default=0,
                        help="参会人数（知道就填，说话人分离会准很多；默认自动估计）")
        sp.add_argument("--no-diarize", action="store_true", help="不区分说话人")
        sp.add_argument("--force", action="store_true", help="已有转写稿时也重新转写")

    def add_backend_opts(sp):
        sp.add_argument("--backend", default="auto", choices=BACKENDS,
                        help="deepseek（需 DEEPSEEK_API_KEY）；openai=任意 OpenAI 兼容服务（需 MEETNOTE_BASE_URL/MEETNOTE_API_KEY）；"
                             "claude-cli；claude-api（需 ANTHROPIC_API_KEY）。auto 按此顺序选第一个已配置的")
        sp.add_argument("--model", help="模型名（deepseek 默认 deepseek-chat，可换 deepseek-reasoner；claude-api 默认 claude-opus-5-5）")

    def add_summary_opts(sp):
        add_backend_opts(sp)
        sp.add_argument("--context", help="会议背景，帮助纠正术语，如 \"产品团队的季度路线讨论\"")
        sp.add_argument("--only", default="both", choices=["both", "synopsis", "highlights"],
                        help="只生成全文梗概或两页精华")
        sp.add_argument("--summary-lang", default="auto",
                        help="梗概和精华用什么语言写：auto=跟随录音语言（默认），或 zh、en 等语言代码")
        sp.add_argument("--pipeline", default="grounded", choices=PIPELINES,
                        help="grounded=逐段抽取→带出处撰写→核查修订（默认，更准，调用次数多）；simple=单次生成（快，易张冠李戴）")

    sp_run = sub.add_parser("run", help="完整流程：转写 + 全文梗概 + 两页精华")
    add_transcribe_opts(sp_run)
    add_summary_opts(sp_run)

    sp_tr = sub.add_parser("transcribe", help="只转写")
    add_transcribe_opts(sp_tr)

    sp_sum = sub.add_parser("summarize", help="对已有转写稿生成梗概和精华")
    sp_sum.add_argument("target", help="输出目录，或转写稿 .txt 文件")
    add_summary_opts(sp_sum)

    sp_chk = sub.add_parser("check", help="评估纪要的忠实度：逐条对照转写稿，统计无依据、对象错误、归属错误等")
    sp_chk.add_argument("target", help="纪要所在的输出目录")
    sp_chk.add_argument("--backend", default="auto", choices=BACKENDS)
    sp_chk.add_argument("--judge-model", help="评审模型（deepseek 后端默认 deepseek-reasoner，与撰写模型不同）")
    sp_chk.add_argument("--docs", nargs="+", default=["两页精华.md", "全文梗概.md"], help="要评估的文件名")

    sp_ren = sub.add_parser("rename", help="把转写稿里的说话人标签换成人名，如 S1=张三 S2=李四；之后用 summarize 重新生成纪要")
    sp_ren.add_argument("target", help="输出目录")
    sp_ren.add_argument("names", nargs="*", help="标签=人名，如 S1=张三；不填则列出现有标签")

    sp_web = sub.add_parser("serve", help="启动网页服务，同事在浏览器里上传录音")
    # `or`, not a get() default: an empty `MEETNOTE_PORT=` line in .env should fall back too
    sp_web.add_argument("--host", default=os.environ.get("MEETNOTE_HOST") or "127.0.0.1",
                        help="监听地址（默认 127.0.0.1，配合 tailscale serve；局域网访问用 0.0.0.0。也可在 .env 里设 MEETNOTE_HOST）")
    sp_web.add_argument("--port", type=int, default=int(os.environ.get("MEETNOTE_PORT") or DEFAULT_PORT),
                        help=f"端口（默认 {DEFAULT_PORT}，也可在 .env 里设 MEETNOTE_PORT）")
    sp_web.add_argument("--data", default=os.environ.get("MEETNOTE_DATA_DIR") or str(Path.home() / "meetnote-data"),
                        help="任务数据目录（默认 ~/meetnote-data，也可在 .env 里设 MEETNOTE_DATA_DIR）")
    sp_web.add_argument("--keep-days", type=int, default=30, help="任务保留天数，过期自动删除（默认 30）")
    sp_web.add_argument("--keep-audio", action="store_true", help="转写后保留原始录音（默认删除，只留文字结果）")
    add_whisper_opts(sp_web)
    add_backend_opts(sp_web)

    args = p.parse_args(argv)
    if getattr(args, "language", None) == "auto":
        args.language = None

    if args.cmd == "transcribe":
        do_transcribe(args)
    elif args.cmd == "summarize":
        do_summarize(args)
    elif args.cmd == "check":
        from .evaluate import evaluate_dir

        out_dir = Path(args.target).resolve()
        try:
            path = evaluate_dir(out_dir, out_dir / TRANSCRIPT, args.backend, args.judge_model, args.docs)
        except SummarizeError as e:
            sys.exit(f"评估失败：{e}")
        log(f"  → {path}")
    elif args.cmd == "rename":
        from .pipeline import rename_speakers

        out_dir = Path(args.target).resolve()
        names = {}
        for item in args.names:
            old, sep, new = item.partition("=")
            if not sep:
                sys.exit(f"格式应为 标签=人名：{item}")
            names[old.strip()] = new.strip()
        try:
            labels = rename_speakers(out_dir, names)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(str(e))
        log(f"说话人：{'、'.join(labels) or '无（转写时没有区分说话人）'}")
        if names:
            log("已更新转写稿。重新生成纪要：meetnote summarize " + str(out_dir))
    elif args.cmd == "serve":
        log(f"配置文件：{env_file or '未找到 .env，只使用环境变量'}")
        do_serve(args)
    else:
        out_dir = do_transcribe(args)
        do_summarize(args, out_dir)
        log(f"\n完成，结果在：{out_dir}")


if __name__ == "__main__":
    main()
