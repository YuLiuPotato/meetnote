"""Transcript → synopsis / highlights, via DeepSeek (or any OpenAI-compatible API) or Claude.

The default "grounded" pipeline exists because a single pass over a long transcript
mixes things up: opinions about two products discussed one after another end up on
the wrong product, and lines get pinned on the wrong person. So it

  1. extracts timestamped notes chunk by chunk, naming the object of every opinion,
  2. writes each document from those notes, citing [mm:ss] for every claim,
  3. has a separate call check every claim against the transcript, and
  4. revises only what the check flagged.

"simple" is the original single call per document, kept for comparison.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .i18n import doc_title, file_path, show_label, tr
from .transcribe import log

PROMPTS_DIR = Path(__file__).parent / "prompts"
API_MODEL = "claude-opus-5-5"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

BACKENDS = ("auto", "deepseek", "openai", "claude-cli", "claude-api")
PIPELINES = ("grounded", "simple")

# document → prompt file (output file names: see i18n.FILE_NAMES)
PROMPT_FILES = {"synopsis": "synopsis.md", "highlights": "highlights.md"}
DOC_TITLES_ZH = {"synopsis": "全文梗概", "highlights": "两页精华"}

# Temperatures for OpenAI-compatible backends (DeepSeek defaults to 1.0, which drifts).
# Claude backends ignore these.
T_EXTRACT, T_WRITE, T_CHECK = 0.2, 0.5, 0.0
CHUNK_CHARS = 6000      # ≈ 10 min of Chinese speech per extraction call
LOOKBACK_CHARS = 1500   # preceding transcript shown for resolving "this one" / "the old one"
CHECK_CHARS = 2500      # document text per verification call


class SummarizeError(RuntimeError):
    pass


def language_rule(output_language: str) -> str:
    """Appended to every prompt; the prompt files themselves stay language-neutral."""
    return (
        "\n\n## 输出语言\n\n"
        f"**整份文档必须使用 {output_language} 书写**，包括标题、章节名、表格表头、标签和附注；"
        "上面结构模板里的中文标题只是示意，输出时要译成目标语言。"
        "不论转写稿是什么语言都照此执行。人名、机构名、产品名等专有名词保留原文写法；引用原话时保留原文语言。\n"
        f"The entire document MUST be written in {output_language}, including all headings and labels.\n"
    )


def build_user_message(transcript: str, context: str | None, meta: str) -> str:
    parts = [f"## 录音信息\n{meta}"]
    if context:
        parts.append(f"## 会议背景（由用户提供）\n{context}")
    parts.append(f"## 转写稿\n<transcript>\n{transcript}\n</transcript>")
    return "\n\n".join(parts)


def _prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- backends

def find_claude_cli() -> str | None:
    override = os.environ.get("MEETNOTE_CLAUDE")
    # str(Path(...)) normalizes forward slashes, which cmd.exe rejects for .cmd paths
    return str(Path(override)) if override else shutil.which("claude")


def has_api_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("MEETNOTE_API_KEY") and os.environ.get("MEETNOTE_BASE_URL"):
        return "openai"
    if find_claude_cli():
        return "claude-cli"
    if has_api_credentials():
        return "claude-api"
    raise SummarizeError(tr(
        "没有配置可用的大模型，在 meetnote\\.env 里任选其一填写：\n"
        "  - DeepSeek：DEEPSEEK_API_KEY（推荐）\n"
        "  - 其他 OpenAI 兼容服务：MEETNOTE_BASE_URL、MEETNOTE_API_KEY、MEETNOTE_MODEL\n"
        "  - Claude：ANTHROPIC_API_KEY，或登录 Claude Code CLI\n"
        "转写稿已保存，配置好之后用 `meetnote summarize <输出目录>` 继续。",
        "No LLM is configured. Set one of these in meetnote\\.env:\n"
        "  - DeepSeek: DEEPSEEK_API_KEY (recommended)\n"
        "  - Any OpenAI-compatible service: MEETNOTE_BASE_URL, MEETNOTE_API_KEY, MEETNOTE_MODEL\n"
        "  - Claude: ANTHROPIC_API_KEY, or a logged-in Claude Code CLI\n"
        "The transcript is saved; continue with `meetnote summarize <output folder>` once configured.",
    ))


def _run_openai_compatible(system: str, user: str, model: str, base_url: str, api_key: str,
                           temperature: float | None) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=900)
    kwargs = {}
    if "reasoner" in model:
        max_tokens = 64000  # reasoning models count their chain of thought against max_tokens
    else:
        max_tokens = 8192
        if temperature is not None:
            kwargs["temperature"] = temperature
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
    except openai.AuthenticationError as e:
        raise SummarizeError(tr(f"API key 无效（{base_url}）：{e}", f"Invalid API key ({base_url}): {e}")) from e
    except openai.NotFoundError as e:
        raise SummarizeError(tr(f"模型 {model} 不存在，可用 --model 换一个：{e}",
                                f"Model {model} not found; pick another with --model: {e}")) from e
    except openai.APIError as e:
        raise SummarizeError(tr(f"调用 {base_url} 失败：{e}", f"Request to {base_url} failed: {e}")) from e
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        log(tr("警告：输出达到长度上限，结果可能不完整", "Warning: output hit the length limit and may be incomplete"))
    text = (choice.message.content or "").strip()
    if not text:
        raise SummarizeError(tr(f"API 返回为空（finish_reason={choice.finish_reason}）",
                                f"Empty API response (finish_reason={choice.finish_reason})"))
    return text


def _run_deepseek(system: str, user: str, model: str | None, temperature: float | None = None) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SummarizeError(tr("没有配置 DEEPSEEK_API_KEY（填在 meetnote\\.env 里）",
                                "DEEPSEEK_API_KEY is not set (put it in meetnote\\.env)"))
    return _run_openai_compatible(system, user, model or DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, key, temperature)


def _run_openai(system: str, user: str, model: str | None, temperature: float | None = None) -> str:
    base_url, key = os.environ.get("MEETNOTE_BASE_URL"), os.environ.get("MEETNOTE_API_KEY")
    model = model or os.environ.get("MEETNOTE_MODEL")
    if not (base_url and key and model):
        raise SummarizeError(tr("openai 后端需要 MEETNOTE_BASE_URL、MEETNOTE_API_KEY，以及 --model 或 MEETNOTE_MODEL",
                                "The openai backend needs MEETNOTE_BASE_URL, MEETNOTE_API_KEY, and --model or MEETNOTE_MODEL"))
    return _run_openai_compatible(system, user, model, base_url, key, temperature)


def _run_cli(system: str, user: str, model: str | None, temperature: float | None = None) -> str:
    exe = find_claude_cli()
    if not exe:
        raise SummarizeError(tr("找不到 claude 命令（可用 MEETNOTE_CLAUDE 环境变量指定路径）",
                                "claude command not found (set its path with MEETNOTE_CLAUDE)"))
    # The system prompt goes through a file: a multi-line argument would be mangled
    # when `claude` is an npm-installed claude.cmd run via cmd.exe.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as f:
        f.write(system)
        system_file = f.name
    cmd = [
        exe, "-p", "请按系统提示处理标准输入中的内容，只输出系统提示要求的最终结果。",
        "--system-prompt-file", system_file,
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format", "text",
    ]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8")
    finally:
        os.unlink(system_file)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SummarizeError(tr(f"claude CLI 调用失败（exit {proc.returncode}）：{detail}",
                                f"claude CLI failed (exit {proc.returncode}): {detail}"))
    return out


def _run_api(system: str, user: str, model: str | None, temperature: float | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=model or API_MODEL,
        max_tokens=32000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "high"},
        # On a safety refusal, the API re-runs the request on a suitable fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise SummarizeError(tr(f"模型拒绝了请求：{getattr(msg, 'stop_details', None)}",
                                f"The model refused the request: {getattr(msg, 'stop_details', None)}"))
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if msg.stop_reason == "max_tokens":
        log(tr("警告：输出达到 max_tokens 上限，结果可能不完整", "Warning: output hit max_tokens and may be incomplete"))
    if not text:
        raise SummarizeError(tr(f"API 返回为空（stop_reason={msg.stop_reason}）",
                                f"Empty API response (stop_reason={msg.stop_reason})"))
    return text


RUNNERS = {"deepseek": _run_deepseek, "openai": _run_openai, "claude-cli": _run_cli, "claude-api": _run_api}


def get_runner(backend: str) -> tuple[str, Callable[..., str]]:
    backend = resolve_backend(backend)
    return backend, RUNNERS[backend]


def parse_json(text: str):
    """Lenient JSON parse: models wrap JSON in ``` fences or add a sentence around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ---------------------------------------------------------------- grounded pipeline

def split_transcript(transcript: str, size: int = CHUNK_CHARS) -> list[tuple[str, str]]:
    """Split on line boundaries into (lookback, chunk) pairs."""
    lines = transcript.splitlines()
    chunks, current, length = [], [], 0
    for line in lines:
        if current and length + len(line) > size:
            chunks.append(current)
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append(current)

    pairs, before = [], ""
    for chunk in chunks:
        pairs.append((before[-LOOKBACK_CHARS:], "\n".join(chunk)))
        before += "\n".join(chunk) + "\n"
    return pairs


def extract_notes(runner, model, transcript: str, context: str | None, meta: str, output_language: str) -> str:
    system = _prompt("extract.md") + language_rule(output_language)
    pairs = split_transcript(transcript)

    def one(i: int) -> str:
        lookback, chunk = pairs[i]
        parts = [f"## 录音信息\n{meta}"]
        if context:
            parts.append(f"## 会议背景（由用户提供）\n{context}")
        if lookback:
            parts.append(f"## 前文（只用于理解指代和当前讨论对象，不要为它写笔记）\n<before>\n{lookback}\n</before>")
        parts.append(f"## 需要整理的片段（第 {i + 1}/{len(pairs)} 段）\n<chunk>\n{chunk}\n</chunk>")
        return runner(system, "\n\n".join(parts), model, temperature=T_EXTRACT)

    log(tr(f"  逐段抽取要点（{len(pairs)} 段）…", f"  Extracting notes ({len(pairs)} sections)…"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        notes = list(pool.map(one, range(len(pairs))))

    def span(chunk: str) -> str:
        stamps = re.findall(r"^\[([\d:]+)\]", chunk, re.MULTILINE)
        return f"{stamps[0]}–{stamps[-1]}" if stamps else ""

    return "\n\n".join(f"## 片段 {i + 1}（{span(pairs[i][1])}）\n\n{n}" for i, n in enumerate(notes))


def split_sections(document: str, size: int = CHECK_CHARS) -> list[str]:
    """Cut at headings into pieces of about `size` characters."""
    pieces, current = [], ""
    for block in re.split(r"(?m)^(?=#{1,3} )", document):
        if current and len(current) + len(block) > size:
            pieces.append(current)
            current = ""
        current += block
    if current.strip():
        pieces.append(current)
    return pieces


def check_document(runner, model, document: str, base_user: str) -> list[dict]:
    """Have separate calls list unsupported or misattributed claims, a few sections at a
    time: shown a whole long document at once, a checker passes nearly everything."""
    def one(piece: str) -> list[dict]:
        user = f"{base_user}\n\n## 待核查的纪要片段\n<document>\n{piece}\n</document>"
        try:
            raw = runner(_prompt("verify.md"), user, model, temperature=T_CHECK)
        except SummarizeError as e:
            log(tr(f"  警告：有一段核查失败（{e}），已跳过", f"  Warning: one check failed ({e}); skipped"))
            return []
        try:
            data = parse_json(raw)
        except ValueError:
            log(tr("  警告：有一段核查结果不是合法 JSON，已跳过", "  Warning: one check returned invalid JSON; skipped"))
            return []
        issues = data.get("issues", []) if isinstance(data, dict) else data
        return [i for i in issues if isinstance(i, dict) and i.get("text")]

    pieces = split_sections(document)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return [issue for part in pool.map(one, pieces) for issue in part]


def revise_document(runner, model, system: str, user: str, draft: str, issues: list[dict]) -> str:
    revise_user = (
        f"{user}\n\n## 初稿\n<draft>\n{draft}\n</draft>\n\n"
        f"## 核查员发现的问题\n```json\n{json.dumps(issues, ensure_ascii=False, indent=1)}\n```\n\n"
        "## 任务\n根据核查意见修订初稿：只改有问题的地方，其他内容保持不变；"
        "若核查意见与转写稿不符，以转写稿为准。只输出修订后的完整文档。"
    )
    return runner(system, revise_user, model, temperature=T_WRITE)


def _format_issues(title: str, issues: list[dict]) -> str:
    if not issues:
        return f"## {title}\n\n{tr('未发现问题。', 'No problems found.')}\n"
    rows = [tr(f"## {title}（{len(issues)} 处）\n", f"## {title} ({len(issues)})\n")]
    evidence, fix = tr("依据", "Evidence"), tr("修改", "Fix")
    for i in issues:
        rows.append(f"- **{show_label(i.get('type', '?'))}**: {i.get('text')}\n  - {evidence}: {i.get('evidence', '')}\n"
                    f"  - {fix}: {i.get('fix', '')}")
    return "\n".join(rows) + "\n"


def summarize(
    transcript_path: Path,
    out_dir: Path,
    backend: str = "auto",
    model: str | None = None,
    context: str | None = None,
    meta: str = "",
    which: tuple[str, ...] = ("synopsis", "highlights"),
    on_step: Callable[[str], None] | None = None,
    output_language: str = "简体中文",
    pipeline: str = "grounded",
) -> list[Path]:
    backend, runner = get_runner(backend)
    transcript = transcript_path.read_text(encoding="utf-8")
    base_user = build_user_message(transcript, context, meta)
    faithfulness = _prompt("faithfulness.md")

    def step(zh: str, en: str) -> None:
        log(tr(f"{zh}（{backend}）…", f"{en} ({backend})…"))
        if on_step:
            on_step(zh)  # the web page translates these itself, so they stay in Chinese

    notes = ""
    if pipeline == "grounded":
        step("逐段整理要点", "Extracting notes section by section")
        notes = extract_notes(runner, model, transcript, context, meta, output_language)
        file_path(out_dir, "notes").write_text(notes + "\n", encoding="utf-8")

    written, check_log = [], []
    for name in which:
        title, title_zh = doc_title(name), DOC_TITLES_ZH[name]
        system = _prompt(PROMPT_FILES[name]) + "\n\n" + faithfulness + language_rule(output_language)
        user = base_user
        if notes:
            user += f"\n\n## 分段要点笔记（从转写稿逐段整理，带时间戳；撰写时以此为主要依据）\n<notes>\n{notes}\n</notes>"

        step(f"撰写《{title_zh}》", f"Writing the {title.lower()}")
        text = runner(system, user, model, temperature=T_WRITE)

        if pipeline == "grounded":
            step(f"核查《{title_zh}》", f"Fact-checking the {title.lower()}")
            issues = check_document(runner, model, text, base_user)
            log(tr(f"  发现 {len(issues)} 处问题", f"  Found {len(issues)} problem(s)"))
            check_log.append(_format_issues(title, issues))
            if issues:
                step(f"修订《{title_zh}》", f"Revising the {title.lower()}")
                text = revise_document(runner, model, system, user, text, issues)

        out_path = file_path(out_dir, name)
        out_path.write_text(text + "\n", encoding="utf-8")
        log(f"  → {out_path}")
        written.append(out_path)

    if check_log:
        file_path(out_dir, "checks").write_text(
            tr("# 核查记录\n\n", "# Fact-check log\n\n") + "\n".join(check_log), encoding="utf-8")
    return written
