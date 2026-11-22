"""Claim-level faithfulness check of a summary against its transcript.

Follows the FineSurE idea (Song et al., ACL 2024): split the summary into statements,
have a separate model label each one against the source, and report the share that is
supported, with a breakdown by error type (object / attribution / direction-number /
over-generalization), the error classes that matter for meeting notes.
"""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .summarize import (DEEPSEEK_MODEL, SummarizeError, _prompt, build_user_message, check_document,
                        get_runner, parse_json)
from .i18n import FILE_NAMES, file_path, find_file, show_label as show, tr
from .transcribe import log

# The judge and checker prompts answer with these Chinese labels; they are translated only for display
LABELS = ("支持", "无依据", "对象错误", "归属错误", "方向或数字错误", "过度概括", "不适用")


def split_claims(markdown: str) -> list[str]:
    """One statement per bullet, table row or paragraph line; headings and rules are dropped."""
    claims = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or re.fullmatch(r"[-|:\s]+", line) or line.startswith("```"):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            line = " ｜ ".join(c for c in cells if c)
        line = re.sub(r"^([-*>]|\d+\.)\s*", "", line).strip()
        if len(re.sub(r"\W", "", line)) >= 6:
            claims.append(line)
    return claims


BATCH = 10  # statements per judge call: graded all at once, a judge waves most of them through


def _judge_batch(runner, model, transcript_msg: str, batch: list[tuple[int, str]]) -> dict[int, dict]:
    numbered = "\n".join(f"{i}. {c}" for i, c in batch)
    user = f"{transcript_msg}\n\n## 待评审的陈述（共 {len(batch)} 条）\n{numbered}"
    try:
        raw = runner(_prompt("judge.md"), user, model, temperature=0.0)
    except SummarizeError as e:
        log(tr(f"  警告：第 {batch[0][0]}–{batch[-1][0]} 条评审失败（{e}），记为未评审",
               f"  Warning: judging statements {batch[0][0]}–{batch[-1][0]} failed ({e}); counted as not judged"))
        return {}
    try:
        data = parse_json(raw)
    except ValueError:
        log(tr(f"  警告：第 {batch[0][0]}–{batch[-1][0]} 条的评审结果不是合法 JSON，记为未评审",
               f"  Warning: the verdict for statements {batch[0][0]}–{batch[-1][0]} is not valid JSON; counted as not judged"))
        return {}
    items = data.get("labels", []) if isinstance(data, dict) else []
    return {int(x["id"]): x for x in items if isinstance(x, dict) and str(x.get("id", "")).isdigit()}


def judge(doc_path: Path, transcript: str, backend: str, model: str | None) -> dict:
    claims = split_claims(doc_path.read_text(encoding="utf-8"))
    if not claims:
        raise SummarizeError(tr(f"{doc_path.name} 里没有可评估的内容", f"Nothing to evaluate in {doc_path.name}"))
    _, runner = get_runner(backend)
    transcript_msg = build_user_message(transcript, None, "")
    numbered = list(enumerate(claims, 1))
    batches = [numbered[i:i + BATCH] for i in range(0, len(numbered), BATCH)]
    labels: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for part in pool.map(lambda b: _judge_batch(runner, model, transcript_msg, b), batches):
            labels.update(part)

    rows, counts = [], Counter()
    for i, claim in enumerate(claims, 1):
        item = labels.get(i, {"label": "未评审"})
        label = item.get("label", "未评审")
        # A "supported" verdict without quoted evidence is not evidence of support
        if label == "支持" and not str(item.get("evidence", "")).strip():
            label = "无依据"
            item["note"] = tr("评审判为支持，但没有给出原文依据", "judged supported, but no transcript quote was given")
        counts[label] += 1
        rows.append((i, claim, label, " ".join(filter(None, [item.get("evidence", ""), item.get("note", "")]))))
    checkable = sum(n for label, n in counts.items() if label not in ("不适用", "未评审"))
    score = counts["支持"] / checkable if checkable else 0.0

    # Second, more discriminating measure: the same model hunts for errors section by section
    document = doc_path.read_text(encoding="utf-8")
    issues = check_document(runner, model, document, transcript_msg)
    issue_types = Counter(i.get("type", "?") for i in issues)
    return {"doc": doc_path.name, "score": score, "checkable": checkable, "counts": counts, "rows": rows,
            "claims": len(claims), "issues": issues, "issue_types": issue_types,
            "per_kchar": len(issues) / max(1, len(document)) * 1000}


ISSUE_TYPES = ("无依据", "对象错误", "归属错误", "方向或数字错误", "过度概括", "时间戳错误")


def report(results: list[dict], judge_model: str) -> str:
    out = [tr(f"# 忠实度评估\n\n评审模型：{judge_model}。两种方法：\n\n"
              "1. **问题密度**：评审模型逐节查找错误，统计每千字的问题数（越低越好；区分度更高）。\n"
              "2. **陈述忠实度**：把纪要拆成单条陈述，每批 10 条逐条判断能否被转写稿支持，"
              "判“支持”必须给出原文（参考 FineSurE）。LLM 判“支持”偏宽松，只适合横向比较。\n",
              f"# Faithfulness evaluation\n\nJudge model: {judge_model}. Two measures:\n\n"
              "1. **Problem density**: the judge searches each section for errors; problems per 1,000 characters "
              "(lower is better; discriminates better).\n"
              "2. **Statement faithfulness**: the notes are split into single statements and judged in batches of 10 "
              "for support by the transcript; \"supported\" requires a quoted line (after FineSurE). LLM judges are "
              "lenient here, so use it to compare versions only.\n")]
    out.append(tr("## 问题密度\n", "## Problem density\n"))
    out.append(tr("| 文档 | 每千字问题数 | 问题总数 | ", "| Document | Problems / 1k chars | Problems | ")
               + " | ".join(show(k) for k in ISSUE_TYPES) + " |")
    out.append("|---|---|---|" + "---|" * len(ISSUE_TYPES))
    for r in results:
        t = r["issue_types"]
        out.append(f"| {r['doc']} | **{r['per_kchar']:.1f}** | {len(r['issues'])} | " + " | ".join(str(t[k]) for k in ISSUE_TYPES) + " |")
    out.append(tr("\n## 陈述忠实度\n", "\n## Statement faithfulness\n"))
    out.append(tr("| 文档 | 忠实度 | 陈述数 | 可核查 | 未评审 | ", "| Document | Faithfulness | Statements | Checkable | Not judged | ")
               + " | ".join(show(l) for l in LABELS[1:-1]) + " |")
    out.append("|---|---|---|---|---|" + "---|" * (len(LABELS) - 2))
    for r in results:
        c = r["counts"]
        out.append(f"| {r['doc']} | **{r['score']:.0%}** | {r['claims']} | {r['checkable']} | {c['未评审']} | "
                   + " | ".join(str(c[l]) for l in LABELS[1:-1]) + " |")
    evidence = tr("依据", "Evidence")
    for r in results:
        out.append(tr(f"\n## {r['doc']}：找到的问题（{len(r['issues'])}）\n", f"\n## {r['doc']}: problems found ({len(r['issues'])})\n"))
        out += [f"- **{show(i.get('type', '?'))}**: {i.get('text')}\n  - {evidence}: {i.get('evidence', '')}"
                for i in r["issues"]] or [tr("无", "None")]
    return "\n".join(out) + "\n"


def evaluate_dir(out_dir: Path, transcript: Path, backend: str, judge_model: str | None, docs: list[str]) -> Path:
    """`docs`: file names, or the keys "highlights" / "synopsis" (found in either language)."""
    backend_name, _ = get_runner(backend)
    # Judge with a different (reasoning) model than the writer, to avoid grading its own homework
    judge_model = judge_model or ("deepseek-reasoner" if backend_name == "deepseek" else None)
    text = transcript.read_text(encoding="utf-8")
    results = []
    for name in docs:
        path = find_file(out_dir, name) if name in FILE_NAMES else out_dir / name
        if not path or not path.exists():
            continue
        log(tr(f"评审《{path.stem}》（{judge_model or backend_name}，每批 {BATCH} 条）…",
               f"Judging {path.name} ({judge_model or backend_name}, {BATCH} statements per batch)…"))
        try:
            r = judge(path, text, backend, judge_model)
        except SummarizeError as e:
            log(tr(f"  评审失败，跳过：{e}", f"  Judging failed, skipped: {e}"))
            continue
        c = r["counts"]
        log(tr(f"  每千字 {r['per_kchar']:.1f} 个问题（共 {len(r['issues'])}）；"
               f"陈述忠实度 {r['score']:.0%}（{c['支持']}/{r['checkable']}，未评审 {c['未评审']}）",
               f"  {r['per_kchar']:.1f} problems per 1k chars ({len(r['issues'])} total); "
               f"statement faithfulness {r['score']:.0%} ({c['支持']}/{r['checkable']}, {c['未评审']} not judged)"))
        results.append(r)
    if not results:
        raise SummarizeError(tr(f"{out_dir} 里没有可评估的纪要", f"No notes to evaluate in {out_dir}"))
    out = file_path(out_dir, "faithfulness")
    out.write_text(report(results, judge_model or backend_name or DEEPSEEK_MODEL), encoding="utf-8")
    return out
