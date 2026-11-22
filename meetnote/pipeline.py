"""Directory-level steps shared by the CLI and the web server.

A meeting directory holds 转写稿.txt / 转写稿.srt / meta.json, and the summaries
written next to them.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Callable

from .summarize import summarize
from .transcribe import fmt_ts, language_name, transcribe, write_transcript

TRANSCRIPT = "转写稿.txt"
SUBTITLES = "转写稿.srt"
META = "meta.json"


def transcribe_to_dir(
    audio: Path,
    out_dir: Path,
    whisper_model: str = "large-v3",
    language: str | None = "zh",
    device: str = "auto",
    terms: str | None = None,
    source_name: str | None = None,
    on_progress: Callable[[float], None] | None = None,
    diarization: bool = True,
    num_speakers: int | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    segments, duration, detected = transcribe(
        audio, model_name=whisper_model, language=language,
        device=device, terms=terms, on_progress=on_progress,
        diarization=diarization, num_speakers=num_speakers,
    )
    write_transcript(segments, out_dir / TRANSCRIPT, out_dir / SUBTITLES)
    meta = {
        "source": source_name or audio.name,
        "duration": duration,
        "language": detected,
        "file_date": dt.datetime.fromtimestamp(audio.stat().st_mtime).strftime("%Y-%m-%d"),
        "whisper_model": whisper_model,
        # Labels as they currently appear in the transcript (S1, S2, … or names given later)
        "speakers": sorted({s.speaker for s in segments if s.speaker}, key=_label_order),
    }
    (out_dir / META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _label_order(label: str):
    m = re.fullmatch(r"S(\d+)", label)
    return (0, int(m.group(1)), "") if m else (1, 0, label)


def rename_speakers(out_dir: Path, names: dict[str, str]) -> list[str]:
    """Replace speaker labels in the transcript and subtitles, e.g. {"S1": "张三"}.

    Keys are the labels currently in the files (see meta.json "speakers"); all labels are
    swapped in one pass, so {"S1": "S2", "S2": "S1"} works. Returns the new label list.
    """
    meta_path = out_dir / META
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    current = meta.get("speakers") or []
    mapping = {}
    for old, new in names.items():
        new = re.sub(r"[\s:：]+", " ", str(new)).strip()[:40]
        if old not in current:
            raise ValueError(f"转写稿里没有说话人 {old}（现有：{', '.join(current) or '无'}）")
        if new and new != old:
            mapping[old] = new
    if not mapping:
        return current
    labels = sorted(current, key=len, reverse=True)
    alt = "|".join(re.escape(x) for x in labels)
    patterns = {
        TRANSCRIPT: re.compile(rf"^(\[[\d:]+\] )({alt})(: )", re.M),
        SUBTITLES: re.compile(rf"^()({alt})(: )", re.M),
    }
    for name, pattern in patterns.items():
        path = out_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            text = pattern.sub(lambda m: m.group(1) + mapping.get(m.group(2), m.group(2)) + m.group(3), text)
            path.write_text(text, encoding="utf-8")
    new_labels = [mapping.get(x, x) for x in current]
    if len(set(new_labels)) < len(new_labels):
        # Two labels given the same name: they were one person split by the diarizer; fine
        new_labels = list(dict.fromkeys(new_labels))
    meta["speakers"] = new_labels
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return new_labels


def summarize_dir(
    out_dir: Path,
    transcript: Path | None = None,
    backend: str = "auto",
    model: str | None = None,
    context: str | None = None,
    which: tuple[str, ...] = ("synopsis", "highlights"),
    on_step: Callable[[str], None] | None = None,
    summary_lang: str = "auto",
    pipeline: str = "grounded",
) -> list[Path]:
    """`summary_lang`: "auto" follows the recording's language; otherwise a code ("zh", "en")
    or a language name the model understands."""
    txt = transcript or out_dir / TRANSCRIPT
    meta_path = out_dir / META
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    source_lang = meta.get("language")
    # Transcripts from before language detection have no language recorded; they were all Chinese
    target = (source_lang or "zh") if summary_lang in ("", "auto", None) else summary_lang

    lines = []
    if source_lang:
        lines.append(f"录音语言：{language_name(source_lang)}")
    if meta.get("duration"):
        lines.append(f"时长：{fmt_ts(meta['duration'])}")
    if meta.get("file_date"):
        lines.append(f"录音文件日期：{meta['file_date']}")
    if meta.get("speakers"):
        lines.append(f"说话人（自动区分，可能已改成姓名）：{'、'.join(meta['speakers'])}")
    if meta.get("source"):
        lines.append(f"文件名：{Path(meta['source']).name}")
    return summarize(
        txt, out_dir, backend=backend, model=model, context=context,
        meta="\n".join(lines) or "未知", which=which, on_step=on_step,
        output_language=language_name(target), pipeline=pipeline,
    )
