"""Interface language: messages, --help text, reports and output file names.

Set MEETNOTE_LANG=zh or en (in .env or the environment). Unset, it follows the system's
display language: Chinese systems get Chinese, everything else English.

This is separate from the *notes* language (--summary-lang), which follows the recording.
Text sent to the LLM stays in Chinese either way: the prompts are tuned in Chinese and
the output language is set explicitly.
"""

from __future__ import annotations

import locale
import os
from pathlib import Path

_cache: dict[str, str] = {}


def _system_language() -> str:
    if os.name == "nt":
        try:
            import ctypes

            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            return "zh" if (lang_id & 0x3FF) == 0x04 else "en"  # primary language 0x04 = Chinese
        except (AttributeError, OSError):
            pass
    name = locale.getlocale()[0] or os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
    return "zh" if name.lower().startswith(("zh", "chinese")) else "en"


def lang() -> str:
    """'zh' or 'en'. Re-read each time, so a MEETNOTE_LANG loaded from .env later still applies."""
    setting = (os.environ.get("MEETNOTE_LANG") or "").strip().lower()
    if setting.startswith("zh"):
        return "zh"
    if setting.startswith("en"):
        return "en"
    if "system" not in _cache:
        _cache["system"] = _system_language()
    return _cache["system"]


def tr(zh: str, en: str) -> str:
    """Pick the Chinese or English version of a message."""
    return zh if lang() == "zh" else en


# Problem and verdict labels: the checker and judge prompts answer in Chinese
LABEL_EN = {
    "支持": "supported", "无依据": "unsupported", "对象错误": "wrong object", "归属错误": "wrong speaker",
    "方向或数字错误": "wrong direction/number", "过度概括": "over-generalized", "不适用": "n/a",
    "未评审": "not judged", "时间戳错误": "wrong timestamp",
}


def show_label(label: str) -> str:
    return label if lang() == "zh" else LABEL_EN.get(label, label)


# key → (Chinese name, English name)
FILE_NAMES = {
    "transcript": ("转写稿.txt", "transcript.txt"),
    "subtitles": ("转写稿.srt", "transcript.srt"),
    "notes": ("要点笔记.md", "notes.md"),
    "synopsis": ("全文梗概.md", "synopsis.md"),
    "highlights": ("两页精华.md", "highlights.md"),
    "checks": ("核查记录.md", "fact-check.md"),
    "faithfulness": ("忠实度评估.md", "faithfulness.md"),
}
DOC_TITLES = {"synopsis": ("全文梗概", "Synopsis"), "highlights": ("两页精华", "Highlights")}
DIR_SUFFIXES = ("_纪要", "_notes")


def find_file(out_dir: Path, key: str) -> Path | None:
    """The existing file for `key` in either language, or None."""
    for name in FILE_NAMES[key]:
        if (out_dir / name).exists():
            return out_dir / name
    return None


def _folder_language(out_dir: Path) -> str | None:
    """The language a folder's existing files are named in, if any."""
    for zh, en in FILE_NAMES.values():
        if (out_dir / zh).exists():
            return "zh"
        if (out_dir / en).exists():
            return "en"
    return None


def file_path(out_dir: Path, key: str) -> Path:
    """Where to read or write `key`. An existing file in either language is reused, and a new
    file follows the language of the folder's other files, so a folder never mixes names;
    an empty folder gets the current interface language."""
    existing = find_file(out_dir, key)
    if existing:
        return existing
    zh, en = FILE_NAMES[key]
    return out_dir / (zh if (_folder_language(out_dir) or lang()) == "zh" else en)


def doc_title(key: str) -> str:
    zh, en = DOC_TITLES[key]
    return tr(zh, en)


def default_out_dir(base: Path, stem: str) -> Path:
    """<stem>_纪要 or <stem>_notes; an existing folder of either name is reused."""
    for suffix in DIR_SUFFIXES:
        if (base / f"{stem}{suffix}").is_dir():
            return base / f"{stem}{suffix}"
    return base / f"{stem}{tr(*DIR_SUFFIXES)}"
