"""Interface language: messages, --help text, reports and output file names.

Set MEETNOTE_LANG=zh, en or de (in .env or the environment). Unset, it follows the system's
display language: Chinese and German systems get Chinese or German, everything else English.

This is separate from the *notes* language (--summary-lang), which follows the recording.
Text sent to the LLM stays in Chinese either way: the prompts are tuned in Chinese and
the output language is set explicitly.
"""

from __future__ import annotations

import locale
import os
from pathlib import Path

LANGS = ("zh", "en", "de")
_cache: dict[str, str] = {}


def _system_language() -> str:
    if os.name == "nt":
        try:
            import ctypes

            primary = ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF
            return {0x04: "zh", 0x07: "de"}.get(primary, "en")  # Windows primary language ids
        except (AttributeError, OSError):
            pass
    name = (locale.getlocale()[0] or os.environ.get("LC_ALL") or os.environ.get("LANG") or "").lower()
    if name.startswith(("zh", "chinese")):
        return "zh"
    if name.startswith(("de", "german")):
        return "de"
    return "en"


def lang() -> str:
    """'zh', 'en' or 'de'. Re-read each time, so a MEETNOTE_LANG loaded from .env later still applies."""
    setting = (os.environ.get("MEETNOTE_LANG") or "").strip().lower()
    for code in LANGS:
        if setting.startswith(code):
            return code
    if "system" not in _cache:
        _cache["system"] = _system_language()
    return _cache["system"]


def tr(zh: str, en: str, de: str | None = None) -> str:
    """Pick the Chinese, English or German version of a message (German falls back to English)."""
    current = lang()
    if current == "zh":
        return zh
    if current == "de" and de is not None:
        return de
    return en


# Problem and verdict labels: the checker and judge prompts answer in Chinese
LABELS = {
    "支持": ("supported", "belegt"),
    "无依据": ("unsupported", "unbelegt"),
    "对象错误": ("wrong object", "falscher Bezug"),
    "归属错误": ("wrong speaker", "falscher Sprecher"),
    "方向或数字错误": ("wrong direction/number", "falsche Richtung/Zahl"),
    "过度概括": ("over-generalized", "übergeneralisiert"),
    "不适用": ("n/a", "n. a."),
    "未评审": ("not judged", "nicht bewertet"),
    "时间戳错误": ("wrong timestamp", "falscher Zeitstempel"),
}


def show_label(label: str) -> str:
    if label not in LABELS:
        return label
    return tr(label, *LABELS[label])


# key → (Chinese name, English name, German name). ASCII-only German names (ae/oe/ue)
# survive zip files and older tools.
FILE_NAMES = {
    "transcript": ("转写稿.txt", "transcript.txt", "Transkript.txt"),
    "subtitles": ("转写稿.srt", "transcript.srt", "Transkript.srt"),
    "notes": ("要点笔记.md", "notes.md", "Notizen.md"),
    "synopsis": ("全文梗概.md", "synopsis.md", "Zusammenfassung.md"),
    "highlights": ("两页精华.md", "highlights.md", "Kernpunkte.md"),
    "checks": ("核查记录.md", "fact-check.md", "Faktencheck.md"),
    "faithfulness": ("忠实度评估.md", "faithfulness.md", "Quellentreue.md"),
}
DOC_TITLES = {
    "synopsis": ("全文梗概", "Synopsis", "Zusammenfassung"),
    "highlights": ("两页精华", "Highlights", "Kernpunkte"),
}
DIR_SUFFIXES = ("_纪要", "_notes", "_Protokoll")


def find_file(out_dir: Path, key: str) -> Path | None:
    """The existing file for `key` in any language, or None."""
    for name in FILE_NAMES[key]:
        if (out_dir / name).exists():
            return out_dir / name
    return None


def _folder_language(out_dir: Path) -> str | None:
    """The language a folder's existing files are named in, if any."""
    for names in FILE_NAMES.values():
        for code, name in zip(LANGS, names):
            if (out_dir / name).exists():
                return code
    return None


def file_path(out_dir: Path, key: str) -> Path:
    """Where to read or write `key`. An existing file in any language is reused, and a new
    file follows the language of the folder's other files, so a folder never mixes names;
    an empty folder gets the current interface language."""
    existing = find_file(out_dir, key)
    if existing:
        return existing
    code = _folder_language(out_dir) or lang()
    return out_dir / FILE_NAMES[key][LANGS.index(code)]


def doc_title(key: str) -> str:
    return tr(*DOC_TITLES[key])


def default_out_dir(base: Path, stem: str) -> Path:
    """<stem>_纪要, <stem>_notes or <stem>_Protokoll; an existing folder of any of them is reused."""
    for suffix in DIR_SUFFIXES:
        if (base / f"{stem}{suffix}").is_dir():
            return base / f"{stem}{suffix}"
    return base / f"{stem}{tr(*DIR_SUFFIXES)}"
