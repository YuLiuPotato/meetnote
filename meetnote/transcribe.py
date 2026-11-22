"""Audio → timestamped transcript using faster-whisper."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .i18n import tr


def _register_cuda_dlls() -> None:
    """Make pip-installed CUDA libs (nvidia-cublas / nvidia-cudnn) loadable on Windows."""
    if os.name != "nt":
        return
    try:
        import nvidia  # namespace package from nvidia-*-cu12 wheels
    except ImportError:
        return
    for root in nvidia.__path__:
        for lib in Path(root).iterdir():
            bin_dir = lib / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None  # "S1", "S2", … when diarization ran


def _decode_audio(path: Path, sr: int = 16000):
    """Decode any audio/video file to mono float32 at `sr` Hz.

    Done here rather than via faster-whisper's decode_audio, which passes an
    argument that newer PyAV releases no longer accept.
    """
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError(tr(f"没有从 {path} 解码出音频", f"No audio could be decoded from {path}"))
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _pick_device(device: str) -> tuple[str, str]:
    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return device, ("float16" if device == "cuda" else "int8")


_models: dict = {}


def _load_model(model_name: str, device: str):
    """Load once and reuse: the web server transcribes many files with one model."""
    _register_cuda_dlls()
    from faster_whisper import WhisperModel

    device, compute_type = _pick_device(device)
    key = (model_name, device)
    if key not in _models:
        log(tr(f"加载 Whisper {model_name}（{device}/{compute_type}）…", f"Loading Whisper {model_name} ({device}/{compute_type})…"))
        _models[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _models[key]


def transcribe(
    audio_path: Path,
    model_name: str = "large-v3",
    language: str | None = "zh",
    device: str = "auto",
    terms: str | None = None,
    on_progress: Callable[[float], None] | None = None,
    diarization: bool = True,
    num_speakers: int | None = None,
) -> tuple[list[Segment], float, str]:
    """Return (segments, duration_seconds, main_language_code).

    `language=None` detects the language, and then also re-detects it per segment, so a
    meeting that switches between Chinese and English keeps each part in its own language
    instead of Whisper translating it. `on_progress` receives 0–100.

    With `diarization`, segments carry a speaker label (S1, S2, …) and a Whisper segment
    that spans a change of speaker is split at the word where the speaker changes.
    `num_speakers` is the head count if known (more reliable than estimating it).
    """
    model = _load_model(model_name, device)

    log(tr("解码音频…", "Decoding audio…"))
    audio = _decode_audio(audio_path)

    # Diarization runs on the CPU while Whisper uses the GPU
    diar: dict = {}
    diar_thread = None
    if diarization:
        def run_diarization():
            from .diarize import diarize
            t = time.time()
            try:
                diar["turns"] = diarize(audio, num_speakers=num_speakers)
                log(tr(f"说话人分离完成，用时 {time.time() - t:.0f}s", f"Speaker separation done in {time.time() - t:.0f}s"))
            except Exception as e:  # never lose the transcript over speaker labels
                diar["error"] = e
        diar_thread = threading.Thread(target=run_diarization, daemon=True)
        diar_thread.start()

    auto = not language
    if auto:
        # Sample several voiced windows: the first 30 s alone is often silence or small talk
        language, prob, _ = model.detect_language(audio, vad_filter=True, language_detection_segments=4)
        log(tr(f"检测到主要语言：{language_name(language)}（置信度 {prob:.0%}），并逐段识别语言",
               f"Main language: {language_name(language)} ({prob:.0%} confidence); detecting per segment"))

    # The prompt steers style, so it must be in the recording's language: a Chinese
    # prompt on English audio makes Whisper output Chinese.
    prompt = _prompt_for(language, terms)

    segments_iter, info = model.transcribe(
        audio,
        language=language,
        multilingual=auto,
        vad_filter=True,
        beam_size=5,
        initial_prompt=prompt,
        hotwords=terms or None,
        # Reduces Whisper's tendency to loop on repeated phrases
        condition_on_previous_text=False,
        # No word_timestamps: with them on, faster-whisper skipped whole sentences of a
        # mixed-language test recording. Speaker changes are handled by re-transcription.
    )
    duration = info.duration
    log(tr(f"音频时长 {fmt_ts(duration)}，开始转写…", f"Audio length {fmt_ts(duration)}, transcribing…"))

    segments: list[Segment] = []
    t0 = time.time()
    last_report = 0.0
    for s in segments_iter:
        text = s.text.strip()
        # Drop exact consecutive repeats and stock phrases Whisper invents over silence
        if text and not (segments and segments[-1].text == text) and not _HALLUCINATION.search(text):
            segments.append(Segment(s.start, s.end, text))
        pct = min(100, s.end / duration * 100) if duration else 0
        if on_progress:
            on_progress(pct)
        if time.time() - last_report > 5:
            last_report = time.time()
            log(tr(f"  {pct:5.1f}%  [{fmt_ts(s.end)}]  已用 {time.time() - t0:.0f}s",
                   f"  {pct:5.1f}%  [{fmt_ts(s.end)}]  {time.time() - t0:.0f}s elapsed"))
    log(tr(f"转写完成，用时 {time.time() - t0:.0f}s，共 {len(segments)} 段",
           f"Transcription done in {time.time() - t0:.0f}s, {len(segments)} segments"))

    if diar_thread:
        if diar_thread.is_alive():
            log(tr("等待说话人分离…", "Waiting for speaker separation…"))
        diar_thread.join()
        if "error" in diar:
            log(tr(f"警告：说话人分离失败（{diar['error']}），转写稿不区分说话人",
                   f"Warning: speaker separation failed ({diar['error']}); the transcript has no speaker labels"))
        elif diar["turns"]:
            segments = assign_speakers(model, audio, segments, diar["turns"], language, terms)
            n = len({s.speaker for s in segments if s.speaker})
            log(tr(f"识别出 {n} 位说话人", f"Found {n} speaker(s)"))
    if auto:
        segments = fix_languages(model, audio, segments, terms)
    return segments, duration, language


def _script(text: str) -> str | None:
    """'cjk' or 'latin': CJK characters vs Latin words (so Chinese with English product names
    still counts as Chinese). None if too short to tell."""
    cjk = sum(1 for ch in text if _is_cjk(ch) and ch.isalpha())
    latin = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    if cjk + latin < 4:
        return None
    return "cjk" if cjk >= latin else "latin"


_CJK_LANGS = {"zh", "yue", "ja"}
_LATIN_LANGS = {"en", "fr", "de", "es", "it", "pt", "nl", "sv", "da", "no", "fi", "pl", "cs", "ro", "hu", "tr", "id", "ms", "vi"}


def fix_languages(model, audio, segments: list[Segment], terms: str | None = None,
                  sr: int = 16000) -> list[Segment]:
    """Re-transcribe stretches that Whisper wrote in the wrong language.

    In multilingual mode Whisper picks one language per 30-second window, so a reply in
    English right after Chinese can come out *translated* into Chinese — sometimes with the
    meaning changed ("I disagree a little" → "我认同一点"). Each stretch of one speaker is
    checked on its own audio; when the detected language and the script of the text
    disagree, that stretch is transcribed again in the detected language.
    """
    groups: list[list[Segment]] = []
    for s in segments:
        g = groups[-1] if groups else None
        if (g and s.speaker == g[-1].speaker and s.start - g[-1].end < 1.0
                and s.end - g[0].start <= 28 and _script(s.text) == _script(g[-1].text)):
            g.append(s)
        else:
            groups.append([s])

    out: list[Segment] = []
    fixed = 0
    for g in groups:
        start, end = g[0].start, g[-1].end
        text = "".join(s.text for s in g)
        script = _script(text)
        if script is None or end - start < 1.5:
            out += g
            continue
        clip = audio[int(max(0, start - 0.2) * sr):int((end + 0.2) * sr)]
        lang, prob, _ = model.detect_language(clip)
        expected = "cjk" if lang in _CJK_LANGS else "latin" if lang in _LATIN_LANGS else None
        if expected is None or expected == script or prob < 0.7:
            out += g
            continue
        new = _retranscribe(model, clip, lang, terms, max(0, start - 0.2), g[0].speaker)
        if not new:
            out += g
            continue
        log(tr(f"  [{fmt_ts(start)}] 按 {language_name(lang)} 重新识别：{text[:30]}… → {new[0].text[:30]}…",
               f"  [{fmt_ts(start)}] re-transcribed as {language_name(lang)}: {text[:30]}… → {new[0].text[:30]}…"))
        out += new
        fixed += 1
    if fixed:
        log(tr(f"重新识别了 {fixed} 段被写成另一种语言的话", f"Re-transcribed {fixed} stretch(es) written in the wrong language"))
    return out


def _speaker_pieces(index, start: float, end: float, min_piece: float = 1.0):
    """Who speaks when inside [start, end]: [(speaker, a, b)], contiguous, short pieces absorbed."""
    turns = [t for t in index.near(start, end) if t.start < end and t.end > start]
    pieces: list[list] = []
    for t in sorted(turns, key=lambda t: t.start):
        a, b = max(start, t.start), min(end, t.end)
        if pieces and a < pieces[-1][2]:  # overlapping speech: split the overlap in the middle
            mid = (a + pieces[-1][2]) / 2
            pieces[-1][2] = mid
            a = mid
        if b > a:
            pieces.append([t.speaker, a, b])
    if not pieces:
        return []
    # Close gaps and stretch to the segment edges, then merge and absorb short pieces
    pieces[0][1], pieces[-1][2] = start, end
    for x, y in zip(pieces, pieces[1:]):
        x[2] = y[1]
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for i in range(len(pieces) - 1):
            if pieces[i][0] == pieces[i + 1][0]:
                pieces[i][2] = pieces[i + 1][2]
                del pieces[i + 1]
                changed = True
                break
        if changed:
            continue
        i = min(range(len(pieces)), key=lambda i: pieces[i][2] - pieces[i][1])
        if pieces[i][2] - pieces[i][1] < min_piece:
            if i == 0 or (i < len(pieces) - 1 and pieces[i + 1][2] - pieces[i + 1][1] > pieces[i - 1][2] - pieces[i - 1][1]):
                pieces[i + 1][1] = pieces[i][1]
            else:
                pieces[i - 1][2] = pieces[i][2]
            del pieces[i]
            changed = True
    return [tuple(x) for x in pieces]


def _guess_language(text: str, main: str) -> str:
    """Language to re-transcribe a piece in when its audio is too short to detect."""
    script = _script(text)
    if script == "cjk":
        return main if main in _CJK_LANGS else "zh"
    if script == "latin":
        return main if main in _LATIN_LANGS else "en"
    return main


def _prompt_for(language: str | None, terms: str | None) -> str | None:
    """Style prompt plus the term list. The terms go in bare ("Acme, Kubernetes."): with a label
    such as "Glossary:" Whisper copied the label into the transcript ("Acme Glossary is
    the co-founder…")."""
    prompt = _PROMPTS.get(language) or ""
    if terms:
        terms = terms.strip().rstrip("。.,，")
        prompt += f"{terms}。" if language in ("zh", "yue") else f"{terms}."
    return prompt or None


def _retranscribe(model, clip, language: str, terms: str | None, offset: float, speaker: str | None,
                  strict: bool = False) -> list[Segment]:
    """Transcribe one clip in a given language. `strict` drops output Whisper itself doubts
    (likely silence, music or noise), for clips nobody had transcribed before."""
    prompt = _prompt_for(language, terms)
    redo, _ = model.transcribe(clip, language=language, beam_size=5, initial_prompt=prompt,
                               condition_on_previous_text=False, vad_filter=False)
    return [Segment(offset + r.start, offset + r.end, r.text.strip(), speaker)
            for r in redo if r.text.strip() and not _HALLUCINATION.search(r.text)
            and not (strict and (r.no_speech_prob > 0.5 or r.avg_logprob < -1.0 or _too_short(r.text)))]


def _tokens(text: str) -> list[str]:
    """CJK characters and lower-cased Latin words, for comparing two transcriptions."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[⺀-鿿]", text.lower())


def _too_short(text: str) -> bool:
    # "Thank you.", "Yes." over applause or silence: too little to be worth the risk
    return len(_tokens(text)) < 4


def _same_words(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, _tokens(a), _tokens(b), autojunk=False).ratio()


def _uncovered(start: float, end: float, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    free = [(start, end)]
    for a, b in spans:
        free = [piece for x, y in free
                for piece in ((x, min(y, a)), (max(x, b), y)) if piece[1] - piece[0] > 0]
    return free


def assign_speakers(model, audio, segments: list[Segment], turns, main_language: str,
                    terms: str | None = None, sr: int = 16000) -> list[Segment]:
    """Label each segment with its speaker. A segment that spans a change of speaker (a
    question and its answer that Whisper put in one line) is cut at the diarization
    boundary and each part transcribed on its own."""
    from .diarize import SpeakerIndex, label_speakers

    index = SpeakerIndex(turns)
    labels = label_speakers(index.turns)
    out: list[Segment] = []
    split = 0
    for seg in segments:
        pieces = _speaker_pieces(index, seg.start, seg.end)
        if len(pieces) <= 1:
            spk = pieces[0][0] if pieces else index.at(seg.start, seg.end, max_gap=30.0)
            out.append(Segment(seg.start, seg.end, seg.text, labels.get(spk)))
            continue
        parts: list[Segment] = []
        for spk, a, b in pieces:
            clip = audio[int(a * sr):int(b * sr)]
            lang = _guess_language(seg.text, main_language)
            if b - a >= 2.0:
                detected, prob, _ = model.detect_language(clip)
                if prob >= 0.5:
                    lang = detected
            parts += _retranscribe(model, clip, lang, terms, a, labels.get(spk))
        # Splitting should only move the line break. If the pieces read differently from the
        # original line (short clips lose context and Whisper starts guessing), keep the
        # original line under the speaker who talks most of it.
        if _same_words(seg.text, " ".join(p.text for p in parts)) < 0.75:
            spk = max(pieces, key=lambda x: x[2] - x[1])[0]
            out.append(Segment(seg.start, seg.end, seg.text, labels.get(spk)))
        else:
            out += parts
            split += 1
    if split:
        log(tr(f"按说话人拆开了 {split} 段跨两人的话", f"Split {split} segment(s) that spanned two speakers"))

    # Whisper's own voice-activity filter sometimes drops real speech (quieter voices, speech
    # over background sound). The diarizer detects speech independently: transcribe what it
    # heard and Whisper did not.
    spans = sorted((s.start - 0.3, s.end + 0.3) for s in out)
    filled = []
    for t in index.turns:
        if t.end - t.start < 1.5:
            continue
        near = [(a, b) for a, b in spans if a < t.end and b > t.start]
        for a, b in _uncovered(t.start, t.end, near):
            if b - a < 1.5:
                continue
            clip = audio[int(a * sr):int(b * sr)]
            lang, prob = main_language, 0.0
            if b - a >= 2.0:
                detected, prob, _ = model.detect_language(clip)
                lang = detected if prob >= 0.5 else main_language
            filled += _retranscribe(model, clip, lang, terms, a, labels.get(t.speaker), strict=True)
    if filled:
        log(tr(f"补上了 {len(filled)} 段 Whisper 漏掉的话", f"Added {len(filled)} line(s) Whisper had skipped"))
        out = sorted(out + filled, key=lambda s: s.start)
    return out


def _is_cjk(ch: str) -> bool:
    return bool(ch) and ("⺀" <= ch <= "鿿" or "＀" <= ch <= "￯" or "　" <= ch <= "〿")


# Initial prompts per language; mainly to get Simplified (not Traditional) Chinese with punctuation
_PROMPTS = {
    "zh": "以下是一段会议录音的普通话转写，使用简体中文和标点。",
    "yue": "以下是一段会议录音的粤语转写，使用标点。",
}

LANGUAGE_NAMES = {
    "zh": "简体中文", "yue": "粤语", "en": "English", "ja": "日本語", "ko": "한국어",
    "fr": "Français", "de": "Deutsch", "es": "Español", "it": "Italiano", "pt": "Português",
    "ru": "Русский", "ar": "العربية", "nl": "Nederlands", "sv": "Svenska", "th": "ไทย", "vi": "Tiếng Việt",
}


def language_name(code: str | None) -> str:
    return LANGUAGE_NAMES.get(code or "", code or "未知")


# Text Whisper is known to hallucinate on silence/music (learned from subtitled videos)
_HALLUCINATION = re.compile(
    r"明镜与点点|明鏡與點點|请不吝点赞|請不吝點贊|订阅.{0,3}转发.{0,3}打赏|字幕(由|提供|製作|制作)|Amara\.org"
    r"|谢谢观看|謝謝觀看|thanks? (you )?for watching|subtitles? by|please subscribe",
    re.IGNORECASE,
)


def write_transcript(segments: list[Segment], txt_path: Path, srt_path: Path | None = None) -> None:
    # One line per Whisper segment. Joining neighbouring segments reads better but glues a
    # question and its answer (two speakers) into one sentence, which the summarizer then
    # attributes to a single person.
    # With diarization each line starts with the speaker label: "[01:23] S2: ..."
    lines = [f"[{fmt_ts(s.start)}] {_said(s)}" for s in segments]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if srt_path:
        blocks = []
        for i, s in enumerate(segments, 1):
            blocks.append(f"{i}\n{_srt_ts(s.start)} --> {_srt_ts(s.end)}\n{_said(s)}\n")
        srt_path.write_text("\n".join(blocks), encoding="utf-8")


def _said(s: Segment) -> str:
    return f"{s.speaker}: {s.text}" if s.speaker else s.text


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _srt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
