"""Join several recordings (e.g. the WAV pieces a recorder splits a meeting into) into one MP3.

Encoding uses the libmp3lame encoder bundled with PyAV, so no ffmpeg install is needed.
Every input is resampled to the first one's sample rate and channel layout (mono or stereo).
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Callable

from .i18n import tr
from .transcribe import fmt_ts, log

AUDIO_EXTS = {
    ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr",
    ".mp4", ".mov", ".mkv", ".webm", ".m4v",
}
MP3_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)


def natural_key(path: Path):
    """REC2.wav before REC10.wav."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def collect_inputs(paths: list[Path]) -> list[Path]:
    """Files in the order given; a folder contributes its audio files in natural name order."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            found = sorted((f for f in p.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS), key=natural_key)
            if not found:
                raise FileNotFoundError(tr(f"文件夹里没有音频文件：{p}", f"No audio files in folder: {p}",
                                                              f"Keine Audiodateien im Ordner: {p}"))
            out += found
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(tr(f"找不到音频文件：{p}", f"Audio file not found: {p}",
                                                       f"Audiodatei nicht gefunden: {p}"))
    return out


def _target_format(first: Path) -> tuple[int, str]:
    import av

    with av.open(str(first)) as c:
        if not c.streams.audio:
            raise RuntimeError(tr(f"没有音轨：{first}", f"No audio track: {first}",
                                                   f"Keine Tonspur: {first}"))
        s = c.streams.audio[0]
        rate = s.rate or s.codec_context.sample_rate or 44100
        channels = s.codec_context.channels or 1
    # MP3 supports a fixed set of rates; take the nearest one that doesn't lose bandwidth
    rate = min((r for r in MP3_RATES if r >= rate), default=48000)
    return rate, ("mono" if channels == 1 else "stereo")


def merge_to_mp3(inputs: list[Path], out_path: Path, bitrate: int = 128_000,
                 on_progress: Callable[[float], None] | None = None) -> list[dict]:
    """Concatenate `inputs` into `out_path` (MP3). Returns where each input starts in the result:
    [{"file": name, "start": seconds, "duration": seconds}]."""
    import av

    if not inputs:
        raise ValueError(tr("没有要合并的文件", "No files to merge", "Keine Dateien zum Zusammenfügen"))
    rate, layout = _target_format(inputs[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    parts: list[dict] = []
    written = 0  # samples per channel

    with av.open(str(tmp), "w", format="mp3") as out:
        stream = out.add_stream("libmp3lame", rate=rate)
        stream.layout = layout
        stream.bit_rate = bitrate if layout == "stereo" else bitrate // 2
        enc = stream.codec_context
        fifo = av.AudioFifo()

        def drain(final: bool = False) -> None:
            nonlocal written
            size = enc.frame_size or 1152
            while fifo.samples >= size or (final and fifo.samples):
                frame = fifo.read(min(size, fifo.samples))
                frame.pts = written
                frame.time_base = enc.time_base
                written += frame.samples
                for packet in stream.encode(frame):
                    out.mux(packet)

        for i, path in enumerate(inputs):
            start = written + fifo.samples
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
            try:
                with av.open(str(path)) as src:
                    if not src.streams.audio:
                        raise RuntimeError(tr(f"没有音轨：{path}", f"No audio track: {path}",
                                                              f"Keine Tonspur: {path}"))
                    for frame in src.decode(audio=0):
                        for r in resampler.resample(frame):
                            r.pts = None  # the FIFO renumbers; source timestamps don't carry over
                            fifo.write(r)
                        drain()
                    for r in resampler.resample(None):
                        r.pts = None
                        fifo.write(r)
                    drain()
            except av.error.FFmpegError as e:
                raise RuntimeError(tr(f"无法读取 {path.name}：{e}", f"Cannot read {path.name}: {e}",
                                                               f"{path.name} kann nicht gelesen werden: {e}")) from e
            end = written + fifo.samples
            parts.append({"file": path.name, "start": round(start / rate, 3), "duration": round((end - start) / rate, 3)})
            log(tr(f"  [{fmt_ts(start / rate)}] {path.name}（{fmt_ts((end - start) / rate)}）",
                   f"  [{fmt_ts(start / rate)}] {path.name} ({fmt_ts((end - start) / rate)})",
                   f"  [{fmt_ts(start / rate)}] {path.name} ({fmt_ts((end - start) / rate)})"))
            if on_progress:
                on_progress((i + 1) / len(inputs) * 100)
        drain(final=True)
        for packet in stream.encode(None):
            out.mux(packet)

    tmp.replace(out_path)
    return parts


def earliest_date(inputs: list[Path]) -> str:
    """Recording date for meta.json: the oldest modification time among the inputs."""
    return dt.datetime.fromtimestamp(min(p.stat().st_mtime for p in inputs)).strftime("%Y-%m-%d")
