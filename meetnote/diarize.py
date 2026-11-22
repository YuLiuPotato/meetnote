"""Speaker diarization ("who spoke when") with sherpa-onnx, and aligning it to Whisper words.

Runs on CPU with ONNX models (pyannote segmentation-3.0 + 3D-Speaker CAM++ embeddings, which
are trained on Chinese and English speech). No Hugging Face token or PyTorch needed; the
models (~35 MB) are downloaded once to the model cache on first use.
"""

from __future__ import annotations

import bisect
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .i18n import tr
from .transcribe import log

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = ("sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
                f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMBEDDING = ("3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
             f"{_RELEASES}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")


def model_dir() -> Path:
    return Path(os.environ.get("MEETNOTE_MODEL_DIR") or Path.home() / ".cache" / "meetnote" / "models")


def _download(url: str, dest: Path) -> None:
    log(tr(f"下载说话人模型 {url.rsplit('/', 1)[-1]} …", f"Downloading speaker model {url.rsplit('/', 1)[-1]}…"))
    root = model_dir()
    root.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=root, delete=False)
    try:
        with tmp, urllib.request.urlopen(url, timeout=120, context=_ssl_context()) as r:
            shutil.copyfileobj(r, tmp)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)  # no half-downloaded file left behind
        raise
    if url.endswith(".tar.bz2"):  # the archive holds its own top-level folder
        with tarfile.open(tmp.name, "r:bz2") as tar:
            tar.extractall(root, filter="data")
        os.unlink(tmp.name)
    else:
        os.replace(tmp.name, dest)


def _ssl_context():
    """Verify with the operating system's certificate store. Python's own check failed on
    this Windows machine ("unable to get issuer certificate") for GitHub's download host."""
    try:
        import ssl

        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return None


def ensure_models() -> tuple[Path, Path]:
    paths = []
    for rel, url in (SEGMENTATION, EMBEDDING):
        path = model_dir() / rel
        if not path.exists():
            _download(url, path)
            if not path.exists():
                raise RuntimeError(tr(f"说话人模型下载后未找到 {path}", f"Speaker model not found after download: {path}"))
        paths.append(path)
    return paths[0], paths[1]


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def diarize(audio, num_speakers: int | None = None, sample_rate: int = 16000) -> list[Turn]:
    """`audio`: mono float32 at 16 kHz. `num_speakers`: known head count, or None to estimate.

    Two stages. sherpa-onnx (pyannote segmentation) finds where speech is and where the
    speaker changes. Its own clustering of those turns is weak — on an AMI meeting it merged
    two speakers and spent a cluster on scraps — so the turns are re-clustered here: each
    turn is cut into ~1.5 s windows, every window gets a CAM++ voice embedding, and the
    windows are grouped by `cluster_speakers`.
    """
    import sherpa_onnx

    seg, emb = ensure_models()
    threads = max(1, min(8, (os.cpu_count() or 4) // 2))
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
        # Low threshold = many small clusters, so neighbouring turns are rarely glued together;
        # only the turn boundaries are used
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(tr("说话人分离配置无效（模型文件损坏？可删除模型缓存后重试）",
                               "Invalid speaker-separation setup (corrupt model file? delete the model cache and retry)"))
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    if sd.sample_rate != sample_rate:
        raise RuntimeError(tr(f"说话人模型需要 {sd.sample_rate} Hz 音频", f"The speaker model needs {sd.sample_rate} Hz audio"))
    turns = [(s.start, s.end) for s in sd.process(audio).sort_by_start_time()]
    if not turns:
        return []

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads))

    def embed(a: float, b: float):
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate, audio[int(a * sample_rate):int(b * sample_rate)])
        stream.input_finished()
        return extractor.compute(stream)

    import numpy as np

    wins = _windows(turns)
    if len(wins) < 2:
        return [Turn(a, b, 0) for a, b in turns]
    E = _normalize(np.asarray([embed(a, b) for _, a, b in wins], dtype=np.float64))
    labels = cluster_speakers(E, [b - a for _, a, b in wins], num_speakers)

    pieces = [(a, b, int(labels[i])) for i, (_, a, b) in enumerate(wins)]
    # Turns too short for a reliable window: nearest speaker centroid
    centroids = _centroids(E, labels)
    covered = {ti for ti, _, _ in wins}
    for ti, (a, b) in enumerate(turns):
        if ti not in covered:
            e = _normalize(np.asarray([embed(a, b)], dtype=np.float64))[0]
            pieces.append((a, b, int(np.argmax(centroids @ e))))
    pieces.sort()
    out: list[Turn] = []
    for a, b, s in pieces:
        if out and out[-1].speaker == s and a - out[-1].end < 0.05:
            out[-1].end = max(out[-1].end, b)
        else:
            out.append(Turn(a, b, s))
    return out


def _normalize(x):
    import numpy as np

    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def _windows(turns, win: float = 1.5, min_len: float = 0.5):
    """Cut each turn into roughly `win`-second pieces: (turn index, start, end)."""
    out = []
    for i, (a, b) in enumerate(turns):
        d = b - a
        if d < min_len:
            continue
        n = max(1, int(round(d / win)))
        out += [(i, a + d * k / n, a + d * (k + 1) / n) for k in range(n)]
    return out


SIMILARITY = 0.5  # cosine similarity at which agglomerative clustering stops merging


def _ahc(E, weights, threshold: float):
    """Average-linkage agglomerative clustering on cosine similarity, weighted by duration.
    Merges the most similar pair of clusters until no pair is at least `threshold` alike."""
    import numpy as np

    n = len(E)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    size = np.asarray(weights, dtype=np.float64).copy()
    labels = np.arange(n)
    while True:
        i, j = np.unravel_index(np.argmax(S), S.shape)
        if S[i, j] < threshold:
            break
        S[i, :] = (size[i] * S[i, :] + size[j] * S[j, :]) / (size[i] + size[j])
        S[:, i] = S[i, :]
        S[i, i] = S[j, :] = S[:, j] = -np.inf
        size[i] += size[j]
        labels[labels == j] = i
    return _renumber(labels)


def _renumber(labels):
    import numpy as np

    used = {v: i for i, v in enumerate(sorted(set(labels.tolist())))}
    return np.array([used[v] for v in labels.tolist()], dtype=int)


def _centroids(E, labels):
    import numpy as np

    return _normalize(np.array([E[labels == j].mean(0) for j in range(labels.max() + 1)]))


def _absorb(E, weights, labels, min_share: float):
    """Hand the windows of clusters below `min_share` of the talk time to the nearest big cluster.
    Real voices form a few big clusters; noise, laughter and crosstalk leave many tiny ones."""
    import numpy as np

    w = np.asarray(weights)
    while True:
        share = np.bincount(labels, weights=w) / w.sum()
        small = share < min_share
        if not small.any() or small.all():
            return labels
        sim = E @ _centroids(E, labels).T
        sim[:, small] = -np.inf
        labels = _renumber(np.where(small[labels], sim.argmax(1), labels))


def _absorb_similar(E, weights, labels, min_share: float, gate: float = 0.3,
                    leftover_similarity: float = 0.3, min_seconds: float = 5.0):
    """Like `_absorb`, but a small cluster only joins a big one whose voice it resembles.

    In a pitch recording the questioners each spoke for well under a minute; folding every
    small cluster into the nearest big one made the whole Q&A "the presenter". Small
    clusters that resemble no big one (similarity < `gate`) are grouped among themselves,
    and each group with at least `min_seconds` of speech is kept as its own speaker.
    """
    import numpy as np

    w = np.asarray(weights, dtype=np.float64)
    share = np.bincount(labels, weights=w) / w.sum()
    big = np.where(share >= min_share)[0]
    if len(big) == 0:
        big = np.array([share.argmax()])
    cent = _centroids(E, labels)
    big_cent = cent[big]
    out = labels.copy()
    leftovers = []
    for j in range(labels.max() + 1):
        if j in big:
            continue
        sim = cent[j] @ big_cent.T
        if sim.max() >= gate:
            out[labels == j] = big[sim.argmax()]
        else:
            leftovers.append(j)
    if leftovers:
        idx = np.where(np.isin(labels, leftovers))[0]
        groups = _ahc(E[idx], w[idx], leftover_similarity) if len(idx) > 1 else np.zeros(1, dtype=int)
        for g in range(groups.max() + 1):
            members = idx[groups == g]
            if w[members].sum() >= min_seconds:
                out[members] = labels.max() + 1 + g
            else:
                out[members] = big[(_normalize(E[members].mean(0, keepdims=True)) @ big_cent.T).argmax()]
    return _renumber(out)


def _merge_down(E, labels, k: int):
    """Merge the two most similar speakers until only k are left."""
    import numpy as np

    while labels.max() + 1 > k:
        c = _centroids(E, labels)
        S = c @ c.T
        np.fill_diagonal(S, -np.inf)
        i, j = np.unravel_index(np.argmax(S), S.shape)
        labels = _renumber(np.where(labels == j, i, labels))
    return labels


def cluster_speakers(E, weights, k: int | None = None):
    """Group voice embeddings (one per window, `weights` = durations) into speakers.

    Unknown head count: clusters with less than about a minute of talk time are folded
    into the speaker they sound like; ones that sound like nobody stay separate speakers. A known head count `k` keeps smaller clusters (and allows
    finer clusters) until there are at least k, then merges the most similar down to k.
    """
    import numpy as np

    total = float(np.sum(weights))
    # ~60 s of talk, but at least 2% and at most 5% of the meeting
    min_share = min(0.05, max(0.02, 60.0 / max(total, 1.0)))
    if k == 1 or len(E) < 2:
        return np.zeros(len(E), dtype=int)
    if k is None:
        return _absorb_similar(E, weights, _ahc(E, weights, SIMILARITY), min_share)
    labels = None
    for threshold in (SIMILARITY, 0.6, 0.7, 0.8):
        fine = _ahc(E, weights, threshold)
        for share in (min_share, 0.02, 0.01, 0.0):
            labels = _absorb(E, weights, fine, share)
            if labels.max() + 1 >= k:
                return _merge_down(E, labels, k)
    return labels  # fewer distinct voices than the given head count


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


class SpeakerIndex:
    """Turns sorted by start, for looking up who speaks during a word."""

    def __init__(self, turns: list[Turn]):
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = [t.start for t in self.turns]
        self.longest = max((t.end - t.start for t in self.turns), default=0.0)

    def near(self, start: float, end: float, margin: float = 0.0) -> list[Turn]:
        """Turns that could overlap [start - margin, end + margin]."""
        lo = bisect.bisect_left(self.starts, start - self.longest - margin)
        hi = bisect.bisect_right(self.starts, end + margin)
        return self.turns[lo:hi]

    def at(self, start: float, end: float, max_gap: float = 1.0) -> int | None:
        return speaker_at(self.near(start, end, max_gap), start, end, max_gap)


def speaker_at(turns: list[Turn], start: float, end: float, max_gap: float = 1.0) -> int | None:
    """The speaker who talks most during [start, end]; else the nearest turn within `max_gap` s."""
    best, best_ov = None, 0.0
    for t in turns:
        if t.start > end:
            break
        ov = _overlap(start, end, t.start, t.end)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    if best is not None:
        return best
    mid = (start + end) / 2
    near = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)), default=None)
    if near and min(abs(mid - near.start), abs(mid - near.end)) <= max_gap:
        return near.speaker
    return None


def label_speakers(turns: list[Turn]) -> dict[int, str]:
    """S1, S2, … in order of first appearance, so S1 is whoever speaks first."""
    labels: dict[int, str] = {}
    for t in turns:
        if t.speaker not in labels:
            labels[t.speaker] = f"S{len(labels) + 1}"
    return labels
