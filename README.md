# meetnote

**English** | [简体中文](README.zh-CN.md)

Meeting recording → **transcript with speakers** + **full synopsis** + **two-page highlights**, in one command.

- **Transcription** runs locally with faster-whisper (large-v3) and uses an NVIDIA GPU automatically. An hour of audio takes a few minutes; no ffmpeg needed.
- **Speaker separation** runs locally on the CPU (sherpa-onnx + 3D-Speaker CAM++), at the same time as transcription.
- **Summaries** come from an LLM: DeepSeek by default, or any OpenAI-compatible service, or Claude. Every point carries a `[mm:ss]` citation, and a separate pass fact-checks the draft against the transcript before it is revised.
- Works with Chinese, English and mixed-language meetings: each part stays in the language it was spoken in.

## Install

Requires Windows and Python 3.10+. An NVIDIA GPU makes it much faster.

```powershell
git clone https://github.com/YuLiuPotato/meetnote.git
cd meetnote
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The script creates a virtual environment in `meetnote\.venv` and puts a `meetnote.cmd` launcher in `%USERPROFILE%\.local\bin` (make sure that folder is on your PATH). The Whisper model (~3 GB) and the speaker models (~35 MB) download on first use.

### Configuration: `.env`

API keys, port and access code live in `meetnote\.env` (created from `.env.example` during install), not in system environment variables. Open it in a text editor and fill in:

```ini
DEEPSEEK_API_KEY=sk-your-key
```

The interface language (messages, `--help`, reports and output file names) follows your system's display language; set `MEETNOTE_LANG=en` or `MEETNOTE_LANG=zh` in `.env` to choose. It is independent of the language of the notes.

Lookup order: the file named by `MEETNOTE_ENV_FILE` → `.env` in the current folder → `meetnote\.env`. Environment variables that are already set win over the file. `.env` is in `.gitignore`; never share it.

**Summaries need one LLM backend** (`--backend auto` picks the first one configured, in this order):

| Backend | Set in `.env` | Default model |
|---|---|---|
| `deepseek` (recommended) | `DEEPSEEK_API_KEY` (from platform.deepseek.com) | `deepseek-chat`; `--model deepseek-reasoner` also works |
| `openai` | `MEETNOTE_BASE_URL` + `MEETNOTE_API_KEY` + `MEETNOTE_MODEL`, for any OpenAI-compatible service (OpenAI, Kimi, Qwen, GLM, …) | your choice |
| `claude-cli` | nothing; needs the standalone Claude Code with `claude` on PATH and logged in | your Claude Code settings |
| `claude-api` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |

## Usage

```powershell
# Full run
meetnote run "D:\recordings\weekly.m4a"

# Names/terms and some background improve both recognition and summaries
meetnote run weekly.m4a --terms "Kubernetes,OKR,Alice Chen" --context "Product team quarterly roadmap review"

# If you know the head count, give it: speaker separation gets much more reliable
meetnote run weekly.m4a --speakers 3

# Summaries in another language than the recording (default: same as the recording)
meetnote run weekly.m4a --summary-lang en

# Transcribe only / re-summarize an existing transcript (e.g. after editing the prompts)
meetnote transcribe weekly.m4a
meetnote summarize weekly_notes

# Replace speaker labels with names, then regenerate the notes
meetnote rename weekly_notes S1=Alice S2=Bob
meetnote summarize weekly_notes
```

Output goes to `<file name>_notes\` in the current folder. File names follow the interface language (with `MEETNOTE_LANG=zh`: `<file name>_纪要\`, `转写稿.txt`, `两页精华.md`, …); a folder that already has files keeps its naming. The content of the notes follows `--summary-lang`.

```
weekly_notes\
  transcript.txt   timestamps and speakers, e.g. "[01:23] S2: …"; new line at every change of speaker
  transcript.srt   subtitles, to check against the audio in a player
  notes.md         section-by-section notes: views/facts/decisions, each with timestamp and what it refers to
  synopsis.md      the whole meeting in order, with citations
  highlights.md    conclusions first, fits on two A4 pages, with citations
  fact-check.md    what the fact-check pass found and fixed
  meta.json        duration, language, speakers, source file
```

### Languages

- The recording language is **detected automatically, segment by segment**: in a mixed Chinese/English meeting, Chinese stays Chinese and English stays English instead of being translated. Each speaker's stretch is re-checked against its own audio, and re-transcribed if Whisper wrote it in the wrong language. Use `--language zh` (or `en`, …) to force one language.
- Summaries follow the recording's language unless you set `--summary-lang`.

### Speakers

Lines are labelled `S1`, `S2`, … in order of first appearance. The notes attribute by label ("S1 argues …") and never guess real names; if someone introduces themselves or is addressed by name, they write e.g. "S2 (introduces themselves as Alice)".

- **Give the head count** with `--speakers N` ("参会人数" on the web page). Without it the count is estimated.
- **Use real names**: check what each label said in the transcript, then run `meetnote rename` (on the web page: "说话人 · 填写姓名"). If one person was split into two labels, give both the same name to merge them.
- How it works: pyannote segmentation-3.0 (via sherpa-onnx) finds speech and speaker changes; every ~1.5 s of speech gets a CAM++ voice embedding (trained on Chinese and English); windows are grouped by average-linkage clustering on voice similarity. Small clusters are folded into a speaker only if they sound like that speaker; otherwise they stay separate, so short-spoken participants such as Q&A questioners are kept. Speech the speaker detector heard but Whisper skipped is transcribed separately. No Hugging Face token needed; models are cached in `%USERPROFILE%\.cache\meetnote\models`.
- `--no-diarize` turns it off.
- Limits: only one speaker per line when people talk over each other; one- or two-word interjections are often mislabelled; far-field and phone recordings do worse.

### Summarization and accuracy

The default `grounded` pipeline targets one failure in particular: opinions attributed to the wrong product, version or person.

1. **Extract**: the transcript is split into ~10-minute sections; each view is noted with *what it is about* and a timestamp ("object unclear" when it can't be told).
2. **Write with citations**: synopsis and highlights are written from those notes, each point followed by `[mm:ss]`.
3. **Fact-check**: a separate call checks every statement against the transcript for unsupported claims, wrong object, wrong speaker, wrong direction/number and over-generalization.
4. **Revise** only what the check flagged.

`--pipeline simple` is the old single-pass mode: faster and cheaper, but long meetings get mixed up.

To score how faithful the notes are (statement by statement against the transcript; the judge defaults to `deepseek-reasoner`):

```powershell
meetnote check weekly_notes
```

The result is written to `faithfulness.md`. LLM judges make mistakes too (TofuEval found binary "is this supported?" judgments unreliable even with GPT-4), so use it to compare versions, not as an absolute score. How the pipeline was evaluated and improved: [docs/accuracy-evaluation.md](docs/accuracy-evaluation.md).

If a transcript already exists, `run` skips transcription; add `--force` to redo it.

## Options

| Option | What it does |
|---|---|
| `-o DIR` | output folder |
| `--terms` | names and terms, comma-separated, to improve recognition |
| `--context` | background about the meeting, to fix terms and understand context |
| `--only synopsis/highlights` | generate only one of the two |
| `--backend` | force a backend: `deepseek` / `openai` / `claude-cli` / `claude-api` |
| `--model` | model name |
| `--whisper-model medium` | faster, slightly less accurate transcription |
| `--device cpu` | no GPU (much slower) |
| `--language zh` | force the recording language (default: detect) |
| `--speakers 3` | number of participants (default: estimate) |
| `--no-diarize` | don't separate speakers |
| `--summary-lang en` | language of the notes (default: same as the recording) |
| `--pipeline simple` | single pass: faster, less accurate |

`MEETNOTE_CLAUDE` in `.env` can point to the `claude` executable.

## Web service (for colleagues)

Run it on one machine with an NVIDIA GPU; colleagues upload recordings and read or download the notes in the browser, with nothing to install. The page switches between Chinese and English (top right) and follows the browser language by default.

- Jobs are queued and transcribed one at a time; an hour of audio takes 3–5 minutes.
- Each job has an unguessable link (`/#job=…`) that can be shared.
- The original recording is deleted after transcription; only text is kept, for 30 days by default (`--keep-days`).
- If summarizing fails, the transcript is kept and the page offers "regenerate".
- Unfinished jobs resume after a restart.

### 1. Configure (once)

In `meetnote\.env` (the key stays on the server; colleagues don't need it):

```ini
DEEPSEEK_API_KEY=sk-your-key
# optional: access code asked for when opening the page
MEETNOTE_ACCESS_CODE=choose-one
# optional: port, default 18765
# MEETNOTE_PORT=18765
```

Restart the service after editing `.env`.

### 2. Start

Double-click `start-server.cmd` or run it in a terminal. The service listens on `http://localhost:18765` (port from `.env`) and stores data in `%USERPROFILE%\meetnote-data`. The first log line shows which `.env` was read.

Start on login (a service window opens after login; closing it stops the service):

```powershell
schtasks /Create /TN meetnote /SC ONLOGON /TR "<meetnote folder>\start-server.cmd"
```

If a laptop is the server, set "When plugged in, sleep after" to "Never".

### 3. Give colleagues access: Tailscale (recommended)

Install and log in to Tailscale on the server, then expose the service inside your tailnet only (not to the internet) with Tailscale's HTTPS proxy:

```powershell
tailscale serve --bg 18765
```

(Use the same port as in `.env`.)

Colleagues then open `https://<machine>.<your-tailnet>.ts.net` (`tailscale serve status` shows the full address). Two ways to let them in:

- **Invite them to the tailnet**: admin console → Users → Invite users.
- **Share only this machine**: admin console → Machines → this machine → Share. They accept with their own Tailscale account and can reach only this machine.

Colleagues need Tailscale installed and logged in. Stop sharing with `tailscale serve --https=443 off`.

### Alternative: office LAN

On a shared LAN you can skip Tailscale: set `MEETNOTE_HOST=0.0.0.0` in `.env`, start the service, and colleagues open `http://<server LAN IP>:18765`. An administrator also needs to allow the port in Windows Firewall (inbound, private network). This is plain http, so set an access code too.

## Customizing the output

The structure and rules of both summaries are in prompt files; edit them directly (the install is editable, so changes apply immediately):

- `meetnote\prompts\synopsis.md`: full synopsis
- `meetnote\prompts\highlights.md`: two-page highlights

The prompts are written in Chinese; the output language is set separately by `--summary-lang`, and the interface language by `MEETNOTE_LANG`.

## Known limitations

- Speaker labels are automatic and some short lines get the wrong speaker; check important "who said what" against the transcript.
- 8 GB of VRAM fits one transcription at a time: running the CLI while the web service transcribes makes both very slow.
- Transcripts are sent to the LLM provider you configure; keep that in mind for sensitive meetings.
- Names, organizations and mixed-language terms are often misheard. `--terms` helps, and the highlights end with a list of names to verify.
- Temporary files from apps such as WeChat are cleaned up automatically; save recordings elsewhere before processing.

## License and credits

Code under the [MIT](LICENSE) license. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [3D-Speaker CAM++](https://github.com/modelscope/3D-Speaker). Models download on first use, are not distributed with this repository, and follow their own licenses.
