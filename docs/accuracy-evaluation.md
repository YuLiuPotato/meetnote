# Meeting-note accuracy: causes, fixes and evaluation (2026-11-04)

**English** | [简体中文](准确性改进与评测_2026-11-04.md)

## Reported problems

- When two products were discussed one after the other, opinions about one ended up attached to the other (e.g. praise for one version written as "the sensor works better").
- Statements were attributed to the wrong person or put in the wrong category.
- The notes drifted too far from what was actually said.

## Causes (largest impact first)

| # | Cause | Evidence | Fix |
|---|---|---|---|
| 1 | Transcription was forced to `language=zh`, so Whisper **translated English speech into Chinese** while transcribing | In an English business conversation, sentences comparing two products came out as vague Chinese that no longer said which one was better; mixed traditional/simplified characters | Detect the language automatically and per segment (`multilingual=True`), so each part of a mixed meeting stays in its own language |
| 2 | Whisper invents stock video-subtitle phrases over silence | First line: "please like, subscribe, share…" | Filter known hallucinated phrases |
| 3 | The transcript joined neighbouring segments with commas | "Is it X? I think it's X, yes": a question and its answer glued into one person's sentence | One line per segment, no merging |
| 4 | An 82-minute transcript summarized in a single pass; the prompt cast the model as a "strategy consultant"; default temperature 1.0; no citations, no verification | The old highlights had 2.2 problems per 1,000 characters, including 3 wrong-object and 4 over-generalization errors | Extract per section → write with citations → fact-check per section → revise (below) |
| 5 | No speaker separation | Almost all remaining attribution errors were "one person's continuous speech split into two sides" or "a restatement taken as the speaker's own view" | Targeted rules first; speaker diarization in round 2 |

## Grounded summarization pipeline (`--pipeline grounded`, default)

1. **Extract per section**: ~10-minute sections, each with the preceding text attached to resolve "this/that/the earlier one". Every view records *what it is about* and a timestamp; "object unclear" when it can't be told; "you mean…" is recorded as a restatement.
2. **Write with citations** from the section notes; every point ends with `[mm:ss]`. Faithfulness rules: only what the transcript says; never move an opinion to another object; don't attribute speakers without evidence; don't invent "one side … the other side"; an opinion is not a consensus and a question is not a conclusion.
3. **Fact-check per section**: a separate call checks ~2,500 characters at a time against the transcript for unsupported claims, wrong object, wrong speaker, wrong direction/number, over-generalization and wrong timestamps.
4. **Revise** only what was flagged. The findings are saved in `核查记录.md`.

Temperatures: extraction 0.2, writing 0.5, checking 0 (DeepSeek's default is 1.0; its docs suggest 0.7 for data analysis).

## Evaluation method

`meetnote check <folder>`, judged by `deepseek-reasoner` (a different model from the writer, `deepseek-chat`, to avoid self-grading):

- **Problem density**: the judge searches each section for errors; we count problems per 1,000 characters by type. It discriminates better and is the main metric.
- **Statement faithfulness**: following FineSurE, the notes are split into single statements and judged in batches of 10 for support by the transcript; "supported" requires a quoted transcript line.

When we first judged whole documents at once, old and new versions both scored 98%, while the same model checking section by section found dozens of real problems. This matches TofuEval: LLMs are lenient at binary "is this supported?" judgments. Hence small batches, mandatory quotes, and problem density as the main metric.

## Results (round 1)

Test meeting: an internal product discussion (Chinese, 60 minutes, two people).

| Version | Highlights: problems / 1k chars | Highlights: statements supported | Synopsis: problems / 1k chars | Synopsis: statements supported |
|---|---|---|---|---|
| Old: single pass | 2.2 (wrong object 3, over-generalization 4) | 93% | 0.6 | 87% |
| New: grounded v2 | 0.9 | 94% | 0.8 | 97% |
| New: grounded v3 (+ attribution rules) | **0.9** (wrong object 2, over-generalization 0) | **96%** | 0.7 (wrong speaker 3→1) | 88% |

- **Highlights** clearly improved: problem density down ~60%, over-generalizations from 4 to 0.
- **Synopsis** about the same as before. Statement faithfulness swung between 88% and 97% across runs, so one meeting and one judging run can't support finer conclusions.
- Most remaining problems in v3 came from **speech recognition** (homophones), **ambiguity in the recording** (numbers that contradict each other), and the **lack of speaker separation**.

Limitations: one test meeting, one judging run per version; the judge makes mistakes too. A fixed set of 5–10 human-checked meetings would be more reliable.

## Round 2: speaker separation

### Approach

1. **Find speech and speaker changes**: sherpa-onnx runs pyannote segmentation-3.0 (ONNX, CPU, no Hugging Face token).
2. **Re-cluster**: sherpa-onnx's own clustering merged two women into one speaker on an AMI meeting and spent a slot on scraps. Instead, each turn is cut into ~1.5 s windows, each window gets a 3D-Speaker CAM++ embedding (trained on Chinese and English), and windows are grouped by duration-weighted average-linkage clustering (stop at cosine similarity 0.5). Clusters with less than ~60 s of speech (bounded to 2–5% of the meeting) are folded into a big cluster only if they sound like it; otherwise they are grouped among themselves and kept as separate speakers. With a known head count, finer clusters are kept and then merged down to that count.
3. **Align with the transcript**: each Whisper segment gets the speaker it overlaps most. A segment spanning two speakers is cut at the change and each part re-transcribed; the split is kept only if the parts read almost the same as the original (similarity ≥ 0.75), so short clips can't introduce hallucinations.
4. Speaker separation runs on the CPU while Whisper runs on the GPU.

### Diarization results

| Recording | Method | Speakers found | DER | of which speaker confusion |
|---|---|---|---|---|
| AMI ES2004a (17 min, 4 people, real English meeting, human labels) | sherpa-onnx built-in clustering, told 4 speakers | 4 | 42.7% | 26.5% |
| same | new method, head count not given | **4** | **22.2%** | **3.2%** |
| same | new method, told 4 speakers | 4 | 22.2% | 3.2% |
| synthetic zh/en dialogue (65 s, 3 voices) | new method, head count not given | **3** | 23.7% | **0%** |
| sherpa Chinese 4-speaker sample (57 s) | new method, head count not given | **4** | no labels | same as when told 4 |

Most of the remaining DER is *missed speech*: about 15% of the AMI meeting has two people talking at once, while each transcript line carries one speaker; on the synthetic dialogue the segmentation model is less sensitive to synthetic voices. What matters for attribution in the notes is *speaker confusion*. For reference, pyannote 3.1 reports ~18.8% DER on AMI under the same protocol. After the similarity-gated folding was added (see the pitch below), AMI gets one extra 8-second cluster and confusion of 3.7%.

### Two transcription bugs found along the way

- **English translated into Chinese, with the meaning flipped**: in multilingual mode Whisper picks one language per 30-second window. In the synthetic dialogue, English right after Chinese was translated: "I disagree a little" became "我认同一点" (I agree a little), "heavier" became "比较厉害" (rather impressive). Now each stretch of one speaker is language-checked on its own audio and re-transcribed when the script of the text doesn't match the detected language. All 10 lines correct after the fix.
- **Whole sentences skipped**: Whisper's voice-activity filter drops some real speech (quieter voices, speech over background sound). In the Chinese sample, 7–17 s and 28–38 s had no text at all. Speech that the diarizer detected but Whisper didn't transcribe is now transcribed separately, keeping only output that Whisper itself is confident about and that is at least 4 words long; the sample went from 5 lines to 11. Separately, Whisper's word-level timestamps made a mixed-language recording lose 4 sentences, so they are not used for splitting speakers.

### Real recording: a 9-minute English startup pitch + Q&A

The old version (forced Chinese, no speaker separation) produced a 29-line transcript with three long gaps, and some English came out translated into Chinese (a university name and an order were mistranslated). The new transcript has 148 lines, covers the whole talk, and separates the two questioners in the Q&A.

Both versions of the notes were judged against the **new** transcript (judge: deepseek-reasoner):

| Version | Highlights: problems / 1k chars | Highlights: statements supported | Synopsis: problems / 1k chars | Synopsis: statements supported |
|---|---|---|---|---|
| Old | 6.5 (unsupported 10, wrong object 3, wrong speaker 3, wrong number 3) | 56% | 6.7 | 58% |
| New, speakers as S1–S5 | **0.5** | **86%** | **0.5** | **100%** |
| New, presenter's two labels given one name | 0.0 | 100% | 0.0 | 100% |

The old version's typical errors all came from the transcript: "no funding was discussed" (the pitch made an explicit funding ask, in a stretch missing from the old transcript); "we make our own hardware" written as "in-house software" (wrong object); customer names and order details mistranslated.

This recording also exposed and fixed three problems: (1) Python's own certificate check failed to download the models on this machine, so downloads now use the OS certificate store (truststore); (2) a term hint written as "Glossary: …" was copied into the transcript by Whisper, so terms are now passed as a bare list; (3) folding every small cluster into the nearest big one merged the Q&A questioners into the presenter, so folding now requires a similar voice.

Remaining speaker errors: the presenter was split into two labels during Q&A (fixed by giving both the same name), and about 5 short lines have the wrong speaker.

## Next steps

1. **Re-evaluate on more real meetings** (Chinese and mixed-language) with `meetnote check`, to see how much attribution errors drop with speaker separation.
2. **Diarization on Chinese meetings**: AliMeeting / M2MeT is the standard dataset for Chinese meeting diarization and transcription; compute DER on a few sessions.
3. **A small fixed test set**: 5–10 real meetings with key facts checked by hand, to track every change with `meetnote check`.
4. **Term lists**: the "names/terms" field directly affects recognition, e.g. company names, people's names and product generations.

## References

- DiaSumFact: Annotating and Detecting Fine-grained Factual Errors for Dialogue Summarization (ACL 2023) — https://arxiv.org/abs/2305.16548
- Analyzing and Evaluating Faithfulness in Dialogue Summarization (EMNLP 2022) — https://arxiv.org/abs/2210.11777
- TofuEval: Evaluating Hallucinations of LLMs on Topic-Focused Dialogue Summarization (2024, includes MeetingBank) — https://arxiv.org/abs/2402.13249
- FineSurE: Fine-grained Summarization Evaluation using LLMs (ACL 2024) — https://arxiv.org/abs/2407.00908
- Chain-of-Verification Reduces Hallucination in Large Language Models (Findings of ACL 2024) — https://arxiv.org/abs/2309.11495
- M2MeT: The ICASSP 2022 Multi-Channel Multi-Party Meeting Transcription Challenge (AliMeeting) — https://arxiv.org/abs/2110.07393
- AMI Meeting Corpus (CC BY 4.0) and pyannote's AMI evaluation setup — https://github.com/pyannote/AMI-diarization-setup
- Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap (Park et al., 2019; tried, unstable on short recordings, not used) — https://arxiv.org/abs/2003.02405
- sherpa-onnx speaker diarization — https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
- DeepSeek API temperature settings — https://platform.deepseek.com/api-docs/zh-cn/temperature
