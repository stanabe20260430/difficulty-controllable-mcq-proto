# Difficulty-Controllable Multimodal MCQ Generation

A pipeline that automatically generates image-based multiple-choice
questions (MCQs) for English-language learning while controlling their
difficulty. Each item shows one image with one correct caption and three
distractor captions, and the task is to choose the caption that best
matches the image.

## Overview

- **Local and leakage-free.** Image and caption generation run entirely on
  local models on a single consumer GPU, so no items are exposed to
  external services.
- **Two independent difficulty axes.** A *visual* axis (people count, camera
  angle, subject distance, lighting, foreground occlusion) and a *language*
  axis (vocabulary and sentence length, in three levels).
- **Per-learner adaptation.** A Level Estimator (MLP) predicts a learner's
  proficiency from their answer history, which can be used to select the
  difficulty of subsequent items.
- **Simulated learner.** In place of human learners, a persona-conditioned
  Claude Haiku 4.5 answers the items; its proficiency is approximated by
  vocabulary masking.

## Pipeline

| Script | Role |
|---|---|
| `src/image_generator.py` | Generate images (FLUX.2-dev NF4) with a YOLO people-count gate |
| `src/mcq_caption_generator.py` | Generate captions (Qwen3-VL-32B, 4-bit): 1 correct + 3 distractors |
| `src/language_difficulty_measurer.py` | Measure language difficulty of each caption |
| `src/visual_difficulty_measurer.py` | Measure visual difficulty (YOLO) of each image |
| `src/make_questions.py` | Assemble one MCQ per item (shuffle choices) |
| `src/solver.py` | Answer a single MCQ with a VLM (Claude API) |
| `src/evaluate_haiku.py` | Batch-evaluate persona Haiku; log per-level accuracy and answer rows |
| `src/caption_masker.py` | Vocabulary masking used to implement personas |
| `src/split_dataset.py` | Merge per-`p` answers and split into train/val/test (70/15/15) |
| `src/train_level_mlp.py` | Train the Level Estimator MLP |
| `plot/*.py` | Figures (accuracy bars, language/visual difficulty, level estimator) |

## Visual axes

| Axis | Meaning | 0 | 1 |
|---|---|---|---|
| `p` | people count | — | 1 / 2 / 4 |
| `a` | camera angle | eye-level | overhead |
| `s` | subject distance | medium | wide |
| `l` | lighting | day | night |
| `o` | foreground occlusion | none | present |

## Caption generation

For each image, the VLM first writes one **correct** caption (subject,
action, location), then rewrites it into three **distractors**, each
corrupting a single semantic axis (`wrong_subject` / `wrong_action` /
`wrong_location`) with a wrong value from the vocabulary. Distractors are
written without re-examining the image. Every caption is verified against
the image and regenerated on failure; the leading axis of each caption is
shuffled to remove positional shortcuts. Language difficulty is set by
three instruction texts (L1/L2/L3) that specify vocabulary and sentence
length.

## Learner simulation (personas)

Personas are implemented by vocabulary masking
(`caption_masker.py`, `configs/personas_config.json`). Content words whose
Zipf frequency is below a threshold (or whose CEFR level is above a
threshold) are masked before the model answers, approximating P1/P2/P3
learners; lower proficiencies mask more words.

## Level estimator

Each answer is encoded as a 12-dimensional vector (visual axes `p,a,s,l,o`,
a one-hot of the vocabulary level, and a one-hot of the chosen option
kind). A window of 100 answers is mean-pooled and passed to an MLP
(12–64–64–3) trained with cross-entropy to predict P1/P2/P3. The discrete
level is the argmax; the continuous level is the expected value, used for
the MAE metric.

## Difficulty metrics

- **Language:** `min_zipf` (lowest-frequency content word), `max_cefr_rank`
  (A1=1 … C2=6), `length`.

## Models

- Image: FLUX.2-dev-bnb-4bit (`guidance_scale=4.0`, `steps=28`, `1024²`)
- Caption: Qwen3-VL-32B-Instruct (4-bit, NF4)
- Detection: YOLO11x
- Solver / simulated learner: Claude Haiku 4.5 (Anthropic API)

Runs on a single RTX 4090 (24 GB). 

## Install

```bash
pip install -U "transformers>=4.57.0" diffusers accelerate sentencepiece \
  protobuf bitsandbytes spacy wordfreq pandas ultralytics anthropic torch
python -m spacy download en_core_web_sm
hf auth login
export ANTHROPIC_API_KEY=...
```

## Run

```bash
# 1. Generate one image (single scene, all visual axes fixed)
python src/image_generator.py --vocab configs/vocab_leveled.json \
  --scenes 1 --p 1 --a 0 --s 0 --l 0 --o 0 --seed 200 --output-dir images/

# 2. Generate captions (1 correct + 3 distractors)
python src/mcq_caption_generator.py --images-dir images/ \
  --vocab configs/vocab_leveled.json

# 3. Measure difficulty
python src/language_difficulty_measurer.py --mcq-dir images/
python src/visual_difficulty_measurer.py --images-dir images/

# 4. Assemble the MCQ
python src/make_questions.py --mcq-dir images/ --n-choices 4 --seed 200
```

## Config files

Wording and parameters are externalized to JSON (`--config` to override).

- `configs/image_generator_config.json` — model/generation params, axis
  phrases, prompt parts
- `configs/mcq_caption_config.json` — model, temperatures, retries, levels,
  semantic-axis vocabulary, prompts
- `configs/personas_config.json` — masking thresholds (Zipf / CEFR / clause
  depth)
- `configs/vocab_leveled.json` — leveled scene vocabulary


## Target venue

IEEE TALE 2026 (Pattaya, Dec 2–4, 2026). Full paper, 4–6 pages, IEEE
Xplore, double-anonymous review.
