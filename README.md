# POTASMS Models

Public recipes for two local SMS filtering baselines: a small character classifier
and a distilled multilingual encoder with a supervised SMS head. Everyone starts
from the same public data, membership, seeds and recipes; personal examples and
adaptation stay on the user's own device.

This repository is being validated against the frozen benchmark before its first
release. Models are research baselines, not a claim of production false-positive
rates. Synthetic, public historical and private personal data must stay distinct.

## Reproduce

Use Python3.10–3.13, CMake and a C++17 compiler. GPU student training was measured
with an RTX4060Ti16GB. Pin dependencies with the project metadata:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[train,test]'
potasms-models --work work/benchmark-v1 prepare
potasms-models --work work/benchmark-v1 fetch-models
potasms-models --work work/benchmark-v1 train-character
potasms-models --work work/benchmark-v1 train-student
```

Preparation rehydrates5261rows byte-for-byte from pinned public UCI/IMC2025 files
and original fictional scenario generators. The default first release also hosts
the frozen semantic snapshot, preserving Tatoeba author/URL/license attribution
and MASSIVE/Banking source records. Never substitute a rotating dataset silently.
Reuse cached downloads only after verifying hashes. Use a fresh work directory
for a new run; training never overwrites an old run.

The character recipe preserves the shipped algorithm: Unicode-scalar char2–5
grams, FNV1a/L2,65536buckets, SGD log-loss, class-balanced weights and symmetric
INT8 storage. Seed42 is primary;17/73 are controls, not test-selected winners.
The student keeps a24layer/H128 LEALLA body with a new train-only32k byteBPE,
distills BGE FP32 semantic representations, then receives supervised SMS training.
Its additional general data, pretrained models and compute are disclosed.

## Data and evaluation boundaries

The main768row EN/ZH/mixed diagnostic is entirely synthetic:32scenes with
correlated slot and language variants. Train1024=768synthetic+133UCI+123IMC25.
Dev192 and calibration768 are also synthetic with separate scene families.
The older public test has2084rows (520UCI+1564IMC25); it is a historical diagnostic,
not fresh market acceptance. Personal-learning probes are fictional, never users'
messages. No private phone history, imports, feedback or personal weights are used.

Report actual TP/FP and per-language denominators, not only accuracy. A0.1%
empirical calibration cap is not a promise of0.1% real-world FPR. Independent
modern normal SMS are still missing. Hardware/library differences can change
float rounding and training bytes; reproducibility includes hashes and numerical
checks, with exact-configuration reference results distinguished from tolerances.

## Licenses

First-party code: Apache-2.0. Vendored utf8proc/PCRE2 retain their own notices.
UCI and IMC2025: CC BY4.0. Tatoeba: per-sentence CC0 or CC BY2.0FR, with author
and source attribution. MASSIVE1.1 and Banking77: CC BY4.0. LEALLA-small's pinned
model card declares Apache-2.0; BGE-M3's pinned model card declares MIT. Dataset
and pretrained-model terms remain separate from this code license.
