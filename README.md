# POTASMS Models

Reproducible public-data baselines for private, on-device SMS filtering:
**Lightweight**, a character classifier, and **Multilingual**, a9.04M-parameter
compact encoder distilled from BGE-M3 and supervised for SMS, exported as Q8.
Both expose the same prediction/private PA-II personalization interface.
No private phone history, imports, feedback or personal weights enter these bases.

The clean reference run reproduced all3 character export hashes and both student
checkpoint hashes exactly. Results are diagnostics, not a promise of real-world
accuracy. On768 fictional EN/ZH/mixed messages, Multilingual Q8 finds326/384 junk
with20/384 normal marked junk; fresh Lightweight seed42 finds54/384 junk with
1/384 normal marked junk. The models have different false-positive rates. See
[full results, languages, seeds, precision and corrections](results/benchmark-v1/README.md).

## Reproduce the shared bases

Python3.10–3.13, CMake, a C++17 compiler and Rust1.95 are needed. Student training
uses NVIDIA CUDA; the reference RTX4060Ti16GB run peaked around7.2GiB allocator
memory. Native export/evaluation and deployed inference use CPU only.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[train,test]'
# For the exact Linux/Python3.10 reference stack, also use:
# pip install -r requirements/reference-linux-py310.txt
potasms-models --work work/benchmark-v1 prepare
potasms-models --work work/benchmark-v1 fetch-models
potasms-models --work work/benchmark-v1 train-character
potasms-models --work work/benchmark-v1 train-student
potasms-models --work work/benchmark-v1 build-runtime
CUDA_VISIBLE_DEVICES='' potasms-models --work work/benchmark-v1 finalize-student
potasms-models --work work/benchmark-v1 package
CUDA_VISIBLE_DEVICES='' potasms-models --work work/benchmark-v1 evaluate-learning
```

Use a fresh work directory; stages refuse to overwrite results. Preparation
rehydrates exactly5261 SMS rows using pinned upstream sources, original fictional
generators and a membership manifest. The frozen8.15MB semantic snapshot is a
release asset with Tatoeba author/source/license attribution. Archives and member
files are SHA256-verified. Use `prepare --semantic-archive FILE` or
`build-runtime --runtime-archive FILE` for already-downloaded assets. Rust
`--offline` requires the Cargo.lock dependencies already cached.

`fetch-models` pins LEALLA-small and BGE-M3 revisions and every file hash.
`--model-cache DIR` accepts an explicit public-only cache with `lealla/` and
`bge/` directories; all listed files are verified. Otherwise it downloads upstream
HuggingFace weights. `HF_ENDPOINT` may select a user-chosen mirror; verification
is unchanged and token=False. The reference run used verified cached public
pretrained files after an upstream timeout, never old trained student weights.
`doctor` records Python/platform/package versions. CUDA/driver versions, GPU,
OS, toolchain and native hashes should also accompany a new run.

Character recipe: Unicode-scalar char2–5/FNV1a/L2,65536buckets, fresh SGD log-loss,
alpha1e-6/max_iter200/tol1e-3, balanced weights, symmetric INT8. Seed42 primary;
17/73 controls, never test-selected. Original probability-floor.995 and matched
calibration policies are reported separately. Student recipe: train-only32k
byteBPE, LEALLA body initialization,24layers/H128,3epochs BGE FP32 semantic
distillation, supervised SMS training with replay/hard negatives, final fixed
float head, then F32/F16/Q8 export and calibration-only threshold selection.
Additional general data, pretraining and compute make this a shared SMS label
budget comparison, not a matched-total-budget experiment.

## Use the trained packages

Download models and optional Apple SDK from the
[benchmark-v1 release](https://github.com/chux0519/potasms-models/releases/tag/benchmark-v1).
Archive SHA256 values are under manifests/. No Python is needed by the native
runtime. For a research CLI adapter (base dependencies only):

```sh
pip install -e .
potasms-models --work work/native build-runtime
potasms-models infer --model /path/to/lightweight --text 'Fictional family dinner reminder.'
potasms-models infer --model /path/to/multilingual --encoder-library work/native/runtime-build/cmake/libpotasms_encoder.so --text '虚构的家庭聚餐提醒。'
```

Use `.dylib` on macOS. Both calls return an authoritative boolean/margin;
probability is informational. The Python API can privately adapt either model:

```python
from potasms_models.model import Model
with Model('models/multilingual', encoder_library='libpotasms_encoder.so') as model:
    original = model.empty()
    personal = model.learn([('An entirely fictional loan advertisement.', 1)], original)
    print(model.predict('Another fictional credit promotion.', personal))
```

Personal state is caller-owned and bound to the exact model fingerprint.
Cancellation leaves input state intact. Do not publish private examples or
personal weights. See [the model/C ABI contract](docs/model-format.md).
Arbitrary custom model import into the iPhone app remains future work.

Apple SDK source rebuild:

```sh
python scripts/build_apple_sdk.py --source /path/to/runtime-source-v1 --out work/apple-sdk --developer-dir /path/to/Xcode.app/Contents/Developer
```

SDK slices are arm64 iPhone/iOS Simulator16 and macOS14; simulator development
currently requires Apple Silicon. CPU4 threads, repack0 and standard attention
are frozen runtime settings. If a new Xcode linker emits rejected host Rust
proc-macro dylibs, `--host-linker /path/to/older/Xcode.../usr/bin/clang` uses that
compiler's matching Mac SDK for host build tools while leaving the target SDK
explicit. It does not modify global Xcode selection. Current SDK has no Metal,
Python, PyTorch or background server dependency.

## Evidence boundaries and licenses

The main768-row diagnostic is entirely fictional and correlated by32scene
families. Historical public2084 rows are not fresh market acceptance. No modern
independent normal-SMS acceptance set is available. The empirical.1% calibration
cap is not a claim of.1% real-world false positives. Hardware/arithmetic changes
can alter rounding and calibration; same checkpoint bytes do not promise every
decision is identical on every device. Foreground iPhone memory measurements do
not establish IdentityLookup limits or locked-screen filtering acceptance.

[Data/attribution/license inventory](docs/data-and-licenses.md) documents sources,
changes and declared terms. First-party code/model adaptations are Apache-2.0;
source datasets, pretrained weights and vendored components retain their own
licenses. UCI/IMC25/MASSIVE/Banking77: CC BY4.0. Tatoeba: per-sentence CC0 or
CC BY2.0FR. LEALLA-small: Apache-2.0. BGE-M3 and embeddings.cpp: MIT. Preserve
all notices and per-row Tatoeba attribution when redistributing text.

Run `python -m pytest -q` after building the native core (`doctor` builds it), and
`ctest --test-dir build/native --output-on-failure` for the signed arithmetic,
PA-II and Unicode native checks. The release includes immutable result reports
and reproduction hashes; failures and historical corrections remain visible.
