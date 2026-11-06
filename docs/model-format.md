# Portable model contract, v1

A model is a directory containing `model.json` and only the declared data files.
The loader checks SHA256, byte counts, filenames, representation, dimensions,
finite scoring parameters and a fingerprint of engine/feature/scoring/file hashes.
No uploaded executable code is run. Supported engines:

| Engine | Representation | Payload |
|---|---|---|
| character-linear-int8-v1 | Unicode scalar char2–5/FNV1a/L2, 65536 dimensions | base.v1.json (INT8 linear head) |
| compact-bert-q8-v1 | 32k byteBPE, compact24×H128 encoder, learned128D unit pooler | encoder.q8_0.gguf, pooler, head.f32.bin |

`Model.predict(text, state)` returns junk, decisionMargin, probability and model
fingerprint. The calibrated decision margin is authoritative: do not recover a
boolean by comparing rounded probabilities near a threshold. Float head and
signed PA-II arithmetic use the portable C ABI; calibration is positive affine
Platt scaling fitted only on calibration data. Each quantized precision receives
its own calibration, but never a refitted SMS head or test-selected threshold.

`Model.learn([(text, 0_or_1), ...], state, cancelled=callback)` copies state before
updating. It returns new private weights; cancellation never changes input state.
Use only explicitly confirmed private preferences, not automatic model decisions.
PA-II fixes bias and threshold, C=1, margin=.05, clip=2, L2norm≤12, 24epochs per
warm replay. A new engine/fingerprint needs its own personal state. Do not carry
deltas between models, even when dimensions happen to match. Delete/rebuild,
rollback, model upgrade and cancellation are application responsibilities. The
reference benchmark uses the same 48 arrival streams/order for both models.

The Python interface is a research adapter. Deployed inference is native C++ and
Rust tokenizer code; it needs no Python or training framework. The Apple static
SDK exposes `pota_semantic_open/encode/close/error` and currently includes arm64
macOS14, arm64 iOS16 and arm64 iOS Simulator16. The portable core's signed
`pota_float_logit` / `pota_pa2_margin_sample` remain separate. CPU4 threads,
repack0, standard attention, max256 tokens / 16KiB UTF8 are frozen settings.
The shipping Flutter host performs both engines' PA-II training; the older
Swift-only character trainer is not the semantic host API.

POTASMS's app-facing BaseModel JSON format2 additionally carries a content-addressed
encoder descriptor, Float32 head, calibration and exact decisionLogit. The app
installs verified bundled public artifacts into App Group storage; the filter
reads them. Private SMS, feedback and versions are never bundled or sent to a
server. Arbitrary user model import is a future app feature, not included here.
