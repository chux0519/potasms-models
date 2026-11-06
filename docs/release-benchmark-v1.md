Public benchmark v1 contains reproducible character and distilled multilingual SMS baselines, frozen data/runtime snapshots, native CPU model packages and an arm64 Apple SDK.

A full clean reconstruction matched all three character export hashes and both student checkpoint hashes. Results include per-language TP/FP counts, Wilson intervals with correlation limits, all seed/precision controls and 48 fictional PA-II preference streams per engine.

These are diagnostic/research baselines, not independent market-accuracy or equal-FPR claims. Multilingual catches more junk on the synthetic multilingual panel with more false positives; private personalization can also affect protected normal messages. Historical evaluator corrections and hardware/arithmetic differences remain explicit.

No private phone history, imports, feedback, personal weights or credentials are included. Preserve source/data/pretrained/runtime licenses and per-sentence Tatoeba attribution. Deployed native inference needs no Python. CPU4/repack0/standard attention; current Apple SDK supports arm64 iPhone/iOS Simulator16 and macOS14.

Use README.md, the manifests and SHA256SUMS to reproduce or verify each release asset. The reference transitive Linux/Python3.10 environment is frozen separately. See results/benchmark-v1/README.md for limitations and numerical evidence.
