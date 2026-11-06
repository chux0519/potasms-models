# Benchmark v1 — public reproduction, 2026-11-06

These are diagnostics, not independent market acceptance. A shared1024-SMS label budget does not equal a shared total training budget: Multilingual also uses pretrained LEALLA/BGE knowledge and132137 general semantic rows. No claim of superiority at equal false-positive rate is supported. Thresholds were frozen using768calibration rows; evaluation never selected seeds or thresholds.

| Model / panel | Correct | FP / normal | FPR95% Wilson | TP / junk | Junk recall |
|---|---:|---:|---:|---:|---:|
| Lightweight seed42, original policy / background | 56.90% | 1/384 | 0.05–1.46% | 54/384 | 14.06% |
| Lightweight seed42, original policy / background-en | 61.72% | 0/128 | 0.00–2.91% | 30/128 | 23.44% |
| Lightweight seed42, original policy / background-zh | 51.56% | 0/128 | 0.00–2.91% | 4/128 | 3.12% |
| Lightweight seed42, original policy / background-mixed | 57.42% | 1/128 | 0.14–4.29% | 20/128 | 15.62% |
| Lightweight seed42, original policy / old-cohort | 80.81% | 1/452 | 0.04–1.24% | 1233/1632 | 75.55% |
| Lightweight seed42, original policy / old-language | 60.00% | 6/72 | 3.88–17.01% | 45/113 | 39.82% |
| Multilingual Q8, native Linux / background | 89.84% | 20/384 | 3.40–7.91% | 326/384 | 84.90% |
| Multilingual Q8, native Linux / background-en | 92.58% | 4/128 | 1.22–7.76% | 113/128 | 88.28% |
| Multilingual Q8, native Linux / background-zh | 89.06% | 8/128 | 3.20–11.85% | 108/128 | 84.38% |
| Multilingual Q8, native Linux / background-mixed | 87.89% | 8/128 | 3.20–11.85% | 105/128 | 82.03% |
| Multilingual Q8, native Linux / old-cohort | 86.71% | 10/452 | 1.21–4.02% | 1365/1632 | 83.64% |
| Multilingual Q8, native Linux / old-language | 79.46% | 10/72 | 7.72–23.71% | 85/113 | 75.22% |

The per-language intervals above are nominal binomial intervals. Parallel synthetic templates are correlated, so these intervals do not establish independent real-world confidence. Only128 fictional normal examples occur per language in this background panel. Modern independent normal SMS acceptance data:0.

## Seeds, calibration and precision

All seeds17/42/73 and both character policies (original probability floor.995 and matched positive-affine Platt/common-language cap.001) are retained in character.json. Seed42 is primary, never a test-selected winner. All student F32/F16/Q8 results and calibration values are in student-native-linux.json; the float-trained SMS head is not refitted for quantization. The empirical calibration cap is not a measured or promised market FPR.

The clean reconstruction matched all3 character export hashes, student initialization/tokenizer tensors, semantic epoch3 checkpoint baddc86b... and SMS epoch5 checkpoint f8ac722c... exactly. See reproduction.json. It used a verified public-only pretrained cache after upstream download timed out. No previous trained checkpoint or private data was imported.

## Historical correction and platform boundary

An earlier internal evaluator compared the shipped character model’s probabilities against a logit threshold, incorrectly reporting50%/0TP/0FP. That arm is invalid. Correct historical shipped-base results on background768 are52.73%,FP168/384,TP189/384; on old2084 they are91.27%,FP0/452,TP1450/1632. This repo publishes a fresh character retrain rather than that historical artifact.

An earlier Mac/NumPy-head Q8 evaluation reported background89.71%,FP20,TP325 and old86.47%,FP10,TP1360. Current Linux/native scalar-C-ABI head gives89.84%,FP20,TP326 and86.71%,FP10,TP1365. The same trained checkpoint hashes do not imply bit-identical inference/calibration across hardware and arithmetic paths. Both records remain explicit; neither is silently overwritten.

## Personalization and device evidence

The public evaluate-learning command runs48 fictional preference arrival streams per engine, two labels/two source languages/three order seeds/four topics, with reports at0/1/5/10 corrections. Novel opposite-language targets and same-topic protected normal notices are measured separately. Learning can improve preference matching while increasing normal-message mistakes; examine the full result before choosing a model.

A prior same-architecture Q8 pilot on an iPhoneX foreground app measured roughly103ms load+first,9.27ms warm inference and28.36MiB sampled peak footprint. This is not a locked-screen SMS-extension memory limit, cold-start SLA or Jetsam acceptance result for the final packaged app. CPU-only, no Python inference dependency.


## Fresh same-data PA-II comparison

| Corrections | Lightweight cross-language preference | Multilingual cross-language preference | Lightweight protected normal FPR (Junk preference) | Multilingual protected normal FPR |
|---|---:|---:|---:|---:|
| 0 | 50.00% | 50.00% | 8.33% | 6.25% |
| 1 | 50.00% | 63.54% | 8.33% | 14.58% |
| 5 | 50.00% | 75.00% | 9.72% | 19.79% |
| 10 | 50.00% | 83.33% | 10.42% | 21.18% |

These48 warm-replay streams per model use fictional preferences only. Cross-language preference accuracy averages both Keep and Junk preferences on novel opposite-language targets; protected FPR is for same-topic normal notices after Junk preferences, averaged over EN/ZH/mixed and arrival orders. These are different populations, not one combined accuracy. Full per-stream counts and background changes are in personal-learning-v1/. No personalization acceptance claim follows from these small correlated probes.
