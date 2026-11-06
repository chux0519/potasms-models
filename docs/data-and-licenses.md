# Frozen public inputs

The SMS membership manifest contains hashes, keys and split metadata, not raw
SMS. `prepare` rehydrates precisely5261 rows and verifies ordered rows SHA256
b4d4b041941c34feaa8595f55dd40c9bf54ba5dc7be0230b51c059f720d21291.
Do not replace the public snapshot with private examples or a rotating download.

| Source | Frozen location / attribution | License |
|---|---|---|
| UCI SMS Spam Collection | Almeida, Hidalgo & Yamakami; DOI10.24432/C5CC84; upstream zip SHA in data.py | UCI declares CC BY4.0 |
| IMC2025 Smishing | Agarwal et al., Fishing for Smishing; DOI10.1145/3730567.3764431; reportsmishing/Smishing-Dataset-IMC25 commit a6175560b57387199871e51fbef6bc523d2516b4 | CC BY4.0 |
| Authored SMS scenarios | Original fictional learning/calibration/diagnostic texts in this repository, November2026 | Apache-2.0 first-party source/data |
| Tatoeba | Selected EN/ZH sentences; sentence IDs, author, source URL and license retained per row | CC0 or CC BY2.0FR per sentence |
| MASSIVE1.1 | Amazon Science multilingual intents; source/locale/split retained | CC BY4.0 |
| Banking77 | PolyAI banking intent corpus; source IDs retained | CC BY4.0 |

The semantic snapshot has132137 rows:94352train/11244dev/26541heldout, with
translation/template/numeric-duplicate family isolation. Selection, normalization,
paired examples and train-only tokenizer construction are adaptations. Per-row
Tatoeba credit remains mandatory when redistributing sentence text. Preserve
licenses/, all source attributions and semantic manifest when sharing this asset.

The SMS train has1024 rows:768 fictional +133UCI +123IMC25. Dev192/calibration768
are fictional with separate scene families. Background768 is fictional EN/ZH/mixed,
32scene families ×4slots ×2classes ×3languages: correlated diagnostics, not768
independent real messages. Old-cohort2084 has520UCI+1564IMC25; old-language185
has144fictional+41IMC25. Personal240 is fictional and labels depend on user
preference. Source and label are highly correlated in these small samples.

LEALLA-small (setu4993, revision8fadf81fe3979f373ba9922ab616468a4184b266,
Apache-2.0) initializes the compact encoder. BGE-M3 (BAAI,
revision5617a9f61b028005a4858fdac845db406aefb181, MIT) is the semantic teacher.
Both model cards and all pinned pretrained file hashes are retained. BGE teacher
licensing and source data terms carry through distillation; no Gemma or private
phone corpus is used. Model artifacts are Apache-2.0 adaptations with all upstream
notices preserved, not a relicensing of source datasets or pretrained weights.

Runtime source: embeddings.cpp commit a349eb1a8b0062a621b981e01db40534a9d19ed9
(MIT), modified CPU loading/tokenizer/pooler C ABI. Its frozen source archive
contains complete code and per-file provenance. Retain the embedded GGML and
third-party notices; HuggingFace tokenizers0.22.2 and tokenizer wrapper are
Apache-2.0. Rust dependency license texts/versions are in licenses/runtime-crates
and runtime-dependencies.json. PCRE2/utf8proc notices are retained under native/.

Source links: [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
[IMC25](https://github.com/reportsmishing/Smishing-Dataset-IMC25),
[Tatoeba](https://tatoeba.org/en/downloads),
[MASSIVE](https://github.com/alexa/massive),
[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data),
[LEALLA](https://huggingface.co/setu4993/LEALLA-small),
[BGE](https://huggingface.co/BAAI/bge-m3),
[embeddings.cpp](https://github.com/chux0519/embeddings.cpp).
