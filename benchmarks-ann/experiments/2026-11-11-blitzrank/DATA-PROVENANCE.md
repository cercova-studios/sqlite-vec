# Data and upstream attribution

The preserved inputs use BEIR's SciFact conversion of **Fact or Fiction:
Verifying Scientific Claims** by David Wadden, Shanchuan Lin, Kyle Lo,
Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan and Hannaneh Hajishirzi (2020).
BEIR is described by Nandan Thakur and colleagues in **BEIR: A Heterogeneous
Benchmark for Zero-shot Evaluation of Information Retrieval Models** (2021).

The [BEIR SciFact dataset card](https://huggingface.co/datasets/BeIR/scifact)
declares [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), checked
2026-11-11. Preserve that attribution and license reference with dataset-derived
text in the recovery snapshot's input records and model prompts. This notice
does not assert separate rights over the underlying scientific publications.
The [original SciFact repository](https://github.com/allenai/scifact) supplies
the dataset's original context and citation.

This experiment joins each title and abstract with a single space, selects
previously defined query IDs, tokenizes lexical claims, and truncates model
passages to the first 1,024 `o200k_base` tokens. It creates ranking prompts and
response/metric receipts from those inputs. No author endorsement is implied.
The raw original zip SHA256 and transformation source hashes are recorded in
the protocol and input inventory.

BlitzRank's tournament modules and RankGPT prompt are from
[ContextualAI/BlitzRank](https://github.com/ContextualAI/BlitzRank/tree/076018cf6153608203029e29562c36f50273b348),
revision `076018cf6153608203029e29562c36f50273b348`, under its MIT license.
The recovery snapshot retains `UPSTREAM-LICENSE`.

The offline tokenizer cache is `o200k_base`, SHA256
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.
The recovery snapshot retains the installed tiktoken package's MIT license as
`TIKTOKEN-LICENSE`. These licenses are distinct from the dataset attribution.
