# WebFace4M Local Target

This folder stores the local WebFace4M download in WebDataset shard form.

- Source: gaunernst/webface4m-wds-gz on Hugging Face
- Format: `.tar.gz` WebDataset shards
- Native manifest-backed training is supported through `Training/dataset.py`
- After download, run `scripts/prepare_webface4m_manifests.py` to build `train.jsonl` and `val.jsonl`
