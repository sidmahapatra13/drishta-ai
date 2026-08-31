# Clinical knowledge base

Source documents for the RAG assistant. Owner: P6.

RAG explains, it never diagnoses. The vision model is the sole source of truth
for the grade; the assistant receives the structured model output as context and
grounds its explanation in these documents.

- `clinical/` - International Clinical DR Severity Scale, WHO screening
  guidance, referral guidelines
- `research/` - DR classification, lesion detection, Grad-CAM, image quality
- `datasets/` - APTOS, IDRiD documentation
- `model_docs/` - training protocol, validation metrics, **model limitations**,
  version notes

`model_docs/model_limitations.md` is the one judges will respect most. Write it
honestly: benchmark and external validation only, no clinical validation,
Grad-CAM is not lesion proof.

Retrieval is numpy cosine similarity over ~40 chunks. At this scale a vector
database is setup cost with no benefit.
