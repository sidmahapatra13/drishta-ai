"""Clinical RAG assistant.

STUB retrieval over a small curated knowledge base. Replaced on day 3 by
embedding-based retrieval (numpy cosine similarity over ~40 chunks - no vector
database needed at this scale).

Architectural rule, and the thing that must not be broken: RAG explains, it
never diagnoses. The vision model is the sole source of truth for the grade.
The assistant receives the structured model output as context and grounds its
explanation in retrieved clinical documents.
"""

from ..contract import Citation, ClinicalAnswer

#: Replaced by ingested PDFs (International DR scale, WHO screening guidance,
#: referral guidelines) chunked and embedded. Kept inline for the skeleton so
#: the endpoint returns real citation shapes on day one.
KNOWLEDGE_BASE = [
    {
        "title": "International Clinical DR Severity Scale",
        "source": "clinical/international_dr_scale.md",
        "text": (
            "Moderate NPDR (Grade 2) is characterised by more than just "
            "microaneurysms but less than severe NPDR. Grade 2 and above is "
            "classified as referable diabetic retinopathy."
        ),
        "keywords": ["moderate", "grade 2", "scale", "severity", "npdr"],
    },
    {
        "title": "WHO Screening Guidance",
        "source": "clinical/who_screening_guidelines.md",
        "text": (
            "Screening programmes should prioritise sensitivity for referable "
            "disease, since a missed referable case carries greater harm than a "
            "false positive that is resolved at specialist review."
        ),
        "keywords": ["referral", "referable", "sensitivity", "screening", "why"],
    },
    {
        "title": "Model Limitations",
        "source": "model_docs/model_limitations.md",
        "text": (
            "The model is validated on public benchmark datasets and external "
            "IDRiD data. It has not undergone clinical validation. Grad-CAM "
            "indicates influential image regions and does not confirm a lesion."
        ),
        "keywords": ["limitation", "validation", "grad-cam", "gradcam", "confidence"],
    },
]


def retrieve(question: str, k: int = 2) -> list[dict]:
    """Keyword overlap scoring. Replaced by cosine similarity over embeddings."""
    q = question.lower()
    scored = [
        (sum(1 for kw in doc["keywords"] if kw in q), doc) for doc in KNOWLEDGE_BASE
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    hits = [doc for score, doc in scored if score > 0][:k]
    return hits or [KNOWLEDGE_BASE[0]]


def ask(question: str, model_output: dict | None = None) -> ClinicalAnswer:
    """Answer a clinician's question, grounded in retrieved documents.

    When a screening is in scope its structured output is quoted verbatim -
    the assistant reports the model's grade, it does not form its own opinion.
    """
    docs = retrieve(question)

    parts = []
    if model_output and model_output.get("grade") is not None:
        parts.append(
            f"The model graded this screening as {model_output.get('grade_label')} "
            f"(Grade {model_output['grade']}), "
            f"{'referable' if model_output.get('referable') else 'non-referable'}, "
            f"with calibrated confidence {model_output.get('confidence')}."
        )
    parts.extend(doc["text"] for doc in docs)

    return ClinicalAnswer(
        answer=" ".join(parts),
        citations=[
            Citation(title=d["title"], source=d["source"], snippet=d["text"][:160])
            for d in docs
        ],
        grounded_in_model_output=model_output is not None,
    )
