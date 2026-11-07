from typing import TypedDict
import json

from langgraph.graph import StateGraph, END
from groq import Groq

from app.config import settings


class InvestigationState(TypedDict, total=False):
    image_name: str
    category: str
    vision: dict
    similar_cases: list
    query: str
    documents: list
    report: str
    llm_provider: str
    llm_error: str | None


def _deterministic_report(state: InvestigationState) -> str:
    vision = state["vision"]
    docs = state.get("documents", [])
    lines = [
        "## Visual findings",
        f"- Category: **{state.get('category', 'unknown')}**",
        f"- Anomaly score (uncalibrated): **{vision.get('anomaly_score', 0):.4f}**",
        "- The anomaly map measures deviation from normal reference features; it does not identify a defect cause.",
        "",
        "## Retrieved documentary evidence",
    ]
    if docs:
        for doc in docs:
            lines.append(
                f"- **{doc['citation']} {doc.get('title') or 'Untitled'}** "
                f"({doc.get('source', 'unknown source')}): {doc.get('text', '')[:500]}"
            )
    else:
        lines.append("- No relevant knowledge passages were retrieved.")
    lines.extend([
        "",
        "## Assessment",
        "- No root cause is asserted automatically. Verify suspected issues against cited sources and physical/process evidence.",
        "",
        "## Limitations",
        "- Similar visual cases are retrieval hints, not proof of a shared cause.",
        "- The anomaly score is not a calibrated defect probability or certified quality decision.",
    ])
    return "\n".join(lines)


def build_graph(knowledge_base):
    def retrieve_context(state: InvestigationState):
        vision = state["vision"]
        query = state.get("query") or (
            f"Industrial quality inspection {state.get('category', 'unknown')} "
            f"manufacturing defect anomaly score {vision['anomaly_score']:.4f}. "
            "Find relevant defect symptoms, failure modes, inspection procedures, "
            "maintenance actions and historical incidents."
        )
        docs = knowledge_base.search(
            query, top_k=settings.retrieval_top_k, semantic=True
        )
        return {"documents": docs, "query": query}

    def validate_evidence(state: InvestigationState):
        cited = []
        for i, doc in enumerate(state.get("documents", []), start=1):
            cited.append({**doc, "citation": f"[S{i}]"})
        return {"documents": cited}

    def generate_report(state: InvestigationState):
        vision = state["vision"]
        docs = state.get("documents", [])
        if not settings.groq_api_key:
            return {
                "report": _deterministic_report(state),
                "llm_provider": "deterministic_fallback",
                "llm_error": "GROQ_API_KEY is not configured",
            }

        payload = {
            "image": state.get("image_name"),
            "category": state.get("category"),
            "visual_observation": {
                "anomaly_score": vision.get("anomaly_score"),
                "feature_map_shape": vision.get("feature_map_shape"),
                "interpretation": (
                    "Higher means greater feature deviation from the normal "
                    "reference bank; this is not a calibrated defect probability."
                ),
            },
            "similar_visual_cases": state.get("similar_cases", []),
            "retrieved_evidence": [
                {
                    "citation": d["citation"],
                    "source": d.get("source"),
                    "title": d.get("title"),
                    "passage": d.get("text"),
                    "retrieval_scores": {
                        "rrf": d.get("rrf_score"),
                        "bm25": d.get("bm25_score"),
                        "semantic": d.get("semantic_score"),
                    },
                }
                for d in docs
            ],
        }
        prompt = (
            "You are an industrial quality investigation assistant. Produce a concise "
            "Markdown report using ONLY the supplied evidence. Never invent a root cause, "
            "machine reading, maintenance action, standard, or procedure. Distinguish "
            "computer-vision observations from retrieved documentary evidence. Any claim "
            "based on a retrieved passage MUST cite its label, e.g. [S1]. If evidence is "
            "insufficient, say so. Similar image filenames are not proof of a common cause. "
            "Include: Visual findings; Retrieved evidence; Evidence-supported hypotheses "
            "(or state none); Recommended verification checks (only if supported by sources); "
            "Limitations.\n\nEVIDENCE JSON:\n"
            + json.dumps(payload, ensure_ascii=False, default=str)
        )
        try:
            client = Groq(
                api_key=settings.groq_api_key,
                timeout=settings.groq_timeout_seconds,
                max_retries=1,
            )
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            report = response.choices[0].message.content or ""
            if not report.strip():
                raise RuntimeError("Groq returned an empty report.")
            return {
                "report": report,
                "llm_provider": "groq",
                "llm_error": None,
            }
        except Exception as exc:
            # Keep investigation available if the external LLM is unavailable.
            return {
                "report": _deterministic_report(state),
                "llm_provider": "deterministic_fallback",
                "llm_error": f"Groq request failed: {type(exc).__name__}",
            }

    graph = StateGraph(InvestigationState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("validate_evidence", validate_evidence)
    graph.add_node("generate_report", generate_report)
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "validate_evidence")
    graph.add_edge("validate_evidence", "generate_report")
    graph.add_edge("generate_report", END)
    return graph.compile()
