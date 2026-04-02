from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from research_agent.config import get_settings
from research_agent.graph.state import GraphState
from research_agent.retrieval.dense import DenseRetriever
from research_agent.schemas import Mode
from research_agent.services.text_generation import TextGenerationService

settings = get_settings()
dense_retriever = DenseRetriever(settings)
text_service = TextGenerationService(settings)

REVIEWER_STATE_KEYS = (
    "attack_vectors",
    "active_vector_id",
    "debate_history",
    "debate_summary",
    "skeptic_position",
    "advocate_position",
    "resolution",
    "turn_count",
    "syntheses",
    "vector_verdicts",
    "vector_judgments",
    "vector_reports",
    "final_report",
    "next_speaker",
    "intervention_mode",
    "vectors_remaining",
)


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error or "").lower()
    return (
        "rate_limit_exceeded" in text
        or "rate limit reached" in text
        or "resource_exhausted" in text
        or "quota exceeded" in text
    )


def _extract_retry_hint(error: Exception) -> str:
    text = str(error or "")
    match = re.search(r"Please try again in ([^\\.]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _prepare_mode_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    instructions = {
        Mode.LOCAL: (
            "You are a strict retrieval assistant. "
            "Answer only from the retrieved paper excerpts. "
            "Do not use outside knowledge. "
            "For every factual claim, attach inline citations like [1], [2]. "
            "For count questions, return a single exact numeric answer when explicit evidence exists. "
            "If the excerpt lists numbered entities (for example Expert 1 ... Expert 6), infer the explicit count as 6. "
            "If evidence is missing, respond with exactly: "
            "'This information is not in your uploaded papers.' "
            "Then briefly name what is missing and offer Global mode."
        ),
        Mode.GLOBAL: (
            "You are a high-quality general-purpose assistant. "
            "Respond naturally, clearly, and directly like a normal LLM. "
            "Use uploaded papers only when they are genuinely relevant to the user request. "
            "Do not force paper grounding for general questions. "
            "Use inline citations [n] only for claims that come from retrieved paper context."
        ),
        Mode.WRITER: (
            "You are a research writing assistant that has studied this researcher's style. "
            "When drafting or rewriting, closely mirror the style profile provided: "
            "match their sentence length, formality level, hedging language, "
            "vocabulary choices, and citation format. "
            "Do not default to generic academic prose - the output should be "
            "indistinguishable from the researcher's own writing. "
            "If no style profile exists yet, say so and ask the user to upload a paper first. "
            "Help with any writing task: drafting sections, rewriting sentences, "
            "generating abstracts, or suggesting citation placements."
        ),
        Mode.REVIEWER: (
            "You are a rigorous top-tier ML conference reviewer. "
            "Produce a critical but fair review with concrete evidence and actionable fixes. "
            "Treat the user's message as a review objective/focus lens. "
            "Explicitly separate major concerns from minor concerns. "
            "Before calling something missing, verify whether the paper already addresses it. "
            "If addressed, mark it covered and explain why it is still sufficient/insufficient. "
            "Cite factual claims with inline citations [n]."
        ),
        Mode.COMPARATOR: (
            "You are a research analyst running a claim-level comparison lab. "
            "Do not produce generic prose. Build explicit claim-to-evidence contrasts, "
            "highlight true conflicts, and end with concrete decisions. "
            "Use paper filenames to disambiguate every point."
        ),
    }
    debug = dict(state.get("debug", {}))
    debug["prepared_mode"] = mode.value
    debug["paper_count"] = len(state.get("paper_ids", []))
    return {
        "mode_instructions": instructions[mode],
        "debug": debug,
    }


def _retrieve_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    paper_ids = state.get("paper_ids", [])
    debug = dict(state.get("debug", {}))
    query = _contextualize_query(
        message=state["message"],
        history=state.get("history", []),
        mode=mode,
    )

    if mode == Mode.REVIEWER and state.get("review_paper_id"):
        paper_ids = [state["review_paper_id"]]
        vector_claim = _active_vector_claim(state)
        if vector_claim:
            query = f"{state['message']} {vector_claim}".strip()
        else:
            query = state["message"]
        hits, reviewer_subqueries = _retrieve_reviewer_hits(
            query=query,
            paper_id=paper_ids[0],
        )
        debug["reviewer_subqueries"] = reviewer_subqueries
    elif mode == Mode.COMPARATOR:
        paper_ids = paper_ids[:3]
        query = f"{state['message']} contributions methods benchmarks results differences"
        hits = _retrieve_comparator_hits(query=query, paper_ids=paper_ids)
    elif mode == Mode.GLOBAL and not _should_use_global_retrieval(query=query, paper_ids=paper_ids):
        hits = []
        debug["global_retrieval"] = "skipped_for_general_query"
    else:
        hits, retrieval_subqueries = _retrieve_general_hits(
            query=query,
            paper_ids=paper_ids or None,
            mode=mode,
        )
        if retrieval_subqueries:
            debug["retrieval_subqueries"] = retrieval_subqueries

    documents = [document for document, _ in hits]
    debug["retrieved_count"] = len(documents)
    debug["retrieval_query"] = query
    debug["retrieval_preview"] = [_citation_from_document(document) for document in documents[:5]]
    debug["retrieval_scores"] = [float(score) if score is not None else None for _, score in hits[:5]]

    return {
        "retrieved_documents": documents,
        "debug": debug,
    }


def _retrieve_comparator_hits(
    *,
    query: str,
    paper_ids: list[str],
) -> list[tuple[Document, float | None]]:
    if not paper_ids:
        return []

    per_paper_top_k = max(4, settings.retrieval_top_k)
    max_total = max(settings.retrieval_top_k * 2, len(paper_ids) * 5)
    combined: list[tuple[Document, float | None]] = []
    for paper_id in paper_ids:
        combined.extend(
            dense_retriever.retrieve(
                query=query,
                paper_ids=[paper_id],
                top_k=per_paper_top_k,
            )
        )

    combined.sort(key=lambda pair: pair[1] if pair[1] is not None else float("-inf"), reverse=True)
    deduped: list[tuple[Document, float | None]] = []
    seen_chunk_ids: set[str] = set()
    for document, score in combined:
        chunk_id = str((document.metadata or {}).get("chunk_id", ""))
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        deduped.append((document, score))
        if len(deduped) >= max_total:
            break
    return deduped


def _retrieve_general_hits(
    *,
    query: str,
    paper_ids: list[str] | None,
    mode: Mode,
) -> tuple[list[tuple[Document, float | None]], list[str]]:
    subqueries = _general_subqueries(query=query, mode=mode)
    if not subqueries:
        return [], []

    per_query_top_k = max(settings.retrieval_top_k, settings.rerank_top_n + 4)
    if mode == Mode.LOCAL:
        per_query_top_k = max(per_query_top_k, settings.retrieval_top_k * 3)
        if _is_math_intent_query(query):
            per_query_top_k = max(per_query_top_k, settings.retrieval_top_k * 4)
    aggregated: list[tuple[Document, float]] = []
    for index, subquery in enumerate(subqueries):
        hits = dense_retriever.retrieve(
            query=subquery,
            paper_ids=paper_ids,
            top_k=per_query_top_k,
        )
        query_weight = max(0.75, 1.0 - (0.08 * index))
        for document, score in hits:
            weighted_score = (float(score) if score is not None else 0.0) * query_weight
            aggregated.append((document, weighted_score))

    best_by_chunk: dict[str, tuple[Document, float]] = {}
    for document, score in aggregated:
        key = _document_identity(document)
        existing = best_by_chunk.get(key)
        if existing is None or score > existing[1]:
            best_by_chunk[key] = (document, score)

    merged = list(best_by_chunk.values())
    merged.sort(key=lambda item: item[1], reverse=True)
    target = max(settings.retrieval_top_k * 3, settings.rerank_top_n + 8)
    if mode == Mode.LOCAL:
        target = max(target, settings.retrieval_top_k * 6)
    return [(document, score) for document, score in merged[:target]], subqueries


def _general_subqueries(*, query: str, mode: Mode) -> list[str]:
    base = (query or "").strip()
    if not base:
        return []

    focused = _focused_retrieval_query(base)
    lower = base.lower()
    quantity_intent = any(marker in lower for marker in ("how many", "number", "count", "how much"))
    expert_intent = any(token.startswith("expert") for token in _tokenize_for_overlap(lower))
    seeds = [base]
    if focused and focused.lower() != lower:
        seeds.append(focused)

    if "mixture of experts" in lower or re.search(r"\bmoe\b", lower):
        seeds.append(f"{focused or base} mixture of experts model MoE")
    if mode == Mode.GLOBAL and _is_global_person_query(base):
        seeds.append(f"{focused or base} author authors corresponding author affiliation email")
    if expert_intent and quantity_intent:
        seeds.append(
            f"{focused or base} number of experts specific experts shared experts parameter analysis ablation"
        )
        seeds.append(f"{focused or base} expert 1 expert 2 expert 3 expert 4 expert 5 expert 6")
    if "transformer" in lower and re.search(r"\bhead\b|\bheads\b", lower):
        seeds.append(f"{focused or base} transformer attention heads multi-head")
    if "eeg" in lower:
        seeds.append(f"{focused or base} EEG model architecture dataset")
    if "model" in lower:
        seeds.append(f"{focused or base} proposed model name architecture")
    if _is_math_intent_query(base):
        seeds.append(f"{focused or base} equation equations objective loss function formula formulation")
        seeds.append(f"{focused or base} optimization training objective regularization derivation")
        seeds.append(f"{focused or base} notation variable definition term interpretation")
        seeds.append(f"{focused or base} where denotes can be formulated as equation ( )")
    if mode == Mode.LOCAL:
        seeds.append(f"{focused or base} exact metric value method model version")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in seeds:
        normalized = " ".join(item.split()).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _is_math_intent_query(query: str) -> bool:
    lower = (query or "").lower()
    markers = (
        "math",
        "equation",
        "equations",
        "formula",
        "formulation",
        "derivation",
        "objective",
        "loss",
        "notation",
        "term does",
        "teach me the math",
    )
    return any(marker in lower for marker in markers)


def _focused_retrieval_query(query: str) -> str:
    tokens = _tokenize_for_overlap(query)
    if not tokens:
        return ""
    drop = {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "many",
        "much",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
    }
    kept = [token for token in tokens if token not in drop]
    if not kept:
        kept = tokens
    return " ".join(kept[:20])


def _contextualize_query(*, message: str, history: list[dict[str, Any]], mode: Mode) -> str:
    current = (message or "").strip()
    if not current:
        return ""
    if mode in {Mode.REVIEWER, Mode.COMPARATOR}:
        return current
    if not _is_followup_style_query(current):
        return current
    previous_user = _latest_user_history_message(history=history, exclude=current)
    if not previous_user:
        return current
    return f"{previous_user} {current}".strip()


def _is_followup_style_query(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    tokens = _tokenize_for_overlap(lower)
    if len(tokens) <= 6:
        return True
    followup_markers = (
        "that",
        "this",
        "it",
        "same",
        "exact number",
        "just number",
        "that's it",
        "thats it",
        "only",
    )
    return any(marker in lower for marker in followup_markers)


def _latest_user_history_message(*, history: list[dict[str, Any]], exclude: str) -> str:
    if not history:
        return ""
    excluded = " ".join((exclude or "").strip().lower().split())
    for item in reversed(history):
        if str(item.get("role", "")).lower() != "user":
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        normalized = " ".join(content.lower().split())
        if normalized == excluded:
            continue
        if _is_auto_reviewer_bootstrap(content):
            continue
        return content
    return ""


def _retrieve_reviewer_hits(
    *,
    query: str,
    paper_id: str,
) -> tuple[list[tuple[Document, float | None]], list[str]]:
    if not paper_id:
        return [], []

    subqueries = _reviewer_subqueries(query)
    per_query_top_k = max(4, settings.retrieval_top_k // 2)
    aggregated: list[tuple[Document, float]] = []
    for index, subquery in enumerate(subqueries):
        hits = dense_retriever.retrieve(
            query=subquery,
            paper_ids=[paper_id],
            top_k=per_query_top_k,
        )
        query_weight = max(0.72, 1.0 - (0.04 * index))
        for document, score in hits:
            weighted_score = (float(score) if score is not None else 0.0) * query_weight
            aggregated.append((document, weighted_score))

    best_by_chunk: dict[str, tuple[Document, float]] = {}
    for document, score in aggregated:
        key = _document_identity(document)
        existing = best_by_chunk.get(key)
        if existing is None or score > existing[1]:
            best_by_chunk[key] = (document, score)

    merged = list(best_by_chunk.values())
    merged.sort(key=lambda item: item[1], reverse=True)
    target = max(settings.retrieval_top_k * 2, settings.rerank_top_n + 4)
    return [(document, score) for document, score in merged[:target]], subqueries


def _reviewer_subqueries(query: str) -> list[str]:
    base = (query or "").strip()
    lower = base.lower()
    seeds = [base]

    if any(marker in lower for marker in ("novel", "contribution", "claim", "scope")):
        seeds.append(f"{base} novelty contribution prior work delta claim support")
    else:
        seeds.append(f"{base} core contribution novelty claims assumptions")

    if any(marker in lower for marker in ("method", "architecture", "training", "objective")):
        seeds.append(f"{base} method architecture training objective implementation details")
    else:
        seeds.append(f"{base} method details algorithm design choices")

    if any(marker in lower for marker in ("benchmark", "dataset", "baseline", "metric", "result", "evaluation")):
        seeds.append(f"{base} datasets benchmarks baselines metrics protocol statistical significance")
    else:
        seeds.append(f"{base} evaluation evidence limitations reproducibility")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in seeds:
        normalized = " ".join(item.split()).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
        if len(deduped) >= 4:
            break
    return deduped


def _document_identity(document: Document) -> str:
    metadata = document.metadata or {}
    chunk_id = str(metadata.get("chunk_id", "")).strip()
    if chunk_id:
        return chunk_id
    filename = str(metadata.get("filename", "unknown.pdf")).strip()
    page = str(metadata.get("page", "")).strip()
    return f"{filename}:{page}:{hash(document.page_content)}"


def _rerank_step(state: GraphState) -> GraphState:
    documents = state.get("retrieved_documents", [])
    if not documents:
        return {}

    mode = state["mode"]
    query_text = state.get("message", "")
    query_terms = set(_tokenize_for_overlap(query_text))
    query_phrases = _query_phrases(query_text)
    anchor_terms = _anchor_terms_for_query(query_text)
    focus_terms = set(_tokenize_for_overlap(" ".join(_mode_keywords(mode))))
    math_query = mode == Mode.LOCAL and _is_math_intent_query(query_text)
    quantity_intent = any(marker in (query_text or "").lower() for marker in ("how many", "number", "count", "how much"))
    expert_count_query = quantity_intent and any(token.startswith("expert") for token in query_terms)
    person_query = mode == Mode.GLOBAL and _is_global_person_query(query_text)

    scored: list[tuple[Document, float]] = []
    for index, document in enumerate(documents):
        text = document.page_content or ""
        lower = text.lower()
        normalized_text = _normalize_for_phrase_match(lower)

        rank_prior = max(0.05, 1.0 - (index / max(1, len(documents))))
        overlap = _overlap_score(lower, query_terms)
        phrase_overlap = _phrase_overlap_score(normalized_text, query_phrases)
        anchor_overlap = _overlap_score(lower, anchor_terms) if anchor_terms else 0.0
        focus_overlap = _overlap_score(lower, focus_terms)

        section_boost = 0.0
        if _looks_like_high_signal_section(text):
            section_boost += 0.12
        if mode == Mode.REVIEWER and any(
            marker in lower for marker in ("ablation", "benchmark", "result", "table", "limitation")
        ):
            section_boost += 0.10
        if mode == Mode.COMPARATOR and any(
            marker in lower for marker in ("dataset", "baseline", "sota", "accuracy", "f1")
        ):
            section_boost += 0.10
        if math_query and _looks_math_dense_chunk(text):
            section_boost += 0.55
        elif math_query and any(
            marker in lower for marker in ("equation", "objective", "loss", "formulated as", "where")
        ):
            section_boost += 0.28
        if person_query and _looks_author_metadata_text(lower):
            section_boost += 0.75
        has_number_phrase = "number of experts" in lower
        has_numbered_experts = _has_numbered_expert_pattern(lower)
        has_any_number = bool(re.search(r"\b\d+\b", lower))
        if expert_count_query and has_numbered_experts:
            section_boost += 0.90
        elif expert_count_query and has_number_phrase:
            section_boost += 0.55

        quality_penalty = _low_signal_penalty(text, allow_numeric_dense=math_query)
        anchor_penalty = 0.0
        if anchor_terms and anchor_overlap < 0.34:
            anchor_penalty += 0.30
        if expert_count_query and (has_number_phrase or has_numbered_experts):
            anchor_penalty = max(0.0, anchor_penalty - 0.30)
        if expert_count_query and "expert" in lower and not has_any_number and not has_number_phrase:
            anchor_penalty += 0.35
        if {"mixture", "experts"}.issubset(anchor_terms):
            if "player" in lower and "novice" in lower and "intermediate" in lower:
                anchor_penalty += 0.35
        if person_query and not _looks_author_metadata_text(lower):
            anchor_penalty += 0.18
        total = (
            rank_prior
            + (0.9 * overlap)
            + (0.7 * phrase_overlap)
            + (1.0 * anchor_overlap)
            + (0.6 * focus_overlap)
            + section_boost
            - quality_penalty
            - anchor_penalty
        )
        scored.append((document, total))

    scored.sort(key=lambda item: item[1], reverse=True)
    rerank_limit = max(1, settings.rerank_top_n)
    if mode == Mode.REVIEWER:
        rerank_limit = max(rerank_limit, 8)
    elif mode == Mode.COMPARATOR:
        rerank_limit = max(rerank_limit, 10)
    reranked_docs = _select_balanced_docs(mode=mode, scored_docs=scored, limit=rerank_limit)
    debug = dict(state.get("debug", {}))
    debug["reranked_count"] = len(reranked_docs)
    debug["rerank_preview"] = [
        {
            "filename": (doc.metadata or {}).get("filename", "unknown.pdf"),
            "page": (doc.metadata or {}).get("page"),
            "chunk_id": (doc.metadata or {}).get("chunk_id"),
            "score": round(score, 4),
        }
        for doc, score in scored[:5]
    ]
    return {
        "retrieved_documents": reranked_docs,
        "citations": [_citation_from_document(document) for document in reranked_docs],
        "debug": debug,
    }


def _draft_answer_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    paper_ids = state.get("paper_ids", [])
    review_paper_id = state.get("review_paper_id")
    documents = state.get("retrieved_documents", [])
    debug = dict(state.get("debug", {}))

    if mode == Mode.REVIEWER and not review_paper_id:
        return {
            "draft_answer": "Reviewer mode requires a selected paper before it can generate a review.",
            "citations": [],
            "debug": {**debug, "response_stage": "validation"},
        }

    if mode == Mode.REVIEWER and not documents:
        return {
            "draft_answer": (
                "I could not retrieve enough evidence from the selected paper to review it. "
                "Try asking a more specific question or re-upload the paper."
            ),
            "citations": [],
            "debug": {**debug, "response_stage": "empty_reviewer_context"},
        }

    if mode == Mode.REVIEWER:
        if not text_service.available:
            draft_answer = _fallback_without_model(state)
            debug["response_stage"] = "fallback"
            return {
                "draft_answer": draft_answer,
                "citations": state.get("citations", []),
                "debug": debug,
            }
        try:
            debate_payload = _run_reviewer_debate(state)
        except Exception as error:
            retry_hint = _extract_retry_hint(error)
            fallback_text = _reviewer_rate_limit_fallback(state, retry_hint=retry_hint)
            debug["response_stage"] = "reviewer_model_fallback"
            debug["model_fallback"] = True
            if retry_hint:
                debug["retry_hint"] = retry_hint
            debug["model_error"] = str(error)[:180]
            return {
                "draft_answer": _prefix_retrieval_fallback(
                    fallback_text,
                    retry_hint=retry_hint,
                    mode=mode,
                ),
                "citations": state.get("citations", []),
                "debug": debug,
            }
        debate_debug = dict(debate_payload.get("debug", {}))
        debate_debug["response_stage"] = "reviewer_debate"
        debate_debug["model_provider"] = text_service.last_provider or debate_debug.get("model_provider")
        debate_payload["debug"] = debate_debug
        return debate_payload

    if mode == Mode.COMPARATOR and len(paper_ids) < 2:
        return {
            "draft_answer": "Comparator mode requires at least two selected papers.",
            "citations": [],
            "debug": {**debug, "response_stage": "validation"},
        }

    comparator_doc_papers = _unique_paper_ids(documents)
    if mode == Mode.COMPARATOR and (len(documents) < 2 or len(comparator_doc_papers) < 2):
        return {
            "draft_answer": (
                "I could not retrieve enough evidence from the selected papers. "
                "Try selecting different papers or ask a more specific comparison question."
            ),
            "citations": state.get("citations", []),
            "debug": {**debug, "response_stage": "insufficient_comparator_context"},
        }

    if mode == Mode.LOCAL and not documents:
        return {
            "draft_answer": "This information is not in your uploaded papers.",
            "citations": [],
            "debug": {**debug, "response_stage": "empty_local_context"},
        }
    if mode == Mode.LOCAL and _insufficient_local_grounding(query=state.get("message", ""), documents=documents):
        debug["response_stage"] = "local_low_relevance"
        return {
            "draft_answer": "This information is not in your uploaded papers.",
            "citations": [],
            "debug": debug,
        }
    if mode == Mode.LOCAL:
        numeric_fastpath = _try_local_numeric_fastpath(
            query=state.get("message", ""),
            documents=documents,
        )
        if numeric_fastpath:
            debug["response_stage"] = "local_numeric_fastpath"
            return {
                "draft_answer": numeric_fastpath,
                "citations": state.get("citations", []),
                "debug": debug,
            }

    if not text_service.available:
        draft_answer = _fallback_without_model(state)
        debug["response_stage"] = "fallback"
        return {
            "draft_answer": draft_answer,
            "citations": state.get("citations", []),
            "debug": debug,
        }

    prompt_documents = documents
    if mode == Mode.GLOBAL and documents and not _is_context_relevant_to_query(state["message"], documents):
        prompt_documents = []
        debug["global_context_relevance"] = "low"
    elif mode == Mode.GLOBAL and documents:
        debug["global_context_relevance"] = "high"

    if mode == Mode.REVIEWER:
        max_output_tokens = 2000
    elif mode == Mode.COMPARATOR:
        max_output_tokens = 2200
    else:
        max_output_tokens = 1400
    try:
        draft_answer = text_service.generate(
            system_prompt=_system_prompt(state),
            user_prompt=_draft_user_prompt(
                mode=mode,
                message=state["message"],
                history=state.get("history", []),
                documents=prompt_documents,
                paper_ids=paper_ids,
            ),
            temperature=_temperature_for_mode(mode),
            max_output_tokens=max_output_tokens,
        )
        debug["model_provider"] = text_service.last_provider or "unknown"
    except Exception as error:
        retry_hint = _extract_retry_hint(error)
        draft_answer = _prefix_retrieval_fallback(
            _rate_limit_fallback_answer(state, retry_hint=retry_hint),
            retry_hint=retry_hint,
            mode=mode,
        )
        debug["response_stage"] = "model_fallback"
        debug["model_fallback"] = True
        if retry_hint:
            debug["retry_hint"] = retry_hint
        debug["model_error"] = str(error)[:180]
    if not debug.get("model_fallback"):
        debug["response_stage"] = "draft_model"
    if mode == Mode.COMPARATOR:
        debug["comparator_retrieved_paper_count"] = len(comparator_doc_papers)
    debug["used_context_docs"] = len(prompt_documents)
    return {
        "draft_answer": draft_answer,
        "citations": state.get("citations", []),
        "debug": debug,
    }


def _validate_answer_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    draft = (state.get("draft_answer") or "").strip()
    documents = state.get("retrieved_documents", [])
    debug = dict(state.get("debug", {}))
    if not draft:
        return {"debug": debug}

    if mode == Mode.LOCAL and debug.get("response_stage") == "local_low_relevance":
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "local_low_relevance_bypassed"},
        }

    if mode == Mode.WRITER:
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "writer_skipped"},
        }

    if mode == Mode.REVIEWER and debug.get("reviewer_debate_mode"):
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "reviewer_debate_bypassed"},
        }

    if mode == Mode.GLOBAL:
        stage = (
            "global_low_context_bypassed"
            if debug.get("global_context_relevance") == "low"
            else "global_normal_llm_bypassed"
        )
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": stage},
        }

    if not text_service.available or not documents:
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "bypassed"},
        }

    try:
        validator_raw = text_service.generate(
            system_prompt=_validation_system_prompt(mode),
            user_prompt=(
                "Validate this answer against the retrieved evidence.\n\n"
                "Draft answer:\n"
                f"{draft}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 8) if mode == Mode.REVIEWER else None)}"
            ),
            temperature=0.0,
            max_output_tokens=1400,
        )
    except Exception as error:
        retry_hint = _extract_retry_hint(error)
        debug["validation_stage"] = "model_bypassed"
        debug["model_fallback"] = True
        if retry_hint:
            debug["retry_hint"] = retry_hint
        debug["model_error"] = str(error)[:180]
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": debug,
        }

    verdict, issues, revised = _parse_validation_payload(validator_raw)
    validated = revised.strip() if revised.strip() else draft
    debug["validation_stage"] = verdict
    debug["validation_issue_count"] = len(issues)
    if issues:
        debug["validation_issues"] = issues[:5]
    return {
        "validated_answer": validated,
        "validation_issues": issues,
        "debug": debug,
    }


def _finalize_answer_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    documents = state.get("retrieved_documents", [])
    draft_answer = (state.get("draft_answer") or "").strip()
    validated_answer = (state.get("validated_answer") or "").strip()
    raw_citations = state.get("citations", [])
    debug = dict(state.get("debug", {}))

    answer = validated_answer or draft_answer
    if not answer and mode == Mode.LOCAL and not documents:
        answer = "This information is not in your uploaded papers."
    if mode == Mode.GLOBAL and _is_global_recommendation_query(state.get("message", "")):
        answer = _strip_inline_reference_markers(answer)
        debug["global_citation_cleanup"] = "recommendation_query"
    if mode == Mode.GLOBAL and debug.get("global_context_relevance") == "low":
        answer = _strip_inline_reference_markers(answer)
        debug["global_citation_cleanup"] = "applied"
    if mode == Mode.COMPARATOR and documents:
        quality_issues = _comparator_answer_quality_issues(
            answer=answer,
            citations=raw_citations,
            documents=documents,
            selected_paper_ids=state.get("paper_ids", []),
        )
        if quality_issues:
            answer = _comparator_structured_fallback(documents=documents)
            debug["comparator_quality_guard"] = quality_issues

    if mode in {Mode.LOCAL, Mode.REVIEWER, Mode.COMPARATOR} and documents and not _has_inline_citations(answer):
        debug["citation_warning"] = "Answer lacked inline citations after validation."

    citations = _select_citations_for_answer(answer=answer, citations=raw_citations, mode=mode)
    reindexed_answer = _reindex_answer_citations(
        answer=answer,
        raw_citations=raw_citations,
        selected_citations=citations,
    )
    if reindexed_answer != answer:
        answer = reindexed_answer
        debug["citation_reindexed"] = True
    debug["citation_count"] = len(citations)
    debug["response_stage"] = "finalized"
    return {
        "answer": answer,
        "citations": citations,
        "debug": debug,
    }


def _system_prompt(state: GraphState) -> str:
    mode = state["mode"]
    style_profile = _load_style_profile()
    base = (state.get("mode_instructions") or "").strip()
    if not base:
        base = "You are a research assistant."

    common = (
        "\n\nGeneral rules:\n"
        "- Be concrete and avoid vague language.\n"
        "- Use inline citations [n] for paper-grounded claims.\n"
        "- If evidence is missing, say so directly.\n"
    )

    if mode == Mode.WRITER:
        style_suffix = (
            f"Stored style profile:\n{style_profile}"
            if style_profile
            else "No stored style profile is available yet. Use a clear academic style."
        )
        return f"{base}\n\n{style_suffix}{common}"

    if mode == Mode.GLOBAL:
        return (
            f"{base}{common}\n"
            "Keep the tone natural and helpful. Do not force paper citations unless the claim comes from paper context."
        )
    return f"{base}{common}"


def _format_context(documents: list[Document], max_docs: int | None = None) -> str:
    if not documents:
        return "No retrieved paper context."

    limit = max_docs if max_docs is not None else settings.rerank_top_n
    blocks = []
    for index, document in enumerate(documents[: max(1, int(limit))], start=1):
        metadata = document.metadata or {}
        filename = metadata.get("filename", "unknown.pdf")
        chunk_id = metadata.get("chunk_id", f"chunk-{index}")
        page = metadata.get("page")
        page_suffix = f", p.{page}" if page else ""
        blocks.append(f"[{index}] {filename} ({chunk_id}{page_suffix})\n{document.page_content}")
    return "\n\n".join(blocks)


def _citation_from_document(document: Document) -> dict[str, Any]:
    metadata = document.metadata or {}
    return {
        "paper_id": str(metadata.get("paper_id", "")),
        "filename": str(metadata.get("filename", "unknown.pdf")),
        "snippet": document.page_content[:500],
        "chunk_id": str(metadata.get("chunk_id")) if metadata.get("chunk_id") else None,
        "page": metadata.get("page"),
    }


def _load_style_profile() -> str:
    if not settings.style_profile_store.exists():
        return ""
    raw = settings.style_profile_store.read_text(encoding="utf-8").strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except Exception:
            return raw
        return str(payload.get("profile", "")).strip()
    return raw


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "No earlier conversation."
    recent = history[-settings.conversation_window :]
    return "\n".join(
        f"{item.get('role', 'user').upper()}: {item.get('content', '').strip()}"
        for item in recent
        if item.get("content")
    )


def _fallback_without_model(state: GraphState) -> str:
    mode = state["mode"]
    documents = state.get("retrieved_documents", [])
    if mode == Mode.LOCAL and documents:
        blocks = [
            f"- {doc.metadata.get('filename', 'unknown.pdf')}: {doc.page_content[:260]}"
            for doc in documents[:3]
        ]
        return (
            "Hybrid retrieval found relevant paper evidence, but no text model is configured yet "
            "(`GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`). "
            "Here are the strongest grounded excerpts:\n\n"
            + "\n\n".join(blocks)
        )
    return (
        "Retrieval is working, but model generation is disabled until `GROQ_API_KEY`, "
        "`GEMINI_API_KEY`, or `OPENROUTER_API_KEY` is set in `backend/.env`."
    )


def _prefix_retrieval_fallback(answer: str, *, retry_hint: str, mode: Mode) -> str:
    guidance = f" Try again in about {retry_hint}." if retry_hint else " Try again shortly."
    if mode == Mode.GLOBAL:
        return (
            "Model generation is temporarily unavailable."
            f"{guidance}\n\n{answer.strip()}"
        ).strip()
    return (
        "Model generation is temporarily unavailable, so this is a retrieval-only fallback."
        f"{guidance}\n\n{answer.strip()}"
    ).strip()


def _rate_limit_fallback_answer(state: GraphState, *, retry_hint: str) -> str:
    mode = state["mode"]
    documents = state.get("retrieved_documents", [])
    if mode == Mode.LOCAL:
        if not documents:
            return "This information is not in your uploaded papers."
        return _local_extractive_fallback(
            query=state.get("message", ""),
            documents=documents,
        )
    if mode == Mode.GLOBAL:
        if documents and _is_context_relevant_to_query(state.get("message", ""), documents):
            return _local_extractive_fallback(
                query=state.get("message", ""),
                documents=documents,
            )
        return (
            "I’m temporarily unable to reach the text-generation provider for Global mode. "
            "Please retry in a moment."
        )
    if mode == Mode.COMPARATOR:
        if not documents:
            return "I could not retrieve enough evidence from the selected papers."
        return _comparator_structured_fallback(documents=documents)
    return (
        _local_extractive_fallback(
            query=state.get("message", ""),
            documents=documents,
        )
        if documents
        else "I could not find enough grounded evidence in the retrieved context."
    )


def _comparator_structured_fallback(*, documents: list[Document]) -> str:
    per_paper: dict[str, list[dict[str, Any]]] = {}
    for index, document in enumerate(documents[:10], start=1):
        metadata = document.metadata or {}
        filename = str(metadata.get("filename", "unknown.pdf")).strip() or "unknown.pdf"
        page = metadata.get("page")
        snippet = _extract_signal_sentence(document.page_content or "")
        if not snippet:
            continue
        per_paper.setdefault(filename, []).append(
            {
                "citation": index,
                "snippet": snippet,
                "page": page,
            }
        )
    if not per_paper:
        return "I could not retrieve enough evidence from the selected papers."

    papers = list(per_paper.keys())[:3]
    claim_lines: list[str] = []
    method_lines: list[str] = []
    benchmark_lines: list[str] = []
    for filename in papers:
        entries = per_paper.get(filename, [])
        first = entries[0] if entries else {"citation": 1, "snippet": "No snippet."}
        second = entries[1] if len(entries) > 1 else first
        claim_lines.append(
            f"- **{filename}**\n"
            f"  - Claim signal: \"{first.get('snippet', '')}\" [{int(first.get('citation', 1))}]\n"
            f"  - Evidence signal: \"{second.get('snippet', '')}\" [{int(second.get('citation', 1))}]"
        )
        method_lines.append(
            f"- **{filename}**: method details are partially available from retrieved snippets; prioritize full-text comparison after model recovery [{int(first.get('citation', 1))}]."
        )
        benchmark_lines.append(
            f"- **{filename}**: benchmark overlap cannot be fully verified from fallback snippets alone [{int(first.get('citation', 1))}]."
        )

    conflict_lines: list[str] = []
    if len(papers) >= 2:
        p1 = papers[0]
        p2 = papers[1]
        s1 = str(per_paper[p1][0].get("snippet", "")).lower()
        s2 = str(per_paper[p2][0].get("snippet", "")).lower()
        overlap = _overlap_score(s1, set(_tokenize_for_overlap(s2)))
        if overlap >= 0.28:
            conflict_lines.append(
                f"- Agreement signal: {p1} and {p2} appear to optimize similar goals in retrieved passages [{int(per_paper[p1][0].get('citation', 1))}][{int(per_paper[p2][0].get('citation', 1))}]."
            )
        else:
            conflict_lines.append(
                f"- Non-overlap signal: {p1} and {p2} target different problem framings in the retrieved passages [{int(per_paper[p1][0].get('citation', 1))}][{int(per_paper[p2][0].get('citation', 1))}]."
            )
        conflict_lines.append("- Contradictions: not confidently detectable in retrieval-only fallback; abstaining.")
    else:
        conflict_lines.append("- Need at least two papers for conflict mapping.")

    use_case_lines: list[str] = []
    if papers:
        use_case_lines.append(f"- If you need strongest available evidence in current snippets: prefer **{papers[0]}** [{int(per_paper[papers[0]][0].get('citation', 1))}].")
    if len(papers) > 1:
        use_case_lines.append(f"- If your task aligns with the second paper framing: prefer **{papers[1]}** [{int(per_paper[papers[1]][0].get('citation', 1))}].")
    use_case_lines.append("- For publication-grade decisions, rerun after model recovery to resolve benchmark and novelty ties.")

    return (
        "## Papers Compared\n"
        + "\n".join(f"- {paper}" for paper in papers)
        + "\n\n## Claim Matrix\n"
        + "\n".join(claim_lines)
        + "\n\n## Conflict Map\n"
        + "\n".join(conflict_lines)
        + "\n\n## Benchmark Verdict Matrix\n"
        + "\n".join(benchmark_lines)
        + "\n\n## Method Trade-offs\n"
        + "\n".join(method_lines)
        + "\n\n## Synthesis Blueprint\n"
        + "- Merge plan: combine the strongest methodological element from each selected paper and evaluate on one shared metric once full model generation is back.\n"
        + "- Blocking risk: fallback mode cannot reliably infer full benchmark comparability.\n"
        + "\n\n## Decision By Use Case\n"
        + "\n".join(use_case_lines)
    )


def _unique_paper_ids(documents: list[Document]) -> set[str]:
    paper_ids: set[str] = set()
    for document in documents:
        paper_id = str((document.metadata or {}).get("paper_id", "")).strip()
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def _comparator_selected_filenames(
    *,
    documents: list[Document],
    selected_paper_ids: list[str] | None,
) -> list[str]:
    lookup = _comparator_filename_lookup(documents)
    selected: list[str] = []
    seen: set[str] = set()
    for paper_id in selected_paper_ids or []:
        pid = str(paper_id or "").strip()
        if not pid:
            continue
        name = lookup.get(pid, "")
        if not name or name in seen:
            continue
        seen.add(name)
        selected.append(name)
    if selected:
        return selected
    for document in documents:
        filename = str((document.metadata or {}).get("filename", "")).strip()
        if not filename or filename in seen:
            continue
        seen.add(filename)
        selected.append(filename)
        if len(selected) >= 3:
            break
    return selected


def _comparator_answer_quality_issues(
    *,
    answer: str,
    citations: list[dict[str, Any]],
    documents: list[Document],
    selected_paper_ids: list[str] | None,
) -> list[str]:
    issues: list[str] = []
    text = (answer or "").strip()
    if not text:
        return ["empty_answer"]

    expected_sections = (
        "## Papers Compared",
        "## Claim Matrix",
        "## Conflict Map",
        "## Benchmark Verdict Matrix",
        "## Method Trade-offs",
        "## Synthesis Blueprint",
        "## Decision By Use Case",
    )
    missing_sections = [section for section in expected_sections if section.lower() not in text.lower()]
    if len(missing_sections) >= 3:
        issues.append("missing_sections")
    if len(text) < 460:
        issues.append("too_short")

    selected_filenames = _comparator_selected_filenames(
        documents=documents,
        selected_paper_ids=selected_paper_ids,
    )
    missing_filename_mentions = [
        filename
        for filename in selected_filenames
        if filename.lower() not in text.lower()
    ]
    if missing_filename_mentions:
        issues.append("missing_selected_paper_mentions")

    referenced_numbers = sorted({int(match) for match in re.findall(r"\[([0-9]+)\]", text) if match.isdigit()})
    if not referenced_numbers:
        issues.append("missing_inline_citations")
        return issues

    cited_sources: set[str] = set()
    for number in referenced_numbers:
        index = number - 1
        if index < 0 or index >= len(citations):
            continue
        citation = citations[index]
        source = str(citation.get("paper_id", "")).strip() or str(citation.get("filename", "")).strip()
        if source:
            cited_sources.add(source)

    if len(_unique_paper_ids(documents)) >= 2 and len(cited_sources) < 2:
        issues.append("citations_single_paper")

    mentioned_files = {match.strip() for match in re.findall(r"\b([A-Za-z0-9_.-]+\.pdf)\b", text)}
    allowed_files = {name.strip() for name in selected_filenames if name.strip()}
    unexpected = [name for name in mentioned_files if name not in allowed_files]
    if allowed_files and unexpected:
        issues.append("mentions_unselected_paper")
    return issues


def _extract_signal_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    for sentence in sentences:
        snippet = sentence.strip()
        if len(snippet) < 40:
            continue
        if _looks_like_reference_snippet(snippet):
            continue
        return snippet[:260]
    return cleaned[:260]


def _local_extractive_fallback(*, query: str, documents: list[Document]) -> str:
    query_terms = set(_tokenize_for_overlap(query))
    query_phrases = _query_phrases(query)
    lower_query = (query or "").lower()
    quantity_intent = _is_quantity_intent_query(query)
    if quantity_intent and any(token.startswith("expert") for token in query_terms):
        quantity_snippet, citation_index = _extract_quantity_snippet(documents=documents, keyword="expert")
        if quantity_snippet:
            expert_count = _infer_expert_count(quantity_snippet)
            if expert_count is not None:
                return (
                    "Based on the uploaded paper: "
                    f"the paper uses {expert_count} experts. {quantity_snippet} [{citation_index}]"
                )
            return f"Based on the uploaded paper: {quantity_snippet} [{citation_index}]"
        return (
            "This information is not in your uploaded papers. "
            "The retrieved context discusses mixture-of-experts concepts but does not provide a clear numeric expert count."
        )
    candidates: list[tuple[float, str]] = []
    for index, document in enumerate(documents[:6]):
        text = (document.page_content or "").strip()
        if not text:
            continue
        doc_overlap = _overlap_score(text.lower(), query_terms)
        doc_phrase = _phrase_overlap_score(_normalize_for_phrase_match(text.lower()), query_phrases)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 28:
                continue
            if _looks_like_reference_snippet(snippet):
                continue
            overlap = _overlap_score(snippet.lower(), query_terms)
            phrase = _phrase_overlap_score(_normalize_for_phrase_match(snippet.lower()), query_phrases)
            rank_prior = max(0.05, 1.0 - (index / max(1, len(documents))))
            score = (0.9 * overlap) + (0.9 * phrase) + (0.25 * doc_overlap) + (0.2 * doc_phrase) + (0.12 * rank_prior)
            snippet_no_citations = re.sub(r"\[[0-9]+\]", " ", snippet)
            if quantity_intent and re.search(r"\b\d+(?:\.\d+)?\b", snippet_no_citations):
                score += 0.35
            score -= min(0.3, _low_signal_penalty(snippet))
            candidates.append((score, snippet))
    if not candidates:
        return "This information is not in your uploaded papers."

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_snippet = candidates[0]
    if best_score < 0.20:
        return "This information is not in your uploaded papers."

    cleaned_best = re.sub(r"\[[0-9]+\]", "", best_snippet)
    cleaned_best = re.sub(r"\s+", " ", cleaned_best).strip()
    return f"Based on the uploaded paper: {cleaned_best} [1]"


def _try_local_numeric_fastpath(*, query: str, documents: list[Document]) -> str | None:
    if not documents:
        return None
    if not _is_quantity_intent_query(query):
        return None

    lower_query = (query or "").lower()
    keyword = "expert"
    if "participant" in lower_query:
        keyword = "participant"
    elif "player" in lower_query:
        keyword = "player"
    elif "subject" in lower_query:
        keyword = "subject"
    elif re.search(r"\bhead\b|\bheads\b", lower_query):
        keyword = "head"

    snippet, citation_index = _extract_quantity_snippet(documents=documents, keyword=keyword)
    if not snippet:
        return None

    count = _extract_keyword_count(snippet=snippet, keyword=keyword)
    if count is None and keyword != "expert":
        count = _extract_keyword_count(snippet=snippet, keyword="expert")
        if count is not None:
            keyword = "expert"
    if count is None:
        return None

    if _is_just_number_request(query):
        return f"{count} [{citation_index}]"

    if keyword == "expert":
        return f"The paper uses {count} experts. [{citation_index}]"
    if keyword == "participant":
        return f"The paper uses {count} participants. [{citation_index}]"
    if keyword == "player":
        return f"The paper uses {count} players. [{citation_index}]"
    if keyword == "subject":
        return f"The paper uses {count} subjects. [{citation_index}]"
    if keyword == "head":
        return f"The paper uses {count} transformer heads. [{citation_index}]"
    return f"The paper reports {count}. [{citation_index}]"


def _looks_like_reference_snippet(text: str) -> bool:
    lower = (text or "").lower()
    markers = (
        "arxiv",
        "preprint",
        "doi:",
        "proc.",
        "proceedings of",
        "ieee trans",
        "et al.",
        "pp.",
    )
    if any(marker in lower for marker in markers):
        return True
    if re.match(r"^\s*\[[0-9]+\]\s*", text or ""):
        return True
    return False


def _looks_like_non_argument_snippet(text: str) -> bool:
    lower = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not lower:
        return True
    junk_markers = (
        "frame-level video-based temporal analysis of fps gameplay without telemetry",
        "start of trial",
        "score =",
        "figure ",
        "table ",
        "(a)",
        "(b)",
        "(c)",
        "copyrights for components of this work",
        "no personally identifiable information",
    )
    if any(marker in lower for marker in junk_markers):
        return True
    if re.match(r"^[\(\[]?[a-z0-9][\)\]]?\s", lower) and len(lower.split()) <= 8:
        return True
    if len(lower) < 45:
        return True
    return False


def _extract_quantity_snippet(*, documents: list[Document], keyword: str) -> tuple[str, int]:
    best_score = float("-inf")
    best_snippet = ""
    best_citation = 1
    for doc_index, document in enumerate(documents[:8], start=1):
        text = (document.page_content or "").strip()
        if not text:
            continue
        cleaned = re.sub(r"\[[0-9]+\]", " ", text)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 24:
                continue
            lower = snippet.lower()
            if keyword not in lower and f"{keyword}s" not in lower:
                continue
            if _looks_like_reference_snippet(snippet):
                continue
            score = 0.0
            if re.search(rf"\b\d+\s+{re.escape(keyword)}s?\b", lower):
                score += 1.0
            if re.search(rf"\bnumber of {re.escape(keyword)}s?\b", lower):
                score += 0.9
            if re.search(rf"\b{re.escape(keyword)}s?\s*(?:is|are|were|was|=|:)?\s*\d+\b", lower):
                score += 1.0
            if keyword == "expert":
                if _has_numbered_expert_pattern(lower):
                    score += 1.1
                if _infer_expert_count(snippet) is not None:
                    score += 0.6
            if re.search(r"\b\d+\b", lower):
                score += 0.25
            score -= min(0.25, _low_signal_penalty(snippet))
            if score > best_score:
                best_score = score
                best_snippet = snippet
                best_citation = doc_index
    if best_score < 0.45:
        return "", 1
    return best_snippet, best_citation


def _extract_keyword_count(*, snippet: str, keyword: str) -> int | None:
    lower = (snippet or "").lower()
    if keyword == "expert":
        return _infer_expert_count(snippet)

    candidates: list[int] = []
    for pattern in (
        rf"\b(\d+)\s+{re.escape(keyword)}s?\b",
        rf"\b{re.escape(keyword)}s?\s*(?:is|are|were|was|=|:)?\s*(\d+)\b",
        rf"\bnumber of {re.escape(keyword)}s?\s*(?:is|:)?\s*(\d+)\b",
    ):
        candidates.extend(int(value) for value in re.findall(pattern, lower))
    if not candidates:
        return None
    return max(candidates)


def _is_quantity_intent_query(query: str) -> bool:
    lower = (query or "").lower()
    return any(marker in lower for marker in ("how many", "number", "count", "how much"))


def _is_just_number_request(query: str) -> bool:
    lower = (query or "").lower()
    markers = ("just number", "only number", "number only", "thats it", "that's it", "just give number")
    return any(marker in lower for marker in markers)


def _infer_expert_count(snippet: str) -> int | None:
    text = (snippet or "").lower()
    if "expert" not in text:
        return None
    values: list[int] = []
    values.extend(int(value) for value in re.findall(r"\bexpert\s+(\d+)\b", text))
    values.extend(int(value) for value in re.findall(r"\bexperts?\s*(?:\(|:)?\s*(\d+)\b", text))
    for match in re.finditer(r"\bexperts?\s+([0-9,\sand]+)", text):
        values.extend(int(value) for value in re.findall(r"\d+", match.group(1)))
    values = [value for value in values if value > 0]
    if not values:
        return None
    return max(values)


def _has_numbered_expert_pattern(text: str) -> bool:
    lower = (text or "").lower()
    if "expert" not in lower:
        return False
    if re.search(r"\bexperts?\s+\d+\b", lower):
        return True
    if re.search(r"\bexpert\s+\d+\b", lower):
        return True
    if re.search(r"\bexperts?\s+\d+\s*,\s*\d+", lower):
        return True
    return False


def _reviewer_rate_limit_fallback(state: GraphState, *, retry_hint: str) -> str:
    documents = state.get("retrieved_documents", [])
    attack_vectors = _normalize_attack_vectors(
        _fallback_attack_vectors(message=state.get("message", ""), documents=documents),
        fallback_count=min(3, settings.reviewer_attack_vector_count),
        documents=documents,
    )
    active = attack_vectors[0] if attack_vectors else {
        "id": "V1",
        "claim": "Core contribution framing vs evidence support.",
        "severity": "high",
        "category": "novelty",
        "quote": _default_quote(documents),
        "skeptic_lead": "Novelty strength is unclear without explicit comparative evidence.",
    }
    quote = active.get("quote", _default_quote(documents))
    return (
        "## Claim Trial Engine\n"
        f"Active Claim: {active.get('id', 'V1')} - {active.get('claim', '')}\n"
        f"Claim Trigger: \"{quote}\"\n\n"
        "### Skeptic\n"
        "- Concern: evidence may be weaker than framing suggests [1].\n"
        "- Ask for a tighter quantitative comparison and clearer scope boundary [1].\n\n"
        "### Advocate\n"
        "- Defense: the paper does provide partial evidence for the claim [1].\n"
        "- Recommend narrowing claim language to what is directly demonstrated [1].\n\n"
        "### Evidence-only Judge\n"
        "- Verdict: contested\n"
        "- Rationale: evidence partially supports feasibility but does not fully settle novelty strength [1].\n\n"
        "### Rewrite Compiler Card\n"
        "Target Section: contribution framing paragraph\n"
        "Target Claim: contribution-level novelty statement\n"
        "Patch Instruction: revise the contribution claim to include one concrete metric/baseline comparison and explicitly state the scope limits.\n"
        "Why: this preserves strengths while reducing overclaim risk.\n\n"
        "Intervention: ask either reviewer to sharpen evidence or move to the next vector."
    )


def _validation_system_prompt(mode: Mode) -> str:
    mode_hint = {
        Mode.LOCAL: "Strictly keep only claims grounded in retrieved paper context.",
        Mode.GLOBAL: (
            "Allow normal general-knowledge responses. "
            "Only enforce citations for claims that explicitly rely on retrieved paper context."
        ),
        Mode.WRITER: "Preserve style while correcting factual inaccuracies.",
        Mode.REVIEWER: (
            "Enforce rigorous review quality. "
            "Remove unsupported claims, and ensure every major concern is evidenced. "
            "Do not mark benchmarks/ablations as missing when context shows they are covered; relabel as covered with caveats if needed. "
            "Ensure the output preserves reviewer structure, concrete actionable feedback, and complete score block."
        ),
        Mode.COMPARATOR: (
            "Keep only comparisons directly supported by retrieved context blocks. "
            "Preserve comparator section structure and abstain where evidence is insufficient."
        ),
    }[mode]
    return (
        "You are a factual validator for research answers.\n"
        f"{mode_hint}\n"
        "Return JSON only with keys:\n"
        "{\n"
        '  "verdict": "pass" or "revise",\n'
        '  "issues": ["short issue", "..."],\n'
        '  "revised_answer": "final corrected answer with inline citations [n] where needed"\n'
        "}\n"
        "If draft is already strong and grounded, set verdict to pass and copy it into revised_answer."
    )


def _parse_validation_payload(raw: str) -> tuple[str, list[str], str]:
    text = (raw or "").strip()
    if not text:
        return "pass", [], ""
    payload = _try_parse_json_object(text)
    if payload is None:
        recovered = _recover_revised_answer(text)
        if recovered:
            return "revise", ["Validator returned malformed JSON; recovered revised answer."], recovered
        return "revise", ["Validator returned non-JSON output."], text
    verdict = str(payload.get("verdict", "pass")).strip().lower()
    if verdict not in {"pass", "revise"}:
        verdict = "pass"
    issues_raw = payload.get("issues", [])
    issues: list[str] = []
    if isinstance(issues_raw, list):
        issues = [str(item).strip() for item in issues_raw if str(item).strip()]
    revised_answer = str(payload.get("revised_answer", "")).strip()
    return verdict, issues, revised_answer


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _recover_revised_answer(text: str) -> str:
    cleaned = _strip_markdown_fence(text).strip()
    if not cleaned:
        return ""

    # Handles malformed JSON where revised_answer is still present in a quoted block.
    quoted_match = re.search(
        r'"revised_answer"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"[^"]+"\s*:|\}\s*$)',
        cleaned,
        flags=re.DOTALL,
    )
    if quoted_match:
        candidate = quoted_match.group(1)
        candidate = candidate.replace('\\"', '"').replace("\\n", "\n").strip()
        if candidate:
            return candidate

    # Fallback for unquoted payload styles.
    raw_match = re.search(
        r'"revised_answer"\s*:\s*([\s\S]*?)\s*(?:,\s*"[^"]+"\s*:|\}\s*$)',
        cleaned,
        flags=re.DOTALL,
    )
    if raw_match:
        candidate = raw_match.group(1).strip().strip('"').strip()
        if candidate:
            return candidate
    return ""


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _draft_user_prompt(
    *,
    mode: Mode,
    message: str,
    history: list[dict[str, str]],
    documents: list[Document],
    paper_ids: list[str] | None = None,
) -> str:
    history_text = _format_history(history)
    reviewer_context_text = _format_context(documents, max_docs=max(settings.rerank_top_n, 8))
    context_text = _format_context(documents)

    if mode == Mode.REVIEWER:
        return (
            "You are writing a rigorous ML conference-style review.\n"
            "Requirements:\n"
            "- Ground factual statements in retrieved evidence and cite with [n].\n"
            "- Do not hallucinate missing experiments; explicitly mark whether each item is covered or missing.\n"
            "- Be specific: name concrete failure points, experimental gaps, and methodological risks.\n"
            "- Prefer precise, actionable recommendations over generic advice.\n"
            "- Output in markdown with exactly these headings:\n"
            "  1) ## Paper Snapshot\n"
            "  2) ## Technical Summary\n"
            "  3) ## Strengths\n"
            "  4) ## Major Concerns\n"
            "  5) ## Minor Concerns\n"
            "  6) ## Coverage Check (Covered vs Missing)\n"
            "  7) ## Reproducibility & Clarity Checklist\n"
            "  8) ## Actionable Revision Plan\n"
            "  9) ## Scores\n"
            "- In `## Scores`, include numeric sub-scores (1-10):\n"
            "  - Novelty\n"
            "  - Technical Soundness\n"
            "  - Empirical Rigor\n"
            "  - Clarity\n"
            "  - Reproducibility\n"
            "  - Weighted Overall Score (compute as: "
            "0.25*Novelty + 0.30*Technical Soundness + 0.25*Empirical Rigor + 0.10*Clarity + 0.10*Reproducibility)\n"
            "- Also include:\n"
            "  - Recommendation: <1-10>\n"
            "  - Confidence: <1-5>\n"
            "  - Risk of Rejection: <Low|Medium|High>\n\n"
            "Conversation history:\n"
            f"{history_text}\n\n"
            "User message:\n"
            f"{message}\n\n"
            "Retrieved context:\n"
            f"{reviewer_context_text}"
        )

    if mode == Mode.GLOBAL:
        recommendation_query = _is_global_recommendation_query(message)
        person_query = _is_global_person_query(message)
        if recommendation_query:
            query_directive = (
                "- The user is asking for recommendations. "
                "Give a useful list with concise relevance reasoning for each item.\n"
            )
        elif person_query:
            query_directive = (
                "- The user is asking about a person/author. "
                "Give a useful profile-style answer (role, domain, notable work themes), "
                "and call out ambiguity if the name could refer to multiple people.\n"
            )
        else:
            query_directive = ""
        return (
            "You are responding in Global mode.\n"
            "Guidelines:\n"
            "- Answer like a normal high-quality LLM: clear, practical, and direct.\n"
            "- Use general knowledge by default.\n"
            "- Use retrieved paper context only when it clearly helps this question.\n"
            "- If a specific statement comes from retrieved paper context, cite it with [n].\n"
            "- Do not force citations or paper framing for general questions.\n"
            "- Do not mention missing uploaded papers unless the user explicitly asks for paper-grounded evidence.\n"
            f"{query_directive}\n"
            "Conversation history:\n"
            f"{history_text}\n\n"
            "User message:\n"
            f"{message}\n\n"
            "Retrieved context:\n"
            f"{context_text}"
        )

    if mode == Mode.COMPARATOR:
        paper_list = _comparator_paper_list(
            documents=documents,
            selected_paper_ids=paper_ids,
        )
        evidence_pack = _comparator_evidence_pack(
            documents=documents,
            selected_paper_ids=paper_ids,
            max_snippets_per_paper=3,
        )
        return (
            "You are producing a deep, decision-useful comparison of selected papers.\n"
            "Requirements:\n"
            "- Cover ALL selected papers explicitly by filename.\n"
            "- Keep comparisons concrete and evidence-backed; cite grounded claims with [n].\n"
            "- Do not claim shared benchmarks unless the retrieved context proves they overlap.\n"
            "- If no shared benchmark appears, state that clearly instead of guessing.\n"
            "- Be specific: include exact metrics/datasets/method names when available.\n"
            "- If evidence for a requested comparison axis is weak, abstain explicitly.\n"
            "- Do not return short summaries. Provide depth under every section.\n"
            "- Never attach a claim to Paper A using a citation that comes from Paper B.\n"
            "- If evidence for one selected paper is thin, say so directly for that paper.\n"
            "- Output in markdown with EXACT sections:\n"
            "  1) ## Papers Compared\n"
            "  2) ## Claim Matrix\n"
            "  3) ## Conflict Map\n"
            "  4) ## Benchmark Verdict Matrix\n"
            "  5) ## Method Trade-offs\n"
            "  6) ## Synthesis Blueprint\n"
            "  7) ## Decision By Use Case\n"
            "- In `## Claim Matrix`, give at least 2 strongest claims per paper with direct evidence.\n"
            "- In `## Conflict Map`, separate `Agreements`, `Contradictions`, and `Non-overlap`.\n"
            "- In `## Benchmark Verdict Matrix`, score each paper (1-10) on novelty, empirical rigor, and reproducibility with one-line justification.\n"
            "- In `## Synthesis Blueprint`, specify what to borrow from each paper and one concrete merged experiment.\n"
            "- In `## Decision By Use Case`, include at least 3 concrete scenarios and a clear winner for each.\n"
            "- If user message is short (for example 'compare'), still deliver full depth.\n\n"
            "Selected papers:\n"
            f"{paper_list}\n\n"
            "Evidence Pack (citation map):\n"
            f"{evidence_pack}\n\n"
            "Conversation policy:\n"
            "- Ignore prior-turn claims about any paper not listed in `Selected papers`.\n"
            "- Resolve this response from the current evidence pack only.\n\n"
            "User message:\n"
            f"{message}\n\n"
            "Retrieved context:\n"
            f"{context_text}"
        )

    return (
        "You are producing a first-draft response.\n"
        "Requirements:\n"
        "- Every concrete factual claim must include at least one inline citation [n].\n"
        "- Do not invent metrics, baselines, or section claims.\n"
        "- If evidence is missing, explicitly state uncertainty.\n\n"
        "Conversation history:\n"
        f"{history_text}\n\n"
        "User message:\n"
        f"{message}\n\n"
        "Retrieved context:\n"
        f"{context_text}"
    )


def _comparator_paper_list(
    *,
    documents: list[Document],
    selected_paper_ids: list[str] | None = None,
) -> str:
    id_to_filename = _comparator_filename_lookup(documents)
    labels: list[str] = []
    seen: set[str] = set()
    for paper_id in selected_paper_ids or []:
        pid = str(paper_id or "").strip()
        if not pid:
            continue
        label = id_to_filename.get(pid, f"paper_id:{pid}")
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    for document in documents:
        filename = str((document.metadata or {}).get("filename", "")).strip()
        if not filename or filename in seen:
            continue
        seen.add(filename)
        labels.append(filename)
        if len(labels) >= 3:
            break
    if not labels:
        return "- Unknown papers from retrieved context"
    return "\n".join(f"- {label}" for label in labels)


def _comparator_filename_lookup(documents: list[Document]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for document in documents:
        metadata = document.metadata or {}
        paper_id = str(metadata.get("paper_id", "")).strip()
        filename = str(metadata.get("filename", "")).strip()
        if paper_id and filename and paper_id not in mapping:
            mapping[paper_id] = filename
    return mapping


def _comparator_evidence_pack(
    *,
    documents: list[Document],
    selected_paper_ids: list[str] | None,
    max_snippets_per_paper: int,
) -> str:
    by_paper: dict[str, list[str]] = {}
    id_to_filename = _comparator_filename_lookup(documents)
    selected = [str(paper_id or "").strip() for paper_id in (selected_paper_ids or []) if str(paper_id or "").strip()]
    selected_set = set(selected)
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        paper_id = str(metadata.get("paper_id", "")).strip() or "unknown"
        if selected_set and paper_id not in selected_set:
            continue
        if len(by_paper.get(paper_id, [])) >= max(1, max_snippets_per_paper):
            continue
        raw_text = (document.page_content or "").strip()
        if _looks_like_reference_snippet(raw_text) or _looks_like_non_argument_snippet(raw_text):
            continue
        snippet = _extract_signal_sentence(raw_text)
        if not snippet:
            continue
        filename = str(metadata.get("filename", "unknown.pdf")).strip() or "unknown.pdf"
        page = metadata.get("page")
        page_suffix = f", p.{page}" if page else ""
        by_paper.setdefault(paper_id, []).append(f"- [{index}] {filename}{page_suffix}: \"{snippet}\"")

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for paper_id in selected:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        ordered_ids.append(paper_id)
    for document in documents:
        paper_id = str((document.metadata or {}).get("paper_id", "")).strip() or "unknown"
        if paper_id in seen:
            continue
        seen.add(paper_id)
        ordered_ids.append(paper_id)

    lines: list[str] = []
    for paper_id in ordered_ids[:3]:
        filename = id_to_filename.get(paper_id, f"paper_id:{paper_id}")
        lines.append(f"### {filename}")
        entries = by_paper.get(paper_id, [])
        if entries:
            lines.extend(entries)
        else:
            lines.append("- No high-signal snippet retrieved for this paper in current context.")

    if not lines:
        return "No comparator evidence pack available."
    return "\n".join(lines)


def _run_reviewer_debate(state: GraphState) -> GraphState:
    raw_message = str(state.get("message", ""))
    message = _normalize_reviewer_message(raw_message)
    intervention_mode = _normalize_intervention_mode(state.get("intervention_mode"))
    documents = state.get("retrieved_documents", [])
    debug = dict(state.get("debug", {}))
    debate_history = deepcopy(state.get("debate_history", []))
    debate_summary = str(state.get("debate_summary", "")).strip()
    syntheses = deepcopy(state.get("syntheses", {}))
    vector_verdicts = deepcopy(state.get("vector_verdicts", {}))
    vector_judgments = deepcopy(state.get("vector_judgments", {}))
    vector_reports = deepcopy(state.get("vector_reports", {}))
    final_report = deepcopy(state.get("final_report", {}))
    attack_vectors = _normalize_attack_vectors(
        state.get("attack_vectors", []),
        fallback_count=settings.reviewer_attack_vector_count,
        documents=documents,
    )
    if _is_new_reviewer_session_signal(raw_message):
        debate_history = []
        debate_summary = ""
        syntheses = {}
        vector_verdicts = {}
        vector_judgments = {}
        vector_reports = {}
        final_report = {}
        debug["reviewer_session_reset"] = True

    if not attack_vectors:
        attack_vectors = _generate_attack_vectors(
            message=message,
            documents=documents,
            count=settings.reviewer_attack_vector_count,
        )
        attack_vectors = _normalize_attack_vectors(
            attack_vectors,
            fallback_count=settings.reviewer_attack_vector_count,
            documents=documents,
        )

    if not attack_vectors:
        attack_vectors = [
            {
                "id": "V1",
                "claim": "Core contribution framing and novelty support.",
                "severity": "high",
                "category": "novelty",
                "quote": _default_quote(documents),
                "skeptic_lead": "The novelty framing is not yet grounded in a direct quantitative delta.",
            }
        ]

    attack_vector_ids = [str(item.get("id", "")).strip() for item in attack_vectors if str(item.get("id", "")).strip()]
    completed_session = bool(final_report) and bool(attack_vector_ids) and len(syntheses) >= len(attack_vector_ids)
    vectors_remaining = [
        vector_id
        for vector_id in state.get("vectors_remaining", [])
        if vector_id in attack_vector_ids and vector_id not in syntheses
    ]
    if not vectors_remaining and not completed_session:
        vectors_remaining = [vector_id for vector_id in attack_vector_ids if vector_id not in syntheses]
    if not vectors_remaining and not completed_session:
        vectors_remaining = attack_vector_ids[:]

    if _user_requested_next_vector(message):
        current_active = str(state.get("active_vector_id", "")).strip()
        if current_active and current_active in vectors_remaining:
            vectors_remaining = [vector for vector in vectors_remaining if vector != current_active] + [current_active]

    explicit_vector = _extract_vector_selection(message, attack_vectors)
    if explicit_vector and explicit_vector in syntheses:
        syntheses.pop(explicit_vector, None)
        vector_verdicts.pop(explicit_vector, None)
        vector_judgments.pop(explicit_vector, None)
        vector_reports.pop(explicit_vector, None)
        if explicit_vector not in vectors_remaining:
            vectors_remaining.insert(0, explicit_vector)

    active_vector_id = (
        explicit_vector
        or state.get("active_vector_id")
        or (vectors_remaining[0] if vectors_remaining else (attack_vector_ids[0] if attack_vector_ids else "V1"))
    )
    if active_vector_id not in attack_vector_ids and attack_vector_ids:
        active_vector_id = attack_vector_ids[0]
    if active_vector_id not in vectors_remaining and active_vector_id not in syntheses:
        vectors_remaining.insert(0, active_vector_id)

    active_vector = _get_attack_vector(attack_vectors, active_vector_id)
    turn_count = _count_vector_turns(debate_history, active_vector_id)
    skeptic_position = str(state.get("skeptic_position", "")).strip() or _latest_speaker_content(
        debate_history, speaker="skeptic", vector_id=active_vector_id
    )
    advocate_position = str(state.get("advocate_position", "")).strip() or _latest_speaker_content(
        debate_history, speaker="advocate", vector_id=active_vector_id
    )
    resolution = _infer_resolution(
        skeptic_position=skeptic_position,
        advocate_position=advocate_position,
        history=debate_history,
        active_vector_id=active_vector_id,
        turn_count=turn_count,
    )

    if _looks_like_score_request(message):
        score_answer = _reviewer_score_response(
            query=message,
            active_vector=active_vector,
            resolution=resolution,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            debate_history=debate_history,
            documents=documents,
        )
        debug["reviewer_debate_mode"] = True
        debug["response_stage"] = "reviewer_scorecard"
        debug["active_vector_id"] = active_vector_id
        debug["resolution"] = resolution
        debug["turn_count"] = turn_count
        debug["warning_turn"] = settings.reviewer_warning_turn
        debug["max_turns"] = settings.reviewer_max_turns
        debug["intervention_mode"] = intervention_mode
        debug["next_speaker"] = state.get("next_speaker", "skeptic")
        return {
            "draft_answer": score_answer,
            "citations": state.get("citations", []),
            "attack_vectors": attack_vectors,
            "active_vector_id": active_vector_id,
            "debate_history": debate_history,
            "debate_summary": debate_summary,
            "skeptic_position": skeptic_position,
            "advocate_position": advocate_position,
            "resolution": resolution,
            "turn_count": turn_count,
            "syntheses": syntheses,
            "vector_verdicts": vector_verdicts,
            "vector_judgments": vector_judgments,
            "vector_reports": vector_reports,
            "final_report": final_report,
            "next_speaker": state.get("next_speaker", "skeptic"),
            "intervention_mode": intervention_mode,
            "vectors_remaining": vectors_remaining,
            "debug": debug,
        }

    # If this session already completed all vectors, keep returning the complete report.
    if not vectors_remaining and isinstance(final_report, dict) and final_report:
        answer = _render_reviewer_debate(
            attack_vectors=attack_vectors,
            active_vector=active_vector,
            vectors_remaining=vectors_remaining,
            syntheses=syntheses,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            vector_reports=vector_reports,
            current_vector_report={},
            final_report=final_report,
            round_events=[],
            debate_history=debate_history,
            debate_summary=debate_summary,
            resolution=resolution,
            turn_count=turn_count,
            next_speaker="user",
        )
        debug["reviewer_debate_mode"] = True
        debug["response_stage"] = "reviewer_complete_report"
        debug["final_report_ready"] = True
        return {
            "draft_answer": answer,
            "citations": state.get("citations", []),
            "attack_vectors": attack_vectors,
            "active_vector_id": active_vector_id,
            "debate_history": debate_history,
            "debate_summary": debate_summary,
            "skeptic_position": skeptic_position,
            "advocate_position": advocate_position,
            "resolution": resolution,
            "turn_count": turn_count,
            "syntheses": syntheses,
            "vector_verdicts": vector_verdicts,
            "vector_judgments": vector_judgments,
            "vector_reports": vector_reports,
            "final_report": final_report,
            "next_speaker": "user",
            "intervention_mode": intervention_mode,
            "vectors_remaining": vectors_remaining,
            "debug": debug,
        }

    user_target = _resolve_user_target(
        message=message,
        intervention_mode=intervention_mode,
    )
    if message and not _is_auto_reviewer_bootstrap(message):
        debate_history.append(
            {
                "speaker": "user",
                "content": message,
                "turn": turn_count + 1,
                "vector_id": active_vector_id,
                "target": user_target,
                "intervention_mode": intervention_mode,
            }
        )

    round_events: list[dict[str, Any]] = []
    next_speaker = str(state.get("next_speaker", "skeptic")).strip().lower() or "skeptic"
    if _is_new_reviewer_session_signal(raw_message):
        next_speaker = "skeptic"
    # Complete-panel mode: run the full multi-vector debate in one call.
    total_vectors = max(1, len(vectors_remaining))
    loops = max(
        2,
        settings.reviewer_max_turns * total_vectors * 2,
        settings.reviewer_turns_per_response,
    )
    for _ in range(loops):
        turn_count = _count_vector_turns(debate_history, active_vector_id)
        if turn_count >= 4:
            next_speaker = "synthesise"
        else:
            next_speaker = str(next_speaker or "").strip().lower() or "skeptic"
        skeptic_position = _latest_speaker_content(debate_history, speaker="skeptic", vector_id=active_vector_id)
        advocate_position = _latest_speaker_content(debate_history, speaker="advocate", vector_id=active_vector_id)
        resolution = _infer_resolution(
            skeptic_position=skeptic_position,
            advocate_position=advocate_position,
            history=debate_history,
            active_vector_id=active_vector_id,
            turn_count=turn_count,
        )
        if next_speaker != "synthesise":
            next_speaker = _route_reviewer_turn(
                history=debate_history,
                active_vector_id=active_vector_id,
                resolution=resolution,
                turn_count=turn_count,
                fallback=next_speaker,
            )

        if next_speaker == "user":
            break
        if next_speaker == "synthesise":
            judgment = _run_evidence_only_judge(
                active_vector=active_vector,
                debate_history=debate_history,
                resolution=resolution,
                documents=documents,
            )
            verdict = str(judgment.get("verdict", "contested"))
            vector_verdicts[active_vector_id] = verdict
            vector_judgments[active_vector_id] = judgment
            round_events.append(
                {
                    "speaker": "judge",
                    "content": _render_judge_card(active_vector_id=active_vector_id, judgment=judgment),
                    "vector_id": active_vector_id,
                }
            )
            synthesis = _synthesise_vector(
                active_vector=active_vector,
                verdict=verdict,
                judgment=judgment,
                debate_history=debate_history,
                documents=documents,
            )
            syntheses[active_vector_id] = synthesis
            vectors_remaining = [vector for vector in vectors_remaining if vector != active_vector_id]
            round_events.append(
                {
                    "speaker": "synthesise",
                    "content": synthesis,
                    "vector_id": active_vector_id,
                }
            )
            if not vectors_remaining:
                break
            active_vector_id = vectors_remaining[0]
            active_vector = _get_attack_vector(attack_vectors, active_vector_id)
            next_speaker = "skeptic"
            resolution = "open"
            continue

        turn_content, route_meta = _generate_reviewer_turn(
            speaker=next_speaker,
            active_vector=active_vector,
            objective=message,
            debate_summary=debate_summary,
            debate_history=debate_history,
            documents=documents,
        )
        if not turn_content:
            break

        turn_count += 1
        turn_payload = {
            "speaker": next_speaker,
            "content": turn_content,
            "turn": turn_count,
            "vector_id": active_vector_id,
            "meta": route_meta,
        }
        debate_history.append(turn_payload)
        round_events.append(turn_payload)

        if _count_vector_turns(debate_history, active_vector_id) % 2 == 0:
            debate_summary = _refresh_debate_summary(
                debate_summary=debate_summary,
                active_vector=active_vector,
                debate_history=debate_history,
            )

    turn_count = _count_vector_turns(debate_history, active_vector_id)
    skeptic_position = _latest_speaker_content(debate_history, speaker="skeptic", vector_id=active_vector_id)
    advocate_position = _latest_speaker_content(debate_history, speaker="advocate", vector_id=active_vector_id)
    resolution = _infer_resolution(
        skeptic_position=skeptic_position,
        advocate_position=advocate_position,
        history=debate_history,
        active_vector_id=active_vector_id,
        turn_count=turn_count,
    )
    if turn_count >= 4:
        next_speaker = "synthesise"
    else:
        next_speaker = _route_reviewer_turn(
            history=debate_history,
            active_vector_id=active_vector_id,
            resolution=resolution,
            turn_count=turn_count,
            fallback=next_speaker,
        )
    current_vector_report = _build_current_vector_report(
        active_vector=active_vector,
        skeptic_position=skeptic_position,
        advocate_position=advocate_position,
        debate_history=debate_history,
        documents=documents,
        existing_report=vector_reports.get(active_vector_id, {}),
    )
    if current_vector_report:
        vector_reports[active_vector_id] = current_vector_report

    if next_speaker == "synthesise":
        judgment = _run_evidence_only_judge(
            active_vector=active_vector,
            debate_history=debate_history,
            resolution=resolution,
            documents=documents,
        )
        verdict = str(judgment.get("verdict", "contested"))
        vector_verdicts[active_vector_id] = verdict
        vector_judgments[active_vector_id] = judgment
        round_events.append(
            {
                "speaker": "judge",
                "content": _render_judge_card(active_vector_id=active_vector_id, judgment=judgment),
                "vector_id": active_vector_id,
            }
        )
        synthesis = _synthesise_vector(
            active_vector=active_vector,
            verdict=verdict,
            judgment=judgment,
            debate_history=debate_history,
            documents=documents,
        )
        syntheses[active_vector_id] = synthesis
        vectors_remaining = [vector for vector in vectors_remaining if vector != active_vector_id]
        round_events.append(
            {
                "speaker": "synthesise",
                "content": synthesis,
                "vector_id": active_vector_id,
            }
        )
        if vectors_remaining:
            active_vector_id = vectors_remaining[0]
            active_vector = _get_attack_vector(attack_vectors, active_vector_id)
            turn_count = _count_vector_turns(debate_history, active_vector_id)
            skeptic_position = _latest_speaker_content(debate_history, speaker="skeptic", vector_id=active_vector_id)
            advocate_position = _latest_speaker_content(debate_history, speaker="advocate", vector_id=active_vector_id)
            resolution = _infer_resolution(
                skeptic_position=skeptic_position,
                advocate_position=advocate_position,
                history=debate_history,
                active_vector_id=active_vector_id,
                turn_count=turn_count,
            )
            next_speaker = "skeptic"
        else:
            next_speaker = "user"

    debate_history = debate_history[-64:]
    if not vectors_remaining and syntheses:
        if not isinstance(final_report, dict) or not final_report:
            final_report = _build_reviewer_final_report(
                attack_vectors=attack_vectors,
                vector_verdicts=vector_verdicts,
                vector_judgments=vector_judgments,
                vector_reports=vector_reports,
                syntheses=syntheses,
                debate_history=debate_history,
                documents=documents,
            )
    else:
        final_report = {}
    answer = _render_reviewer_debate(
        attack_vectors=attack_vectors,
        active_vector=active_vector,
        vectors_remaining=vectors_remaining,
        syntheses=syntheses,
        vector_verdicts=vector_verdicts,
        vector_judgments=vector_judgments,
        vector_reports=vector_reports,
        current_vector_report=current_vector_report,
        final_report=final_report,
        round_events=round_events,
        debate_history=debate_history,
        debate_summary=debate_summary,
        resolution=resolution,
        turn_count=turn_count,
        next_speaker=next_speaker,
    )
    debug["reviewer_debate_mode"] = True
    debug["active_vector_id"] = active_vector_id
    debug["resolution"] = resolution
    debug["turn_count"] = turn_count
    debug["warning_turn"] = settings.reviewer_warning_turn
    debug["max_turns"] = settings.reviewer_max_turns
    debug["intervention_mode"] = intervention_mode
    debug["next_speaker"] = next_speaker
    debug["vectors_remaining"] = vectors_remaining[:]
    debug["round_speakers"] = [str(item.get("speaker", "")) for item in round_events]
    debug["round_events"] = [
        {
            "speaker": str(item.get("speaker", "")).strip().lower(),
            "vector_id": str(item.get("vector_id", "")).strip(),
            "turn": int(item.get("turn", 0) or 0),
            "content": _compact_turn_text(str(item.get("content", "")), max_chars=520),
        }
        for item in round_events
        if str(item.get("speaker", "")).strip().lower() in {"skeptic", "advocate", "judge", "synthesise"}
    ]
    if isinstance(final_report, dict) and final_report and not vectors_remaining:
        debug["round_events"] = []
    debug["round_event_count"] = len(debug["round_events"])
    debug["vector_verdicts"] = vector_verdicts
    debug["vector_judgments"] = vector_judgments
    if current_vector_report:
        debug["current_vector_report"] = current_vector_report
    debug["final_report_ready"] = bool(final_report)
    if final_report:
        debug["final_report"] = final_report
    if turn_count >= settings.reviewer_warning_turn:
        debug["turn_warning"] = "debate_closing"

    return {
        "draft_answer": answer,
        "citations": state.get("citations", []),
        "attack_vectors": attack_vectors,
        "active_vector_id": active_vector_id,
        "debate_history": debate_history,
        "debate_summary": debate_summary,
        "skeptic_position": skeptic_position,
        "advocate_position": advocate_position,
        "resolution": resolution,
        "turn_count": turn_count,
        "syntheses": syntheses,
        "vector_verdicts": vector_verdicts,
        "vector_judgments": vector_judgments,
        "vector_reports": vector_reports,
        "final_report": final_report,
        "next_speaker": next_speaker,
        "intervention_mode": intervention_mode,
        "vectors_remaining": vectors_remaining,
        "debug": debug,
    }


def _normalize_reviewer_message(message: str) -> str:
    raw = (message or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"\s+", " ", raw)
    if normalized.lower().startswith("[start debate]"):
        lens_match = re.search(r"focus lens:\s*(.+)$", normalized, flags=re.IGNORECASE)
        lens = lens_match.group(1).strip() if lens_match else ""
        return f"Focus lens: {lens}" if lens else ""
    if normalized.lower().startswith("[lens:"):
        return normalized
    if normalized.lower().startswith("act as a top-tier ml conference reviewer."):
        lens_match = re.search(r"focus lens:\s*([^\.]+)", normalized, flags=re.IGNORECASE)
        lens = lens_match.group(1).strip() if lens_match else ""
        if lens:
            return f"Focus lens: {lens}"
        return ""
    return raw


def _is_new_reviewer_session_signal(message: str) -> bool:
    lower = (message or "").strip().lower()
    return lower.startswith("[start debate]") or lower.startswith("/restart debate")


def _is_auto_reviewer_bootstrap(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return True
    if lower.startswith("act as a top-tier ml conference reviewer."):
        return True
    if lower.startswith("[auto review]"):
        return True
    if lower.startswith("[start debate]"):
        return True
    return False


def _extract_user_target(message: str) -> str | None:
    lower = (message or "").strip().lower()
    if not lower:
        return None
    if lower.startswith("skeptic:") or lower.startswith("@skeptic"):
        return "skeptic"
    if lower.startswith("advocate:") or lower.startswith("@advocate"):
        return "advocate"
    if "skeptic" in lower and "address" in lower:
        return "skeptic"
    if "advocate" in lower and "address" in lower:
        return "advocate"
    return None


def _normalize_intervention_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"defend", "ask", "redirect"}:
        return mode
    return "ask"


def _resolve_user_target(*, message: str, intervention_mode: str) -> str | None:
    explicit = _extract_user_target(message)
    if explicit:
        return explicit
    if intervention_mode == "defend":
        return "skeptic"
    if intervention_mode == "redirect":
        return "skeptic"
    if intervention_mode == "ask":
        return None
    return None


def _user_requested_next_vector(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    markers = ("next", "move on", "another vector", "new vector", "skip this")
    return any(marker in lower for marker in markers)


def _extract_vector_selection(message: str, attack_vectors: list[dict[str, Any]]) -> str | None:
    if not message:
        return None
    vector_ids = [str(vector.get("id", "")).strip() for vector in attack_vectors if str(vector.get("id", "")).strip()]
    if not vector_ids:
        return None

    id_lookup = {vector_id.lower(): vector_id for vector_id in vector_ids}
    normalized = (message or "").lower()
    for key, vector_id in id_lookup.items():
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            return vector_id

    number_match = re.search(r"\b(?:vector|v)?\s*([0-9]{1,2})\b", normalized)
    if number_match:
        index = int(number_match.group(1)) - 1
        if 0 <= index < len(vector_ids):
            return vector_ids[index]
    return None


def _normalize_attack_vectors(
    vectors: list[dict[str, Any]],
    fallback_count: int,
    documents: list[Document],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    quote_cursor = 0
    quote_pool = _quote_candidates(documents)
    for index, raw in enumerate(vectors or [], start=1):
        claim = str(raw.get("claim", "")).strip()
        if not claim:
            continue
        vector_id = str(raw.get("id", "")).strip() or f"V{index}"
        if vector_id in seen:
            continue
        seen.add(vector_id)
        severity = str(raw.get("severity", "medium")).strip().lower()
        if severity not in {"low", "medium", "high", "critical"}:
            severity = "medium"
        category = str(raw.get("category", "method")).strip().lower() or "method"
        quote = str(raw.get("quote", "")).strip()
        if not quote and quote_pool:
            quote = quote_pool[quote_cursor % len(quote_pool)]
            quote_cursor += 1
        if not quote:
            continue
        skeptic_lead = str(raw.get("skeptic_lead", "")).strip()
        if not skeptic_lead:
            skeptic_lead = f"Challenge whether the evidence behind '{claim}' is sufficient."
        normalized.append(
            {
                "id": vector_id,
                "claim": claim,
                "severity": severity,
                "category": category,
                "quote": quote,
                "skeptic_lead": skeptic_lead,
            }
        )
        if len(normalized) >= max(1, fallback_count):
            break
    return normalized


def _generate_attack_vectors(*, message: str, documents: list[Document], count: int) -> list[dict[str, Any]]:
    if not text_service.available or not documents:
        return []
    user_focus = message or "Run a full adversarial-vs-charitable review."
    try:
        raw = text_service.generate(
            system_prompt=(
                "You are generating attack vectors for a paper-review debate.\n"
                "Produce vectors that are contestable and evidence-checkable from paper excerpts.\n"
                "Every vector must include an exact quote from the paper and a skeptic opener.\n"
                "Return JSON only as an array with objects using keys: id, claim, severity, category, quote, skeptic_lead.\n"
                "Use id format V1, V2, ... and severity in {low, medium, high, critical}."
            ),
            user_prompt=(
                f"User focus: {user_focus}\n"
                f"Target vector count: {max(3, count)}\n"
                "Retrieved paper context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 8))}"
            ),
            temperature=0.1,
            max_output_tokens=420,
        )
    except Exception:
        return _fallback_attack_vectors(message=message, documents=documents)
    payload = _try_parse_json_payload(raw)
    if isinstance(payload, list):
        vectors = [item for item in payload if isinstance(item, dict)]
        return vectors
    if isinstance(payload, dict):
        maybe_list = payload.get("attack_vectors")
        if isinstance(maybe_list, list):
            return [item for item in maybe_list if isinstance(item, dict)]
    return _fallback_attack_vectors(message=message, documents=documents)


def _fallback_attack_vectors(message: str, documents: list[Document]) -> list[dict[str, Any]]:
    focus = (message or "").lower()
    quote_pool = _quote_candidates(documents)
    quote = quote_pool[0] if quote_pool else _default_quote(documents)
    q1 = quote_pool[1] if len(quote_pool) > 1 else quote
    q2 = quote_pool[2] if len(quote_pool) > 2 else quote
    q3 = quote_pool[3] if len(quote_pool) > 3 else quote
    q4 = quote_pool[4] if len(quote_pool) > 4 else quote
    vectors = [
        {
            "id": "V1",
            "claim": "Novelty framing may be broader than the directly demonstrated evidence.",
            "severity": "high",
            "category": "novelty",
            "quote": quote,
            "skeptic_lead": "This novelty statement needs a sharper delta against closest prior work.",
        },
        {
            "id": "V2",
            "claim": "Method assumptions and implementation choices need stronger justification.",
            "severity": "high",
            "category": "method",
            "quote": q1,
            "skeptic_lead": "The methodology may be plausible, but the paper has to justify why this setup is the right one.",
        },
        {
            "id": "V3",
            "claim": "Evaluation coverage may not fully support the claimed scope.",
            "severity": "high",
            "category": "evaluation",
            "quote": q2,
            "skeptic_lead": "Claimed scope appears broader than what the benchmark suite actually supports.",
        },
        {
            "id": "V4",
            "claim": "Robustness and ablation evidence appears incomplete.",
            "severity": "medium",
            "category": "ablation",
            "quote": q3,
            "skeptic_lead": "Without targeted ablations, the source of the gains remains unclear.",
        },
        {
            "id": "V5",
            "claim": "Reproducibility detail may be insufficient for clean replication.",
            "severity": "medium",
            "category": "reproducibility",
            "quote": q4,
            "skeptic_lead": "Key replication details are currently too sparse for a reliable reproduction.",
        },
    ]
    if "novelty" in focus:
        return vectors[:3]
    return vectors


def _quote_candidates(documents: list[Document]) -> list[str]:
    candidates: list[str] = []
    for document in documents[:8]:
        text = (document.page_content or "").strip()
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 70 or len(snippet) > 220:
                continue
            lowered = snippet.lower()
            if lowered.startswith("references"):
                continue
            if any(
                marker in lowered
                for marker in (
                    "copyright",
                    "permission",
                    "licensed to acm",
                    "request permissions",
                    "doi:",
                    "all rights reserved",
                    "manuscript submitted",
                )
            ):
                continue
            if lowered.startswith("figure ") or lowered.startswith("table "):
                continue
            if re.search(r"\([a-z]\)\s+[a-z]", lowered):
                continue
            tokens = re.findall(r"[A-Za-z0-9_]+", snippet)
            if len(tokens) < 9:
                continue
            alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
            if len(alpha_tokens) < 7:
                continue
            numeric_ratio = sum(1 for token in tokens if token.isdigit()) / max(1, len(tokens))
            if numeric_ratio > 0.33:
                continue
            candidates.append(snippet)
            if len(candidates) >= 12:
                return candidates
    return candidates


def _default_quote(documents: list[Document]) -> str:
    candidates = _quote_candidates(documents)
    if candidates:
        return candidates[0]
    if documents and (documents[0].page_content or "").strip():
        return (documents[0].page_content or "").strip()[:180]
    return "The paper claims its contribution is effective and broadly applicable."


def _get_attack_vector(vectors: list[dict[str, Any]], vector_id: str) -> dict[str, Any]:
    for item in vectors:
        if str(item.get("id", "")).strip() == vector_id:
            return item
    return vectors[0] if vectors else {"id": "V1", "claim": "Unspecified vector.", "severity": "medium", "category": "method"}


def _count_vector_turns(history: list[dict[str, Any]], vector_id: str) -> int:
    return sum(
        1
        for item in history
        if str(item.get("vector_id", "")) == vector_id and str(item.get("speaker", "")) in {"skeptic", "advocate"}
    )


def _latest_speaker_content(history: list[dict[str, Any]], *, speaker: str, vector_id: str) -> str:
    for item in reversed(history):
        if str(item.get("vector_id", "")) != vector_id:
            continue
        if str(item.get("speaker", "")) != speaker:
            continue
        content = str(item.get("content", "")).strip()
        if content:
            return content
    return ""


def _route_reviewer_turn(
    *,
    history: list[dict[str, Any]],
    active_vector_id: str,
    resolution: str,
    turn_count: int,
    fallback: str,
) -> str:
    if turn_count >= settings.reviewer_max_turns:
        return "synthesise"
    if resolution == "force_closed":
        return "synthesise"
    if turn_count == 0:
        return "skeptic"

    last_turn = _last_vector_turn(history, active_vector_id)
    if last_turn and str(last_turn.get("speaker", "")) == "user":
        target = str(last_turn.get("target", "")).strip().lower()
        if target in {"skeptic", "advocate"}:
            return target
        previous = _last_non_user_speaker(history, active_vector_id)
        if previous == "skeptic":
            return "advocate"
        if previous == "advocate":
            return "skeptic"
        return "skeptic"

    last_meta = _last_route_meta(history, active_vector_id)
    if last_meta.get("addressed_to") == "user":
        if turn_count >= max(3, settings.reviewer_warning_turn - 1):
            return "synthesise"
        previous = _last_non_user_speaker(history, active_vector_id)
        return "advocate" if previous == "skeptic" else "skeptic"
    if last_meta.get("addressed_to") in {"advocate", "skeptic"}:
        return str(last_meta.get("addressed_to"))

    if _speaker_conceded(_latest_speaker_content(history, speaker="skeptic", vector_id=active_vector_id), "skeptic"):
        return "advocate"
    if bool(last_meta.get("concession")):
        return "synthesise"

    if resolution in {"resolved", "deadlocked", "force_closed"}:
        return "synthesise"

    if _is_deadlock(history, active_vector_id):
        return "synthesise"

    skeptic_turns = _speaker_turn_count(history, "skeptic", active_vector_id)
    advocate_turns = _speaker_turn_count(history, "advocate", active_vector_id)
    if skeptic_turns == 0:
        return "skeptic"
    if advocate_turns == 0:
        return "advocate"

    if last_turn and str(last_turn.get("speaker", "")) == "skeptic":
        return "advocate"
    if last_turn and str(last_turn.get("speaker", "")) == "advocate":
        return "skeptic"

    return fallback if fallback in {"skeptic", "advocate", "user", "synthesise"} else "skeptic"


def _last_vector_turn(history: list[dict[str, Any]], vector_id: str) -> dict[str, Any] | None:
    for item in reversed(history):
        if str(item.get("vector_id", "")) == vector_id:
            return item
    return None


def _last_non_user_speaker(history: list[dict[str, Any]], vector_id: str) -> str | None:
    for item in reversed(history):
        if str(item.get("vector_id", "")) != vector_id:
            continue
        speaker = str(item.get("speaker", ""))
        if speaker in {"skeptic", "advocate"}:
            return speaker
    return None


def _last_route_meta(history: list[dict[str, Any]], vector_id: str) -> dict[str, Any]:
    for item in reversed(history):
        if str(item.get("vector_id", "")) != vector_id:
            continue
        speaker = str(item.get("speaker", ""))
        if speaker not in {"skeptic", "advocate"}:
            continue
        meta = item.get("meta", {})
        if isinstance(meta, dict):
            return meta
    return {"addressed_to": "none", "concession": False, "confidence": 0.0}


def _speaker_turn_count(history: list[dict[str, Any]], speaker: str, vector_id: str) -> int:
    return sum(
        1
        for item in history
        if str(item.get("vector_id", "")) == vector_id and str(item.get("speaker", "")) == speaker
    )


def _infer_resolution(
    *,
    skeptic_position: str,
    advocate_position: str,
    history: list[dict[str, Any]],
    active_vector_id: str,
    turn_count: int,
) -> str:
    if turn_count >= settings.reviewer_max_turns:
        return "force_closed"

    last_meta = _last_route_meta(history, active_vector_id)
    if bool(last_meta.get("concession")):
        return "resolved"
    if _speaker_conceded(advocate_position, "advocate"):
        return "resolved"
    if _speaker_conceded(skeptic_position, "skeptic"):
        return "resolved"
    if _is_deadlock(history, active_vector_id):
        if turn_count >= max(4, settings.reviewer_warning_turn - 1):
            return "force_closed"
        return "deadlocked"
    return "open"


def _speaker_conceded(text: str, speaker: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    generic = (
        "i concede",
        "concession: yes",
        "point conceded",
        "this point is conceded",
        "i retract",
    )
    if any(marker in lowered for marker in generic):
        return True
    if speaker == "skeptic":
        skeptic_markers = (
            "claim is sufficiently supported",
            "this concern is covered",
            "concern resolved",
        )
        return any(marker in lowered for marker in skeptic_markers)
    advocate_markers = (
        "cannot defend",
        "defense fails",
        "insufficient evidence",
        "skeptic is correct",
    )
    return any(marker in lowered for marker in advocate_markers)


def _is_deadlock(history: list[dict[str, Any]], vector_id: str) -> bool:
    skeptic_turns = _last_n_speaker_turns(history, speaker="skeptic", vector_id=vector_id, n=2)
    advocate_turns = _last_n_speaker_turns(history, speaker="advocate", vector_id=vector_id, n=2)
    if len(skeptic_turns) < 2 or len(advocate_turns) < 2:
        return False
    skeptic_unchanged = _normalize_turn_text(skeptic_turns[0]) == _normalize_turn_text(skeptic_turns[1])
    advocate_unchanged = _normalize_turn_text(advocate_turns[0]) == _normalize_turn_text(advocate_turns[1])
    if skeptic_unchanged and advocate_unchanged:
        return True

    skeptic_similarity = _turn_similarity_score(skeptic_turns[0], skeptic_turns[1])
    advocate_similarity = _turn_similarity_score(advocate_turns[0], advocate_turns[1])
    if skeptic_similarity >= 0.88 and advocate_similarity >= 0.88:
        return True
    if skeptic_similarity >= 0.94 and advocate_similarity >= 0.80:
        return True
    if advocate_similarity >= 0.94 and skeptic_similarity >= 0.80:
        return True
    return False


def _last_n_speaker_turns(
    history: list[dict[str, Any]],
    *,
    speaker: str,
    vector_id: str,
    n: int,
) -> list[str]:
    collected: list[str] = []
    for item in reversed(history):
        if str(item.get("vector_id", "")) != vector_id:
            continue
        if str(item.get("speaker", "")) != speaker:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        collected.append(content)
        if len(collected) >= n:
            break
    return collected


def _normalize_turn_text(text: str) -> str:
    condensed = re.sub(r"\s+", " ", (text or "").strip().lower())
    return condensed[:240]


def _turn_similarity_score(first: str, second: str) -> float:
    first_tokens = set(_tokenize_for_overlap(first))
    second_tokens = set(_tokenize_for_overlap(second))
    if not first_tokens or not second_tokens:
        return 0.0
    overlap = len(first_tokens & second_tokens)
    union = len(first_tokens | second_tokens)
    if union <= 0:
        return 0.0
    return overlap / union


def _looks_like_score_request(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    markers = (
        "what score",
        "give score",
        "overall score",
        "recommendation score",
        "acceptance score",
        "rate this",
        "what would you rate",
        "rating",
    )
    return any(marker in lower for marker in markers)


def _reviewer_score_response(
    *,
    query: str,
    active_vector: dict[str, Any],
    resolution: str,
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> str:
    fallback = _fallback_scorecard(
        query=query,
        active_vector=active_vector,
        resolution=resolution,
        vector_verdicts=vector_verdicts,
    )
    if not text_service.available:
        return fallback
    try:
        response = text_service.generate(
            system_prompt=(
                "You are a strict ML conference reviewer producing a concise scorecard.\n"
                "Return markdown only with these headings:\n"
                "## Scorecard\n"
                "- Overall: <x.x/10>\n"
                "- Recommendation: <Reject|Weak Reject|Borderline|Weak Accept|Accept>\n"
                "- Confidence: <1-5>\n"
                "- Rationale: <3 bullets tied to evidence with [n] citations>"
            ),
            user_prompt=(
                f"User ask: {query}\n"
                f"Active vector: {active_vector.get('id', 'V?')} - {active_vector.get('claim', '')}\n"
                f"Resolution: {resolution}\n"
                f"Vector verdicts: {json.dumps(vector_verdicts)}\n"
                f"Judge cards: {json.dumps(vector_judgments)}\n"
                "Debate excerpt:\n"
                f"{_format_vector_history(debate_history=debate_history, vector_id=str(active_vector.get('id','')), max_turns=6)}\n\n"
                "Context:\n"
                f"{_format_context(documents, max_docs=4)}"
            ),
            temperature=0.1,
            max_output_tokens=260,
        )
        cleaned = (response or "").strip()
        return cleaned if cleaned else fallback
    except Exception:
        return fallback


def _fallback_scorecard(
    *,
    query: str,
    active_vector: dict[str, Any],
    resolution: str,
    vector_verdicts: dict[str, str],
) -> str:
    verdict = vector_verdicts.get(str(active_vector.get("id", "")), "contested")
    base = 6.2
    if verdict == "skeptic_prevailed":
        base = 5.2
    elif verdict == "advocate_prevailed":
        base = 7.1
    elif resolution in {"deadlocked", "force_closed"}:
        base = 5.8
    score = max(1.0, min(9.5, base))
    if score >= 7.5:
        recommendation = "Accept"
    elif score >= 6.5:
        recommendation = "Weak Accept"
    elif score >= 5.8:
        recommendation = "Borderline"
    elif score >= 4.8:
        recommendation = "Weak Reject"
    else:
        recommendation = "Reject"
    return (
        "## Scorecard\n"
        f"- Overall: {score:.1f}/10\n"
        f"- Recommendation: {recommendation}\n"
        "- Confidence: 2/5\n"
        "- Rationale:\n"
        f"  - The active concern is `{active_vector.get('claim', 'claim strength vs evidence')}` and remains unresolved.\n"
        "  - Current evidence supports parts of the contribution but claim wording should be narrowed.\n"
        "  - A stronger score needs explicit baseline/metric framing and cleaner scope boundaries."
    )


def _generate_reviewer_turn(
    *,
    speaker: str,
    active_vector: dict[str, Any],
    objective: str,
    debate_summary: str,
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> tuple[str, dict[str, Any]]:
    history_excerpt = _format_vector_history(
        debate_history=debate_history,
        vector_id=str(active_vector.get("id", "")),
        max_turns=2,
    )
    turn_docs = _select_turn_documents(
        documents=documents,
        active_vector=active_vector,
        objective=objective,
    )
    evidence_pack = _build_vector_evidence_pack(
        active_vector=active_vector,
        documents=turn_docs,
        limit=3,
    )
    if speaker == "advocate" and len(evidence_pack) > 1:
        evidence_pack = evidence_pack[1:] + evidence_pack[:1]
    previous_same_speaker = _latest_speaker_content(
        debate_history,
        speaker=speaker,
        vector_id=str(active_vector.get("id", "")),
    )
    opponent_speaker = "advocate" if speaker == "skeptic" else "skeptic"
    opponent_latest = _latest_speaker_content(
        debate_history,
        speaker=opponent_speaker,
        vector_id=str(active_vector.get("id", "")),
    )
    vector_id = str(active_vector.get("id", ""))
    speaker_turn_number = _speaker_turn_count(debate_history, speaker, vector_id) + 1
    deterministic = _deterministic_reviewer_turn(
        speaker=speaker,
        active_vector=active_vector,
        evidence_pack=evidence_pack,
        opponent_turn=opponent_latest,
        speaker_turn_number=speaker_turn_number,
        debate_summary=debate_summary,
        history_excerpt=history_excerpt,
    )
    deduped = _reduce_reviewer_repetition(
        speaker=speaker,
        turn_text=deterministic,
        previous_turn=previous_same_speaker,
        active_vector=active_vector,
        evidence_pack=evidence_pack,
    )
    grounded = _enforce_grounded_reviewer_turn(
        speaker=speaker,
        turn_text=deduped,
        active_vector=active_vector,
        evidence_pack=evidence_pack,
        opponent_turn=opponent_latest,
    )
    return grounded, {
        "addressed_to": "advocate" if speaker == "skeptic" else "skeptic",
        "concession": False,
        "confidence": 0.66,
    }


def _deterministic_reviewer_turn(
    *,
    speaker: str,
    active_vector: dict[str, Any],
    evidence_pack: list[dict[str, Any]],
    opponent_turn: str,
    speaker_turn_number: int,
    debate_summary: str,
    history_excerpt: str,
) -> str:
    primary = evidence_pack[0] if evidence_pack else {}
    secondary = evidence_pack[1] if len(evidence_pack) > 1 else primary
    p_text = _compact_turn_text(str(primary.get("snippet", "")).strip() or str(active_vector.get("quote", "")), max_chars=200)
    s_text = _compact_turn_text(str(secondary.get("snippet", "")).strip() or p_text, max_chars=200)
    p_cite = int(primary.get("citation_index", 1)) if primary else 1
    s_cite = int(secondary.get("citation_index", 1)) if secondary else p_cite
    category = str(active_vector.get("category", "method")).strip().lower()
    opponent_short = _extract_opponent_position(opponent_turn) or _compact_turn_text(
        opponent_turn or "No direct opposing argument yet.",
        max_chars=160,
    )
    opponent_gap = _extract_opponent_gap(opponent_turn)

    skeptic_openers = [
        "Position: Evidence supports feasibility, but claim strength still exceeds what is directly shown.",
        "Position: The current claim remains under-justified relative to the evidence presented.",
        "Position: The paper is promising, yet the claim framing is still too broad for the reported support.",
    ]
    advocate_openers = [
        "Position: The claim is defensible when explicitly bounded to reported evidence and scope.",
        "Position: The paper supports a scoped version of the claim with credible evidence.",
        "Position: The contribution can stand if framed conservatively around measured outcomes.",
    ]
    skeptic_gaps = {
        "novelty": "novelty delta versus closest prior work is not explicit in quantified terms",
        "method": "method assumptions are not yet justified strongly enough for the full claim",
        "evaluation": "evaluation coverage does not fully match the breadth of the claim",
        "ablation": "robustness/ablation evidence is too thin for stronger wording",
        "reproducibility": "replication-critical details remain insufficiently specified",
    }
    advocate_strengths = {
        "novelty": "accessibility and telemetry-free analysis provide a meaningful scoped contribution",
        "method": "method choices are reasonable for an exploratory scoped study",
        "evaluation": "evidence supports a narrower claim tied to reported settings",
        "ablation": "feasibility is supported, with robustness claims needing scoped language",
        "reproducibility": "contribution can be retained with explicit implementation caveats",
    }

    if speaker == "skeptic":
        opener = skeptic_openers[(speaker_turn_number - 1) % len(skeptic_openers)]
        gap = skeptic_gaps.get(category, "evidence-claim alignment is still incomplete")
        return (
            f"{opener}\n"
            "Argument:\n"
            f"- Rebuttal target: {opponent_gap or opponent_short}.\n"
            f"- Evidence anchor: \"{p_text}\" [{p_cite}].\n"
            f"- Why rebuttal fails: {gap}. Supporting context: \"{s_text}\" [{s_cite}].\n"
            "- Required revision: narrow the claim sentence and add one explicit comparator on a reported metric."
        )

    opener = advocate_openers[(speaker_turn_number - 1) % len(advocate_openers)]
    strength = advocate_strengths.get(category, "claim can be defended when scoped to reported evidence")
    return (
        f"{opener}\n"
        "Argument:\n"
        f"- Counter to skeptic: {opponent_gap or opponent_short}.\n"
        f"- Evidence anchor: \"{p_text}\" [{p_cite}].\n"
        f"- Defense logic: {strength}. Boundary signal: \"{s_text}\" [{s_cite}].\n"
        "- Accepted limitation: contribution should be stated as scoped to the reported setup and measurements."
    )


def _extract_opponent_position(turn: str) -> str:
    text = (turn or "").strip()
    if not text:
        return ""
    match = re.search(r"Position:\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    sentence = re.sub(r"\s+", " ", match.group(1)).strip()
    return sentence[:150]


def _extract_opponent_gap(turn: str) -> str:
    text = (turn or "").strip()
    if not text:
        return ""
    patterns = (
        r"Unresolved gap:\s*(.+?)(?:\n|$)",
        r"Remaining issue:\s*(.+?)(?:\n|$)",
        r"Why rebuttal fails:\s*(.+?)(?:\n|$)",
        r"Counter to skeptic:\s*(.+?)(?:\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            snippet = re.sub(r"\s+", " ", match.group(1)).strip()
            if snippet:
                return snippet[:180]
    return ""


def _enforce_grounded_reviewer_turn(
    *,
    speaker: str,
    turn_text: str,
    active_vector: dict[str, Any],
    evidence_pack: list[dict[str, Any]],
    opponent_turn: str = "",
) -> str:
    text = (turn_text or "").strip()
    if not text:
        return _grounded_reviewer_template(
            speaker=speaker,
            active_vector=active_vector,
            evidence_pack=evidence_pack,
            opponent_turn=opponent_turn,
        )
    if _reviewer_turn_low_quality(text=text, evidence_pack=evidence_pack):
        return _grounded_reviewer_template(
            speaker=speaker,
            active_vector=active_vector,
            evidence_pack=evidence_pack,
            opponent_turn=opponent_turn,
        )
    if "position:" not in text.lower() or "argument:" not in text.lower():
        return _grounded_reviewer_template(
            speaker=speaker,
            active_vector=active_vector,
            evidence_pack=evidence_pack,
            opponent_turn=opponent_turn,
        )
    return text


def _reviewer_turn_low_quality(*, text: str, evidence_pack: list[dict[str, Any]]) -> bool:
    lower = (text or "").lower()
    if len(lower) < 80:
        return True
    if len(lower) > 2200:
        return True
    if not re.search(r"\[[0-9]+\]", lower):
        return True

    evidence_tokens: set[str] = set()
    evidence_overlaps: list[float] = []
    for item in evidence_pack:
        snippet = str(item.get("snippet", ""))
        evidence_tokens.update(_tokenize_for_overlap(snippet))
        evidence_overlaps.append(_overlap_score(lower, set(_tokenize_for_overlap(snippet))))
    if evidence_tokens:
        overlap = _overlap_score(lower, evidence_tokens)
        if overlap < 0.14:
            return True
    if evidence_overlaps and max(evidence_overlaps) < 0.22:
        return True

    generic_markers = (
        "real-world scenarios",
        "practical applications",
        "specific examples would strengthen",
        "case studies",
        "significant impact",
        "broader implications",
        "methodological shift",
        "intriguing",
        "it is difficult to assess",
        "would strengthen this argument significantly",
    )
    generic_hits = sum(1 for marker in generic_markers if marker in lower)
    if generic_hits >= 1:
        return True

    if lower.count("?") >= 3:
        return True
    if lower.count("while") >= 3 and "response to" not in lower:
        return True
    return False


def _grounded_reviewer_template(
    *,
    speaker: str,
    active_vector: dict[str, Any],
    evidence_pack: list[dict[str, Any]],
    opponent_turn: str = "",
) -> str:
    quote = str(active_vector.get("quote", "")).strip()
    claim = str(active_vector.get("claim", "")).strip() or "the active claim"

    primary = evidence_pack[0] if evidence_pack else {}
    secondary = evidence_pack[1] if len(evidence_pack) > 1 else primary
    p_text = _compact_turn_text(str(primary.get("snippet", "")).strip() or quote or claim, max_chars=180)
    p_cite = int(primary.get("citation_index", 1)) if primary else 1
    s_text = _compact_turn_text(str(secondary.get("snippet", "")).strip() or p_text, max_chars=180)
    s_cite = int(secondary.get("citation_index", 1)) if secondary else p_cite
    category = str(active_vector.get("category", "method")).strip().lower()
    opponent_short = _compact_turn_text(opponent_turn, max_chars=140) if opponent_turn else ""

    skeptic_gap_by_category = {
        "novelty": "the novelty delta versus closest prior work is still not explicit in quantified terms.",
        "method": "key method assumptions are not yet justified strongly enough for the full claim.",
        "evaluation": "evaluation coverage does not yet fully match the breadth of the claim.",
        "ablation": "ablation/robustness evidence is still too thin to support stronger wording.",
        "reproducibility": "replication-critical details remain underspecified for a strong claim.",
    }
    advocate_defense_by_category = {
        "novelty": "the contribution can be defended as a scoped accessibility/telemetry-free approach.",
        "method": "the method is defensible when described as an exploratory, scoped design choice.",
        "evaluation": "evaluation can support a narrower claim aligned to reported settings.",
        "ablation": "current evidence supports feasibility; stronger robustness framing should be conditional.",
        "reproducibility": "claim is defendable when bounded and accompanied by explicit implementation caveats.",
    }
    skeptic_gap = skeptic_gap_by_category.get(category, "the claim still extends beyond what is directly demonstrated.")
    advocate_defense = advocate_defense_by_category.get(category, "the claim is defensible when explicitly scoped.")

    if speaker == "skeptic":
        return (
            "Position: Evidence supports feasibility, but the claim is still broader than what is directly demonstrated.\n"
            "Argument:\n"
            f"- Evidence shown: \"{p_text}\" [{p_cite}].\n"
            f"- Gap: {skeptic_gap} Supporting context: \"{s_text}\" [{s_cite}].\n"
            f"- Response to advocate: {opponent_short if opponent_short else 'scope-limited defense is reasonable, but still needs tighter evidence linkage.'}"
        )
    return (
        "Position: The claim is defensible when explicitly bounded to the reported setup and evidence.\n"
        "Argument:\n"
        f"- Supporting evidence: \"{p_text}\" [{p_cite}].\n"
        f"- Scope support: \"{s_text}\" [{s_cite}] indicates the paper already states limits that can bound the claim.\n"
        f"- Response to skeptic: {advocate_defense} Keep wording scoped to reported measurements."
    )


def _format_vector_history(
    *,
    debate_history: list[dict[str, Any]],
    vector_id: str,
    max_turns: int,
) -> str:
    relevant = [item for item in debate_history if str(item.get("vector_id", "")) == vector_id]
    if not relevant:
        return "No prior turns."
    trimmed = relevant[-max_turns:]
    lines = []
    for item in trimmed:
        speaker = str(item.get("speaker", "unknown")).upper()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "No prior turns."


def _format_vector_history_compact(
    *,
    debate_history: list[dict[str, Any]],
    vector_id: str,
    max_turns: int,
    max_chars_per_turn: int = 220,
) -> str:
    relevant = [item for item in debate_history if str(item.get("vector_id", "")) == vector_id]
    if not relevant:
        return "No prior turns."
    trimmed = relevant[-max_turns:]
    lines: list[str] = []
    for item in trimmed:
        speaker = str(item.get("speaker", "unknown")).upper()
        content = _compact_turn_text(str(item.get("content", "")), max_chars=max_chars_per_turn)
        if not content:
            continue
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "No prior turns."


def _compact_turn_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"ROUTE_JSON:\s*\{[\s\S]*\}\s*$", "", text or "", flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _select_turn_documents(
    *,
    documents: list[Document],
    active_vector: dict[str, Any],
    objective: str,
) -> list[Document]:
    if not documents:
        return []
    query_terms = set(
        _tokenize_for_overlap(
            " ".join(
                [
                    str(active_vector.get("claim", "")),
                    str(active_vector.get("category", "")),
                    str(active_vector.get("quote", "")),
                    objective or "",
                ]
            )
        )
    )
    scored: list[tuple[Document, float]] = []
    for index, document in enumerate(documents):
        text = document.page_content or ""
        overlap = _overlap_score(text, query_terms)
        rank_prior = max(0.05, 1.0 - (index / max(1, len(documents))))
        penalty = _low_signal_penalty(text)
        scored.append((document, overlap + (0.4 * rank_prior) - (0.6 * penalty)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    selected: list[Document] = []
    seen_pages: set[tuple[str, Any]] = set()
    for document, _score in scored:
        metadata = document.metadata or {}
        filename = str(metadata.get("filename", "")).strip()
        page = metadata.get("page")
        key = (filename, page)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        selected.append(document)
        if len(selected) >= 4:
            break
    return selected if selected else [document for document, _ in scored[:2]]


def _extract_route_meta(text: str, default_target: str) -> dict[str, Any]:
    payload = {"addressed_to": default_target, "concession": False, "confidence": 0.55}
    match = re.search(r"ROUTE_JSON:\s*(\{[\s\S]*\})", text or "", flags=re.IGNORECASE)
    if not match:
        return payload
    parsed = _try_parse_json_payload(match.group(1))
    if not isinstance(parsed, dict):
        return payload
    addressed = str(parsed.get("addressed_to", default_target)).strip().lower()
    if addressed not in {"advocate", "skeptic", "user", "none"}:
        addressed = default_target
    concession = bool(parsed.get("concession", False))
    confidence = parsed.get("confidence", 0.55)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.55
    confidence = max(0.0, min(1.0, confidence))
    return {
        "addressed_to": addressed,
        "concession": concession,
        "confidence": confidence,
    }


def _strip_route_json_footer(text: str) -> str:
    cleaned = re.sub(r"ROUTE_JSON:\s*\{[\s\S]*\}\s*$", "", text or "", flags=re.IGNORECASE).strip()
    return cleaned


def _fallback_reviewer_turn(
    *,
    speaker: str,
    active_vector: dict[str, Any],
    documents: list[Document],
) -> str:
    quote = str(active_vector.get("quote", "")).strip() or _default_quote(documents)
    evidence = _fallback_vector_evidence(active_vector=active_vector, documents=documents, limit=2)
    primary = evidence[0] if evidence else quote
    secondary = evidence[1] if len(evidence) > 1 else primary
    if speaker == "skeptic":
        return (
            "Position: Evidence supports feasibility, but not the full strength of the claim as currently worded.\n"
            "Argument:\n"
            f"- Triggered claim: \"{quote}\"\n"
            f"- Evidence check: \"{primary}\" [1].\n"
            f"- Scope gap: \"{secondary}\" [1] indicates limits that should be reflected in claim wording.\n"
            "- Request: tighten claim scope and anchor it to one explicit reported metric.\n"
            'ROUTE_JSON: {"addressed_to":"advocate","concession":false,"confidence":0.54}'
        )
    return (
        "Position: The claim is supportable when bounded to the paper's measured scope.\n"
        "Argument:\n"
        f"- Triggered claim: \"{quote}\"\n"
        f"- Supporting evidence: \"{primary}\" [1].\n"
        f"- Scope caveat already present: \"{secondary}\" [1].\n"
        "- Revision path: keep contribution framing and rewrite novelty language to match exactly what is measured.\n"
        'ROUTE_JSON: {"addressed_to":"skeptic","concession":false,"confidence":0.56}'
    )


def _fallback_vector_evidence(
    *,
    active_vector: dict[str, Any],
    documents: list[Document],
    limit: int,
) -> list[str]:
    if not documents:
        return []
    query = " ".join(
        [
            str(active_vector.get("claim", "")),
            str(active_vector.get("category", "")),
            str(active_vector.get("quote", "")),
            str(active_vector.get("skeptic_lead", "")),
        ]
    )
    query_terms = set(_tokenize_for_overlap(query))
    query_phrases = _query_phrases(query)
    candidates: list[tuple[float, str]] = []
    for doc_index, document in enumerate(documents[:4]):
        text = (document.page_content or "").strip()
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        doc_prior = max(0.05, 1.0 - (doc_index / max(1, len(documents))))
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 30:
                continue
            if _looks_like_reference_snippet(snippet):
                continue
            if _looks_like_non_argument_snippet(snippet):
                continue
            overlap = _overlap_score(snippet.lower(), query_terms)
            phrase = _phrase_overlap_score(_normalize_for_phrase_match(snippet.lower()), query_phrases)
            numeric_bonus = 0.15 if re.search(r"\b\d+(?:\.\d+)?%?\b", snippet) else 0.0
            score = (0.9 * overlap) + (0.8 * phrase) + (0.2 * doc_prior) + numeric_bonus
            score -= min(0.3, _low_signal_penalty(snippet))
            candidates.append((score, snippet))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen_norm: set[str] = set()
    for score, snippet in candidates:
        if score < 0.18:
            continue
        normalized = re.sub(r"\s+", " ", re.sub(r"\[[0-9]+\]", "", snippet)).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_norm:
            continue
        seen_norm.add(key)
        selected.append(normalized)
        if len(selected) >= max(1, limit):
            break
    return selected


def _build_vector_evidence_pack(
    *,
    active_vector: dict[str, Any],
    documents: list[Document],
    limit: int,
) -> list[dict[str, Any]]:
    snippets = _fallback_vector_evidence(
        active_vector=active_vector,
        documents=documents,
        limit=max(2, limit + 1),
    )
    if not snippets and documents:
        fallback_text = _compact_turn_text(documents[0].page_content or "", max_chars=220)
        if fallback_text:
            snippets = [fallback_text]
    pack: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, snippet in enumerate(snippets, start=1):
        normalized = re.sub(r"\s+", " ", snippet).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        doc_index = _best_doc_index_for_snippet(snippet=normalized, documents=documents)
        citation_index = doc_index + 1 if doc_index >= 0 else 1
        metadata = documents[doc_index].metadata if 0 <= doc_index < len(documents) else {}
        pack.append(
            {
                "id": f"E{index}",
                "snippet": normalized,
                "citation_index": citation_index,
                "filename": str((metadata or {}).get("filename", "unknown.pdf")),
                "page": (metadata or {}).get("page"),
                "chunk_id": str((metadata or {}).get("chunk_id", "")),
            }
        )
        if len(pack) >= max(1, limit):
            break
    return pack


def _best_doc_index_for_snippet(*, snippet: str, documents: list[Document]) -> int:
    if not documents:
        return -1
    lowered = snippet.lower()
    query_terms = set(_tokenize_for_overlap(lowered))
    best_index = 0
    best_score = float("-inf")
    for index, document in enumerate(documents):
        text = (document.page_content or "").lower()
        if not text:
            continue
        contains_bonus = 1.2 if lowered and lowered in text else 0.0
        overlap = _overlap_score(text, query_terms)
        score = contains_bonus + overlap
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _format_evidence_pack(pack: list[dict[str, Any]]) -> str:
    if not pack:
        return "- No direct evidence snippets available from current retrieval."
    lines: list[str] = []
    for item in pack:
        evidence_id = str(item.get("id", "E?"))
        citation_index = int(item.get("citation_index", 1))
        filename = str(item.get("filename", "unknown.pdf"))
        page = item.get("page")
        page_text = f", p.{page}" if page else ""
        snippet = str(item.get("snippet", "")).strip()
        lines.append(
            f"- {evidence_id} -> [{citation_index}] {filename}{page_text}: \"{snippet}\""
        )
    return "\n".join(lines)


def _clean_reviewer_turn_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\s*#+\s*Reviewer\s+[AB]\s*\([^)]+\)\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Reviewer\s+[AB]\s*\([^)]+\)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _reduce_reviewer_repetition(
    *,
    speaker: str,
    turn_text: str,
    previous_turn: str,
    active_vector: dict[str, Any],
    evidence_pack: list[dict[str, Any]],
) -> str:
    current = (turn_text or "").strip()
    previous = (previous_turn or "").strip()
    if not current or not previous:
        return current
    similarity = _turn_similarity_score(current, previous)
    if similarity < 0.70:
        return current

    alternate = evidence_pack[1] if len(evidence_pack) > 1 else (evidence_pack[0] if evidence_pack else {})
    alt_snippet = str(alternate.get("snippet", "")).strip()
    alt_citation = int(alternate.get("citation_index", 1)) if alternate else 1
    claim = str(active_vector.get("claim", "")).strip() or "this claim"
    category = str(active_vector.get("category", "method")).strip().lower()

    skeptic_gap_by_category = {
        "novelty": "the novelty delta versus closest prior work still needs an explicit quantitative statement",
        "method": "the method rationale still needs stronger evidence for the full framing",
        "evaluation": "evaluation support is still narrower than the breadth of the current claim",
        "ablation": "robustness evidence is still too limited for stronger wording",
        "reproducibility": "replication-critical details remain incomplete for a stronger claim",
    }
    advocate_support_by_category = {
        "novelty": "the contribution can still be defended as a scoped telemetry-free analysis approach",
        "method": "the method remains defensible if framed as a scoped design choice",
        "evaluation": "the reported setup supports a narrower claim aligned to measured outcomes",
        "ablation": "the evidence supports feasibility, with robustness framed as future work",
        "reproducibility": "the claim can stand with explicit implementation boundaries",
    }

    if speaker == "skeptic":
        return (
            "Position: The unresolved evidence gap still blocks a stronger claim.\n"
            "Argument:\n"
            f"- Evidence anchor: \"{alt_snippet or claim}\" [{alt_citation}].\n"
            f"- Remaining issue: {skeptic_gap_by_category.get(category, 'evidence-claim alignment is still incomplete')}.\n"
            "- Required revision: narrow wording and attach one concrete metric/baseline comparator."
        )

    return (
        "Position: The claim remains defendable with explicit scope boundaries.\n"
        "Argument:\n"
        f"- Supporting anchor: \"{alt_snippet or claim}\" [{alt_citation}].\n"
        f"- Defense: {advocate_support_by_category.get(category, 'the claim is defensible when tied directly to reported evidence')}.\n"
        "- Revision path: keep the contribution, but state limits and comparator in the same paragraph."
    )


def _refresh_debate_summary(
    *,
    debate_summary: str,
    active_vector: dict[str, Any],
    debate_history: list[dict[str, Any]],
) -> str:
    vector_id = str(active_vector.get("id", "V?"))
    claim = str(active_vector.get("claim", "")).strip() or "the active claim"
    skeptic_latest = _compact_turn_text(
        _latest_speaker_content(debate_history, speaker="skeptic", vector_id=vector_id),
        max_chars=180,
    )
    advocate_latest = _compact_turn_text(
        _latest_speaker_content(debate_history, speaker="advocate", vector_id=vector_id),
        max_chars=180,
    )
    if not skeptic_latest and not advocate_latest:
        return debate_summary
    sentence_1 = f"Debate on {vector_id} centers on whether {claim.lower()} is adequately supported."
    sentence_2 = f"Skeptic focus: {skeptic_latest or 'insufficient concrete evidence.'}"
    sentence_3 = f"Advocate focus: {advocate_latest or 'scope-limited defense with partial support.'}"
    return " ".join([sentence_1, sentence_2, sentence_3]).strip()


def _run_evidence_only_judge(
    *,
    active_vector: dict[str, Any],
    debate_history: list[dict[str, Any]],
    resolution: str,
    documents: list[Document],
) -> dict[str, Any]:
    vector_id = str(active_vector.get("id", "V?"))
    evidence_pack = _build_vector_evidence_pack(
        active_vector=active_vector,
        documents=documents,
        limit=3,
    )
    if resolution in {"deadlocked", "force_closed"}:
        return {
            "verdict": "contested",
            "confidence": 0.42,
            "rationale": "Debate hit stopping criteria without a clean evidence-based resolution.",
            "decisive_evidence": [item.get("id", "E1") for item in evidence_pack[:1]],
            "evidence_pack": evidence_pack,
        }

    fallback = {
        "verdict": "contested",
        "confidence": 0.5,
        "rationale": "Evidence was mixed and no side clearly prevailed on the quoted claim.",
        "decisive_evidence": [item.get("id", "E1") for item in evidence_pack[:1]],
        "evidence_pack": evidence_pack,
    }
    if not text_service.available:
        return fallback
    try:
        response = text_service.generate(
            system_prompt=(
                "You are an evidence-only judge for a paper-review trial.\n"
                "Decide ONLY from the provided evidence pack and debate excerpt.\n"
                "If evidence does not settle the claim, return contested.\n"
                "Return JSON only with keys: verdict, confidence, rationale, decisive_evidence.\n"
                "Allowed verdict values: skeptic_prevailed, advocate_prevailed, contested."
            ),
            user_prompt=(
                f"Claim vector: {vector_id} | {active_vector.get('claim', '')}\n"
                f"Claim trigger quote: {active_vector.get('quote', '')}\n\n"
                "Evidence pack:\n"
                f"{_format_evidence_pack(evidence_pack)}\n\n"
                "Debate excerpt:\n"
                f"{_format_vector_history(debate_history=debate_history, vector_id=vector_id, max_turns=8)}\n\n"
                "Rules:\n"
                "- decisive_evidence must be an array of evidence ids from the pack (for example [\"E1\",\"E2\"]).\n"
                "- rationale must be one short sentence."
            ),
            temperature=0.0,
            max_output_tokens=180,
        )
    except Exception:
        return fallback

    payload = _try_parse_json_payload(response)
    if not isinstance(payload, dict):
        return fallback

    verdict = str(payload.get("verdict", "contested")).strip().lower()
    if verdict not in {"skeptic_prevailed", "advocate_prevailed", "contested"}:
        verdict = "contested"
    try:
        confidence = float(payload.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(payload.get("rationale", "")).strip() or fallback["rationale"]
    decisive_raw = payload.get("decisive_evidence", [])
    decisive: list[str] = []
    allowed_ids = {str(item.get("id", "")) for item in evidence_pack}
    if isinstance(decisive_raw, list):
        for item in decisive_raw:
            evidence_id = str(item).strip()
            if evidence_id in allowed_ids and evidence_id not in decisive:
                decisive.append(evidence_id)
    if not decisive and evidence_pack:
        decisive = [str(evidence_pack[0].get("id", "E1"))]
    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
        "decisive_evidence": decisive,
        "evidence_pack": evidence_pack,
    }


def _render_judge_card(*, active_vector_id: str, judgment: dict[str, Any]) -> str:
    verdict = str(judgment.get("verdict", "contested"))
    confidence = float(judgment.get("confidence", 0.0))
    rationale = str(judgment.get("rationale", "No rationale.")).strip()
    evidence_pack = judgment.get("evidence_pack", [])
    evidence_lookup: dict[str, str] = {}
    if isinstance(evidence_pack, list):
        for item in evidence_pack:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("id", "")).strip()
            citation_index = int(item.get("citation_index", 1))
            if evidence_id:
                evidence_lookup[evidence_id] = f"{evidence_id} [{citation_index}]"
    decisive_labels: list[str] = []
    decisive_raw = judgment.get("decisive_evidence", [])
    if isinstance(decisive_raw, list):
        for item in decisive_raw:
            evidence_id = str(item).strip()
            if not evidence_id:
                continue
            decisive_labels.append(evidence_lookup.get(evidence_id, evidence_id))
    decisive_text = ", ".join(decisive_labels) if decisive_labels else "none"
    return (
        "### Evidence-only Judge\n"
        f"Vector: {active_vector_id}\n"
        f"Verdict: {verdict}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Decisive Evidence: {decisive_text}\n"
        f"Rationale: {rationale}"
    )


def _synthesise_vector(
    *,
    active_vector: dict[str, Any],
    verdict: str,
    judgment: dict[str, Any],
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> str:
    vector_id = str(active_vector.get("id", "V?"))
    if not text_service.available:
        return _build_grounded_rewrite_card(active_vector=active_vector, verdict=verdict, judgment=judgment)
    try:
        evidence_text = _format_evidence_pack(judgment.get("evidence_pack", []))
        response = text_service.generate(
            system_prompt=(
                "You are a rewrite compiler that converts a judged review claim into one concrete patch.\n"
                "Do not summarize the debate. Output exactly one actionable patch tied to paper evidence.\n"
                "Do not invent metrics, datasets, or numeric results not present in the evidence pack."
            ),
            user_prompt=(
                f"Vector: {vector_id} | {active_vector.get('claim', '')}\n"
                f"Verdict: {verdict}\n"
                f"Judge rationale: {judgment.get('rationale', '')}\n"
                f"Judge decisive evidence ids: {judgment.get('decisive_evidence', [])}\n"
                f"Quoted trigger sentence: {active_vector.get('quote', '')}\n"
                "Evidence pack:\n"
                f"{evidence_text}\n\n"
                "Debate transcript:\n"
                f"{_format_vector_history(debate_history=debate_history, vector_id=vector_id, max_turns=10)}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 8))}\n\n"
                "Output format:\n"
                "### Rewrite Compiler Card\n"
                "Target Section: <section name or approximate location>\n"
                "Target Claim: <the claim being rewritten>\n"
                "Patch Instruction: <one concrete instruction sentence with metric/baseline/clarification target>\n"
                "Patch (Before -> After):\n"
                "- Before: <short phrase for current wording weakness>\n"
                "- After: <short phrase for corrected wording>\n"
                "Why: <one sentence>"
            ),
            temperature=0.1,
            max_output_tokens=280,
        )
        candidate = (response or "").strip()
        if not candidate:
            return _build_grounded_rewrite_card(active_vector=active_vector, verdict=verdict, judgment=judgment)
        if _rewrite_card_low_quality(candidate, judgment.get("evidence_pack", [])):
            return _build_grounded_rewrite_card(active_vector=active_vector, verdict=verdict, judgment=judgment)
        return candidate
    except Exception:
        return _build_grounded_rewrite_card(active_vector=active_vector, verdict=verdict, judgment=judgment)


def _build_grounded_rewrite_card(
    *,
    active_vector: dict[str, Any],
    verdict: str,
    judgment: dict[str, Any],
) -> str:
    claim = str(active_vector.get("claim", "")).strip() or "the active claim"
    evidence_pack = judgment.get("evidence_pack", []) if isinstance(judgment, dict) else []
    snippet = ""
    citation = 1
    if isinstance(evidence_pack, list) and evidence_pack:
        first = evidence_pack[0] if isinstance(evidence_pack[0], dict) else {}
        snippet = _compact_turn_text(str(first.get("snippet", "")).strip(), max_chars=220)
        try:
            citation = int(first.get("citation_index", 1))
        except Exception:
            citation = 1
    section_hint = _rewrite_section_hint(claim)
    evidence_line = f'Key Evidence: "{snippet}" [{citation}]' if snippet else "Key Evidence: use the strongest cited sentence for this claim."
    return (
        "### Rewrite Compiler Card\n"
        f"Verdict: {verdict}\n"
        f"Target Section: {section_hint}\n"
        f"Target Claim: {claim}\n"
        f"{evidence_line}\n"
        "Patch Instruction: rewrite the claim to stay within measured scope, then add one explicit metric/comparator already reported in the cited evidence.\n"
        "Patch (Before -> After):\n"
        "- Before: broad claim wording that exceeds direct support.\n"
        "- After: scoped claim wording tied to cited measurement and stated limitation.\n"
        "Why: this converts a contested claim into an evidence-aligned statement without overreach."
    )


def _rewrite_section_hint(claim: str) -> str:
    lower = (claim or "").lower()
    if "novel" in lower or "contribution" in lower:
        return "Introduction / Contribution Framing"
    if "method" in lower or "implementation" in lower:
        return "Methods"
    if "evaluation" in lower or "benchmark" in lower:
        return "Results / Evaluation"
    if "ablation" in lower or "robust" in lower:
        return "Results / Limitations"
    return "Discussion"


def _rewrite_card_low_quality(text: str, evidence_pack: list[dict[str, Any]]) -> bool:
    lower = (text or "").lower()
    if "patch instruction:" not in lower:
        return True
    if "before" not in lower or "after" not in lower:
        return True
    evidence_text = " ".join(str((item or {}).get("snippet", "")) for item in evidence_pack if isinstance(item, dict))
    allowed_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", evidence_text))
    output_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text or ""))
    if output_numbers and allowed_numbers:
        unseen = {value for value in output_numbers if value not in allowed_numbers}
        if unseen:
            return True
    banned_markers = (
        "significant improvement",
        "state of the art",
        "outperform",
        "novel benchmark gain",
    )
    return any(marker in lower for marker in banned_markers)


def _render_reviewer_debate(
    *,
    attack_vectors: list[dict[str, Any]],
    active_vector: dict[str, Any],
    vectors_remaining: list[str],
    syntheses: dict[str, str],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    current_vector_report: dict[str, Any],
    final_report: dict[str, Any],
    round_events: list[dict[str, Any]],
    debate_history: list[dict[str, Any]],
    debate_summary: str,
    resolution: str,
    turn_count: int,
    next_speaker: str,
) -> str:
    active_vector_id = str(active_vector.get("id", "V?"))
    active_claim = str(active_vector.get("claim", "")).strip() or "Unspecified claim."
    total_claims = max(1, len(attack_vectors))
    active_rank = 1
    for idx, vector in enumerate(attack_vectors, start=1):
        if str(vector.get("id", "")) == active_vector_id:
            active_rank = idx
            break

    if isinstance(final_report, dict) and final_report and not vectors_remaining:
        complete_reports = dict(vector_reports or {})
        if isinstance(current_vector_report, dict) and current_vector_report:
            complete_reports[active_vector_id] = current_vector_report
        return _render_reviewer_complete_report(
            attack_vectors=attack_vectors,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            vector_reports=complete_reports,
            syntheses=syntheses,
            debate_history=debate_history,
            final_report=final_report,
        )

    status_map = {
        "open": "Open",
        "resolved": "Resolved",
        "deadlocked": "Deadlocked",
        "force_closed": "Force Closed",
    }
    status_label = status_map.get(str(resolution).lower(), str(resolution).title())

    skeptic_latest = _latest_speaker_content(
        debate_history,
        speaker="skeptic",
        vector_id=active_vector_id,
    )
    advocate_latest = _latest_speaker_content(
        debate_history,
        speaker="advocate",
        vector_id=active_vector_id,
    )

    judgment = vector_judgments.get(active_vector_id, {})
    has_judgment = isinstance(judgment, dict) and bool(judgment)
    judge_verdict = str(judgment.get("verdict", "pending")) if has_judgment else "pending"
    try:
        judge_confidence = float(judgment.get("confidence", 0.0)) if has_judgment else 0.0
    except Exception:
        judge_confidence = 0.0
    judge_rationale = str(judgment.get("rationale", "")).strip() if has_judgment else ""

    active_rewrite = syntheses.get(active_vector_id, "")
    if not active_rewrite and syntheses:
        # Show the latest completed rewrite card if current claim is still open.
        latest_completed_id = list(syntheses.keys())[-1]
        active_rewrite = syntheses.get(latest_completed_id, "")

    queued_claims: list[str] = []
    for idx, vector in enumerate(attack_vectors, start=1):
        vector_id = str(vector.get("id", "V?"))
        if vector_id == active_vector_id:
            continue
        claim_text = str(vector.get("claim", "")).strip()
        if not claim_text:
            continue
        state = "resolved" if vector_id in syntheses else "queued"
        queued_claims.append(f"- Claim {idx} ({state}): {claim_text}")
        if len(queued_claims) >= 3:
            break

    next_move = _human_next_move(next_speaker=next_speaker, vector_id=active_vector_id)
    this_round = _render_round_events_compact(round_events=round_events, vector_id=active_vector_id)

    blocks = [
        "## Review Panel",
        f"Primary Claim: {active_claim}",
        f"Claim {active_rank}/{total_claims} | Status: {status_label} | Turn {turn_count}/{settings.reviewer_max_turns}",
        "",
        "### Skeptic (Latest)",
        skeptic_latest or "No skeptic argument yet.",
        "",
        "### Advocate (Latest)",
        advocate_latest or "No advocate response yet.",
    ]

    if has_judgment:
        blocks.extend(
            [
                "",
                "### Judge",
                f"Verdict: {judge_verdict}",
                f"Confidence: {judge_confidence:.2f}",
                f"Why: {judge_rationale or 'No rationale available.'}",
            ]
        )

    if active_rewrite:
        blocks.extend(
            [
                "",
                "### Recommended Rewrite",
                active_rewrite,
            ]
        )

    if isinstance(current_vector_report, dict) and current_vector_report:
        blocks.extend(
            [
                "",
                "### Claim Intelligence",
                _render_current_vector_report_brief(current_vector_report),
            ]
        )

    if queued_claims:
        blocks.extend(
            [
                "",
                "### Other Claims",
                "\n".join(queued_claims),
            ]
        )

    blocks.extend(
        [
            "",
            "### Round Timeline",
            this_round,
            "",
            "### Next Step",
            next_move,
            f"Claims remaining: {len(vectors_remaining)}",
            "Use Reviewer Controls: `Next Turn` to continue, `Restart` to reset this debate.",
        ]
    )
    return "\n".join(blocks).strip()


def _render_reviewer_complete_report(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
    debate_history: list[dict[str, Any]],
    final_report: dict[str, Any],
) -> str:
    overview = str(final_report.get("overview", "")).strip() or "Panel review completed."
    final_decision = str(final_report.get("final_decision", "")).strip() or "No final decision available."
    confidence = final_report.get("confidence", 0.0)
    try:
        confidence_text = f"{float(confidence):.2f}"
    except Exception:
        confidence_text = "0.00"
    agreements = [str(item).strip() for item in final_report.get("agreements", []) if str(item).strip()]
    disagreements = [str(item).strip() for item in final_report.get("disagreements", []) if str(item).strip()]
    suggestions = [str(item).strip() for item in final_report.get("final_suggestions", []) if str(item).strip()]

    lines: list[str] = [
        "## Reviewer Complete Report",
        overview,
        "",
        f"Final Decision: {final_decision}",
        f"Panel Confidence: {confidence_text}",
    ]
    if agreements:
        lines.extend(["", "### Agreements"])
        lines.extend(f"- {item}" for item in agreements[:5])
    if disagreements:
        lines.extend(["", "### Major Disagreements"])
        lines.extend(f"- {item}" for item in disagreements[:5])
    if suggestions:
        lines.extend(["", "### Final Suggestions"])
        lines.extend(f"- {item}" for item in suggestions[:6])

    detailed_guidance = _build_detailed_author_guidance(
        attack_vectors=attack_vectors,
        vector_verdicts=vector_verdicts,
        vector_judgments=vector_judgments,
        vector_reports=vector_reports,
        syntheses=syntheses,
    )
    if detailed_guidance:
        lines.extend(["", "### Detailed Author Guidance"])
        lines.extend(detailed_guidance)

    for idx, vector in enumerate(attack_vectors, start=1):
        vector_id = str(vector.get("id", "V?"))
        claim = str(vector.get("claim", "")).strip() or "Unspecified claim."
        verdict = str(vector_verdicts.get(vector_id, "contested"))
        judgment = vector_judgments.get(vector_id, {})
        rationale = str(judgment.get("rationale", "")).strip()
        skeptic_latest = _latest_speaker_content(debate_history, speaker="skeptic", vector_id=vector_id)
        advocate_latest = _latest_speaker_content(debate_history, speaker="advocate", vector_id=vector_id)
        report = vector_reports.get(vector_id, {})
        joint = str(report.get("joint_conclusion", "")).strip()
        actions = [str(item).strip() for item in report.get("author_action_plan", []) if str(item).strip()]
        rewrite = str(syntheses.get(vector_id, "")).strip()

        lines.extend(
            [
                "",
                f"### Claim {idx}: {claim}",
                f"- Verdict: {verdict}",
                f"- Judge rationale: {rationale or 'No rationale recorded.'}",
                f"- Skeptic strongest point: {_compact_turn_text(skeptic_latest, max_chars=1200) or 'Not available.'}",
                f"- Advocate strongest point: {_compact_turn_text(advocate_latest, max_chars=1200) or 'Not available.'}",
            ]
        )
        if joint:
            lines.append(f"- Joint conclusion: {joint}")
        if actions:
            lines.append("- Action plan:")
            lines.extend(f"  - {item}" for item in actions[:3])
        if rewrite:
            lines.extend(["- Rewrite instruction:", rewrite])

        transcript = _render_full_vector_transcript(debate_history=debate_history, vector_id=vector_id)
        if transcript:
            lines.extend(["- Full debate transcript:", transcript])

    return "\n".join(lines).strip()


def _render_full_vector_transcript(*, debate_history: list[dict[str, Any]], vector_id: str) -> str:
    turns = [
        item
        for item in debate_history
        if str(item.get("vector_id", "")) == vector_id and str(item.get("speaker", "")) in {"skeptic", "advocate"}
    ]
    if not turns:
        return "No transcript available."
    lines: list[str] = []
    for item in turns:
        speaker = str(item.get("speaker", "")).strip().title() or "Reviewer"
        turn = int(item.get("turn", 0) or 0)
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"  - Turn {turn} {speaker}: {_compact_turn_text(content, max_chars=1400)}")
    return "\n".join(lines) if lines else "No transcript available."


def _build_detailed_author_guidance(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
) -> list[str]:
    guidance: list[str] = []
    for idx, vector in enumerate(attack_vectors, start=1):
        vector_id = str(vector.get("id", "V?"))
        claim = str(vector.get("claim", "")).strip() or "Unspecified claim."
        verdict = str(vector_verdicts.get(vector_id, "contested"))
        judgment = vector_judgments.get(vector_id, {})
        rationale = str(judgment.get("rationale", "")).strip() or "Evidence remains mixed for this claim."
        report = vector_reports.get(vector_id, {}) if isinstance(vector_reports, dict) else {}
        actions = [str(item).strip() for item in report.get("author_action_plan", []) if str(item).strip()]
        skeptic_conclusion = str(report.get("skeptic_conclusion", "")).strip()
        advocate_conclusion = str(report.get("advocate_conclusion", "")).strip()
        patch_instruction = _extract_patch_instruction(syntheses.get(vector_id, ""))

        what_to_change = patch_instruction or (
            actions[0]
            if actions
            else "Rewrite the claim sentence so the scope and evidence are explicitly aligned."
        )
        why_it_matters = (
            f"Judge outcome is `{verdict}` and the unresolved risk is: {rationale}"
            if verdict == "contested"
            else f"Judge outcome is `{verdict}`; this edit preserves strengths while reducing reviewer risk."
        )
        skeptic_point = skeptic_conclusion or "Current framing still appears broader than direct evidence support."
        advocate_point = advocate_conclusion or "Core contribution can stand when claim language is explicitly bounded."
        implementation_steps = actions[:3] if actions else [
            "Edit the claim sentence to name the exact evaluated setting.",
            "Add one explicit quantitative comparator next to the claim.",
            "Add one limitation sentence that bounds generalization beyond measured data.",
        ]

        block_lines = [
            f"- **Claim {idx} ({vector_id})**: {claim}",
            f"  - What to change: {what_to_change}",
            f"  - Why it matters: {why_it_matters}",
            f"  - Skeptic concern to resolve: {skeptic_point}",
            f"  - Advocate condition to preserve: {advocate_point}",
            "  - Implementation steps:",
        ]
        block_lines.extend(f"    1. {step}" for step in implementation_steps)
        guidance.append("\n".join(block_lines))
    return guidance


def _build_current_vector_report(
    *,
    active_vector: dict[str, Any],
    skeptic_position: str,
    advocate_position: str,
    debate_history: list[dict[str, Any]],
    documents: list[Document],
    existing_report: dict[str, Any],
) -> dict[str, Any]:
    skeptic = (skeptic_position or "").strip()
    advocate = (advocate_position or "").strip()
    if not skeptic or not advocate:
        return existing_report if isinstance(existing_report, dict) else {}

    fingerprint = _normalize_turn_text(skeptic) + "||" + _normalize_turn_text(advocate)
    if isinstance(existing_report, dict) and existing_report.get("fingerprint") == fingerprint:
        return existing_report

    fallback = _fallback_current_vector_report(
        active_vector=active_vector,
        skeptic_position=skeptic,
        advocate_position=advocate,
    )
    fallback["fingerprint"] = fingerprint
    if not text_service.available:
        return fallback

    vector_id = str(active_vector.get("id", "V?"))
    claim = str(active_vector.get("claim", "")).strip() or "active claim"
    try:
        response = text_service.generate(
            system_prompt=(
                "You are a debate analyst for a paper reviewer panel.\n"
                "Return JSON only with keys:\n"
                "agreements (array), disagreements (array), common_points (array),\n"
                "skeptic_conclusion (string), advocate_conclusion (string),\n"
                "joint_conclusion (string), author_action_plan (array).\n"
                "Keep output concrete and grounded in the two arguments."
            ),
            user_prompt=(
                f"Vector: {vector_id} | Claim: {claim}\n\n"
                f"Skeptic:\n{skeptic}\n\n"
                f"Advocate:\n{advocate}\n\n"
                "Recent debate transcript:\n"
                f"{_format_vector_history_compact(debate_history=debate_history, vector_id=vector_id, max_turns=6)}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=3)}"
            ),
            temperature=0.1,
            max_output_tokens=360,
        )
        payload = _try_parse_json_payload(response)
        if isinstance(payload, dict):
            report = {
                "agreements": [str(item).strip() for item in payload.get("agreements", []) if str(item).strip()],
                "disagreements": [str(item).strip() for item in payload.get("disagreements", []) if str(item).strip()],
                "common_points": [str(item).strip() for item in payload.get("common_points", []) if str(item).strip()],
                "skeptic_conclusion": str(payload.get("skeptic_conclusion", "")).strip(),
                "advocate_conclusion": str(payload.get("advocate_conclusion", "")).strip(),
                "joint_conclusion": str(payload.get("joint_conclusion", "")).strip(),
                "author_action_plan": [
                    str(item).strip() for item in payload.get("author_action_plan", []) if str(item).strip()
                ],
                "fingerprint": fingerprint,
            }
            if report["skeptic_conclusion"] and report["advocate_conclusion"] and report["joint_conclusion"]:
                if not report["agreements"]:
                    report["agreements"] = fallback["agreements"]
                if not report["disagreements"]:
                    report["disagreements"] = fallback["disagreements"]
                if not report["common_points"]:
                    report["common_points"] = fallback["common_points"]
                if not report["author_action_plan"]:
                    report["author_action_plan"] = fallback["author_action_plan"]
                return report
    except Exception:
        pass
    return fallback


def _fallback_current_vector_report(
    *,
    active_vector: dict[str, Any],
    skeptic_position: str,
    advocate_position: str,
) -> dict[str, Any]:
    claim = str(active_vector.get("claim", "")).strip() or "the active claim"
    return {
        "agreements": [
            "Both reviewers agree the claim should be bounded to directly measured evidence.",
            "Both sides agree clearer wording improves credibility.",
        ],
        "disagreements": [
            "Skeptic argues current evidence is insufficient for the full claim strength.",
            "Advocate argues the claim is defendable once scope is explicit.",
        ],
        "common_points": [
            "Evidence exists for feasibility.",
            "Claim wording should match measured scope.",
        ],
        "skeptic_conclusion": f"The paper overstates {claim.lower()} without enough direct support.",
        "advocate_conclusion": f"The paper can keep {claim.lower()} if scope is narrowed to measured settings.",
        "joint_conclusion": "Promising contribution, but strongest claims need tighter evidence-aligned framing.",
        "author_action_plan": [
            "Rewrite claim language to match reported setup and measurements.",
            "Add one explicit metric/baseline comparator in the same paragraph as the claim.",
            "State one explicit limitation boundary adjacent to the claim.",
        ],
    }


def _render_current_vector_report_markdown(report: dict[str, Any]) -> str:
    agreements = report.get("agreements", [])
    disagreements = report.get("disagreements", [])
    common_points = report.get("common_points", [])
    actions = report.get("author_action_plan", [])
    lines = ["Shared Ground:"]
    if isinstance(agreements, list) and agreements:
        lines.extend(f"- {str(item).strip()}" for item in agreements if str(item).strip())
    else:
        lines.append("- No explicit agreements yet.")
    lines.extend(["", "Core Disagreement:"])
    if isinstance(disagreements, list) and disagreements:
        lines.extend(f"- {str(item).strip()}" for item in disagreements if str(item).strip())
    else:
        lines.append("- No explicit disagreements yet.")
    lines.extend(["", "Alignment Signals:"])
    if isinstance(common_points, list) and common_points:
        lines.extend(f"- {str(item).strip()}" for item in common_points if str(item).strip())
    else:
        lines.append("- No common ground identified yet.")
    lines.extend(
        [
            "",
            "Joint Conclusion:",
            str(report.get("joint_conclusion", "Not available.")).strip(),
            "",
            "Action Plan:",
        ]
    )
    if isinstance(actions, list) and actions:
        lines.extend(f"- {str(item).strip()}" for item in actions if str(item).strip())
    else:
        lines.append("- No action plan available.")
    return "\n".join(lines).strip()


def _render_current_vector_report_brief(report: dict[str, Any]) -> str:
    agreements = [str(item).strip() for item in report.get("agreements", []) if str(item).strip()]
    disagreements = [str(item).strip() for item in report.get("disagreements", []) if str(item).strip()]
    common_points = [str(item).strip() for item in report.get("common_points", []) if str(item).strip()]
    actions = [str(item).strip() for item in report.get("author_action_plan", []) if str(item).strip()]

    lines: list[str] = []
    if agreements:
        lines.append(f"- Agreement: {agreements[0]}")
    if disagreements:
        lines.append(f"- Main disagreement: {disagreements[0]}")
    if common_points:
        lines.append(f"- Common ground: {common_points[0]}")
    joint = str(report.get("joint_conclusion", "")).strip()
    if joint:
        lines.append(f"- Joint conclusion: {joint}")
    if actions:
        lines.append("- Immediate action plan:")
        for item in actions[:2]:
            lines.append(f"  - {item}")
    if not lines:
        return "No intelligence summary available yet."
    return "\n".join(lines).strip()


def _render_round_events_compact(*, round_events: list[dict[str, Any]], vector_id: str) -> str:
    if not round_events:
        return "No new debate turns in this round."
    lines: list[str] = []
    for event in round_events:
        if str(event.get("vector_id", "")) != vector_id:
            continue
        speaker = str(event.get("speaker", "")).strip().lower()
        if speaker not in {"skeptic", "advocate"}:
            continue
        label = "Skeptic" if speaker == "skeptic" else "Advocate"
        content = _compact_turn_text(str(event.get("content", "")), max_chars=280)
        if not content:
            continue
        lines.append(f"- **{label}:** {content}")
    return "\n".join(lines) if lines else "No new debate turns in this round."


def _build_reviewer_final_report(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> dict[str, Any]:
    fallback = _fallback_reviewer_final_report(
        attack_vectors=attack_vectors,
        vector_verdicts=vector_verdicts,
        vector_judgments=vector_judgments,
        vector_reports=vector_reports,
        syntheses=syntheses,
    )
    if not text_service.available:
        return fallback
    try:
        response = text_service.generate(
            system_prompt=(
                "You are generating a final panel report for a two-reviewer paper debate.\n"
                "Return JSON only with keys:\n"
                "overview (string), agreements (array of strings), disagreements (array of strings),\n"
                "common_points (array of strings), skeptic_conclusion (string), advocate_conclusion (string),\n"
                "joint_conclusion (string), final_suggestions (array of strings), final_decision (string), confidence (number 0..1).\n"
                "Requirements: concise, evidence-grounded, no generic filler."
            ),
            user_prompt=(
                "Attack vectors with verdicts:\n"
                f"{json.dumps(vector_verdicts)}\n\n"
                "Judge rationales:\n"
                f"{json.dumps({k: v.get('rationale', '') for k, v in vector_judgments.items()})}\n\n"
                "Per-vector intelligence reports:\n"
                f"{json.dumps(vector_reports)}\n\n"
                "Rewrite cards:\n"
                f"{json.dumps(syntheses)}\n\n"
                "Debate transcript excerpt:\n"
                f"{_format_panel_history_compact(debate_history=debate_history, max_turns=14)}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=4)}"
            ),
            temperature=0.1,
            max_output_tokens=520,
        )
        payload = _try_parse_json_payload(response)
        if isinstance(payload, dict):
            report = {
                "overview": str(payload.get("overview", "")).strip(),
                "agreements": [str(item).strip() for item in payload.get("agreements", []) if str(item).strip()],
                "disagreements": [str(item).strip() for item in payload.get("disagreements", []) if str(item).strip()],
                "common_points": [str(item).strip() for item in payload.get("common_points", []) if str(item).strip()],
                "skeptic_conclusion": str(payload.get("skeptic_conclusion", "")).strip(),
                "advocate_conclusion": str(payload.get("advocate_conclusion", "")).strip(),
                "joint_conclusion": str(payload.get("joint_conclusion", "")).strip(),
                "final_suggestions": [
                    str(item).strip() for item in payload.get("final_suggestions", []) if str(item).strip()
                ],
                "final_decision": str(payload.get("final_decision", "")).strip(),
                "confidence": float(payload.get("confidence", 0.55)),
            }
            report["confidence"] = max(0.0, min(1.0, report["confidence"]))
            if report["overview"] and report["final_decision"]:
                if not report["agreements"]:
                    report["agreements"] = fallback["agreements"]
                if not report["disagreements"]:
                    report["disagreements"] = fallback["disagreements"]
                if not report["common_points"]:
                    report["common_points"] = fallback["common_points"]
                if not report["skeptic_conclusion"]:
                    report["skeptic_conclusion"] = fallback["skeptic_conclusion"]
                if not report["advocate_conclusion"]:
                    report["advocate_conclusion"] = fallback["advocate_conclusion"]
                if not report["joint_conclusion"]:
                    report["joint_conclusion"] = fallback["joint_conclusion"]
                if not report["final_suggestions"]:
                    report["final_suggestions"] = fallback["final_suggestions"]
                return report
    except Exception:
        pass
    return fallback


def _fallback_reviewer_final_report(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
) -> dict[str, Any]:
    skeptic_wins = sum(1 for verdict in vector_verdicts.values() if verdict == "skeptic_prevailed")
    advocate_wins = sum(1 for verdict in vector_verdicts.values() if verdict == "advocate_prevailed")
    contested = sum(1 for verdict in vector_verdicts.values() if verdict == "contested")
    total = max(1, len(vector_verdicts))

    agreements: list[str] = []
    disagreements: list[str] = []
    common_points: list[str] = []
    suggestions: list[str] = []
    for vector in attack_vectors:
        vector_id = str(vector.get("id", ""))
        claim = str(vector.get("claim", "")).strip()
        verdict = str(vector_verdicts.get(vector_id, "contested"))
        rationale = str(vector_judgments.get(vector_id, {}).get("rationale", "")).strip()
        if verdict == "advocate_prevailed":
            line = f"{vector_id}: reviewers converged that this claim is sufficiently supported."
            if rationale:
                line = f"{line} {rationale}".strip()
            agreements.append(line)
        elif verdict == "skeptic_prevailed":
            line = f"{vector_id}: skeptic challenge held; support was weaker than framing."
            if rationale:
                line = f"{line} {rationale}".strip()
            disagreements.append(line)
        else:
            line = f"{vector_id}: remained contested and needs clearer evidence framing."
            if rationale:
                line = f"{line} {rationale}".strip()
            disagreements.append(line)
        patch = _extract_patch_instruction(syntheses.get(vector_id, ""))
        if patch:
            suggestions.append(patch)
        elif claim:
            suggestions.append(f"Tighten claim wording for {vector_id} and add one concrete metric/baseline reference.")

    if not agreements:
        agreements.append("Both sides agreed that clearer scope boundaries improve claim credibility.")
    if not disagreements:
        disagreements.append("No major unresolved disagreements were recorded.")
    if vector_reports:
        for report in vector_reports.values():
            points = report.get("common_points", [])
            if not isinstance(points, list):
                continue
            for point in points:
                text = str(point).strip()
                if text and text not in common_points:
                    common_points.append(text)
                    if len(common_points) >= 5:
                        break
            if len(common_points) >= 5:
                break
    if not common_points:
        common_points.append("The contribution is promising, but strongest claims need tighter evidence alignment.")
    if not suggestions:
        suggestions.append("Convert each debated claim into one scoped statement tied to a measurable comparator.")

    if skeptic_wins > advocate_wins:
        decision = "Weak Reject until major claim-overreach and evidence gaps are revised."
    elif advocate_wins > skeptic_wins and skeptic_wins == 0:
        decision = "Weak Accept with targeted clarity edits."
    else:
        decision = "Borderline: promising contribution, but contested claims need tighter evidence framing."

    return {
        "overview": (
            f"Panel completed {total} claim trials: skeptic_prevailed={skeptic_wins}, "
            f"advocate_prevailed={advocate_wins}, contested={contested}."
        ),
        "agreements": agreements[:4],
        "disagreements": disagreements[:5],
        "common_points": common_points[:5],
        "skeptic_conclusion": "Skeptic view: major risks are overclaiming and insufficient direct support for broader statements.",
        "advocate_conclusion": "Advocate view: core contribution stands when claims are explicitly scoped to measured evidence.",
        "joint_conclusion": "Joint panel view: strong potential, but claim wording and evidence linking need tightening before acceptance.",
        "final_suggestions": suggestions[:6],
        "final_decision": decision,
        "confidence": max(0.35, min(0.9, 0.5 + ((advocate_wins - skeptic_wins) * 0.08))),
    }


def _extract_patch_instruction(card: str) -> str:
    text = (card or "").strip()
    if not text:
        return ""
    match = re.search(r"Patch Instruction:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _render_reviewer_final_report_markdown(report: dict[str, Any]) -> str:
    agreements = report.get("agreements", [])
    disagreements = report.get("disagreements", [])
    suggestions = report.get("final_suggestions", [])
    confidence = float(report.get("confidence", 0.0))
    lines = [
        "## Final Debate Report",
        str(report.get("overview", "Final panel summary is ready.")).strip(),
        "",
        "### Agreements",
    ]
    if isinstance(agreements, list) and agreements:
        lines.extend(f"- {str(item).strip()}" for item in agreements if str(item).strip())
    else:
        lines.append("- No explicit agreements captured.")
    lines.extend(["", "### Major Disagreements"])
    if isinstance(disagreements, list) and disagreements:
        lines.extend(f"- {str(item).strip()}" for item in disagreements if str(item).strip())
    else:
        lines.append("- No major disagreements captured.")
    lines.extend(["", "### Final Suggestions"])
    if isinstance(suggestions, list) and suggestions:
        lines.extend(f"- {str(item).strip()}" for item in suggestions if str(item).strip())
    else:
        lines.append("- No final suggestions captured.")
    lines.extend(
        [
            "",
            "### Final Decision",
            str(report.get("final_decision", "Decision not available.")).strip(),
            f"Confidence: {confidence:.2f}",
        ]
    )
    return "\n".join(lines).strip()


def _format_panel_history_compact(*, debate_history: list[dict[str, Any]], max_turns: int) -> str:
    if not debate_history:
        return "No prior turns."
    trimmed = debate_history[-max_turns:]
    lines: list[str] = []
    for item in trimmed:
        speaker = str(item.get("speaker", "unknown")).upper()
        vector_id = str(item.get("vector_id", "")).strip()
        prefix = f"[{vector_id}] " if vector_id else ""
        content = _compact_turn_text(str(item.get("content", "")), max_chars=220)
        if not content:
            continue
        lines.append(f"{speaker}: {prefix}{content}")
    return "\n".join(lines) if lines else "No prior turns."


def _human_next_move(*, next_speaker: str, vector_id: str) -> str:
    if next_speaker == "skeptic":
        return "The skeptic will challenge the weakest unresolved evidence next."
    if next_speaker == "advocate":
        return "The advocate will respond to the latest criticism next."
    if next_speaker == "synthesise":
        return "This claim is ready for a final rewrite recommendation."
    return "Your input is needed to break the tie or redirect the debate."


def _try_parse_json_payload(text: str) -> Any:
    cleaned = _strip_markdown_fence(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    object_match = re.search(r"\{[\s\S]*\}", cleaned)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            return None
    return None


def _active_vector_claim(state: GraphState) -> str:
    active_id = str(state.get("active_vector_id", "")).strip()
    if not active_id:
        return ""
    for item in state.get("attack_vectors", []) or []:
        if str(item.get("id", "")).strip() != active_id:
            continue
        return str(item.get("claim", "")).strip()
    return ""


def _should_use_global_retrieval(*, query: str, paper_ids: list[str]) -> bool:
    if paper_ids:
        return True
    lower = (query or "").lower()
    if ".pdf" in lower:
        return True
    if _looks_paper_specific_query(query):
        return True
    explicit_markers = (
        "uploaded paper",
        "uploaded papers",
        "in your papers",
        "in this paper",
        "according to the paper",
        "from the paper",
    )
    return any(marker in lower for marker in explicit_markers)


def _is_context_relevant_to_query(query: str, documents: list[Document]) -> bool:
    if not documents:
        return False
    person_query = _is_global_person_query(query)
    recommendation_query = _is_global_recommendation_query(query)
    if not _looks_paper_specific_query(query) and not person_query and not recommendation_query:
        return False
    query_terms = set(_tokenize_for_overlap(query))
    if not query_terms:
        return False
    if person_query and _has_author_metadata_in_docs(documents):
        return True

    best_overlap = 0.0
    for document in documents:
        overlap = _overlap_score(document.page_content or "", query_terms)
        if overlap > best_overlap:
            best_overlap = overlap

    threshold = 0.14
    if person_query:
        threshold = 0.06
    elif recommendation_query:
        threshold = 0.10
    return best_overlap >= threshold


def _looks_paper_specific_query(query: str) -> bool:
    lower = (query or "").lower()
    markers = (
        "paper",
        "author",
        "authors",
        "this work",
        "this study",
        "uploaded",
        "document",
        "in the paper",
        "according to",
        "approach",
        "method",
        "methodology",
        "results",
        "benchmark",
        "dataset",
        "participant",
        "players",
        "precision",
        "recall",
        "easyocr",
        "ocr",
        "valorant",
        "section",
        "table",
        "figure",
        "game was this done on",
    )
    return any(marker in lower for marker in markers)


def _is_global_recommendation_query(query: str) -> bool:
    lower = (query or "").lower()
    markers = (
        "related papers",
        "similar papers",
        "recommend papers",
        "more papers",
        "paper recommendations",
        "literature",
        "survey",
        "state of the art",
        "sota",
    )
    return any(marker in lower for marker in markers)


def _is_global_person_query(query: str) -> bool:
    lower = (query or "").lower()
    return (
        "who is" in lower
        or "tell me about" in lower
        or ("author" in lower and "paper" not in lower)
    )


def _has_author_metadata_in_docs(documents: list[Document]) -> bool:
    for document in documents[:8]:
        lower = (document.page_content or "").lower()
        if _looks_author_metadata_text(lower):
            return True
    return False


def _looks_author_metadata_text(lower_text: str) -> bool:
    markers = (
        "corresponding author",
        "the authors are",
        "authors are",
        "e-mail",
        "email",
        "affiliation",
    )
    return any(marker in (lower_text or "") for marker in markers)


def _strip_inline_reference_markers(text: str) -> str:
    cleaned = re.sub(r"\s*\[[0-9]+\]", "", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _tokenize_for_overlap(text: str) -> list[str]:
    normalized = (text or "")
    normalized = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", normalized)
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return re.findall(r"[a-zA-Z0-9_]+", normalized)


def _normalize_for_phrase_match(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _query_phrases(query: str) -> list[str]:
    tokens = _tokenize_for_overlap(query)
    if len(tokens) < 2:
        return []
    ignored = {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "many",
        "much",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "and",
        "or",
        "with",
    }
    phrases: list[str] = []
    seen: set[str] = set()
    max_n = min(4, len(tokens))
    for n in range(max_n, 1, -1):
        for idx in range(0, len(tokens) - n + 1):
            window = tokens[idx : idx + n]
            if all(token in ignored for token in window):
                continue
            phrase = " ".join(window).strip()
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= 16:
                return phrases
    return phrases


def _anchor_terms_for_query(query: str) -> set[str]:
    lower = (query or "").lower()
    anchors: set[str] = set()
    if "mixture of experts" in lower or re.search(r"\bmoe\b", lower):
        anchors.update({"mixture", "experts", "moe"})
    if "transformer" in lower:
        anchors.add("transformer")
    if re.search(r"\bhead\b|\bheads\b", lower):
        anchors.update({"head", "heads"})
    if "easyocr" in lower or "ocr" in lower:
        anchors.update({"ocr", "easyocr"})
    if "precision" in lower and "recall" in lower:
        anchors.update({"precision", "recall"})
    if _is_math_intent_query(query):
        anchors.update({"equation", "objective", "loss", "formulation"})
    if not anchors:
        return anchors
    return {token for token in anchors if token not in {"the", "a", "an", "and"}}


def _insufficient_local_grounding(*, query: str, documents: list[Document]) -> bool:
    if not documents:
        return True
    if _is_math_intent_query(query):
        # Math/derivation asks often use varied terminology; avoid over-pruning these.
        return False
    anchor_terms = _anchor_terms_for_query(query)
    if not anchor_terms:
        return False
    best_overlap = 0.0
    for document in documents[:8]:
        text = (document.page_content or "").lower()
        best_overlap = max(best_overlap, _overlap_score(text, anchor_terms))
    return best_overlap < 0.34


def _phrase_overlap_score(text: str, phrases: list[str]) -> float:
    if not text or not phrases:
        return 0.0
    padded = f" {text} "
    score = 0.0
    for phrase in phrases:
        words = phrase.split()
        if len(words) < 2:
            continue
        if f" {phrase} " not in padded:
            continue
        if len(words) >= 4:
            score += 0.55
        elif len(words) == 3:
            score += 0.38
        else:
            score += 0.22
    return min(1.0, score)


def _overlap_score(text: str, terms: set[str]) -> float:
    if not terms:
        return 0.0
    text_terms = set(_tokenize_for_overlap(text))
    if not text_terms:
        return 0.0
    overlap = len(text_terms & terms)
    return overlap / max(1, len(terms))


def _mode_keywords(mode: Mode) -> list[str]:
    if mode == Mode.REVIEWER:
        return [
            "contribution",
            "novelty",
            "benchmark",
            "ablation",
            "limitation",
            "table",
            "experiment",
            "reproducibility",
            "hyperparameter",
            "baseline",
            "statistical significance",
        ]
    if mode == Mode.COMPARATOR:
        return ["method", "dataset", "baseline", "result", "accuracy", "f1", "comparison"]
    if mode == Mode.WRITER:
        return ["style", "tone", "structure", "clarity"]
    if mode == Mode.GLOBAL:
        return ["background", "context", "evidence"]
    return ["evidence", "paper", "claim"]


def _looks_like_high_signal_section(text: str) -> bool:
    header_window = (text or "")[:280].lower()
    markers = (
        "abstract",
        "introduction",
        "method",
        "experiment",
        "results",
        "conclusion",
        "limitation",
        "discussion",
    )
    return any(marker in header_window for marker in markers)


def _looks_math_dense_chunk(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    if any(marker in lower for marker in ("equation", "formulated as", "objective", "loss", "where")):
        return True
    if re.search(r"\bl(?:pretrain|aux|1|2)\b", lower):
        return True
    if re.search(r"\btop[-\s]?k\b", lower):
        return True
    if re.search(r"\b\w+\s*=\s*[^=]+", lower):
        return True
    return False


def _low_signal_penalty(text: str, *, allow_numeric_dense: bool = False) -> float:
    lower = (text or "").lower()
    penalty = 0.0
    boilerplate_markers = (
        "permission to make digital or hard copies",
        "copyrights for components",
        "manuscript submitted to acm",
        "publication rights licensed to acm",
        "request permissions from",
        "doi:",
        "references",
    )
    if any(marker in lower for marker in boilerplate_markers):
        penalty += 0.35
    tokens = re.findall(r"[a-zA-Z0-9_]+", lower)
    if tokens:
        numeric_ratio = sum(1 for token in tokens if token.isdigit()) / len(tokens)
        if numeric_ratio > 0.38 and not (allow_numeric_dense and _looks_math_dense_chunk(text)):
            penalty += 0.25
    return penalty


def _select_balanced_docs(
    *,
    mode: Mode,
    scored_docs: list[tuple[Document, float]],
    limit: int,
) -> list[Document]:
    if not scored_docs:
        return []

    if mode == Mode.REVIEWER:
        return _select_reviewer_coverage_docs(scored_docs=scored_docs, limit=limit)

    if mode == Mode.LOCAL:
        return _select_local_coverage_docs(scored_docs=scored_docs, limit=limit)

    if mode != Mode.COMPARATOR:
        return [document for document, _ in scored_docs[:limit]]

    filtered_scored_docs = [
        (document, score)
        for document, score in scored_docs
        if not _looks_like_reference_snippet(document.page_content or "")
        and not _looks_like_non_argument_snippet(document.page_content or "")
    ]
    if filtered_scored_docs:
        scored_docs = filtered_scored_docs

    by_paper: dict[str, list[tuple[Document, float]]] = {}
    for document, score in scored_docs:
        paper_id = str((document.metadata or {}).get("paper_id", ""))
        by_paper.setdefault(paper_id, []).append((document, score))

    selected: list[Document] = []
    for paper_docs in by_paper.values():
        if paper_docs:
            selected.append(paper_docs[0][0])
        if len(selected) >= limit:
            return selected[:limit]

    for document, _ in scored_docs:
        if document in selected:
            continue
        selected.append(document)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _select_local_coverage_docs(
    *,
    scored_docs: list[tuple[Document, float]],
    limit: int,
) -> list[Document]:
    if not scored_docs:
        return []

    top_score = scored_docs[0][1]
    best_by_paper: dict[str, tuple[Document, float]] = {}
    for document, score in scored_docs:
        paper_id = str((document.metadata or {}).get("paper_id", "")).strip() or "unknown"
        if paper_id not in best_by_paper:
            best_by_paper[paper_id] = (document, score)

    if len(best_by_paper) <= 1:
        return [document for document, _ in scored_docs[:limit]]

    selected: list[Document] = []
    seen: set[str] = set()
    for document, score in sorted(best_by_paper.values(), key=lambda pair: pair[1], reverse=True):
        if score < (top_score - 0.22):
            continue
        identity = _document_identity(document)
        if identity in seen:
            continue
        selected.append(document)
        seen.add(identity)
        if len(selected) >= limit:
            return selected[:limit]

    for document, _ in scored_docs:
        identity = _document_identity(document)
        if identity in seen:
            continue
        selected.append(document)
        seen.add(identity)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _select_reviewer_coverage_docs(
    *,
    scored_docs: list[tuple[Document, float]],
    limit: int,
) -> list[Document]:
    if not scored_docs:
        return []

    buckets: dict[str, list[Document]] = {}
    for document, _ in scored_docs:
        bucket = _reviewer_bucket(document.page_content or "")
        buckets.setdefault(bucket, []).append(document)

    priority = [
        "summary",
        "method",
        "experiments",
        "ablation",
        "limitations",
        "reproducibility",
        "other",
    ]
    selected: list[Document] = []
    for bucket in priority:
        candidates = buckets.get(bucket, [])
        if not candidates:
            continue
        selected.append(candidates[0])
        if len(selected) >= limit:
            return selected[:limit]

    for document, _ in scored_docs:
        if document in selected:
            continue
        selected.append(document)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _reviewer_bucket(text: str) -> str:
    lower = (text or "")[:450].lower()
    if any(marker in lower for marker in ("abstract", "introduction", "contribution", "novelty", "motivation")):
        return "summary"
    if any(marker in lower for marker in ("method", "approach", "architecture", "model", "algorithm", "training")):
        return "method"
    if any(marker in lower for marker in ("experiment", "results", "dataset", "benchmark", "baseline", "table")):
        return "experiments"
    if any(marker in lower for marker in ("ablation", "sensitivity", "error analysis", "robustness")):
        return "ablation"
    if any(marker in lower for marker in ("limitation", "failure", "threat", "bias", "ethic")):
        return "limitations"
    if any(marker in lower for marker in ("implementation", "hyperparameter", "seed", "compute", "reproducibility", "code")):
        return "reproducibility"
    return "other"


def _has_inline_citations(text: str) -> bool:
    return bool(re.search(r"\[[0-9]+\]", text or ""))


def _select_citations_for_answer(
    *,
    answer: str,
    citations: list[dict[str, Any]],
    mode: Mode,
) -> list[dict[str, Any]]:
    if not citations:
        return []

    referenced_numbers = sorted({int(match) for match in re.findall(r"\[([0-9]+)\]", answer or "") if match.isdigit()})
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for number in referenced_numbers:
        index = number - 1
        if index < 0 or index >= len(citations):
            continue
        citation = citations[index]
        key = f"{citation.get('paper_id','')}|{citation.get('chunk_id','')}|{citation.get('page','')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(citation)

    if selected:
        if mode == Mode.GLOBAL:
            return selected[: min(2, len(selected))]
        return selected

    if mode == Mode.REVIEWER:
        fallback_limit = 1
    elif mode == Mode.LOCAL:
        fallback_limit = 1
    elif mode == Mode.COMPARATOR:
        fallback_limit = min(4, len(citations))
    else:
        fallback_limit = 0
    return citations[: min(fallback_limit, len(citations))]


def _citation_identity(citation: dict[str, Any]) -> str:
    return f"{citation.get('paper_id','')}|{citation.get('chunk_id','')}|{citation.get('page','')}"


def _reindex_answer_citations(
    *,
    answer: str,
    raw_citations: list[dict[str, Any]],
    selected_citations: list[dict[str, Any]],
) -> str:
    text = answer or ""
    if not _has_inline_citations(text):
        return text
    if not selected_citations:
        return _strip_inline_reference_markers(text)

    key_to_new_index: dict[str, int] = {}
    for index, citation in enumerate(selected_citations, start=1):
        key_to_new_index[_citation_identity(citation)] = index

    old_to_new: dict[int, int] = {}
    for index, citation in enumerate(raw_citations or [], start=1):
        new_index = key_to_new_index.get(_citation_identity(citation))
        if new_index is not None:
            old_to_new[index] = new_index

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if not token.isdigit():
            return ""
        old_index = int(token)
        new_index = old_to_new.get(old_index)
        if new_index is None:
            return ""
        return f"[{new_index}]"

    reindexed = re.sub(r"\[([0-9]+)\]", _replace, text)
    reindexed = re.sub(r"(\[[0-9]+\])(?:\s*\1)+", r"\1", reindexed)
    reindexed = re.sub(r"\s+([,.;:])", r"\1", reindexed)
    reindexed = re.sub(r"[ \t]{2,}", " ", reindexed)
    reindexed = re.sub(r"\n{3,}", "\n\n", reindexed)
    return reindexed.strip()


def _temperature_for_mode(mode: Mode) -> float:
    return {
        Mode.LOCAL: 0.0,
        Mode.GLOBAL: 0.35,
        Mode.WRITER: 0.4,
        Mode.REVIEWER: 0.1,
        Mode.COMPARATOR: 0.1,
    }[mode]


def extract_reviewer_state(state: GraphState) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in REVIEWER_STATE_KEYS:
        if key in state:
            snapshot[key] = deepcopy(state[key])
    return snapshot


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("prepare_mode", _prepare_mode_step)
    graph.add_node("retrieve", _retrieve_step)
    graph.add_node("rerank", _rerank_step)
    graph.add_node("draft_answer", _draft_answer_step)
    graph.add_node("validate_answer", _validate_answer_step)
    graph.add_node("finalize_answer", _finalize_answer_step)
    graph.add_edge(START, "prepare_mode")
    graph.add_edge("prepare_mode", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "draft_answer")
    graph.add_edge("draft_answer", "validate_answer")
    graph.add_edge("validate_answer", "finalize_answer")
    graph.add_edge("finalize_answer", END)
    return graph.compile()
