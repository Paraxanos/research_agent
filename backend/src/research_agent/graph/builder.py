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
            "For count questions, if the excerpt lists numbered entities (for example Expert 1 ... Expert 6), "
            "infer the minimum explicit count from the highest listed index and state it as 'at least N'. "
            "If evidence is missing, respond with exactly: "
            "'This information is not in your uploaded papers.' "
            "Then briefly name what is missing and offer Global mode."
        ),
        Mode.GLOBAL: (
            "You are an expert, conversational research assistant. "
            "Respond naturally like a normal high-quality LLM. "
            "Use uploaded papers when they are relevant to the user request, and cite those paper-grounded claims with [n]. "
            "If the request is general and not paper-specific, answer freely from your broader knowledge without forcing citations."
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
            "You are a research analyst comparing the selected papers. "
            "Structure responses around these five axes:\n"
            "  1. Core contribution and claimed novelty of each paper.\n"
            "  2. Shared benchmarks - list datasets both or all papers evaluate on "
            "and directly compare their reported numbers.\n"
            "  3. Methodological similarities and differences - "
            "architecture choices, training setup, evaluation protocol.\n"
            "  4. Relative strength - which paper is stronger and why, "
            "with specific evidence from the text.\n"
            "  5. Unique aspects - what each paper does that the others do not.\n"
            "Be explicit and direct. Use paper filenames to disambiguate. "
            "Avoid vague statements and always state the exact differences."
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

    per_paper_top_k = max(2, settings.retrieval_top_k // max(1, len(paper_ids)))
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
        if len(deduped) >= settings.retrieval_top_k:
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
    if mode == Mode.REVIEWER:
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
    seeds = [
        base,
        f"{base} core contribution novelty claims problem setup assumptions",
        f"{base} method architecture training objective algorithm implementation details",
        f"{base} datasets benchmarks baselines protocol metrics statistical significance",
        f"{base} ablation sensitivity analysis error analysis robustness",
        f"{base} limitations failure cases threats to validity biases",
        f"{base} reproducibility code data hyperparameters compute seeds",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in seeds:
        normalized = " ".join(item.split()).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
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
        if person_query and _looks_author_metadata_text(lower):
            section_boost += 0.75
        has_number_phrase = "number of experts" in lower
        has_numbered_experts = _has_numbered_expert_pattern(lower)
        has_any_number = bool(re.search(r"\b\d+\b", lower))
        if expert_count_query and has_numbered_experts:
            section_boost += 0.90
        elif expert_count_query and has_number_phrase:
            section_boost += 0.55

        quality_penalty = _low_signal_penalty(text)
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
                ),
                "citations": state.get("citations", []),
                "debug": debug,
            }
        debate_debug = dict(debate_payload.get("debug", {}))
        debate_debug["response_stage"] = "reviewer_debate"
        debate_payload["debug"] = debate_debug
        return debate_payload

    if mode == Mode.COMPARATOR and len(paper_ids) < 2:
        return {
            "draft_answer": "Comparator mode requires at least two selected papers.",
            "citations": [],
            "debug": {**debug, "response_stage": "validation"},
        }

    if mode == Mode.COMPARATOR and len(documents) < 2:
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

    max_output_tokens = 2000 if mode == Mode.REVIEWER else 1400
    try:
        draft_answer = text_service.generate(
            system_prompt=_system_prompt(state),
            user_prompt=_draft_user_prompt(
                mode=mode,
                message=state["message"],
                history=state.get("history", []),
                documents=prompt_documents,
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
        )
        debug["response_stage"] = "model_fallback"
        debug["model_fallback"] = True
        if retry_hint:
            debug["retry_hint"] = retry_hint
        debug["model_error"] = str(error)[:180]
    if not debug.get("model_fallback"):
        debug["response_stage"] = "draft_model"
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

    if mode == Mode.GLOBAL and debug.get("global_context_relevance") == "low":
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "global_low_context_bypassed"},
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

    if mode in {Mode.LOCAL, Mode.REVIEWER, Mode.COMPARATOR} and documents and not _has_inline_citations(answer):
        debug["citation_warning"] = "Answer lacked inline citations after validation."

    citations = _select_citations_for_answer(answer=answer, citations=raw_citations, mode=mode)
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


def _prefix_retrieval_fallback(answer: str, *, retry_hint: str) -> str:
    guidance = f" Try again in about {retry_hint}." if retry_hint else " Try again shortly."
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
        if documents:
            return _local_extractive_fallback(
                query=state.get("message", ""),
                documents=documents,
            )
        return (
            "I could not find enough grounded evidence in the uploaded papers for this request. "
            "Try a paper-specific question or upload a relevant source."
        )
    if mode == Mode.COMPARATOR:
        if not documents:
            return "I could not retrieve enough evidence from the selected papers."
        return (
            "Here are the strongest comparison excerpts from the selected papers:\n\n"
            f"{_format_context(documents, max_docs=min(3, len(documents)))}"
        )
    return (
        _local_extractive_fallback(
            query=state.get("message", ""),
            documents=documents,
        )
        if documents
        else "I could not find enough grounded evidence in the retrieved context."
    )


def _local_extractive_fallback(*, query: str, documents: list[Document]) -> str:
    query_terms = set(_tokenize_for_overlap(query))
    query_phrases = _query_phrases(query)
    lower_query = (query or "").lower()
    quantity_intent = any(marker in lower_query for marker in ("how many", "number", "count", "how much"))
    if quantity_intent and any(token.startswith("expert") for token in query_terms):
        quantity_snippet = _extract_quantity_snippet(documents=documents, keyword="expert")
        if quantity_snippet:
            expert_count = _infer_expert_count(quantity_snippet)
            if expert_count is not None:
                return (
                    "Based on the uploaded paper: "
                    f"at least {expert_count} experts are explicitly described. {quantity_snippet} [1]"
                )
            return f"Based on the uploaded paper: {quantity_snippet} [1]"
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


def _extract_quantity_snippet(*, documents: list[Document], keyword: str) -> str:
    for document in documents[:6]:
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
            if re.search(rf"\b\d+\s+{re.escape(keyword)}s?\b", lower):
                return snippet
            if re.search(rf"\b{re.escape(keyword)}s?\s*(?:is|are|were|was|=|:)?\s*\d+\b", lower):
                return snippet
            if keyword == "expert" and "expert" in lower:
                if re.search(r"\b\d+\b", lower):
                    return snippet
    return ""


def _infer_expert_count(snippet: str) -> int | None:
    text = (snippet or "").lower()
    if "expert" not in text:
        return None
    values = [int(value) for value in re.findall(r"\b\d+\b", text)]
    if not values:
        return None
    high = max(values)
    if high <= 0:
        return None
    return high


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
        "## Reviewer Arena\n"
        f"Active Vector: {active.get('id', 'V1')} - {active.get('claim', '')}\n"
        f"Quote Trigger: \"{quote}\"\n\n"
        "### Skeptic\n"
        "- Concern: evidence may be weaker than framing suggests [1].\n"
        "- Ask for a tighter quantitative comparison and clearer scope boundary [1].\n\n"
        "### Advocate\n"
        "- Defense: the paper does provide partial evidence for the claim [1].\n"
        "- Recommend narrowing claim language to what is directly demonstrated [1].\n\n"
        "### Action Card\n"
        "Target Section: contribution framing paragraph\n"
        "Rewrite Instruction: revise the contribution claim to include one concrete metric/baseline comparison and explicitly state the scope limits.\n"
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
        Mode.COMPARATOR: "Keep only comparisons directly supported by retrieved context blocks.",
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
                "- The user is asking for related-paper recommendations. "
                "Provide 8-12 concrete papers (title + year + one-line relevance). "
                "Use broader model knowledge and do not invent [n] citations.\n"
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
            "You are producing a response in Global mode.\n"
            "Requirements:\n"
            "- Answer naturally and directly.\n"
            "- You are not limited to retrieved paper context; use broader model knowledge when helpful.\n"
            "- Treat retrieved context as optional support, not a hard boundary.\n"
            "- If a claim is grounded in retrieved paper context, cite it with [n].\n"
            "- Never invent [n] citations for general-knowledge or conversational parts.\n"
            "- For general questions (for example definitions or everyday knowledge), answer directly and confidently.\n"
            "- Do not claim you are missing uploaded papers when a general answer is possible.\n"
            "- Keep paper citations minimal in Global mode (typically 1-2 strongest references).\n"
            "- If prior turns were paper-constrained, do not carry that constraint into this global response.\n"
            f"{query_directive}\n"
            "Conversation history:\n"
            f"{history_text}\n\n"
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
    vectors_remaining = [
        vector_id
        for vector_id in state.get("vectors_remaining", [])
        if vector_id in attack_vector_ids and vector_id not in syntheses
    ]
    if not vectors_remaining:
        vectors_remaining = [vector_id for vector_id in attack_vector_ids if vector_id not in syntheses]
    if not vectors_remaining:
        vectors_remaining = attack_vector_ids[:]

    if _user_requested_next_vector(message):
        current_active = str(state.get("active_vector_id", "")).strip()
        if current_active and current_active in vectors_remaining:
            vectors_remaining = [vector for vector in vectors_remaining if vector != current_active] + [current_active]

    explicit_vector = _extract_vector_selection(message, attack_vectors)
    if explicit_vector and explicit_vector in syntheses:
        syntheses.pop(explicit_vector, None)
        vector_verdicts.pop(explicit_vector, None)
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
            "next_speaker": state.get("next_speaker", "skeptic"),
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
    loops = max(1, settings.reviewer_turns_per_response)
    for _ in range(loops):
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
            verdict = _compute_vector_verdict(
                active_vector=active_vector,
                debate_history=debate_history,
                resolution=resolution,
            )
            vector_verdicts[active_vector_id] = verdict
            synthesis = _synthesise_vector(
                active_vector=active_vector,
                verdict=verdict,
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
    next_speaker = _route_reviewer_turn(
        history=debate_history,
        active_vector_id=active_vector_id,
        resolution=resolution,
        turn_count=turn_count,
        fallback=next_speaker,
    )

    if next_speaker == "synthesise":
        verdict = _compute_vector_verdict(
            active_vector=active_vector,
            debate_history=debate_history,
            resolution=resolution,
        )
        vector_verdicts[active_vector_id] = verdict
        synthesis = _synthesise_vector(
            active_vector=active_vector,
            verdict=verdict,
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
    answer = _render_reviewer_debate(
        attack_vectors=attack_vectors,
        active_vector=active_vector,
        vectors_remaining=vectors_remaining,
        syntheses=syntheses,
        vector_verdicts=vector_verdicts,
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
    debug["vector_verdicts"] = vector_verdicts
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
        return "user"
    if last_meta.get("addressed_to") in {"advocate", "skeptic"}:
        return str(last_meta.get("addressed_to"))

    if _speaker_conceded(_latest_speaker_content(history, speaker="skeptic", vector_id=active_vector_id), "skeptic"):
        return "advocate"
    if bool(last_meta.get("concession")):
        return "synthesise"

    if resolution in {"resolved", "deadlocked", "force_closed"}:
        return "synthesise"

    if _is_deadlock(history, active_vector_id):
        return "user"

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
    return skeptic_unchanged and advocate_unchanged


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
    persona = "Reviewer B (Skeptic)" if speaker == "skeptic" else "Reviewer A (Advocate)"
    mission = (
        "You are unconvinced. You require numerical or concrete evidence and you do not accept vague qualitative defenses."
        if speaker == "skeptic"
        else "You defend the paper charitably using strongest available evidence and explicit scope constraints."
    )
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
    prompt = (
        f"Active vector: {active_vector.get('id', 'V?')} | {active_vector.get('claim', '')}\n"
        f"Severity: {active_vector.get('severity', 'medium')} | Category: {active_vector.get('category', 'method')}\n"
        f"Quoted claim trigger: \"{active_vector.get('quote', '')}\"\n"
        f"Skeptic lead-in: {active_vector.get('skeptic_lead', '')}\n"
        f"User objective: {objective or 'General full-paper review'}\n\n"
        "Debate summary (compressed memory):\n"
        f"{debate_summary or 'No summary yet.'}\n\n"
        "Most recent turns:\n"
        f"{history_excerpt}\n\n"
        "Retrieved context:\n"
        f"{_format_context(turn_docs, max_docs=3)}\n\n"
        "Reply as this reviewer only. Use citations [n] when factual.\n"
        "Output format:\n"
        "Position: <single sentence stance>\n"
        "Argument: <2-4 concise bullets or short paragraphs>\n"
        "Routing footer JSON (single line):\n"
        'ROUTE_JSON: {"addressed_to":"advocate|skeptic|user|none","concession":true|false,"confidence":0.0}'
    )
    try:
        response = text_service.generate(
            system_prompt=(
                f"You are {persona}. {mission}\n"
                "You are in a live two-reviewer debate. Directly respond to the opposing argument and avoid monologue."
            ),
            user_prompt=prompt,
            temperature=0.1,
            max_output_tokens=360,
        )
        content = (response or "").strip()
    except Exception:
        content = _fallback_reviewer_turn(
            speaker=speaker,
            active_vector=active_vector,
            documents=turn_docs,
        )
    route_meta = _extract_route_meta(content, default_target="advocate" if speaker == "skeptic" else "skeptic")
    return _clean_reviewer_turn_text(_strip_route_json_footer(content)), route_meta


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
        scored.append((document, overlap + (0.4 * rank_prior)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [document for document, _ in scored[:3]]


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
            "Position: Evidence for this claim is still not tight enough.\n"
            "Argument:\n"
            f"- Triggered claim: \"{quote}\"\n"
            f"- Evidence check: \"{primary}\" [1].\n"
            "- Concern: this supports feasibility, but not a strong novelty delta versus prior work [1].\n"
            "- Request: add one quantitative comparator and explicitly bound the scope claim [1].\n"
            'ROUTE_JSON: {"addressed_to":"advocate","concession":false,"confidence":0.54}'
        )
    return (
        "Position: The claim has partial support but should be framed with clearer limits.\n"
        "Argument:\n"
        f"- Triggered claim: \"{quote}\"\n"
        f"- Supporting evidence: \"{primary}\" [1].\n"
        f"- Scope caveat already present: \"{secondary}\" [1].\n"
        "- Revision path: keep the contribution framing but tighten novelty wording to the measured scope [1].\n"
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


def _clean_reviewer_turn_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\s*#+\s*Reviewer\s+[AB]\s*\([^)]+\)\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Reviewer\s+[AB]\s*\([^)]+\)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _refresh_debate_summary(
    *,
    debate_summary: str,
    active_vector: dict[str, Any],
    debate_history: list[dict[str, Any]],
) -> str:
    if not text_service.available:
        return debate_summary
    vector_id = str(active_vector.get("id", "V?"))
    excerpt = _format_vector_history(debate_history=debate_history, vector_id=vector_id, max_turns=6)
    try:
        response = text_service.generate(
            system_prompt=(
                "Compress debate context into exactly 3 short sentences capturing disagreement, strongest evidence, and unresolved item."
            ),
            user_prompt=(
                f"Existing summary:\n{debate_summary or 'None'}\n\n"
                f"New turns for {vector_id}:\n{excerpt}\n\n"
                "Return exactly 3 sentences."
            ),
            temperature=0.0,
            max_output_tokens=120,
        )
        return (response or debate_summary or "").strip()
    except Exception:
        return debate_summary


def _compute_vector_verdict(
    *,
    active_vector: dict[str, Any],
    debate_history: list[dict[str, Any]],
    resolution: str,
) -> str:
    if resolution in {"deadlocked", "force_closed"}:
        return "contested"
    if not text_service.available:
        return "contested"
    vector_id = str(active_vector.get("id", "V?"))
    try:
        response = text_service.generate(
            system_prompt=(
                "Return JSON only with key verdict. Allowed verdict values: "
                "skeptic_prevailed, advocate_prevailed, contested."
            ),
            user_prompt=(
                f"Vector {vector_id}: {active_vector.get('claim', '')}\n"
                f"Debate excerpt:\n{_format_vector_history(debate_history=debate_history, vector_id=vector_id, max_turns=8)}"
            ),
            temperature=0.0,
            max_output_tokens=80,
        )
    except Exception:
        return "contested"
    payload = _try_parse_json_payload(response)
    if isinstance(payload, dict):
        verdict = str(payload.get("verdict", "contested")).strip().lower()
        if verdict in {"skeptic_prevailed", "advocate_prevailed", "contested"}:
            return verdict
    return "contested"


def _synthesise_vector(
    *,
    active_vector: dict[str, Any],
    verdict: str,
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> str:
    vector_id = str(active_vector.get("id", "V?"))
    if not text_service.available:
        return (
            f"### Action Card {vector_id}\n"
            f"Verdict: {verdict}\n"
            "Rewrite Instruction: Rewrite the claim paragraph to align scope with demonstrated evidence and add one concrete quantitative comparison."
        )
    try:
        response = text_service.generate(
            system_prompt=(
                "You generate one concrete rewrite instruction after a debate.\n"
                "Do not summarize the debate. Produce exactly one actionable edit request tied to paper text."
            ),
            user_prompt=(
                f"Vector: {vector_id} | {active_vector.get('claim', '')}\n"
                f"Verdict: {verdict}\n"
                f"Quoted trigger sentence: {active_vector.get('quote', '')}\n"
                "Debate transcript:\n"
                f"{_format_vector_history(debate_history=debate_history, vector_id=vector_id, max_turns=10)}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 8))}\n\n"
                "Output format:\n"
                "### Action Card\n"
                "Target Section: <section name or approximate location>\n"
                "Rewrite Instruction: <one concrete instruction sentence with metric/baseline/clarification target>\n"
                "Why: <one sentence>"
            ),
            temperature=0.1,
            max_output_tokens=220,
        )
        return (response or "").strip()
    except Exception:
        return (
            "### Action Card\n"
            "Target Section: contribution/claim paragraph\n"
            "Rewrite Instruction: replace broad claim wording with a bounded statement and add one quantitative comparison against the closest baseline on the reported metric.\n"
            "Why: this preserves the contribution while reducing overclaim risk."
        )


def _render_reviewer_debate(
    *,
    attack_vectors: list[dict[str, Any]],
    active_vector: dict[str, Any],
    vectors_remaining: list[str],
    syntheses: dict[str, str],
    vector_verdicts: dict[str, str],
    round_events: list[dict[str, Any]],
    debate_history: list[dict[str, Any]],
    debate_summary: str,
    resolution: str,
    turn_count: int,
    next_speaker: str,
) -> str:
    progress = "".join("*" if idx < turn_count else "o" for idx in range(settings.reviewer_max_turns))
    vector_lines: list[str] = []
    for vector in attack_vectors:
        vector_id = str(vector.get("id", "V?"))
        status = "resolved" if vector_id in syntheses else ("active" if vector_id == str(active_vector.get("id", "")) else "queued")
        verdict = vector_verdicts.get(vector_id, "pending")
        vector_lines.append(
            f"- {vector_id} [{status}] verdict={verdict} ({vector.get('severity', 'medium')}/{vector.get('category', 'method')}): {vector.get('claim', '')}\n"
            f"  Quote: \"{vector.get('quote', '')}\""
        )

    if not round_events:
        round_text = "- No new reviewer turns generated in this round."
    else:
        fragments: list[str] = []
        for event in round_events:
            speaker = str(event.get("speaker", "")).lower()
            if speaker == "synthesise":
                fragments.append(f"### Synthesis ({event.get('vector_id', '')})\n{event.get('content', '')}")
            else:
                heading = "Skeptic" if speaker == "skeptic" else "Advocate" if speaker == "advocate" else speaker.title()
                meta = event.get("meta", {}) if isinstance(event.get("meta"), dict) else {}
                tag = (
                    f"Tag: addressed_to={meta.get('addressed_to', 'none')} | "
                    f"conceded={str(bool(meta.get('concession', False))).lower()} | "
                    f"confidence={float(meta.get('confidence', 0.0)):.2f}"
                )
                fragments.append(f"### {heading}\n{event.get('content', '')}\n\n{tag}")
        round_text = "\n\n".join(fragments)

    active_vector_id = str(active_vector.get("id", "V?"))
    active_synthesis = syntheses.get(active_vector_id, "")
    action_cards = [
        f"### {vector_id}\n{content}"
        for vector_id, content in syntheses.items()
    ]
    latest_excerpt = _format_vector_history_compact(
        debate_history=debate_history,
        vector_id=active_vector_id,
        max_turns=4,
    )
    next_move = _human_next_move(next_speaker=next_speaker, vector_id=active_vector_id)

    blocks = [
        "## Reviewer Arena",
        f"Active Vector: {active_vector_id} - {active_vector.get('claim', '')}",
        f"Resolution: {resolution} | Debate Turns: {turn_count}/{settings.reviewer_max_turns}",
        f"Progress: {progress}",
        "",
        "## Attack Vectors",
        "\n".join(vector_lines),
        "",
        "## Compressed Context",
        debate_summary or "No compressed summary yet.",
        "",
        "## This Round",
        round_text,
        "",
        "## Latest Exchange",
        latest_excerpt,
    ]
    if active_synthesis:
        blocks.extend(["", "## Active Vector Synthesis", active_synthesis])
    if action_cards:
        blocks.extend(["", "## Action Cards", "\n\n".join(action_cards)])
    blocks.extend(
        [
            "",
            "## Next Move",
            next_move,
            f"Vectors remaining: {len(vectors_remaining)}",
            "You can type `skeptic: ...`, `advocate: ...`, `vector 2`, or `next`.",
        ]
    )
    return "\n".join(blocks).strip()


def _human_next_move(*, next_speaker: str, vector_id: str) -> str:
    if next_speaker == "skeptic":
        return f"Skeptic will press the strongest unresolved point on {vector_id}."
    if next_speaker == "advocate":
        return f"Advocate will rebut the latest challenge on {vector_id}."
    if next_speaker == "synthesise":
        return f"The current debate on {vector_id} is ready for synthesis."
    return "User intervention requested to break a deadlock or steer the debate."


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
    if not anchors:
        return anchors
    return {token for token in anchors if token not in {"the", "a", "an", "and"}}


def _insufficient_local_grounding(*, query: str, documents: list[Document]) -> bool:
    if not documents:
        return True
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


def _low_signal_penalty(text: str) -> float:
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
        if numeric_ratio > 0.38:
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
        fallback_limit = 2
    else:
        fallback_limit = 0
    return citations[: min(fallback_limit, len(citations))]


def _temperature_for_mode(mode: Mode) -> float:
    return {
        Mode.LOCAL: 0.0,
        Mode.GLOBAL: 0.2,
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
