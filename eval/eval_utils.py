"""Fonctions utilitaires pour l'evaluation du pipeline RAG LogiStore."""

import time
import numpy as np
import pandas as pd
from openai import OpenAI
from ragas.metrics import ContextPrecision, ContextRecall, ContextEntityRecall
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory

# --- Constantes ---
SCORE_COLS = ["context_precision", "context_recall", "context_entity_recall"]
METHODS = ["dense", "sparse", "hybrid"]
FILTER_MODES = ["no_filter", "type_filter", "priority_filter", "queue_filter"]


# ============================================================
# COLLECTE DE RESULTATS
# ============================================================

def collect_results(search_fn, test_cases, method="hybrid", limit=3, **kwargs):
    """Collecte les resultats RAG pour une liste de test cases.

    Args:
        search_fn: fonction search(query, method, limit, **kwargs) -> list[dict]
        test_cases: list de dicts avec 'query' et 'reference'
        method: methode de recherche
        limit: nombre de resultats (K)
        **kwargs: arguments supplementaires (collection, etc.)
    """
    dataset = []
    for tc in test_cases:
        results = search_fn(tc["query"], method=method, limit=limit, **kwargs)
        contexts = [r["body"] for r in results] if results else []
        response = results[0]["answer"] if results else ""
        dataset.append({
            "user_input": tc["query"],
            "retrieved_contexts": contexts,
            "response": response,
            "reference": tc["reference"],
        })
    return dataset


def collect_with_case_filter(search_fn, test_cases, method, limit, filter_field, **kwargs):
    """Collecte avec un filtre dynamique par test case.

    Chaque cas utilise sa propre valeur de filtre (ex: son type, sa queue).
    """
    dataset = []
    for tc in test_cases:
        filter_value = tc.get(filter_field, None)
        filter_key = "type_" if filter_field == "type" else filter_field
        case_kwargs = {**kwargs, filter_key: filter_value}
        results = search_fn(tc["query"], method=method, limit=limit, **case_kwargs)
        contexts = [r["body"] for r in results] if results else []
        response = results[0]["answer"] if results else ""
        dataset.append({
            "user_input": tc["query"],
            "retrieved_contexts": contexts,
            "response": response,
            "reference": tc["reference"],
        })
    return dataset


def collect_all_configs(search_fn, test_cases, methods=None, limit=3, **kwargs):
    """Collecte toutes les combinaisons methode x filtre.

    Returns:
        dict[(method, filter_mode)] -> list[dict]
    """
    methods = methods or METHODS
    results = {}
    for method in methods:
        for filter_mode in FILTER_MODES:
            print(f"  {method} / {filter_mode} ...")
            if filter_mode == "no_filter":
                results[(method, filter_mode)] = collect_results(
                    search_fn, test_cases, method=method, limit=limit, **kwargs
                )
            else:
                field = filter_mode.replace("_filter", "")
                results[(method, filter_mode)] = collect_with_case_filter(
                    search_fn, test_cases, method=method, limit=limit,
                    filter_field=field, **kwargs
                )
    return results


# ============================================================
# METRIQUES INSTANTANEES (pas de LLM)
# ============================================================

def hit_at_1(data):
    """Hit@1 : le top-1 resultat est-il exactement le bon ticket ?"""
    hits = sum(
        1 for s in data
        if s["response"].strip() == s["reference"].strip()
    )
    return hits / len(data) if data else 0


def hit_at_k(data):
    """Hit@K : le bon ticket est-il parmi les K contextes recuperes ?"""
    hits = 0
    for s in data:
        if s["reference"].strip() in [c.strip() for c in s["retrieved_contexts"]]:
            hits += 1
        elif s["response"].strip() == s["reference"].strip():
            hits += 1
    return hits / len(data) if data else 0


def mrr(data):
    """Mean Reciprocal Rank (base sur le top-1 exact match)."""
    rr = [
        1.0 if s["response"].strip() == s["reference"].strip() else 0.0
        for s in data
    ]
    return np.mean(rr) if rr else 0


def compute_hitk_table(all_results):
    """Calcule Hit@1 et MRR pour toutes les configs.

    Args:
        all_results: dict[(method, filter_mode)] -> list[dict]

    Returns:
        DataFrame avec colonnes [method, filter, hit_at_1, hit_at_k, mrr]
    """
    rows = []
    for (method, filter_mode), data in all_results.items():
        rows.append({
            "method": method,
            "filter": filter_mode,
            "hit_at_1": hit_at_1(data),
            "hit_at_k": hit_at_k(data),
            "mrr": mrr(data),
        })
    return pd.DataFrame(rows).sort_values("hit_at_1", ascending=False).reset_index(drop=True)


# ============================================================
# RAGAS EVALUATION (avec LLM)
# ============================================================

def make_openrouter_client():
    """Cree un client OpenAI pointe vers OpenRouter."""
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY manquante dans .env")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "ragas-eval"},
    )


def make_judge(model_id, client=None):
    """Cree un LLM juge RAGAS via OpenRouter.

    Args:
        model_id: ex. 'google/gemini-2.0-flash-001'
        client: OpenAI client (si None, cree automatiquement)
    """
    if client is None:
        client = make_openrouter_client()
    return llm_factory(model_id, client=client)


def make_metrics(llm):
    """Cree les 3 metriques RAGAS avec un LLM juge donne."""
    return [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        ContextEntityRecall(llm=llm),
    ]


def run_ragas(data, llm, batch_size=1):
    """Lance l'evaluation RAGAS sur un dataset.

    Args:
        data: list[dict] avec user_input, retrieved_contexts, response, reference
        llm: instance LLM RAGAS
        batch_size: taille de batch

    Returns:
        DataFrame avec les scores par sample
    """
    eval_dataset = EvaluationDataset.from_list(data)
    results = evaluate(
        dataset=eval_dataset,
        metrics=make_metrics(llm),
        raise_exceptions=False,
        show_progress=True,
        batch_size=batch_size,
    )
    return results.to_pandas()


def run_ragas_multi_judge(data, judges_dict, pause=10):
    """Evalue un dataset avec plusieurs juges LLM.

    Args:
        data: list[dict] de samples
        judges_dict: dict[name -> llm_instance]
        pause: secondes entre chaque juge (rate limit)

    Returns:
        dict[name -> DataFrame]
    """
    results = {}
    for i, (name, llm) in enumerate(judges_dict.items()):
        print(f"\n[{i+1}/{len(judges_dict)}] {name} ...")
        df = run_ragas(data, llm)
        results[name] = df
        means = {c: f"{df[c].mean():.4f}" for c in SCORE_COLS if c in df.columns}
        print(f"  Scores : {means}")
        if i < len(judges_dict) - 1:
            print(f"  Pause {pause}s ...")
            time.sleep(pause)
    return results


def ragas_summary(eval_results):
    """Tableau recapitulatif des scores moyens.

    Args:
        eval_results: dict[label -> DataFrame]

    Returns:
        DataFrame avec colonnes [label, context_precision, ..., mean_score]
    """
    rows = []
    for label, df in eval_results.items():
        row = {"label": label}
        for col in SCORE_COLS:
            row[col] = df[col].mean() if col in df.columns else float("nan")
        row["mean_score"] = np.nanmean([row[c] for c in SCORE_COLS])
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# CORRELATIONS INTER-JUGES
# ============================================================

def judge_correlations(eval_results):
    """Calcule les correlations Pearson/Spearman entre chaque paire de juges.

    Args:
        eval_results: dict[judge_name -> DataFrame]

    Returns:
        DataFrame avec colonnes [pair, metric, pearson, pearson_p, spearman]
    """
    from scipy.stats import pearsonr, spearmanr

    names = list(eval_results.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            df_a = eval_results[names[i]]
            df_b = eval_results[names[j]]
            for col in SCORE_COLS:
                sa = df_a[col].dropna()
                sb = df_b[col].dropna()
                common = sa.index.intersection(sb.index)
                if len(common) > 5:
                    r_p, p_p = pearsonr(sa[common], sb[common])
                    r_s, _ = spearmanr(sa[common], sb[common])
                    rows.append({
                        "pair": f"{names[i]} vs {names[j]}",
                        "metric": col,
                        "pearson": r_p,
                        "pearson_p": p_p,
                        "spearman": r_s,
                    })
    return pd.DataFrame(rows)
