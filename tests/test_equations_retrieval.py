"""
tests/test_equations_retrieval.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pytest suite for gateway/equations/retrieval_math.py.
Verifies mathematical properties: bounds, monotonicity, degenerate-input
behaviour, and specific numerical identities.
"""

import math
import pytest

from gateway.equations.retrieval_math import (
    mmr,
    mmr_score,
    min_max_normalize,
    z_score_normalize,
    sigmoid_normalize,
    normalize_scores,
    cosine_to_similarity,
    rrf_score,
    rrf,
    weighted_reciprocal_rank,
    pointwise_mutual_information,
    mi_gate,
    novelty_score,
    curiosity_score,
    greedy_diverse_select,
    chunk_cache_key,
    embedding_similarity_from_embeddings,
    relevance_gate,
    qf_similarity,
    hybrid_relevance,
    topk_stable,
    select_within_token_budget,
)


# ---- Eq R1 -- MMR ----

class TestMMR:
    def test_empty(self):
        assert mmr([], [], top_k=5) == []

    def test_lambda_1_returns_pure_relevance(self):
        """With lambda=1.0, selection order matches pure relevance rank."""
        sim_doc = [0.1, 0.9, 0.5]
        sim_docs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = mmr(sim_doc, sim_docs, lambda_relevance=1.0, top_k=3)
        assert result == [1, 2, 0]

    def test_lambda_0_maximizes_diversity(self):
        """With lambda=0.0, penalty dominates: first pick is highest sim_doc,
        then picks minimize max similarity to already-selected."""
        sim_doc = [0.9, 0.8, 0.7]
        sim_docs = [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
        result = mmr(sim_doc, sim_docs, lambda_relevance=0.0, top_k=3)
        assert result[0] == 0
        # doc 2 has lowest penalty to doc 0 (0.1) so picked second;
        # doc 1 is most similar to doc 0 so picked last.
        assert result == [0, 2, 1]

    def test_returns_subset(self):
        sim_doc = [0.1, 0.2, 0.3, 0.4, 0.5]
        sim_docs = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
        result = mmr(sim_doc, sim_docs, top_k=3)
        assert len(result) == 3
        assert all(0 <= idx < 5 for idx in result)
        assert len(set(result)) == 3


class TestMMRScore:
    def test_no_selection_penalty(self):
        sim = [0.5, 0.8, 0.3]
        rel = [0.6, 0.9, 0.4]
        score = mmr_score(sim, rel, 0.7, selected=[], candidate=1)
        assert score == pytest.approx(0.7 * 0.9)

    def test_with_selection(self):
        # pairwise_sim is the candidate's row: sim to each doc by index.
        # candidate=1, selected=[0] → penalty = pairwise_sim[0] = 0.5
        sim = [0.5, 1.0, 0.3]
        rel = [0.8, 0.9, 0.6]
        score = mmr_score(sim, rel, 0.7, selected=[0], candidate=1)
        expected = 0.7 * 0.9 - 0.3 * 0.5
        assert score == pytest.approx(expected)


# ---- Eq R2 -- Score normalization ----

class TestMinMaxNormalize:
    def test_maps_min_to_0_max_to_1(self):
        result = min_max_normalize([1.0, 2.0, 3.0, 4.0])
        assert result[0] == pytest.approx(0.0)
        assert result[3] == pytest.approx(1.0)

    def test_all_equal(self):
        result = min_max_normalize([5.0, 5.0, 5.0])
        assert all(v == pytest.approx(0.5) for v in result)

    def test_empty(self):
        assert min_max_normalize([]) == []

    def test_single(self):
        result = min_max_normalize([7.0])
        assert result == [0.5]


class TestZScoreNormalize:
    def test_empty(self):
        assert z_score_normalize([]) == []

    def test_all_equal(self):
        result = z_score_normalize([3.0, 3.0, 3.0])
        assert all(v == pytest.approx(0.0) for v in result)

    def test_mean_zero(self):
        result = z_score_normalize([1.0, 2.0, 3.0])
        mean = sum(result) / len(result)
        assert mean == pytest.approx(0.0, abs=1e-10)


class TestSigmoidNormalize:
    def test_output_in_unit_interval(self):
        for val in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            out = sigmoid_normalize(val)
            assert 0.0 < out < 1.0

    def test_monotonic(self):
        vals = [-5.0, -1.0, 0.0, 1.0, 5.0]
        sigs = [sigmoid_normalize(v) for v in vals]
        for i in range(len(sigs) - 1):
            assert sigs[i] < sigs[i + 1]


class TestNormalizeScores:
    def test_minmax_dispatcher(self):
        result = normalize_scores([0.0, 5.0, 10.0], method="minmax")
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_empty(self):
        assert normalize_scores([]) == []
        assert normalize_scores([], method="zscore") == []
        assert normalize_scores([], method="sigmoid") == []


class TestCosineToSimilarity:
    def test_maps_neg1_to_0(self):
        assert cosine_to_similarity(-1.0) == pytest.approx(0.0)

    def test_maps_0_to_05(self):
        assert cosine_to_similarity(0.0) == pytest.approx(0.5)

    def test_maps_1_to_1(self):
        assert cosine_to_similarity(1.0) == pytest.approx(1.0)

    def test_nan_guard(self):
        assert cosine_to_similarity(float("nan")) == 0.5

    def test_inf_guard(self):
        assert cosine_to_similarity(float("inf")) == 0.5


# ---- Eq R3 -- RRF ----

class TestRRF:
    def test_rrf_score_basic(self):
        assert rrf_score(1, 60) == pytest.approx(1.0 / 61)
        assert rrf_score(5, 60) == pytest.approx(1.0 / 65)

    def test_empty(self):
        assert rrf([]) == []

    def test_doc_in_many_lists_ranks_higher(self):
        lists = [[1, 2, 3], [1, 4, 5], [1, 6, 7]]
        result = rrf(lists)
        doc_ids = [d for d, _ in result]
        assert doc_ids[0] == 1

    def test_fusion_score_positive(self):
        lists = [[10, 20], [20, 10]]
        result = rrf(lists)
        for _, sc in result:
            assert sc > 0


class TestWeightedReciprocalRank:
    def test_empty(self):
        assert weighted_reciprocal_rank([], []) == []

    def test_higher_weight_boosts(self):
        lists_a = [[1, 2]]
        lists_b = [[2, 1]]
        result_a = weighted_reciprocal_rank(lists_a, [2.0])
        result_b = weighted_reciprocal_rank(lists_b, [1.0])
        score_a = dict(result_a)
        score_b = dict(result_b)
        assert score_a[1] > score_b[1]


# ---- Eq R5 -- MI gate ----

class TestMIGate:
    def test_high_signal_low_noise(self):
        assert mi_gate(signal_entropy=0.1, noise_entropy=0.9, threshold=0.5) is True

    def test_low_signal_high_noise(self):
        assert mi_gate(signal_entropy=0.9, noise_entropy=0.5, threshold=0.5) is False

    def test_pointwise_mi(self):
        assert pointwise_mutual_information(0.5, 0.5, 0.5) == pytest.approx(1.0)
        assert pointwise_mutual_information(0.25, 0.5, 0.5) == pytest.approx(0.0)
        assert pointwise_mutual_information(0.0, 0.5, 0.5) == pytest.approx(0.0)


# ---- Eq R6 -- Curiosity / novelty ----

class TestCuriosity:
    def test_novelty_no_known(self):
        assert novelty_score([1, 0, 0], []) == 1.0

    def test_novelty_identical(self):
        assert novelty_score([1, 0, 0], [[1, 0, 0]]) == pytest.approx(0.0)

    def test_curiosity_balanced(self):
        score = curiosity_score(0.8, 0.6, alpha=0.5, beta=0.5)
        assert score == pytest.approx(0.7)

    def test_curiosity_zero_weights(self):
        assert curiosity_score(0.5, 0.5, alpha=0.0, beta=0.0) == 0.0


# ---- Eq R7 -- Greedy diverse select ----

class TestGreedyDiverseSelect:
    def test_empty(self):
        assert greedy_diverse_select([], [], 5) == []

    def test_single(self):
        result = greedy_diverse_select([0.5], [[1.0]], 1)
        assert result == [0]

    def test_lambda_1_is_pure_topk(self):
        scores = [0.1, 0.9, 0.5]
        sim_matrix = [[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]]
        result = greedy_diverse_select(scores, sim_matrix, top_k=3, lambda_d=1.0)
        assert result == [1, 2, 0]


# ---- Eq R8 -- Cache key ----

class TestCacheKey:
    def test_deterministic(self):
        k1 = chunk_cache_key("Hello World")
        k2 = chunk_cache_key("Hello World")
        assert k1 == k2

    def test_case_insensitive(self):
        k1 = chunk_cache_key("foo bar")
        k2 = chunk_cache_key("FOO BAR")
        assert k1 == k2

    def test_empty(self):
        key = chunk_cache_key("")
        assert isinstance(key, str) and len(key) == 64


class TestEmbeddingSimilarity:
    def test_identical(self):
        assert embedding_similarity_from_embeddings([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert embedding_similarity_from_embeddings([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_empty(self):
        assert embedding_similarity_from_embeddings([], [1, 0]) == 0.0


# ---- Eq R9 -- Relevance gate ----

class TestRelevanceGate:
    def test_above_threshold(self):
        assert relevance_gate(0.8, threshold=0.5) is True

    def test_below_threshold(self):
        assert relevance_gate(0.3, threshold=0.5) is False

    def test_strict(self):
        assert relevance_gate(0.5, threshold=0.5, strict=True) is False
        assert relevance_gate(0.5001, threshold=0.5, strict=True) is True

    def test_nan(self):
        assert relevance_gate(float("nan"), threshold=0.5) is False


# ---- Eq R10 -- Query reformulation ----

class TestQFSimilarity:
    def test_identical(self):
        score = qf_similarity("machine learning", "machine learning")
        assert score > 0.9

    def test_different(self):
        score = qf_similarity("cats", "quantum physics")
        assert score < 0.5

    def test_empty(self):
        assert qf_similarity("", "anything") == 0.0


class TestHybridRelevance:
    def test_balanced(self):
        assert hybrid_relevance(0.6, 0.4, alpha=0.5) == pytest.approx(0.5)

    def test_all_weight_semantic(self):
        assert hybrid_relevance(0.0, 1.0, alpha=1.0) == pytest.approx(1.0)

    def test_all_weight_lexical(self):
        assert hybrid_relevance(1.0, 0.0, alpha=0.0) == pytest.approx(1.0)


# ---- Eq R11 -- Top-k stable ----

class TestTopkStable:
    def test_empty(self):
        assert topk_stable([], 5) == []

    def test_deterministic(self):
        scores = [0.5, 0.5, 0.5]
        r1 = topk_stable(scores, 2)
        r2 = topk_stable(scores, 2)
        assert r1 == r2
        assert r1 == [0, 1]

    def test_respects_scores(self):
        scores = [0.1, 0.9, 0.5]
        result = topk_stable(scores, 2)
        assert result == [1, 2]


# ---- Eq R12 -- Budgeted selection ----

class TestSelectWithinTokenBudget:
    def test_empty(self):
        assert select_within_token_budget([], [], 100) == []

    def test_zero_budget(self):
        assert select_within_token_budget([0.9, 0.8], [10, 10], 0) == []

    def test_never_exceeds_budget(self):
        scores = [0.9, 0.8, 0.7, 0.6]
        tokens = [50, 50, 50, 50]
        budget = 100
        selected = select_within_token_budget(scores, tokens, budget)
        total = sum(tokens[i] for i in selected)
        assert total <= budget

    def test_picks_highest_score_per_token(self):
        scores = [0.5, 0.9]
        tokens = [100, 10]
        budget = 20
        selected = select_within_token_budget(scores, tokens, budget)
        assert 1 in selected
        assert 0 not in selected
