import math
from typing import Iterable, List

import networkx as nx
import numpy as np

from sim_hdt_siot.entities import Agent, DiscoveryCandidate, RankedSource


def _is_accessible(candidate: DiscoveryCandidate, profile_access_policy: str) -> bool:
    if profile_access_policy == "public":
        return True
    if profile_access_policy == "self":
        return False
    if profile_access_policy == "relational":
        return candidate.hops <= 2 and candidate.access_score >= 0.55
    if profile_access_policy == "trusted":
        return candidate.path_trust >= 0.68 and candidate.access_score >= 0.60
    if profile_access_policy == "restricted":
        return candidate.path_trust >= 0.80 and candidate.access_score >= 0.75 and candidate.hops == 1
    return False


def rank_sources(
    graph: nx.Graph,
    ego_id: str,
    candidates: Iterable[DiscoveryCandidate],
    agents: dict[str, Agent],
    policy: str,
    trigger_threshold: float,
    max_selected_sources: int | None = None,
    latency_penalty_scale: float = 1.0,
    privacy_penalty_scale: float = 1.0,
    broaden_threshold_factor: float = 1.0,
    privacy_penalty_strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> List[RankedSource]:
    del ego_id
    ranked: List[RankedSource] = []
    for candidate in candidates:
        profile = agents[candidate.source_id].profile
        if policy != "opportunistic_all" and not _is_accessible(candidate, profile.access_policy):
            continue

        path_latency = _path_latency_contribution(graph, candidate.path)
        latency_proxy = profile.base_latency + path_latency + (0.18 * max(0, candidate.graph_distance - 1))
        freshness = math.exp(-math.log(2.0) * latency_proxy / max(profile.freshness_halflife, 0.1))
        latency_penalty = 1.0 / (1.0 + (latency_penalty_scale * latency_proxy))
        distance_penalty = 1.0 / (1.0 + (0.20 * max(0, candidate.graph_distance - 1)))
        relevance = profile.base_relevance or (profile.base_accuracy * profile.availability)
        trust = candidate.path_trust * profile.trust
        score = (
            relevance
            * trust
            * freshness
            * latency_penalty
            * distance_penalty
            * candidate.relation_utility
            * candidate.access_score
        )
        if policy == "siot_aware_trust_privacy":
            score /= 1.0 + (privacy_penalty_strength * privacy_penalty_scale * profile.privacy_cost)
        elif policy == "siot_aware":
            relaxed_privacy_weight = max(0.0, 1.0 - privacy_penalty_scale)
            if relaxed_privacy_weight > 0.0:
                score /= 1.0 + (0.25 * relaxed_privacy_weight * profile.privacy_cost)
        ranked.append(
            RankedSource(
                source_id=candidate.source_id,
                score=score,
                relevance=relevance,
                trust=trust,
                freshness=freshness,
                latency=latency_proxy,
                access_score=candidate.access_score,
                privacy_cost=profile.privacy_cost,
                hops=candidate.graph_distance,
                relation_utility=candidate.relation_utility,
            )
        )

    if policy == "opportunistic_all":
        return sorted(ranked, key=lambda item: (item.hops, item.source_id))
    if policy == "budgeted_opportunistic_k":
        ranked.sort(key=lambda item: (item.hops, item.source_id))
        return ranked[: max_selected_sources or 0]
    if policy == "random_k":
        if not ranked or not max_selected_sources:
            return []
        deterministic = sorted(ranked, key=lambda item: item.source_id)
        generator = rng or np.random.default_rng(7)
        count = min(max_selected_sources, len(deterministic))
        indices = sorted(generator.choice(len(deterministic), size=count, replace=False).tolist())
        return [deterministic[index] for index in indices]

    effective_threshold = trigger_threshold * broaden_threshold_factor
    filtered = [item for item in ranked if item.score >= effective_threshold]
    if not filtered and ranked:
        filtered = [max(ranked, key=lambda item: item.score)]
    filtered.sort(key=lambda item: item.score, reverse=True)
    if max_selected_sources is not None:
        filtered = filtered[:max_selected_sources]
    return filtered


def _path_latency_contribution(graph: nx.Graph, path: tuple[str, ...]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for source, target in zip(path[:-1], path[1:]):
        total += float(graph.edges[source, target].get("latency_contribution", 0.0))
    return total
