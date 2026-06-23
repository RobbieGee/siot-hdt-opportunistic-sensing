from typing import Dict, List

import networkx as nx

from sim_hdt_siot.entities import Agent, ContextState, RankedSource, SourceObservation
from sim_hdt_siot.fusion import weighted_vote_fusion


def estimate_context(
    ego_observation: SourceObservation,
    selected_sources: List[RankedSource],
    external_observations: Dict[str, SourceObservation],
    policy: str,
    graph: nx.Graph,
    agents: dict[str, Agent],
    ego_id: str,
) -> tuple[ContextState, Dict[str, float], Dict[str, float]]:
    del graph, agents, ego_id
    ego_weight = 1.0 if ego_observation.values else 0.35
    weighted_observations: List[tuple[float, Dict[str, str]]] = [(ego_weight, ego_observation.values)]
    total_latency = 0.0
    total_privacy = 0.0
    observed_sources = 0

    if policy != "ego_only":
        for ranked_source in selected_sources:
            observation = external_observations[ranked_source.source_id]
            total_latency += observation.latency
            total_privacy += ranked_source.privacy_cost
            if not observation.values:
                continue
            weight = ranked_source.relevance * ranked_source.trust * observation.freshness * ranked_source.access_score
            weight *= ranked_source.relation_utility / (1.0 + observation.latency)
            if policy == "opportunistic_all":
                weight = max(weight, 0.15)
            elif policy == "siot_aware_trust_privacy":
                weight /= 1.0 + ranked_source.privacy_cost
            weighted_observations.append((weight, observation.values))
            observed_sources += 1

    recruited_sources = len(selected_sources) if policy != "ego_only" else 0
    estimate, confidence = weighted_vote_fusion(weighted_observations)
    telemetry = {
        "selected_source_count": float(recruited_sources),
        "observed_external_source_count": float(observed_sources),
        "estimated_latency": 0.0 if recruited_sources == 0 else total_latency / recruited_sources,
        "privacy_exposure_cost": total_privacy,
    }
    return estimate, confidence, telemetry
