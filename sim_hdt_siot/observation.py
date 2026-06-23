import math
from typing import Dict

import numpy as np

from sim_hdt_siot.config import CONTEXT_VARIABLES
from sim_hdt_siot.entities import ContextState, DiscoveryCandidate, SourceObservation, SourceProfile


def sample_observation(
    source_id: str,
    truth: ContextState,
    timeline: list[ContextState],
    step: int,
    profile: SourceProfile,
    rng: np.random.Generator,
) -> SourceObservation:
    age = 0
    if step > 0 and rng.random() < profile.stale_probability:
        age = int(min(step, max(1, round(rng.uniform(1.0, profile.base_latency + 2.5)))))
    reference_truth = timeline[max(0, step - age)]
    freshness = math.exp(-math.log(2.0) * age / max(profile.freshness_halflife, 0.1))
    latency = profile.base_latency + (0.35 * age)
    observation: Dict[str, str] = {}
    for variable, truth_value in reference_truth.as_dict().items():
        if profile.sensing_capabilities and variable not in profile.sensing_capabilities:
            continue
        states = CONTEXT_VARIABLES[variable]
        availability = _state_adjusted_availability(profile, variable, truth, profile.variable_availability.get(variable, profile.availability))
        if rng.random() > availability:
            continue

        accuracy = _state_adjusted_accuracy(profile, variable, truth, profile.variable_accuracy.get(variable, profile.base_accuracy))
        effective_accuracy = min(1.0, max(0.0, accuracy * (0.55 + 0.45 * freshness)))
        if rng.random() <= effective_accuracy and rng.random() > profile.misleading_probability:
            observation[variable] = truth_value
            continue

        preferred_wrong = profile.confusion_bias.get(variable, {}).get(truth_value)
        alternatives = [state for state in states if state != truth_value]
        if preferred_wrong is not None and preferred_wrong in alternatives and rng.random() < 0.75:
            observation[variable] = preferred_wrong
        else:
            observation[variable] = str(rng.choice(alternatives))

    return SourceObservation(
        source_id=source_id,
        values=observation,
        age=age,
        freshness=freshness,
        latency=latency,
        trust=profile.trust,
        access_score=profile.access_level,
        privacy_cost=profile.privacy_cost,
        hops=0,
        relation_utility=1.0,
    )


def apply_discovery_metadata(observation: SourceObservation, candidate: DiscoveryCandidate) -> SourceObservation:
    return SourceObservation(
        source_id=observation.source_id,
        values=observation.values,
        age=observation.age,
        freshness=observation.freshness,
        latency=observation.latency + (0.18 * max(0, candidate.graph_distance - 1)),
        trust=observation.trust * candidate.path_trust,
        access_score=min(observation.access_score, candidate.access_score) / (1.0 + (0.05 * max(0, candidate.graph_distance - 1))),
        privacy_cost=observation.privacy_cost,
        hops=candidate.graph_distance,
        relation_utility=candidate.relation_utility,
    )


def _state_adjusted_availability(profile: SourceProfile, variable: str, truth: ContextState, base_availability: float) -> float:
    availability = base_availability
    if profile.node_type == "ego_device":
        if truth.resource_state == "degraded":
            availability *= 0.78
        elif truth.resource_state == "sparse":
            availability *= 0.60
        if truth.env_load == "high" and variable in {"place", "activity"}:
            availability *= 0.88
    elif profile.degraded:
        availability *= 0.88
    return float(min(1.0, max(0.0, availability)))


def _state_adjusted_accuracy(profile: SourceProfile, variable: str, truth: ContextState, base_accuracy: float) -> float:
    accuracy = base_accuracy
    if truth.env_load == "high":
        accuracy *= 0.90 if variable in {"place", "activity"} else 0.96
    if profile.node_type == "ego_device":
        if truth.resource_state == "degraded":
            accuracy *= 0.86
        elif truth.resource_state == "sparse":
            accuracy *= 0.76
    elif profile.degraded:
        accuracy *= 0.90
    return float(min(1.0, max(0.0, accuracy)))
