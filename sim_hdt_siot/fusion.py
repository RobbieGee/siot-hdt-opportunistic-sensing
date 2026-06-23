from typing import Dict, Iterable, Tuple

from sim_hdt_siot.config import CONTEXT_VARIABLES
from sim_hdt_siot.entities import ContextState


def weighted_vote_fusion(weighted_observations: Iterable[Tuple[float, Dict[str, str]]]) -> tuple[ContextState, Dict[str, float]]:
    votes = {variable: {state: 0.0 for state in states} for variable, states in CONTEXT_VARIABLES.items()}
    total_weight = {variable: 0.0 for variable in CONTEXT_VARIABLES}

    for weight, observation in weighted_observations:
        for variable, value in observation.items():
            votes[variable][value] += weight
            total_weight[variable] += weight

    estimated = {}
    confidence = {}
    for variable, state_votes in votes.items():
        best_state = max(state_votes, key=state_votes.get)
        estimated[variable] = best_state
        confidence[variable] = 0.0 if total_weight[variable] == 0 else state_votes[best_state] / total_weight[variable]

    return ContextState(**estimated), confidence
