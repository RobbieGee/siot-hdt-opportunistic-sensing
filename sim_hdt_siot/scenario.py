from typing import Dict, List, Sequence, Tuple

import numpy as np

from sim_hdt_siot.config import CONTEXT_VARIABLES
from sim_hdt_siot.entities import Agent, ContextState, Relationship, ScenarioDefinition, ScenarioSettings, SourceProfile


def _weighted_choice(rng: np.random.Generator, weighted_items: Sequence[Tuple[str, float]]) -> str:
    values = [item for item, _ in weighted_items]
    weights = np.asarray([weight for _, weight in weighted_items], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(values, p=weights))


PLACE_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    "home": [("home", 0.62), ("transit", 0.20), ("office", 0.08), ("public_place", 0.08), ("meeting_room", 0.02)],
    "transit": [("transit", 0.34), ("office", 0.28), ("home", 0.20), ("public_place", 0.14), ("meeting_room", 0.04)],
    "office": [("office", 0.55), ("meeting_room", 0.20), ("transit", 0.12), ("public_place", 0.08), ("home", 0.05)],
    "meeting_room": [("meeting_room", 0.42), ("office", 0.38), ("public_place", 0.08), ("transit", 0.08), ("home", 0.04)],
    "public_place": [("public_place", 0.48), ("transit", 0.25), ("home", 0.12), ("office", 0.10), ("meeting_room", 0.05)],
}

ACTIVITY_BY_PLACE: Dict[str, List[Tuple[str, float]]] = {
    "home": [("resting", 0.58), ("focused_work", 0.16), ("waiting", 0.14), ("interacting", 0.10), ("commuting", 0.02)],
    "office": [("focused_work", 0.58), ("interacting", 0.20), ("waiting", 0.10), ("resting", 0.08), ("commuting", 0.04)],
    "transit": [("commuting", 0.70), ("waiting", 0.18), ("interacting", 0.06), ("resting", 0.04), ("focused_work", 0.02)],
    "meeting_room": [("interacting", 0.62), ("focused_work", 0.20), ("waiting", 0.12), ("resting", 0.04), ("commuting", 0.02)],
    "public_place": [("waiting", 0.40), ("interacting", 0.24), ("commuting", 0.18), ("resting", 0.12), ("focused_work", 0.06)],
}


def _sample_activity(place: str, previous_activity: str | None, rng: np.random.Generator, scenario_name: str) -> str:
    distribution = list(ACTIVITY_BY_PLACE[place])
    if previous_activity is not None:
        distribution.append((previous_activity, 0.22))
    if scenario_name == "ambiguous_local_context" and place in {"office", "meeting_room"}:
        distribution = [
            ("focused_work", 0.42),
            ("interacting", 0.42),
            ("waiting", 0.08),
            ("resting", 0.05),
            ("commuting", 0.03),
        ]
    return _weighted_choice(rng, distribution)


def _sample_env_load(place: str, activity: str, rng: np.random.Generator, scenario_name: str) -> str:
    high_weight = 0.18
    medium_weight = 0.45
    low_weight = 0.37
    if place in {"meeting_room", "public_place", "transit"}:
        high_weight += 0.18
        low_weight -= 0.10
    if activity in {"interacting", "commuting"}:
        high_weight += 0.12
        medium_weight += 0.03
        low_weight -= 0.10
    if scenario_name in {"ambiguous_local_context", "noisy_untrusted_external"}:
        high_weight += 0.08
        low_weight -= 0.05
    weights = np.clip([low_weight, medium_weight, high_weight], 0.03, None)
    weights = weights / weights.sum()
    return str(rng.choice(CONTEXT_VARIABLES["env_load"], p=weights))


def _sample_resource_state(activity: str, env_load: str, rng: np.random.Generator, scenario_name: str) -> str:
    degraded = 0.14
    sparse = 0.10
    if activity == "commuting":
        sparse += 0.12
    if env_load == "high":
        degraded += 0.15
        sparse += 0.05
    if scenario_name == "degraded_ego":
        degraded += 0.12
        sparse += 0.10
    nominal = max(0.08, 1.0 - degraded - sparse)
    weights = np.asarray([nominal, degraded, sparse], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(CONTEXT_VARIABLES["resource_state"], p=weights))


def build_context_timeline(name: str, steps: int, rng: np.random.Generator) -> List[ContextState]:
    place = _weighted_choice(
        rng,
        [
            ("home", 0.28),
            ("office", 0.26),
            ("transit", 0.16),
            ("meeting_room", 0.12),
            ("public_place", 0.18),
        ],
    )
    activity: str | None = None
    timeline: List[ContextState] = []
    for _ in range(steps):
        if timeline:
            place = _weighted_choice(rng, PLACE_TRANSITIONS[place])
        activity = _sample_activity(place, activity, rng, name)
        env_load = _sample_env_load(place, activity, rng, name)
        resource_state = _sample_resource_state(activity, env_load, rng, name)
        timeline.append(ContextState(place, activity, env_load, resource_state))
    return timeline


def _scenario_settings(name: str) -> ScenarioSettings:
    if name == "nominal":
        return ScenarioSettings(
            ego_availability=0.90,
            ego_base_accuracy=0.70,
            ego_variable_accuracy={"place": 0.72, "activity": 0.68, "env_load": 0.70, "resource_state": 0.66},
            external_degraded_fraction=0.05,
        )
    if name == "degraded_ego":
        return ScenarioSettings(
            ego_availability=0.58,
            ego_base_accuracy=0.50,
            ego_variable_accuracy={"place": 0.52, "activity": 0.46, "env_load": 0.52, "resource_state": 0.46},
            ego_stale_probability=0.12,
            external_degraded_fraction=0.08,
        )
    if name == "ambiguous_local_context":
        return ScenarioSettings(
            ego_availability=0.82,
            ego_base_accuracy=0.58,
            ego_variable_accuracy={"place": 0.42, "activity": 0.46, "env_load": 0.72, "resource_state": 0.70},
            ego_confusion_bias={
                "place": {"office": "meeting_room", "meeting_room": "office"},
                "activity": {"focused_work": "interacting", "interacting": "focused_work"},
            },
            external_degraded_fraction=0.06,
            ambiguity_focus=("place", "activity"),
        )
    if name == "noisy_untrusted_external":
        return ScenarioSettings(
            ego_availability=0.82,
            ego_base_accuracy=0.64,
            ego_variable_accuracy={"place": 0.66, "activity": 0.62, "env_load": 0.66, "resource_state": 0.62},
            external_accuracy_multiplier=0.86,
            external_trust_multiplier=0.84,
            external_availability_multiplier=0.92,
            external_degraded_fraction=0.25,
            external_stale_increment=0.18,
            external_misleading_increment=0.10,
        )
    raise ValueError(f"Unsupported scenario: {name}")


def _ego_agent(settings: ScenarioSettings) -> Agent:
    return Agent(
        node_id="ego",
        kind="ego_device",
        profile=SourceProfile(
            node_id="ego",
            relation_type="self",
            trust=1.0,
            availability=settings.ego_availability,
            access_policy="self",
            access_level=1.0,
            privacy_cost=0.0,
            base_latency=0.0,
            freshness_halflife=8.0,
            base_accuracy=settings.ego_base_accuracy,
            stale_probability=settings.ego_stale_probability,
            misleading_probability=settings.ego_misleading_probability,
            variable_accuracy=settings.ego_variable_accuracy,
            variable_availability={variable: settings.ego_availability for variable in CONTEXT_VARIABLES},
            confusion_bias=settings.ego_confusion_bias,
            node_type="ego_device",
            owner_group="ego",
            object_class="class_1_smart_object",
            sensing_capabilities=tuple(CONTEXT_VARIABLES.keys()),
            base_relevance=settings.ego_base_accuracy,
            base_freshness=1.0,
            degraded=False,
        ),
    )


def build_scenario(name: str, steps: int, rng: np.random.Generator) -> ScenarioDefinition:
    settings = _scenario_settings(name)
    return ScenarioDefinition(
        name=name,
        timeline=build_context_timeline(name, steps, rng),
        agents={"ego": _ego_agent(settings)},
        relations=[],
        settings=settings,
    )


SCENARIO_BUILDERS = {name: build_scenario for name in ("nominal", "degraded_ego", "ambiguous_local_context", "noisy_untrusted_external")}
