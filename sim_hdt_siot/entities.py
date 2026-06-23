from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ContextState:
    place: str
    activity: str
    env_load: str
    resource_state: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "place": self.place,
            "activity": self.activity,
            "env_load": self.env_load,
            "resource_state": self.resource_state,
        }


@dataclass(frozen=True)
class SourceProfile:
    node_id: str
    relation_type: str
    trust: float
    availability: float
    access_policy: str
    access_level: float
    privacy_cost: float
    base_latency: float
    freshness_halflife: float
    base_accuracy: float
    stale_probability: float = 0.0
    misleading_probability: float = 0.0
    variable_accuracy: Dict[str, float] = field(default_factory=dict)
    variable_availability: Dict[str, float] = field(default_factory=dict)
    confusion_bias: Dict[str, Dict[str, str]] = field(default_factory=dict)
    node_type: str = "opportunistic_object"
    owner_group: str = "external"
    object_class: str = "class_2_simple_sensor"
    sensing_capabilities: Tuple[str, ...] = field(default_factory=tuple)
    base_relevance: float = 0.0
    base_freshness: float = 1.0
    degraded: bool = False


@dataclass(frozen=True)
class Agent:
    node_id: str
    kind: str
    profile: SourceProfile


@dataclass(frozen=True)
class Relationship:
    source: str
    target: str
    relation_type: str
    trust: float
    access_level: float
    relation_utility: float


@dataclass(frozen=True)
class ScenarioSettings:
    ego_availability: float
    ego_base_accuracy: float
    ego_variable_accuracy: Dict[str, float]
    ego_confusion_bias: Dict[str, Dict[str, str]] = field(default_factory=dict)
    ego_stale_probability: float = 0.0
    ego_misleading_probability: float = 0.0
    external_accuracy_multiplier: float = 1.0
    external_trust_multiplier: float = 1.0
    external_availability_multiplier: float = 1.0
    external_degraded_fraction: float = 0.0
    external_stale_increment: float = 0.0
    external_misleading_increment: float = 0.0
    ambiguity_focus: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    timeline: List[ContextState]
    agents: Dict[str, Agent]
    relations: List[Relationship]
    settings: ScenarioSettings


@dataclass(frozen=True)
class DiscoveryCandidate:
    source_id: str
    hops: int
    graph_distance: int
    relation_type: str
    path_trust: float
    access_score: float
    relation_utility: float
    path: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: List[DiscoveryCandidate]
    nodes_visited: int
    traversed_edges: int = 0
    relation_traversal_cost: float = 0.0
    discovery_mode: str = "exhaustive_hop"
    discovery_node_budget_active: int = 0
    discovery_edge_budget_active: int = 0
    max_candidates_to_score_active: int = 0
    stopped_by_budget: bool = False
    stopped_by_coverage: bool = False
    stopped_by_quality: bool = False
    stopped_by_frontier_empty: bool = False
    candidate_variable_coverage_count: int = 0
    all_variables_covered_by_candidates: bool = False


@dataclass(frozen=True)
class SourceObservation:
    source_id: str
    values: Dict[str, str]
    age: int
    freshness: float
    latency: float
    trust: float
    access_score: float
    privacy_cost: float
    hops: int
    relation_utility: float


@dataclass(frozen=True)
class RankedSource:
    source_id: str
    score: float
    relevance: float
    trust: float
    freshness: float
    latency: float
    access_score: float
    privacy_cost: float
    hops: int
    relation_utility: float
