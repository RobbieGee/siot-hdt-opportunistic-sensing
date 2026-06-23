from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

from sim_hdt_siot.config import CONTEXT_VARIABLES, DEFAULT_RELATION_UTILITIES, ExperimentConfig, RELATION_TYPES
from sim_hdt_siot.entities import Agent, ScenarioDefinition, SourceProfile


@dataclass(frozen=True)
class SiotGraphBundle:
    graph: nx.Graph
    agents: Dict[str, Agent]
    graph_summary: Dict[str, object]
    relation_distribution: List[Dict[str, object]]
    candidate_pool_summary: Dict[str, object]


NODE_TYPE_BASES: Dict[str, Dict[str, float]] = {
    "personal_device": {"trust": 0.88, "availability": 0.86, "privacy": 0.34, "latency": 0.55, "accuracy": 0.75},
    "wearable": {"trust": 0.86, "availability": 0.84, "privacy": 0.38, "latency": 0.50, "accuracy": 0.76},
    "environmental_sensor": {"trust": 0.78, "availability": 0.84, "privacy": 0.08, "latency": 0.75, "accuracy": 0.74},
    "infrastructure_sensor": {"trust": 0.81, "availability": 0.88, "privacy": 0.12, "latency": 0.90, "accuracy": 0.73},
    "vehicle_sensor": {"trust": 0.72, "availability": 0.78, "privacy": 0.18, "latency": 1.10, "accuracy": 0.68},
    "opportunistic_object": {"trust": 0.58, "availability": 0.68, "privacy": 0.24, "latency": 1.55, "accuracy": 0.60},
}

CAPABILITY_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "personal_device": ("place", "activity", "resource_state"),
    "wearable": ("activity", "resource_state", "place"),
    "environmental_sensor": ("env_load", "place"),
    "infrastructure_sensor": ("place", "env_load", "resource_state"),
    "vehicle_sensor": ("place", "activity", "env_load"),
    "opportunistic_object": ("place", "activity", "env_load", "resource_state"),
}

RELATION_LATENCY_CONTRIBUTION = {
    "OOR": 0.08,
    "CLOR": 0.18,
    "CWOR": 0.22,
    "SOR": 0.40,
    "POR": 0.46,
}


def build_explicit_siot_graph(
    scenario: ScenarioDefinition,
    config: ExperimentConfig,
    rng: np.random.Generator,
    seed: int,
    episode: int,
) -> SiotGraphBundle:
    graph = nx.Graph()
    ego = scenario.agents[config.ego_id]
    graph.add_node(
        config.ego_id,
        node_id=config.ego_id,
        node_type="ego_device",
        owner_group="ego",
        object_class="class_1_smart_object",
        sensing_capabilities=tuple(CONTEXT_VARIABLES.keys()),
        base_relevance=ego.profile.base_accuracy,
        base_trust=1.0,
        base_freshness=1.0,
        base_latency=0.0,
        base_privacy_cost=0.0,
        mobility_group="ego",
        context_group="ego",
        degraded=False,
        node_index=0,
    )

    nodes_by_type = _create_external_nodes(config, rng)
    for index, attrs in enumerate(nodes_by_type, start=1):
        attrs["node_index"] = index
        graph.add_node(attrs["node_id"], **attrs)

    _connect_required_structure(graph, config, rng)
    _connect_additional_typed_edges(graph, config, rng)
    _ensure_reachable_layers(graph, config, rng)

    degraded_sources = _select_degraded_sources(graph, scenario, config, rng)
    agents = dict(scenario.agents)
    for node_id, attrs in graph.nodes(data=True):
        if node_id == config.ego_id:
            continue
        degraded = node_id in degraded_sources
        attrs["degraded"] = degraded
        agents[node_id] = Agent(
            node_id=node_id,
            kind=str(attrs["node_type"]),
            profile=_profile_from_node_attrs(node_id, attrs, scenario, config, rng, degraded),
        )
        attrs["base_trust"] = agents[node_id].profile.trust
        attrs["base_latency"] = agents[node_id].profile.base_latency
        attrs["base_privacy_cost"] = agents[node_id].profile.privacy_cost
        attrs["base_relevance"] = agents[node_id].profile.base_relevance
        attrs["base_freshness"] = agents[node_id].profile.base_freshness

    graph_summary = summarize_graph(graph, config, scenario.name, seed, episode)
    relation_distribution = summarize_relation_distribution(graph, scenario.name, seed, episode)
    candidate_pool_summary = summarize_candidate_pool(graph, config, scenario.name, seed, episode)
    return SiotGraphBundle(graph, agents, graph_summary, relation_distribution, candidate_pool_summary)


def _create_external_nodes(config: ExperimentConfig, rng: np.random.Generator) -> List[Dict[str, object]]:
    personal_count = max(3, config.ego_personal_device_count)
    environmental_count = config.environmental_source_count
    infrastructure_count = config.infrastructure_source_count
    vehicle_count = config.vehicle_source_count
    used = personal_count + environmental_count + infrastructure_count + vehicle_count
    opportunistic_count = max(0, config.source_node_count - used)

    node_specs: List[Tuple[str, str]] = []
    for idx in range(personal_count):
        node_specs.append((f"personal_{idx + 1:02d}", "wearable" if idx == 0 else "personal_device"))
    node_specs.extend((f"env_sensor_{idx + 1:02d}", "environmental_sensor") for idx in range(environmental_count))
    node_specs.extend((f"infra_sensor_{idx + 1:02d}", "infrastructure_sensor") for idx in range(infrastructure_count))
    node_specs.extend((f"vehicle_sensor_{idx + 1:02d}", "vehicle_sensor") for idx in range(vehicle_count))
    node_specs.extend((f"opportunistic_{idx + 1:02d}", "opportunistic_object") for idx in range(opportunistic_count))

    nodes: List[Dict[str, object]] = []
    for node_id, node_type in node_specs[: config.source_node_count]:
        object_class = "class_1_smart_object" if rng.random() < config.smart_object_fraction else "class_2_simple_sensor"
        capabilities = _sample_capabilities(node_type, object_class, rng)
        base = NODE_TYPE_BASES[node_type]
        nodes.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "owner_group": _owner_group_for_type(node_type, rng),
                "object_class": object_class,
                "sensing_capabilities": capabilities,
                "base_relevance": float(base["accuracy"]),
                "base_trust": float(base["trust"]),
                "base_freshness": 1.0,
                "base_latency": float(base["latency"]),
                "base_privacy_cost": float(base["privacy"]),
                "mobility_group": _mobility_group_for_type(node_type, rng),
                "context_group": _context_group_for_type(node_type, rng),
                "degraded": False,
            }
        )
    _ensure_capability_coverage(nodes)
    return nodes


def _sample_capabilities(node_type: str, object_class: str, rng: np.random.Generator) -> Tuple[str, ...]:
    template = list(CAPABILITY_TEMPLATES[node_type])
    if object_class == "class_1_smart_object":
        count = int(rng.integers(2, min(len(template), 4) + 1))
    else:
        count = int(rng.integers(1, min(len(template), 2) + 1))
    selected = rng.choice(template, size=count, replace=False)
    return tuple(sorted(str(item) for item in selected))


def _ensure_capability_coverage(nodes: List[Dict[str, object]]) -> None:
    for variable in CONTEXT_VARIABLES:
        if any(variable in attrs["sensing_capabilities"] for attrs in nodes):
            continue
        for attrs in nodes:
            capabilities = tuple(sorted(set(attrs["sensing_capabilities"]) | {variable}))
            attrs["sensing_capabilities"] = capabilities
            break


def _owner_group_for_type(node_type: str, rng: np.random.Generator) -> str:
    if node_type in {"personal_device", "wearable"}:
        return "ego"
    if node_type in {"environmental_sensor", "infrastructure_sensor"}:
        return str(rng.choice(["facility", "service_provider", "public"]))
    if node_type == "vehicle_sensor":
        return str(rng.choice(["transport_operator", "personal_vehicle", "public"]))
    return str(rng.choice(["peer", "unknown", "service_provider"]))


def _mobility_group_for_type(node_type: str, rng: np.random.Generator) -> str:
    if node_type in {"personal_device", "wearable"}:
        return "ego_mobile"
    if node_type == "vehicle_sensor":
        return "transit_mobile"
    if node_type == "opportunistic_object":
        return str(rng.choice(["mobile_peer", "semi_static", "public_mobile"]))
    return "fixed"


def _context_group_for_type(node_type: str, rng: np.random.Generator) -> str:
    if node_type in {"personal_device", "wearable"}:
        return "personal"
    if node_type == "environmental_sensor":
        return str(rng.choice(["home_env", "office_env", "public_env"]))
    if node_type == "infrastructure_sensor":
        return str(rng.choice(["office_infra", "public_infra", "service_infra"]))
    if node_type == "vehicle_sensor":
        return "transit"
    return str(rng.choice(["social", "public", "service"]))


def _connect_required_structure(graph: nx.Graph, config: ExperimentConfig, rng: np.random.Generator) -> None:
    personal_nodes = _nodes_of_type(graph, {"personal_device", "wearable"})
    env_nodes = _nodes_of_type(graph, {"environmental_sensor"})
    infra_nodes = _nodes_of_type(graph, {"infrastructure_sensor"})
    vehicle_nodes = _nodes_of_type(graph, {"vehicle_sensor"})
    opportunistic_nodes = _nodes_of_type(graph, {"opportunistic_object"})
    all_nodes = [node for node in graph.nodes if node != config.ego_id]
    layer1_count = min(len(all_nodes), int(rng.integers(config.ego_layer1_candidate_min, config.ego_layer1_candidate_max + 1)))
    layer1 = _dedupe_preserve_order(
        personal_nodes[: config.ego_personal_device_count]
        + env_nodes[:2]
        + infra_nodes[:1]
        + vehicle_nodes[:1]
        + opportunistic_nodes[:1]
    )[:layer1_count]
    remaining = [node for node in all_nodes if node not in set(layer1)]
    if len(layer1) < layer1_count:
        needed = layer1_count - len(layer1)
        layer1.extend(remaining[:needed])
        remaining = remaining[needed:]

    if config.source_node_count <= 80:
        layer2_target = min(len(remaining), max(20, round(0.40 * config.source_node_count)))
        layer3_target = min(len(remaining) - layer2_target, max(12, round(0.38 * config.source_node_count)))
    else:
        layer2_target = min(len(remaining), max(layer1_count * 3, round(0.32 * config.source_node_count)))
        layer3_target = min(len(remaining) - layer2_target, max(layer1_count * 4, round(0.43 * config.source_node_count)))
    layer2 = remaining[:layer2_target]
    layer3 = remaining[layer2_target : layer2_target + layer3_target]
    layer4 = remaining[layer2_target + layer3_target :]

    graph.nodes[config.ego_id]["graph_layer"] = 0
    for layer_index, layer_nodes in ((1, layer1), (2, layer2), (3, layer3), (4, layer4)):
        for node in layer_nodes:
            graph.nodes[node]["graph_layer"] = layer_index

    for node in layer1:
        _add_or_update_edge(graph, config.ego_id, node, _relation_for_parent_child(graph, config.ego_id, node), rng, config)

    _connect_layer(graph, layer1, layer2, rng, config)
    _connect_layer(graph, layer2, layer3, rng, config)
    _connect_layer(graph, layer3, layer4, rng, config)
    _connect_local_layer_edges(graph, layer2, rng, config, probability=0.035)
    _connect_local_layer_edges(graph, layer3, rng, config, probability=0.025)
    _connect_local_layer_edges(graph, layer4, rng, config, probability=0.015)


def _connect_additional_typed_edges(graph: nx.Graph, config: ExperimentConfig, rng: np.random.Generator) -> None:
    nodes = [node for node in graph.nodes if node != config.ego_id]
    edge_probability = config.graph_density_edge_probabilities.get(config.graph_density_mode, 0.032)
    relation_probabilities = np.asarray([config.relation_type_probabilities[relation] for relation in RELATION_TYPES], dtype=float)
    relation_probabilities = relation_probabilities / relation_probabilities.sum()
    pair_iterable: Iterable[Tuple[str, str]]
    if len(nodes) <= 250:
        pair_iterable = ((source, target) for idx, source in enumerate(nodes) for target in nodes[idx + 1 :])
    else:
        sampled_pairs: set[Tuple[str, str]] = set()
        attempts_per_node = max(1, round(edge_probability * len(nodes) * 2.0))
        for source in nodes:
            possible_targets = [node for node in nodes if node != source]
            for target in rng.choice(possible_targets, size=min(attempts_per_node, len(possible_targets)), replace=False):
                sampled_pairs.add(tuple(sorted((str(source), str(target)))))
        pair_iterable = sampled_pairs
    for source, target in pair_iterable:
        if graph.has_edge(source, target) or (len(nodes) <= 250 and rng.random() > edge_probability):
            continue
        source_layer = int(graph.nodes[source].get("graph_layer", 2))
        target_layer = int(graph.nodes[target].get("graph_layer", 2))
        if abs(source_layer - target_layer) > 1:
            continue
        if min(source_layer, target_layer) == 1 and rng.random() < 0.70:
            continue
        relation = str(rng.choice(RELATION_TYPES, p=relation_probabilities))
        if relation == "OOR" and graph.nodes[source]["owner_group"] != graph.nodes[target]["owner_group"]:
            relation = "SOR"
        if relation == "POR" and graph.nodes[source]["object_class"] != graph.nodes[target]["object_class"]:
            relation = "CLOR" if graph.nodes[source]["context_group"] == graph.nodes[target]["context_group"] else "SOR"
        _add_or_update_edge(graph, source, target, relation, rng, config)


def _ensure_reachable_layers(graph: nx.Graph, config: ExperimentConfig, rng: np.random.Generator) -> None:
    path_lengths = nx.single_source_shortest_path_length(graph, config.ego_id, cutoff=3)
    if not any(distance == 2 for node, distance in path_lengths.items() if node != config.ego_id):
        candidates = [node for node in graph.nodes if node != config.ego_id and not graph.has_edge(config.ego_id, node)]
        if candidates:
            _add_or_update_edge(
                graph,
                str(rng.choice(_nodes_of_type(graph, {"personal_device", "wearable"}))),
                str(rng.choice(candidates)),
                "CLOR",
                rng,
                config,
            )
    path_lengths = nx.single_source_shortest_path_length(graph, config.ego_id, cutoff=3)
    if not any(distance == 3 for node, distance in path_lengths.items() if node != config.ego_id):
        distance_two = [node for node, distance in path_lengths.items() if distance == 2]
        far_candidates = [node for node in graph.nodes if node not in path_lengths and node != config.ego_id]
        if distance_two and far_candidates:
            _add_or_update_edge(graph, str(rng.choice(distance_two)), str(rng.choice(far_candidates)), "SOR", rng, config)


def _add_or_update_edge(
    graph: nx.Graph,
    source: str,
    target: str,
    relation_type: str,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> None:
    if source == target:
        return
    relation_utility = DEFAULT_RELATION_UTILITIES[relation_type]
    current = graph.get_edge_data(source, target)
    if current and float(current.get("relation_utility", 0.0)) >= relation_utility:
        return
    trust = float(np.clip(0.42 + (0.52 * relation_utility) + rng.normal(0.0, 0.035), 0.25, 1.0))
    access_level = float(np.clip(0.35 + (0.58 * relation_utility) + rng.normal(0.0, 0.04), 0.20, 1.0))
    graph.add_edge(
        source,
        target,
        relation_type=relation_type,
        relation_utility=relation_utility,
        relation_weight=relation_utility,
        trust=trust,
        access_level=access_level,
        latency_contribution=RELATION_LATENCY_CONTRIBUTION[relation_type],
        relation_traversal_cost=config.relation_traversal_costs[relation_type],
    )


def _connect_layer(
    graph: nx.Graph,
    parent_layer: List[str],
    child_layer: List[str],
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> None:
    if not parent_layer:
        return
    parent_load = {node: 0 for node in parent_layer}
    for child in child_layer:
        parent = min(parent_layer, key=lambda node: (parent_load[node], rng.random()))
        parent_load[parent] += 1
        relation = _relation_for_parent_child(graph, parent, child)
        _add_or_update_edge(graph, parent, child, relation, rng, config)
        if rng.random() < 0.16 and len(parent_layer) > 1:
            alternate = str(rng.choice([node for node in parent_layer if node != parent]))
            _add_or_update_edge(graph, alternate, child, _relation_for_parent_child(graph, alternate, child), rng, config)


def _connect_local_layer_edges(
    graph: nx.Graph,
    layer_nodes: List[str],
    rng: np.random.Generator,
    config: ExperimentConfig,
    probability: float,
) -> None:
    for idx, source in enumerate(layer_nodes):
        for target in layer_nodes[idx + 1 :]:
            if graph.has_edge(source, target) or rng.random() > probability:
                continue
            if graph.nodes[source].get("context_group") == graph.nodes[target].get("context_group"):
                relation = "CLOR"
            elif graph.nodes[source].get("object_class") == graph.nodes[target].get("object_class"):
                relation = "POR"
            else:
                relation = "SOR"
            _add_or_update_edge(graph, source, target, relation, rng, config)


def _relation_for_parent_child(graph: nx.Graph, parent: str, child: str) -> str:
    child_attrs = graph.nodes[child]
    if parent == "ego":
        if child_attrs.get("owner_group") == "ego":
            return "OOR"
        if child_attrs.get("node_type") == "infrastructure_sensor":
            return "CWOR"
        if child_attrs.get("node_type") in {"environmental_sensor", "vehicle_sensor"}:
            return "CLOR"
        return "SOR"
    parent_attrs = graph.nodes[parent]
    if child_attrs.get("owner_group") == parent_attrs.get("owner_group") == "ego":
        return "OOR"
    if child_attrs.get("context_group") == parent_attrs.get("context_group"):
        return "CLOR"
    if child_attrs.get("node_type") == "infrastructure_sensor" or parent_attrs.get("node_type") == "infrastructure_sensor":
        return "CWOR"
    if child_attrs.get("object_class") == parent_attrs.get("object_class"):
        return "POR"
    return "SOR"


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _nodes_of_type(graph: nx.Graph, node_types: Iterable[str]) -> List[str]:
    node_type_set = set(node_types)
    return [str(node) for node, attrs in graph.nodes(data=True) if attrs.get("node_type") in node_type_set]


def _select_degraded_sources(
    graph: nx.Graph,
    scenario: ScenarioDefinition,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> set[str]:
    if scenario.name == "noisy_untrusted_external":
        fraction = config.degraded_external_fraction
    else:
        fraction = scenario.settings.external_degraded_fraction
    candidates = [node for node in graph.nodes if node != config.ego_id]
    count = int(round(len(candidates) * fraction))
    if count <= 0:
        return set()
    weights = []
    for node in candidates:
        attrs = graph.nodes[node]
        incident_relations = [graph.edges[node, neighbor]["relation_type"] for neighbor in graph.neighbors(node)]
        weight = 1.0
        if attrs["node_type"] == "opportunistic_object":
            weight += 1.1
        if any(relation in {"SOR", "POR"} for relation in incident_relations):
            weight += 1.0
        if any(relation in {"OOR", "CLOR", "CWOR"} for relation in incident_relations):
            weight -= 0.25
        weights.append(max(0.15, weight))
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    selected = rng.choice(candidates, size=min(count, len(candidates)), replace=False, p=probabilities)
    return {str(node) for node in selected}


def _profile_from_node_attrs(
    node_id: str,
    attrs: Dict[str, object],
    scenario: ScenarioDefinition,
    config: ExperimentConfig,
    rng: np.random.Generator,
    degraded: bool,
) -> SourceProfile:
    node_type = str(attrs["node_type"])
    base = NODE_TYPE_BASES[node_type]
    caps = tuple(str(item) for item in attrs["sensing_capabilities"])
    smart_bonus = 0.05 if attrs["object_class"] == "class_1_smart_object" else 0.0
    trust = float(np.clip(base["trust"] + rng.normal(0.0, 0.04), 0.20, 0.98))
    availability = float(np.clip(base["availability"] + rng.normal(0.0, 0.05), 0.20, 0.98))
    accuracy = float(np.clip(base["accuracy"] + smart_bonus + rng.normal(0.0, 0.04), 0.25, 0.94))
    latency = float(max(0.05, base["latency"] + rng.normal(0.0, 0.16)))
    privacy = float(np.clip(base["privacy"] + rng.normal(0.0, 0.04), 0.02, 0.75))

    trust *= scenario.settings.external_trust_multiplier
    availability *= scenario.settings.external_availability_multiplier
    accuracy *= scenario.settings.external_accuracy_multiplier
    stale_probability = 0.05 + scenario.settings.external_stale_increment
    misleading_probability = 0.02 + scenario.settings.external_misleading_increment
    freshness_halflife = 5.0 if node_type in {"personal_device", "wearable", "environmental_sensor"} else 3.2

    if degraded:
        trust *= 0.58
        availability *= 0.78
        accuracy *= 0.68
        latency *= 1.30
        stale_probability += 0.20
        misleading_probability += 0.16
        freshness_halflife *= 0.72

    variable_accuracy = {}
    variable_availability = {}
    for variable in CONTEXT_VARIABLES:
        if variable not in caps:
            variable_accuracy[variable] = 0.0
            variable_availability[variable] = 0.0
            continue
        capability_boost = _capability_boost(node_type, variable)
        variable_accuracy[variable] = float(np.clip(accuracy + capability_boost, 0.20, 0.96))
        variable_availability[variable] = float(np.clip(availability + (0.03 if variable in {"place", "env_load"} else 0.0), 0.10, 0.99))

    access_policy = _access_policy_for_node(node_type, attrs)
    relation_type = _dominant_incident_relation_type(attrs, node_id)
    base_relevance = float(np.mean([variable_accuracy[var] * variable_availability[var] for var in caps])) if caps else 0.0
    base_freshness = float(np.exp(-np.log(2.0) * latency / max(freshness_halflife, 0.1)))
    return SourceProfile(
        node_id=node_id,
        relation_type=relation_type,
        trust=float(np.clip(trust, 0.10, 0.99)),
        availability=float(np.clip(availability, 0.05, 0.99)),
        access_policy=access_policy,
        access_level=_access_level_for_policy(access_policy),
        privacy_cost=privacy,
        base_latency=latency,
        freshness_halflife=freshness_halflife,
        base_accuracy=float(np.clip(accuracy, 0.10, 0.98)),
        stale_probability=float(np.clip(stale_probability, 0.0, 0.75)),
        misleading_probability=float(np.clip(misleading_probability, 0.0, 0.60)),
        variable_accuracy=variable_accuracy,
        variable_availability=variable_availability,
        confusion_bias=_confusion_bias_for_node(node_type, degraded),
        node_type=node_type,
        owner_group=str(attrs["owner_group"]),
        object_class=str(attrs["object_class"]),
        sensing_capabilities=caps,
        base_relevance=base_relevance,
        base_freshness=base_freshness,
        degraded=degraded,
    )


def _capability_boost(node_type: str, variable: str) -> float:
    if node_type == "environmental_sensor" and variable == "env_load":
        return 0.12
    if node_type == "infrastructure_sensor" and variable in {"place", "env_load"}:
        return 0.08
    if node_type == "vehicle_sensor" and variable in {"place", "activity"}:
        return 0.06
    if node_type in {"personal_device", "wearable"} and variable in {"activity", "resource_state"}:
        return 0.06
    return 0.0


def _access_policy_for_node(node_type: str, attrs: Dict[str, object]) -> str:
    if attrs["owner_group"] == "ego":
        return "self" if node_type == "ego_device" else "trusted"
    if node_type in {"environmental_sensor", "infrastructure_sensor"}:
        return "public"
    if node_type == "vehicle_sensor":
        return "relational"
    if node_type == "opportunistic_object":
        return "public" if attrs["owner_group"] in {"service_provider", "public"} else "relational"
    return "relational"


def _access_level_for_policy(access_policy: str) -> float:
    return {
        "self": 1.0,
        "trusted": 0.88,
        "relational": 0.72,
        "public": 0.82,
        "restricted": 0.55,
    }.get(access_policy, 0.60)


def _dominant_incident_relation_type(attrs: Dict[str, object], node_id: str) -> str:
    del attrs, node_id
    return "SIoT"


def _confusion_bias_for_node(node_type: str, degraded: bool) -> Dict[str, Dict[str, str]]:
    if node_type == "vehicle_sensor":
        return {"place": {"public_place": "transit", "office": "transit"}}
    if node_type == "opportunistic_object" and degraded:
        return {
            "place": {"office": "public_place", "meeting_room": "public_place"},
            "activity": {"focused_work": "waiting", "interacting": "waiting"},
        }
    return {}


def summarize_graph(graph: nx.Graph, config: ExperimentConfig, scenario: str, seed: int, episode: int) -> Dict[str, object]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    connected = nx.is_connected(graph) if node_count else False
    ego_component = nx.node_connected_component(graph, config.ego_id)
    ego_component_graph = graph.subgraph(ego_component).copy()
    average_shortest_path, effective_diameter = _path_length_summary(ego_component_graph, seed + episode)
    centrality_sample = min(96, node_count)
    betweenness = nx.betweenness_centrality(graph, k=centrality_sample if node_count > centrality_sample else None, seed=seed)
    summary: Dict[str, object] = {
        "scenario": scenario,
        "seed": seed,
        "episode": episode,
        "nodes": node_count,
        "source_nodes": node_count - 1,
        "edges": edge_count,
        "density": nx.density(graph),
        "average_degree": float(np.mean(degrees)) if degrees else 0.0,
        "average_clustering_coefficient": nx.average_clustering(graph),
        "graph_connected": bool(connected),
        "average_shortest_path_length": float(average_shortest_path),
        "effective_diameter": float(effective_diameter),
        "ego_component_size": len(ego_component),
        "ego_component_fraction": len(ego_component) / max(1, node_count),
        "ego_degree": int(graph.degree(config.ego_id)),
        "ego_degree_centrality": nx.degree_centrality(graph).get(config.ego_id, 0.0),
        "ego_betweenness_centrality": betweenness.get(config.ego_id, 0.0),
    }
    candidate_summary = summarize_candidate_pool(graph, config, scenario, seed, episode)
    for hops in (1, 2, 3, 4):
        summary[f"candidate_sources_within_{hops}_hop"] = candidate_summary[f"candidate_sources_within_{hops}_hop"]
    for relation in RELATION_TYPES:
        summary[f"edge_count_{relation}"] = sum(1 for _, _, attrs in graph.edges(data=True) if attrs["relation_type"] == relation)
    for node_type in sorted({attrs["node_type"] for _, attrs in graph.nodes(data=True)}):
        summary[f"node_type_count_{node_type}"] = sum(1 for _, attrs in graph.nodes(data=True) if attrs["node_type"] == node_type)
    for variable in CONTEXT_VARIABLES:
        summary[f"capability_count_{variable}"] = sum(
            1
            for node, attrs in graph.nodes(data=True)
            if node != config.ego_id and variable in attrs.get("sensing_capabilities", ())
        )
    return summary


def summarize_relation_distribution(graph: nx.Graph, scenario: str, seed: int, episode: int) -> List[Dict[str, object]]:
    total_edges = max(1, graph.number_of_edges())
    rows = []
    for relation in RELATION_TYPES:
        count = sum(1 for _, _, attrs in graph.edges(data=True) if attrs["relation_type"] == relation)
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "episode": episode,
                "relation_type": relation,
                "edge_count": count,
                "edge_share": count / total_edges,
                "relation_utility": DEFAULT_RELATION_UTILITIES[relation],
            }
        )
    return rows


def summarize_candidate_pool(graph: nx.Graph, config: ExperimentConfig, scenario: str, seed: int, episode: int) -> Dict[str, object]:
    path_lengths = nx.single_source_shortest_path_length(graph, config.ego_id, cutoff=4)
    summary: Dict[str, object] = {"scenario": scenario, "seed": seed, "episode": episode}
    for hops in (1, 2, 3, 4):
        summary[f"candidate_sources_within_{hops}_hop"] = sum(
            1
            for node, distance in path_lengths.items()
            if node != config.ego_id
            and distance <= hops
            and bool(graph.nodes[node].get("sensing_capabilities"))
        )
    return summary


def _path_length_summary(graph: nx.Graph, seed: int, percentile: float = 0.90) -> Tuple[float, float]:
    if graph.number_of_nodes() <= 1:
        return 0.0, 0.0
    distances: List[int] = []
    nodes = list(graph.nodes)
    if graph.number_of_nodes() > 180:
        rng = np.random.default_rng(seed)
        nodes = [str(node) for node in rng.choice(nodes, size=min(120, len(nodes)), replace=False)]
    for source in nodes:
        lengths = nx.single_source_shortest_path_length(graph, source)
        distances.extend(distance for distance in lengths.values() if distance > 0)
    if not distances:
        return 0.0, 0.0
    values = np.asarray(distances, dtype=float)
    return float(np.mean(values)), float(np.quantile(values, percentile))
