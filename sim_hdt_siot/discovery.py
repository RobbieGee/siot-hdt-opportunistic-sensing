import heapq
from typing import Dict, Iterable, List, Sequence

import networkx as nx

from sim_hdt_siot.config import BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY, CONTEXT_VARIABLES, EXHAUSTIVE_HOP_DISCOVERY
from sim_hdt_siot.entities import DiscoveryCandidate, DiscoveryResult


def discover_sources(graph: nx.Graph, ego_id: str, policy: str, max_hops: int) -> DiscoveryResult:
    if policy == "ego_only":
        return DiscoveryResult(
            candidates=[],
            nodes_visited=1,
            traversed_edges=0,
            relation_traversal_cost=0.0,
            discovery_mode=EXHAUSTIVE_HOP_DISCOVERY,
        )

    paths = nx.single_source_shortest_path(graph, ego_id, cutoff=max_hops)
    path_lengths = {node_id: len(path) - 1 for node_id, path in paths.items()}
    traversed_edges = set()
    relation_traversal_cost = 0.0
    for source, distance in path_lengths.items():
        if distance >= max_hops:
            continue
        for target in graph.neighbors(source):
            if target not in path_lengths or path_lengths[target] > max_hops:
                continue
            edge_key = tuple(sorted((str(source), str(target))))
            if edge_key in traversed_edges:
                continue
            traversed_edges.add(edge_key)
            relation_traversal_cost += float(graph.edges[source, target].get("relation_traversal_cost", 1.0))
    candidates: List[DiscoveryCandidate] = []
    for node_id, shortest_path in paths.items():
        if node_id == ego_id:
            continue
        if not graph.nodes[node_id].get("sensing_capabilities"):
            continue
        candidate = _candidate_from_path(graph, str(node_id), [str(item) for item in shortest_path])
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item.hops, item.source_id))
    return DiscoveryResult(
        candidates=candidates,
        nodes_visited=len(paths),
        traversed_edges=len(traversed_edges),
        relation_traversal_cost=relation_traversal_cost,
        discovery_mode=EXHAUSTIVE_HOP_DISCOVERY,
        candidate_variable_coverage_count=_coverage_count(graph, candidates),
        all_variables_covered_by_candidates=_coverage_count(graph, candidates) == len(CONTEXT_VARIABLES),
        stopped_by_frontier_empty=True,
    )


def discover_sources_bounded_relationship_guided(
    graph: nx.Graph,
    ego_id: str,
    max_hops: int,
    node_budget: int,
    edge_budget: int,
    max_candidates_to_score: int,
    min_candidates_required: int,
    min_variable_coverage: int,
    early_stop_when_all_variables_covered: bool,
    early_stop_min_quality: float,
    max_neighbors_per_expansion: int,
    relation_priority_weights: Dict[str, float],
    discovery_privacy_penalty: float,
    discovery_latency_penalty: float,
    unresolved_variables: Iterable[str] | None = None,
) -> DiscoveryResult:
    """Discover sources with a bounded SIoT traversal guided by relation semantics.

    The traversal is deterministic: ties are broken by node id. It expands from the ego
    node through high-utility relations and high-quality nodes first, while preferring
    candidates that add coverage over still-uncovered HDT context variables.
    """
    node_budget = max(1, int(node_budget))
    edge_budget = max(0, int(edge_budget))
    max_candidates_to_score = max(0, int(max_candidates_to_score))
    max_neighbors_per_expansion = max(1, int(max_neighbors_per_expansion))
    if max_candidates_to_score == 0:
        return DiscoveryResult(
            candidates=[],
            nodes_visited=1,
            traversed_edges=0,
            relation_traversal_cost=0.0,
            discovery_mode=BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY,
            discovery_node_budget_active=node_budget,
            discovery_edge_budget_active=edge_budget,
            max_candidates_to_score_active=max_candidates_to_score,
            stopped_by_budget=True,
        )

    unresolved_set = set(unresolved_variables or ())
    visited = {ego_id}
    traversed_edges: set[tuple[str, str]] = set()
    relation_traversal_cost = 0.0
    candidates: List[DiscoveryCandidate] = []
    covered_variables: set[str] = set()
    best_quality = 0.0
    frontier: list[tuple[tuple[float, ...], tuple[str, ...], str]] = []

    def push_neighbors(current_node: str, path: Sequence[str]) -> bool:
        nonlocal relation_traversal_cost
        current_distance = len(path) - 1
        if current_distance >= max_hops:
            return False
        stopped_on_budget = False
        neighbor_entries = []
        for neighbor in graph.neighbors(current_node):
            neighbor_id = str(neighbor)
            if neighbor_id in visited:
                continue
            next_distance = current_distance + 1
            if next_distance > max_hops:
                continue
            edge = graph.edges[current_node, neighbor_id]
            priority = _frontier_priority(
                graph=graph,
                node_id=neighbor_id,
                edge=edge,
                distance=next_distance,
                covered_variables=covered_variables,
                unresolved_variables=unresolved_set,
                relation_priority_weights=relation_priority_weights,
                discovery_privacy_penalty=discovery_privacy_penalty,
                discovery_latency_penalty=discovery_latency_penalty,
            )
            neighbor_entries.append((priority, neighbor_id))
        neighbor_entries.sort(key=lambda item: (item[0], item[1]))
        for priority, neighbor_id in neighbor_entries[:max_neighbors_per_expansion]:
            edge_key = tuple(sorted((str(current_node), neighbor_id)))
            if edge_key in traversed_edges:
                continue
            if len(traversed_edges) >= edge_budget:
                stopped_on_budget = True
                break
            traversed_edges.add(edge_key)
            relation_traversal_cost += float(graph.edges[current_node, neighbor_id].get("relation_traversal_cost", 1.0))
            heapq.heappush(frontier, (priority, tuple([*path, neighbor_id]), neighbor_id))
        return stopped_on_budget

    stopped_by_budget = push_neighbors(ego_id, (ego_id,))
    stopped_by_coverage = False
    stopped_by_quality = False

    while frontier:
        if len(visited) >= node_budget:
            stopped_by_budget = True
            break
        priority, path, node_id = heapq.heappop(frontier)
        del priority
        if node_id in visited:
            continue
        visited.add(node_id)
        attrs = graph.nodes[node_id]
        capabilities = tuple(str(item) for item in attrs.get("sensing_capabilities", ()))
        if capabilities:
            candidate = _candidate_from_path(graph, node_id, list(path))
            if candidate is not None:
                candidates.append(candidate)
                covered_variables.update(variable for variable in capabilities if variable in CONTEXT_VARIABLES)
                best_quality = max(
                    best_quality,
                    _candidate_expected_quality(
                        graph,
                        candidate,
                        relation_priority_weights,
                        discovery_privacy_penalty,
                        discovery_latency_penalty,
                    ),
                )

        coverage_count = len(covered_variables)
        candidate_threshold_met = len(candidates) >= min_candidates_required
        coverage_threshold_met = coverage_count >= min_variable_coverage
        quality_threshold_met = best_quality >= early_stop_min_quality
        all_variables_covered = coverage_count == len(CONTEXT_VARIABLES)
        if (
            early_stop_when_all_variables_covered
            and all_variables_covered
            and candidate_threshold_met
            and quality_threshold_met
        ):
            stopped_by_coverage = True
            break
        if (
            len(candidates) >= max_candidates_to_score
            and candidate_threshold_met
            and coverage_threshold_met
            and quality_threshold_met
        ):
            stopped_by_quality = True
            break
        if push_neighbors(node_id, path):
            stopped_by_budget = True
            break

    stopped_by_frontier_empty = not frontier and not stopped_by_budget and not stopped_by_coverage and not stopped_by_quality
    candidates = _prioritize_final_candidates(
        graph,
        candidates,
        relation_priority_weights,
        discovery_privacy_penalty,
        discovery_latency_penalty,
        max_candidates_to_score,
    )
    coverage_count = _coverage_count(graph, candidates)
    return DiscoveryResult(
        candidates=candidates,
        nodes_visited=len(visited),
        traversed_edges=len(traversed_edges),
        relation_traversal_cost=relation_traversal_cost,
        discovery_mode=BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY,
        discovery_node_budget_active=node_budget,
        discovery_edge_budget_active=edge_budget,
        max_candidates_to_score_active=max_candidates_to_score,
        stopped_by_budget=stopped_by_budget,
        stopped_by_coverage=stopped_by_coverage,
        stopped_by_quality=stopped_by_quality,
        stopped_by_frontier_empty=stopped_by_frontier_empty,
        candidate_variable_coverage_count=coverage_count,
        all_variables_covered_by_candidates=coverage_count == len(CONTEXT_VARIABLES),
    )


def _candidate_from_path(
    graph: nx.Graph,
    node_id: str,
    path: List[str],
) -> DiscoveryCandidate | None:
    distance = len(path) - 1
    if distance <= 0:
        return None
    path_trust = 1.0
    access_score = 1.0
    relation_utility = 1.0
    strongest_relation = "SOR"
    strongest_utility = -1.0
    for source, target in zip(path[:-1], path[1:]):
        edge = graph.get_edge_data(source, target, default={})
        edge_utility = float(edge.get("relation_utility", 1.0))
        relation_utility *= edge_utility
        path_trust = min(path_trust, float(edge.get("trust", 1.0)))
        access_score = min(access_score, float(edge.get("access_level", 1.0)))
        if edge_utility > strongest_utility:
            strongest_utility = edge_utility
            strongest_relation = str(edge.get("relation_type", "SOR"))
    return DiscoveryCandidate(
        source_id=node_id,
        hops=distance,
        graph_distance=distance,
        relation_type=strongest_relation,
        path_trust=path_trust,
        access_score=access_score,
        relation_utility=relation_utility,
        path=tuple(path),
    )


def _frontier_priority(
    graph: nx.Graph,
    node_id: str,
    edge: Dict[str, object],
    distance: int,
    covered_variables: set[str],
    unresolved_variables: set[str],
    relation_priority_weights: Dict[str, float],
    discovery_privacy_penalty: float,
    discovery_latency_penalty: float,
) -> tuple[float, ...]:
    relation_type = str(edge.get("relation_type", "SOR"))
    relation_priority = float(relation_priority_weights.get(relation_type, edge.get("relation_utility", 0.5)))
    attrs = graph.nodes[node_id]
    capabilities = {str(item) for item in attrs.get("sensing_capabilities", ())}
    uncovered_gain = len((set(CONTEXT_VARIABLES) - covered_variables) & capabilities)
    unresolved_gain = len(unresolved_variables & capabilities)
    quality = _node_expected_quality(
        attrs,
        relation_priority,
        discovery_privacy_penalty,
        discovery_latency_penalty,
    )
    privacy = float(attrs.get("base_privacy_cost", 0.0))
    latency = float(attrs.get("base_latency", 0.0))
    # Short paths and high-utility SIoT relations dominate; coverage and expected
    # source quality break ties before lower cost and deterministic node id order.
    return (
        float(distance),
        -relation_priority,
        -float(uncovered_gain),
        -float(unresolved_gain),
        -quality,
        privacy,
        latency,
    )


def _node_expected_quality(
    attrs: Dict[str, object],
    relation_priority: float,
    discovery_privacy_penalty: float,
    discovery_latency_penalty: float,
) -> float:
    relevance = float(attrs.get("base_relevance", 0.0))
    trust = float(attrs.get("base_trust", 0.0))
    freshness = float(attrs.get("base_freshness", 1.0))
    latency = float(attrs.get("base_latency", 0.0))
    privacy = float(attrs.get("base_privacy_cost", 0.0))
    degraded_penalty = 0.72 if bool(attrs.get("degraded", False)) else 1.0
    denominator = 1.0 + (discovery_privacy_penalty * privacy) + (discovery_latency_penalty * latency)
    return float((relevance * trust * freshness * relation_priority * degraded_penalty) / max(denominator, 1.0e-9))


def _candidate_expected_quality(
    graph: nx.Graph,
    candidate: DiscoveryCandidate,
    relation_priority_weights: Dict[str, float],
    discovery_privacy_penalty: float,
    discovery_latency_penalty: float,
) -> float:
    attrs = graph.nodes[candidate.source_id]
    relation_priority = float(relation_priority_weights.get(candidate.relation_type, candidate.relation_utility))
    path_penalty = 1.0 / (1.0 + 0.15 * max(0, candidate.graph_distance - 1))
    return _node_expected_quality(attrs, relation_priority, discovery_privacy_penalty, discovery_latency_penalty) * path_penalty


def _prioritize_final_candidates(
    graph: nx.Graph,
    candidates: List[DiscoveryCandidate],
    relation_priority_weights: Dict[str, float],
    discovery_privacy_penalty: float,
    discovery_latency_penalty: float,
    max_candidates_to_score: int,
) -> List[DiscoveryCandidate]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -_candidate_expected_quality(
                graph,
                candidate,
                relation_priority_weights,
                discovery_privacy_penalty,
                discovery_latency_penalty,
            ),
            candidate.graph_distance,
            candidate.source_id,
        ),
    )
    return ranked[:max_candidates_to_score]


def _coverage_count(graph: nx.Graph, candidates: List[DiscoveryCandidate]) -> int:
    covered: set[str] = set()
    for candidate in candidates:
        covered.update(
            variable
            for variable in graph.nodes[candidate.source_id].get("sensing_capabilities", ())
            if variable in CONTEXT_VARIABLES
        )
    return len(covered)
