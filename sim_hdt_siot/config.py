from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


PLACE_STATES: List[str] = ["home", "office", "transit", "meeting_room", "public_place"]
ACTIVITY_STATES: List[str] = ["focused_work", "commuting", "waiting", "interacting", "resting"]
ENV_LOAD_STATES: List[str] = ["low", "medium", "high"]
RESOURCE_STATES: List[str] = ["nominal", "degraded", "sparse"]

CONTEXT_VARIABLES: Dict[str, List[str]] = {
    "place": PLACE_STATES,
    "activity": ACTIVITY_STATES,
    "env_load": ENV_LOAD_STATES,
    "resource_state": RESOURCE_STATES,
}

DEFAULT_SCENARIOS: Tuple[str, ...] = (
    "nominal",
    "degraded_ego",
    "ambiguous_local_context",
    "noisy_untrusted_external",
)

DEFAULT_POLICIES: Tuple[str, ...] = (
    "ego_only",
    "opportunistic_all",
    "siot_aware",
    "siot_aware_trust_privacy",
)

BUDGET_MATCHED_POLICIES: Tuple[str, ...] = (
    "random_k",
    "budgeted_opportunistic_k",
    "siot_aware",
    "siot_aware_trust_privacy",
)

DISCOVERY_MODE_COMPARISON_POLICIES: Tuple[str, ...] = (
    "siot_aware_exhaustive_discovery",
    "siot_aware_bounded_discovery",
    "siot_aware_trust_privacy_exhaustive_discovery",
    "siot_aware_trust_privacy_bounded_discovery",
)

POLICY_LABELS_LONG: Dict[str, str] = {
    "ego_only": "Ego-only",
    "opportunistic_all": "Opportunistic all",
    "siot_aware": "SIoT-aware",
    "siot_aware_trust_privacy": "Privacy-aware SIoT",
    "random_k": "Random-k",
    "budgeted_opportunistic_k": "Budgeted opportunistic",
    "siot_aware_exhaustive_discovery": "SIoT-aware exhaustive discovery",
    "siot_aware_bounded_discovery": "SIoT-aware bounded discovery",
    "siot_aware_trust_privacy_exhaustive_discovery": "Privacy-aware SIoT exhaustive discovery",
    "siot_aware_trust_privacy_bounded_discovery": "Privacy-aware SIoT bounded discovery",
}

POLICY_LABELS_SHORT: Dict[str, str] = {
    "ego_only": "EGO",
    "opportunistic_all": "OPP-ALL",
    "siot_aware": "SIoT",
    "siot_aware_trust_privacy": "P-SIoT",
    "random_k": "R-k",
    "budgeted_opportunistic_k": "B-OPP",
    "siot_aware_exhaustive_discovery": "SIoT-EXH",
    "siot_aware_bounded_discovery": "SIoT-BND",
    "siot_aware_trust_privacy_exhaustive_discovery": "P-SIoT-EXH",
    "siot_aware_trust_privacy_bounded_discovery": "P-SIoT-BND",
}

POLICY_LABELS: Dict[str, str] = POLICY_LABELS_LONG

SCENARIO_LABELS: Dict[str, str] = {
    "nominal": "Nominal",
    "degraded_ego": "Degraded ego",
    "ambiguous_local_context": "Ambiguous local",
    "noisy_untrusted_external": "Noisy external",
}

RELATION_TYPES: Tuple[str, ...] = ("OOR", "CLOR", "CWOR", "SOR", "POR")

EXHAUSTIVE_HOP_DISCOVERY = "exhaustive_hop"
BOUNDED_RELATIONSHIP_GUIDED_DISCOVERY = "bounded_relationship_guided"

DEFAULT_RELATION_UTILITIES: Dict[str, float] = {
    "OOR": 1.0,
    "CLOR": 0.8,
    "CWOR": 0.8,
    "SOR": 0.6,
    "POR": 0.5,
}

DEFAULT_SOURCE_SELECTION_LIMITS: Dict[str, int] = {
    "ego_only": 0,
    "opportunistic_all": 999,
    "siot_aware": 3,
    "siot_aware_trust_privacy": 3,
    "random_k": 3,
    "budgeted_opportunistic_k": 3,
    "siot_aware_exhaustive_discovery": 3,
    "siot_aware_bounded_discovery": 3,
    "siot_aware_trust_privacy_exhaustive_discovery": 3,
    "siot_aware_trust_privacy_bounded_discovery": 3,
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    seeds: Tuple[int, ...] = (7, 11, 13, 17, 19)
    episodes: int = 50
    timesteps: int = 48
    baseline_hop_radius: int = 2
    adaptive_hop_radius: int = 3
    max_hops: int = 2
    trigger_threshold: float = 0.18
    ego_confidence_threshold: float = 0.35
    ego_missingness_threshold: float = 0.50
    unresolved_variable_threshold: int = 2
    collapse_ego_conf_threshold: float = 0.35
    collapse_missing_threshold: float = 0.50
    collapse_unresolved_threshold: int = 2
    collapse_require_joint_condition: bool = True
    adaptive_extra_hops: int = 1
    default_selection_cap: int = 3
    adaptive_max_selected_sources: int = 3
    adaptive_selection_cap: int = 5
    adaptive_latency_penalty_scale: float = 0.72
    adaptive_privacy_penalty_scale: float = 0.80
    privacy_penalty_strength: float = 1.0
    adaptive_enabled: bool = True
    scenarios: Tuple[str, ...] = DEFAULT_SCENARIOS
    policies: Tuple[str, ...] = DEFAULT_POLICIES
    ego_id: str = "ego"
    results_dir: Path = Path(".")
    data_dir_name: str = "data"
    figures_dir_name: str = "figures"
    logs_dir_name: str = "logs"
    raw_outputs_dir_name: str = "raw_or_step_outputs"
    aggregate_outputs_dir_name: str = "aggregate_outputs"
    sensitivity_dir_name: str = "sensitivity"
    error_analysis_dir_name: str = "error_analysis"
    plots_dir_name: str = "figures"
    raw_results_name: str = "raw_step_results.csv"
    raw_results_archive_name: str = "raw_step_results.csv.gz"
    metrics_name: str = "aggregated_metrics.csv"
    robustness_name: str = "robustness_by_missingness.csv"
    episode_summary_name: str = "episode_summary.csv"
    overall_table_name: str = "overall_performance_by_scenario_policy.csv"
    per_variable_summary_name: str = "per_variable_macro_f1_by_scenario_policy.csv"
    convergence_summary_name: str = "convergence_by_scenario_policy.csv"
    overhead_summary_name: str = "overhead_by_scenario_policy.csv"
    statistical_summary_name: str = "statistical_summary.csv"
    pairwise_comparisons_name: str = "pairwise_policy_comparisons.csv"
    main_results_table_name: str = "table1_main_results.csv"
    overhead_table_name: str = "table2_overhead_results.csv"
    adaptive_ablation_table_name: str = "table3_adaptive_ablation_results.csv"
    per_variable_table_name: str = "per_variable_table.csv"
    experiment_summary_name: str = "experiment_summary.txt"
    validation_report_name: str = "validation_report.txt"
    plot_dpi: int = 300
    source_node_count: int = 60
    ego_personal_device_count: int = 3
    environmental_source_count: int = 16
    infrastructure_source_count: int = 14
    vehicle_source_count: int = 8
    relation_type_probabilities: Dict[str, float] = field(
        default_factory=lambda: {
            "OOR": 0.12,
            "CLOR": 0.34,
            "CWOR": 0.24,
            "SOR": 0.18,
            "POR": 0.12,
        }
    )
    graph_density_mode: str = "default"
    graph_density_edge_probabilities: Dict[str, float] = field(
        default_factory=lambda: {
            "sparse": 0.004,
            "default": 0.010,
            "dense": 0.020,
        }
    )
    ego_layer1_candidate_min: int = 5
    ego_layer1_candidate_max: int = 8
    target_average_shortest_path_low: float = 3.0
    target_average_shortest_path_high: float = 4.0
    target_effective_diameter_low: float = 3.0
    target_effective_diameter_high: float = 5.0
    degraded_external_fraction: float = 0.25
    smart_object_fraction: float = 0.45
    relation_traversal_costs: Dict[str, float] = field(
        default_factory=lambda: {
            "OOR": 0.5,
            "CLOR": 0.8,
            "CWOR": 0.8,
            "SOR": 1.2,
            "POR": 1.5,
        }
    )
    bounded_discovery_node_budget_small: int = 20
    bounded_discovery_edge_budget_small: int = 40
    bounded_discovery_score_cap_small: int = 12
    bounded_discovery_node_budget_medium: int = 30
    bounded_discovery_edge_budget_medium: int = 70
    bounded_discovery_score_cap_medium: int = 18
    bounded_discovery_node_budget_large: int = 40
    bounded_discovery_edge_budget_large: int = 100
    bounded_discovery_score_cap_large: int = 24
    bounded_discovery_medium_graph_threshold: int = 120
    bounded_discovery_large_graph_threshold: int = 400
    adaptive_discovery_budget_multiplier: float = 1.35
    bounded_discovery_min_candidates_required: int = 8
    bounded_discovery_min_variable_coverage: int = 4
    bounded_discovery_early_stop_when_all_variables_covered: bool = True
    bounded_discovery_early_stop_min_quality: float = 0.10
    bounded_discovery_max_neighbors_per_expansion: int = 8
    bounded_discovery_privacy_penalty: float = 0.35
    bounded_discovery_latency_penalty: float = 0.25
    relation_priority_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RELATION_UTILITIES))
    discovery_edge_lambda: float = 0.25
    recruitment_latency_lambda: float = 0.35
    recruitment_privacy_lambda: float = 0.75
    selection_cost_lambda: float = 0.10
    recovery_discovery_lambda: float = 1.0
    recovery_recruitment_lambda: float = 1.0
    recovery_latency_lambda: float = 0.35
    net_utility_cost_lambda: float = 0.25
    net_utility_privacy_lambda: float = 0.15
    net_utility_latency_lambda: float = 0.10
    cost_epsilon: float = 1.0
    bootstrap_samples: int = 1000
    bootstrap_basis: str = "seed_episode_combinations"
    run_sensitivity: bool = True
    budget_matched_episodes: int = 30
    budget_matched_timesteps: int = 48
    budget_matched_seeds: Tuple[int, ...] = (7, 11, 13, 17, 19)
    discovery_mode_comparison_episodes: int = 20
    discovery_mode_comparison_timesteps: int = 48
    discovery_mode_comparison_seeds: Tuple[int, ...] = (7, 11, 13)
    sensitivity_episodes: int = 8
    sensitivity_timesteps: int = 24
    sensitivity_seeds: Tuple[int, ...] = (7, 11)
    sensitivity_scenarios: Tuple[str, ...] = (
        "degraded_ego",
        "ambiguous_local_context",
        "noisy_untrusted_external",
    )
    sensitivity_hop_radius_values: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 3), (2, 4))
    sensitivity_selection_cap_values: Tuple[int, ...] = (1, 2, 3, 5)
    sensitivity_privacy_penalty_values: Dict[str, float] = field(
        default_factory=lambda: {
            "weak": 0.5,
            "default": 1.0,
            "strong": 1.8,
        }
    )
    sensitivity_degraded_fraction_values: Tuple[float, ...] = (0.10, 0.25, 0.40, 0.50)
    sensitivity_graph_density_values: Tuple[str, ...] = ("sparse", "default", "dense")
    scalability_graph_sizes: Tuple[int, ...] = (60, 200, 800)
    scalability_episodes: int = 12
    scalability_timesteps: int = 48
    scalability_seeds: Tuple[int, ...] = (7, 11, 13)
    run_scalability: bool = True
    plot_names: Dict[str, str] = field(
        default_factory=lambda: {
            "overall_accuracy": "overall_accuracy_by_scenario_policy.png",
            "macro_f1_by_variable": "macro_f1_by_variable_policy.png",
            "accuracy_vs_missing_ego": "performance_vs_ego_missingness.png",
            "convergence_time": "convergence_time_by_policy_scenario.png",
            "effective_recovery_cost": "effective_recovery_cost_by_policy_scenario.png",
            "budget_matched": "budget_matched_oca_cost_comparison.png",
            "pareto_cost": "pareto_oca_vs_total_cost.png",
            "pareto_privacy": "pareto_oca_vs_privacy_cost.png",
            "oca_per_cost": "oca_per_cost_by_policy_scenario.png",
            "scalability_cost": "scalability_total_cost_vs_graph_size.png",
            "scalability_oca": "scalability_oca_vs_graph_size.png",
            "scalability_oca_per_cost": "scalability_oca_per_cost_vs_graph_size.png",
            "discovery_mode_comparison": "discovery_mode_oca_cost_comparison.png",
        }
    )
