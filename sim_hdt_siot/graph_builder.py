import numpy as np

from sim_hdt_siot.config import ExperimentConfig
from sim_hdt_siot.entities import ScenarioDefinition
from sim_hdt_siot.siot_graph import SiotGraphBundle, build_explicit_siot_graph


def build_siot_graph(
    scenario: ScenarioDefinition,
    config: ExperimentConfig,
    rng: np.random.Generator,
    seed: int,
    episode: int,
) -> SiotGraphBundle:
    return build_explicit_siot_graph(scenario, config, rng, seed, episode)
