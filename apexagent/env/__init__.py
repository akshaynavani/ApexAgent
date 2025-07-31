from gymnasium.envs.registration import register

from apexagent.env.race_env import EnvConfig, RaceEnv
from apexagent.env.reward import RewardConfig, RewardShaper, StepContext

register(id="ApexRace-v0", entry_point="apexagent.env.race_env:RaceEnv")

__all__ = ["EnvConfig", "RaceEnv", "RewardConfig", "RewardShaper", "StepContext"]
