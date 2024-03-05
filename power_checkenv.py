from stable_baselines3.common.env_checker import check_env
from power_env import ComputeClusterEnv

env = ComputeClusterEnv()

check_env(env)

print('done')
