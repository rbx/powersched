from stable_baselines3.common.env_checker import check_env
from environment import ComputeClusterEnv

env = ComputeClusterEnv()

check_env(env)

print('done')
