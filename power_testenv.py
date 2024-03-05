from power_env import ComputeClusterEnv

env = ComputeClusterEnv()
episodes = 1

for episode in range(episodes):
    print("episode: ", episode)
    terminated = False
    obs = env.reset()
    while not terminated:
        random_action = env.action_space.sample()
        print("  action: ", random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        print("  reward: [", reward, "]")

print('done')
