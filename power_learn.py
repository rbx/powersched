from stable_baselines3 import PPO
import os
from power_env import ComputeClusterEnv
import re
import glob

models_dir = "models/power/"
logdir = "logs/power/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = ComputeClusterEnv()
env.reset()

# Check if there are any saved models in models_dir
model_files = glob.glob(models_dir + "*.zip")
latest_model_file = None
if model_files:
    # Sort the files by extracting the timestep number from the filename and converting it to an integer
    model_files.sort(key=lambda filename: int(re.match(r"(\d+)", os.path.basename(filename)).group()))
    latest_model_file = model_files[-1]  # Get the last file after sorting, which should be the one with the most timesteps
    print(f"Found a saved model: {latest_model_file}, continuing training from it.")
    model = PPO.load(latest_model_file, env=env, tensorboard_log=logdir)
else:
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)

# # Overall architecture of the policy network
# print("model.policy:")
# print(model.policy)

# # Detailed architecture, including layers:
# print("model.policy.net_arch:")
# print(model.policy.net_arch)

# # Parameters (weights and biases)
# print("name, param in model.policy.named_parameters:")
# for name, param in model.policy.named_parameters():
#     print(name, param.shape)

TIMESTEPS = 10000
iters = 0

# If we're continuing from a saved model, adjust iters so that filenames continue sequentially
if latest_model_file:
    try:
        # Assumes the filename format is "{models_dir}/{TIMESTEPS * iters}.zip"
        iters = int(os.path.basename(latest_model_file).split('.')[0]) // TIMESTEPS
    except ValueError:
        # If the filename doesn't follow expected format, default to 0
        iters = 0

try:
    while True:
        print(f"iter {iters}")
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{models_dir}/{TIMESTEPS * iters}.zip")
except KeyboardInterrupt:
    print("Training interrupted by user.")
    # print("Training interrupted by user, attempting to save the current model...")
    # model.save(f"{models_dir}/interrupted_{TIMESTEPS * iters}.zip")
    # print("Model saved successfully.")
finally:
    print("Exiting training...")
    env.close()
