from stable_baselines3 import PPO
import os
from datetime import datetime
from environment import ComputeClusterEnv, Weights
import re
import glob
import argparse
import pandas as pd

TIMESTEPS = 100000

def main():
    parser = argparse.ArgumentParser(description="Run the Compute Cluster Environment with optional rendering.")
    parser.add_argument('--render', type=str, default='none', choices=['human', 'none'], help='Render mode for the environment (default: none).')
    parser.add_argument('--quick-plot', action='store_true', help='In "human" render mode, skip quickly to the plot (default: False).')
    parser.add_argument('--prices', type=str, nargs='?', const="", default="", help='Path to the CSV file containing electricity prices (Date,Price)')
    parser.add_argument('--plot-rewards', action='store_true', help='Per step, plot rewards for all possible num_idle_nodes & num_used_nodes (default: False).')
    parser.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient for the loss calculation (default: 0.0) (Passed to PPO).')
    parser.add_argument("--efficiency-weight", type=float, default=0.7, help="Weight for efficiency reward")
    parser.add_argument("--price-weight", type=float, default=0.2, help="Weight for price reward")
    parser.add_argument("--idle-weight", type=float, default=0.1, help="Weight for idle penalty")
    parser.add_argument("--iter-limit", type=int, default=0, help=f"Max number of training iterations (1 iteration = {TIMESTEPS} steps)")
    parser.add_argument("--session", default="default", help="Session ID")

    args = parser.parse_args()
    csv_file_path = args.prices

    if csv_file_path:
        df = pd.read_csv(csv_file_path, parse_dates=['Date'])
        prices = df['Price'].values.tolist()
        # Print the first few prices to verify
        print(f"Loaded {len(prices)} prices from CSV.")
        print("First few prices:", prices[:19])
    else:
        prices = None
        print("No CSV file provided. Using default price generation.")

    weights = Weights(
        efficiency_weight=args.efficiency_weight,
        price_weight=args.price_weight,
        idle_weight=args.idle_weight
    )

    models_dir = f"models/{args.session}_e{args.efficiency_weight}_p{args.price_weight}_i{args.idle_weight}/"
    log_dir = f"logs/{args.session}_e{args.efficiency_weight}_p{args.price_weight}_i{args.idle_weight}/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = "plots/"
    plots_filepath = f"{plots_dir}{args.session}_e{args.efficiency_weight}_p{args.price_weight}_i{args.idle_weight}_{timestamp}.png"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    env = ComputeClusterEnv(weights=weights,
                            session=args.session,
                            render_mode=args.render,
                            quick_plot=args.quick_plot,
                            external_prices=prices,
                            plot_rewards=args.plot_rewards,
                            plots_filepath=plots_filepath)
    env.reset()

    # Check if there are any saved models in models_dir
    model_files = glob.glob(models_dir + "*.zip")
    latest_model_file = None
    if model_files:
        # Sort the files by extracting the timestep number from the filename and converting it to an integer
        model_files.sort(key=lambda filename: int(re.match(r"(\d+)", os.path.basename(filename)).group()))
        latest_model_file = model_files[-1]  # Get the last file after sorting, which should be the one with the most timesteps
        print(f"Found a saved model: {latest_model_file}, continuing training from it.")
        model = PPO.load(latest_model_file, env=env, tensorboard_log=log_dir)
    else:
        model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, ent_coef=args.ent_coef)

    iters = 0

    # If we're continuing from a saved model, adjust iters so that filenames continue sequentially
    if latest_model_file:
        try:
            # Assumes the filename format is "{models_dir}/{TIMESTEPS * iters}.zip"
            iters = int(os.path.basename(latest_model_file).split('.')[0]) // TIMESTEPS
        except ValueError:
            # If the filename doesn't follow expected format, default to 0
            iters = 0

    env.set_progress(iters, TIMESTEPS)

    try:
        while True:
            print(f"iter {iters}")
            iters += 1
            if args.iter_limit > 0 and iters > args.iter_limit:
                print(f"iterations limit ({args.iter_limit}) reached: {iters}.")
                break
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

if __name__ == "__main__":
    main()
