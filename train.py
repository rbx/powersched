from stable_baselines3 import PPO
import os
from environment import ComputeClusterEnv, Weights, PlottingComplete
from callbacks import ComputeClusterCallback
import re
import glob
import argparse
import pandas as pd

STEPS_PER_ITERATION = 100000

def main():
    parser = argparse.ArgumentParser(description="Run the Compute Cluster Environment with optional rendering.")
    parser.add_argument('--render', type=str, default='none', choices=['human', 'none'], help='Render mode for the environment (default: none).')
    parser.add_argument('--quick-plot', action='store_true', help='In "human" render mode, skip quickly to the plot (default: False).')
    parser.add_argument('--plot-once', action='store_true', help='In "human" render mode, exit after the first plot.')
    parser.add_argument('--prices', type=str, nargs='?', const="", default="", help='Path to the CSV file containing electricity prices (Date,Price)')
    parser.add_argument('--job-durations', type=str, nargs='?', const="", default="", help='Path to a file containing job duration samples (for use with duration_sampler)')
    parser.add_argument('--plot-rewards', action='store_true', help='Per step, plot rewards for all possible num_idle_nodes & num_used_nodes (default: False).')
    parser.add_argument('--plot-eff-reward', action='store_true', help='Include efficiency reward in the plot (dashed line).')
    parser.add_argument('--plot-price-reward', action='store_true', help='Include price reward in the plot (dashed line).')
    parser.add_argument('--plot-idle-penalty', action='store_true', help='Include idle penalty in the plot (dashed line).')
    parser.add_argument('--plot-job-age-penalty', action='store_true', help='Include job age penalty in the plot (dashed line).')
    parser.add_argument('--skip-plot-price', action='store_true', help='Skip electricity price in the plot (blue line).')
    parser.add_argument('--skip-plot-online-nodes', action='store_true', help='Skip online nodes in the plot (blue line).')
    parser.add_argument('--skip-plot-used-nodes', action='store_true', help='Skip used nodes in the plot (blue line).')
    parser.add_argument('--skip-plot-job-queue', action='store_true', help='Skip job queue in the plot (blue line).')
    parser.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient for the loss calculation (default: 0.0) (Passed to PPO).')
    parser.add_argument("--efficiency-weight", type=float, default=0.7, help="Weight for efficiency reward")
    parser.add_argument("--price-weight", type=float, default=0.2, help="Weight for price reward")
    parser.add_argument("--idle-weight", type=float, default=0.1, help="Weight for idle penalty")
    parser.add_argument("--job-age-weight", type=float, default=0.0, help="Weight for job age penalty")
    parser.add_argument("--iter-limit", type=int, default=0, help=f"Max number of training iterations (1 iteration = {STEPS_PER_ITERATION} steps)")
    parser.add_argument("--session", default="default", help="Session ID")

    args = parser.parse_args()
    prices_file_path = args.prices
    job_durations_file_path = args.job_durations

    if prices_file_path:
        df = pd.read_csv(prices_file_path, parse_dates=['Date'])
        prices = df['Price'].values.tolist()
        print(f"Loaded {len(prices)} prices from CSV: {prices_file_path}")
        # print("First few prices:", prices[:30])
    else:
        prices = None
        print("No CSV file provided. Using default price generation.")

    weights = Weights(
        efficiency_weight=args.efficiency_weight,
        price_weight=args.price_weight,
        idle_weight=args.idle_weight,
        job_age_weight=args.job_age_weight
    )

    weights_prefix = f"e{weights.efficiency_weight}_p{weights.price_weight}_i{weights.idle_weight}_d{weights.job_age_weight}"

    models_dir = f"sessions/{args.session}/models/{weights_prefix}/"
    log_dir = f"sessions/{args.session}/logs/{weights_prefix}/"
    plots_dir = f"sessions/{args.session}/plots/"

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
                            external_durations=job_durations_file_path,
                            plot_rewards=args.plot_rewards,
                            plots_dir=plots_dir,
                            plot_once=args.plot_once,
                            plot_eff_reward=args.plot_eff_reward,
                            plot_price_reward=args.plot_price_reward,
                            plot_idle_penalty=args.plot_idle_penalty,
                            plot_job_age_penalty=args.plot_job_age_penalty,
                            skip_plot_price=args.skip_plot_price,
                            skip_plot_online_nodes=args.skip_plot_online_nodes,
                            skip_plot_used_nodes=args.skip_plot_used_nodes,
                            skip_plot_job_queue=args.skip_plot_job_queue,
                            steps_per_iteration=STEPS_PER_ITERATION)
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
        print(f"Starting a new model training...")
        model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, ent_coef=args.ent_coef)

    iters = 0

    # If we're continuing from a saved model, adjust iters so that filenames continue sequentially
    if latest_model_file:
        try:
            # Assumes the filename format is "{models_dir}/{STEPS_PER_ITERATION * iters}.zip"
            iters = int(os.path.basename(latest_model_file).split('.')[0]) // STEPS_PER_ITERATION
        except ValueError:
            # If the filename doesn't follow expected format, default to 0
            iters = 0

    env.set_progress(iters)

    try:
        while True:
            print(f"Training iteration {iters + 1} ({STEPS_PER_ITERATION * (iters + 1)} steps)...")
            iters += 1
            if args.iter_limit > 0 and iters > args.iter_limit:
                print(f"iterations limit ({args.iter_limit}) reached: {iters}.")
                break
            try:
                model.learn(total_timesteps=STEPS_PER_ITERATION, reset_num_timesteps=False, tb_log_name=f"PPO", callback=ComputeClusterCallback())
                model.save(f"{models_dir}/{STEPS_PER_ITERATION * iters}.zip")
            except PlottingComplete:
                print("Plotting complete, terminating training...")
                break
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print("Exiting training...")
        env.close()

if __name__ == "__main__":
    main()
