import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def plot(env, num_hours, max_nodes, save=True, show=True, suffix=""):
    hours = np.arange(num_hours)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis for electricity price
    color = 'tab:blue'
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Electricity Price (€/MWh)', color=color)
    if not env.skip_plot_price:
        ax1.plot(hours, env.price_stats, color=color, label='Electricity Price (€/MWh)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Right y-axis for counts and rewards
    ax2 = ax1.twinx()
    ax2.set_ylabel('Count / Rewards', color='tab:orange')

    # Original metrics
    if not env.skip_plot_online_nodes:
        ax2.plot(hours, env.on_nodes, color='orange', label='Online Nodes')
    if not env.skip_plot_used_nodes:
        ax2.plot(hours, env.used_nodes, color='green', label='Used Nodes')
    if not env.skip_plot_job_queue:
        ax2.plot(hours, env.job_queue_sizes, color='red', label='Job Queue Size')

    # New metrics with dashed lines
    if env.plot_eff_reward:
        ax2.plot(hours, env.eff_rewards, color='brown', linestyle='--', label='Efficiency Rewards')
    if env.plot_price_reward:
        ax2.plot(hours, env.price_rewards, color='blue', linestyle='--', label='Price Rewards')
    if env.plot_idle_penalty:
        ax2.plot(hours, env.idle_penalties, color='green', linestyle='--', label='Idle Penalties')
    if env.plot_job_age_penalty:
        ax2.plot(hours, env.job_age_penalties, color='yellow', linestyle='--', label='Job Age Penalties Penalties')

    ax2.tick_params(axis='y')
    if env.plot_idle_penalty or env.plot_job_age_penalty:
        ax2.set_ylim(-100, max_nodes)
    else:
        ax2.set_ylim(0, max_nodes)

    plt.title(f"session: {env.session}, "
              f"episode: {env.current_episode}, step: {env.current_step}\n"
              f"{env.weights}\n"
              f"Cost: €{env.total_cost:.2f}, "
              f"Base_Cost: €{env.baseline_cost:.2f} "
              f"({'+' if env.baseline_cost - env.total_cost >= 0 else '-'}"
              f"{abs(env.baseline_cost - env.total_cost):.2f}), "
              f"Base_Cost_Off: €{env.baseline_cost_off:.2f} "
              f"({'+' if env.baseline_cost_off - env.total_cost >= 0 else '-'}"
              f"{abs(env.baseline_cost_off - env.total_cost):.2f})")

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    prefix = f"e{env.weights.efficiency_weight}_p{env.weights.price_weight}_i{env.weights.idle_weight}_d{env.weights.job_age_weight}"

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{env.plots_dir}{prefix}_{suffix:09d}_{timestamp}.png")
        print(f"Figure saved as: {env.plots_dir}{prefix}_{suffix:09d}_{timestamp}.png\nExpecting next save after {env.next_plot_save + env.steps_per_iteration}")
    if show:
        plt.show()

    plt.close(fig)

def plot_reward(env, num_used_nodes, num_idle_nodes, current_price, num_off_nodes, average_future_price, num_processed_jobs, num_node_changes, job_queue_2d, max_nodes):
    used_nodes, idle_nodes, rewards = [], [], []

    for i in range(max_nodes + 1):
        for j in range(max_nodes + 1 - i):
            reward, _ = env.calculate_reward(i, j, current_price, average_future_price, num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d)
            used_nodes.append(i)
            idle_nodes.append(j)
            rewards.append(reward)

    plt.figure(figsize=(14, 12))

    scatter = plt.scatter(used_nodes, idle_nodes, c=rewards, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Reward')

    plt.xlabel('Number of Used Nodes')
    plt.ylabel('Number of Idle Nodes')

    title = f"session: {env.session}, step: {env.current_step}, episode: {env.current_episode}\ncurrent_price: {current_price:.2f}, average_future_price: {average_future_price:.2f}\nnum_processed_jobs: {num_processed_jobs}, num_node_changes: {num_node_changes}, num_off_nodes: {num_off_nodes}"
    plt.title(title, fontsize=10)

    plt.plot([0, max_nodes], [max_nodes, 0], 'r--', linewidth=2, label='Max Nodes Constraint')
    plt.plot([0, max_nodes - num_off_nodes], [max_nodes - num_off_nodes, 0], 'b--', linewidth=2, label='Online/Offline Separator')

    current_reward, _ = env.calculate_reward(num_used_nodes, num_idle_nodes, current_price, average_future_price, max_nodes - num_used_nodes - num_idle_nodes, num_processed_jobs, num_node_changes, job_queue_2d)
    plt.scatter(num_used_nodes, num_idle_nodes, color='red', s=100, zorder=5, label=f'Current Reward: {current_reward:.2f}')

    plt.legend()
    plt.tight_layout()
    plt.show()