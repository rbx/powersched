import matplotlib.pyplot as plt
import numpy as np

def plot(env, num_hours, max_nodes):
    hours = np.arange(num_hours)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Electricity Price ($/MWh)', color=color)
    ax1.plot(hours, env.price_stats, color=color, label='Electricity Price ($/MWh)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Count', color=color)

    ax2.plot(hours, env.on_nodes, color='orange', label='Online Nodes')
    ax2.plot(hours, env.used_nodes, color='green', label='Used Nodes')
    ax2.plot(hours, env.job_queue_sizes, color='red', label='Job Queue Size')

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max_nodes)

    plt.title(f"session: {env.session}, step: {env.current_step}, episode: {env.current_episode}\nweights: {env.weights}\nEff: {env.eff_score}, Base_Eff: {env.baseline_eff_score}, Base_Eff_Off: {env.baseline_eff_score_off}")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.savefig(env.plots_filepath)
    print(f"Figure saved as: {env.plots_filepath}")
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

    current_reward, _ = env.calculate_reward(num_used_nodes, num_idle_nodes, current_price, average_future_price, max_nodes - num_used_nodes - num_idle_nodes, num_processed_jobs, num_node_changes)
    plt.scatter(num_used_nodes, num_idle_nodes, color='red', s=100, zorder=5, label=f'Current Reward: {current_reward:.2f}')

    plt.legend()
    plt.tight_layout()
    plt.show()