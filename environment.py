import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt

WEEK_HOURS = 168

MAX_NODES = 100  # Maximum number of nodes
MAX_QUEUE_SIZE = 100  # Maximum number of jobs in the queue
MAX_CHANGE = 100
MAX_JOB_DURATION = 1 # maximum job runtime
MAX_JOB_AGE = WEEK_HOURS # job waits maximum a week
MAX_NEW_JOBS_PER_HOUR = 5

ELECTRICITY_PRICE_BASE = 20
MAX_PRICE = 24
MIN_PRICE = 16

COST_IDLE = 150
COST_USED = 450

EPISODE_HOURS = WEEK_HOURS * 2

# Reward components
REWARD_TURN_OFF_NODE = 0.1 # Reward for each node turned off
REWARD_PROCESSED_JOB = 1   # Reward for processing jobs under favorable prices
PENALTY_WAITING_JOB = -0.1  # Penalty for each hour a job is delayed
PENALTY_NODE_CHANGE = -0.05 # Penalty for changing node state
PENALTY_IDLE_NODE = -0.1 # Penalty for idling nodes

class ComputeClusterEnv(gym.Env):
    """An toy environment for scheduling compute jobs based on electricity price predictions."""

    metadata = { 'render.modes': ['human', 'none'] }

    def render(self, mode='human'):
        self.render_mode = mode

    def set_progress(self, iterations=0, timesteps=0):
        self.current_step = iterations * timesteps
        self.episode = self.current_step // EPISODE_HOURS

    def env_print(self, *args):
        """Prints only if the render mode is 'human'."""
        if self.render_mode == 'human':
            print(*args)

    def __init__(self, render_mode='none', quick_plot=False, external_prices=None, plot_rewards=False):
        super().__init__()

        self.render_mode = render_mode
        self.quick_plot = quick_plot
        self.external_prices = external_prices
        self.plot_rewards = plot_rewards
        self.current_step = 0
        self.episode = 0

        self.hour = 0
        self.week = 0
        self.episode_reward = 0
        self.price_index = 0

        self.on_nodes = []
        self.used_nodes = []
        self.idle_nodes = []
        self.job_queue_sizes = []
        self.prices = []

        # actions: - change number of available nodes:
        #   direction: 0: decrease, 1: maintain, 2: increase
        #   num nodes: 0-9 (+1ed in the action)
        self.action_space = spaces.MultiDiscrete([3, MAX_CHANGE])

        # - predicted allocation
        # - predicted green/conventional ratio
        # - predicted usage/load
        self.observation_space = spaces.Dict({
            # - nodes: [state (available, active, off), # of hours scheduled]
            'nodes': spaces.Box(
                low=-1, # -1: off, 0: available, >0: booked for n hours
                high=MAX_JOB_DURATION, # 24h max
                shape=(MAX_NODES,),  # Correct shape to (100,)
                dtype=np.int32
            ),
            # job queue: [job duration, job age, job duration, job age, ...]
            'job_queue': spaces.Box(
                low=0,
                high=max(MAX_JOB_DURATION, MAX_JOB_AGE),  # the highest from MAX_JOB_DURATION, MAX_JOB_AGE
                shape=(MAX_QUEUE_SIZE * 2,),  # Each job is a: duration, age
                dtype=np.int32
            ),
            # predicted prices for the next 24h
            'predicted_prices': spaces.Box(
                low=-1000,
                high=1000, # Assuming there's no maximum price
                shape=(24,), # Prices for the next 24 hours
                dtype=np.float32
            ),
        })

    def reset(self, seed = None, options = None):
        # Initialize all nodes to be 'online but free' (0)
        initial_nodes_state = np.zeros(MAX_NODES, dtype=np.int32)

        # Initialize job queue to be empty
        initial_job_queue = np.zeros((MAX_QUEUE_SIZE * 2), dtype=np.int32)

        # Initialize predicted prices array
        initial_predicted_prices = np.zeros(24, dtype=np.float32)  # Pre-allocate array for 24 hours

        # Set the first hour's price to a fixed value
        initial_predicted_prices[0] = ELECTRICITY_PRICE_BASE

        if self.external_prices is not None:
            for i in range(24):
                initial_predicted_prices[i] = self.external_prices[(self.price_index + i) % len(self.external_prices)]
        else:
            # set the price for each subsequent hour
            for i in range(1, 24):
                # put in a simple day/night pattern with a +/- 20% variation
                initial_predicted_prices[i] = ELECTRICITY_PRICE_BASE * (1 + 0.2 * np.sin(i / 24 * 2 * np.pi))
                # Ensure prices do not go negative, temporary
                initial_predicted_prices[i] = max(1.0, initial_predicted_prices[i])

        self.hour = 0
        self.episode_reward = 0

        self.on_nodes = []
        self.used_nodes = []
        self.idle_nodes = []
        self.job_queue_sizes = []
        self.prices = []

        self.state = {
            'nodes': initial_nodes_state,
            'job_queue': initial_job_queue,
            'predicted_prices': initial_predicted_prices,
        }

        return self.state, {}

    def step(self, action):
        self.env_print(f"week: {self.week}, hour: {self.hour}, step: {self.current_step}, episode: {self.episode}")
        self.current_step += 1

        if self.external_prices is not None:
            new_price = self.external_prices[(self.price_index + 24) % len(self.external_prices)]
            self.price_index = (self.price_index + 1) % len(self.external_prices)
        else:
            new_price = ELECTRICITY_PRICE_BASE * (1 + 0.2 * np.sin((self.hour % 24) / 24 * 2 * np.pi))

        current_price = self.state['predicted_prices'][0]
        self.state['predicted_prices'] = np.roll(self.state['predicted_prices'], -1)
        self.state['predicted_prices'][-1] = new_price

        self.env_print("predicted_prices: ", np.array2string(self.state['predicted_prices'], separator=" ", max_line_width=np.inf, formatter={'float_kind': lambda x: "{:05.2f}".format(x)}))

        # reshape the 1d job_queue array into 2d for cleaner code
        job_queue_2d = self.state['job_queue'].reshape(-1, 2)

        # Decrement booked time for nodes and complete running jobs
        for i, node_state in enumerate(self.state['nodes']):
            if node_state > 0:  # If node is booked
                self.state['nodes'][i] -= 1  # Decrement the booked time

        # Update job queue with 0-1 new jobs. If queue is full, do nothing
        new_jobs_count = np.random.randint(0, MAX_NEW_JOBS_PER_HOUR + 1)
        new_jobs_durations = np.random.randint(1, MAX_JOB_DURATION + 1, size=new_jobs_count)
        if new_jobs_count > 0:
            for i, new_job_duration in enumerate(new_jobs_durations):
                for j in range(len(job_queue_2d)):
                    if job_queue_2d[j][0] == 0:
                        job_queue_2d[j] = [new_job_duration, 0]
                        break

        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        self.env_print("job_queue: ", ' '.join(['[{}]'.format(' '.join(map(str, pair))) for pair in job_queue_2d if not np.array_equal(pair, np.array([0, 0]))]))
        self.env_print(f"new_jobs_durations: {new_jobs_durations}")

        action_type, action_magnitude = action # Unpack the action array
        action_magnitude += 1

        num_node_changes = 0
        # Adjust nodes based on action
        if action_type == 0: # Decrease number of available nodes
            self.env_print(f"   >>> turning OFF up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(self.state['nodes'])):
                if self.state['nodes'][i] == 0: # Find the first available node and turn it off
                    self.state['nodes'][i] = -1
                    nodes_modified += 1
                    num_node_changes += 1
                    if nodes_modified == action_magnitude:  # Stop if enough nodes have been modified
                        break
        elif action_type == 1:
            self.env_print(f"   >>> Not touching any nodes")
            pass # maintain node count = do nothing
        elif action_type == 2: # Increase number of available nodes
            self.env_print(f"   >>> turning ON up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(self.state['nodes'])):
                if self.state['nodes'][i] == -1: # Find the first off node and make it available
                    self.state['nodes'][i] = 0
                    nodes_modified += 1
                    num_node_changes += 1
                    if nodes_modified == action_magnitude:  # Stop if enough nodes have been modified
                        break

        # assign jobs to available nodes
        num_processed_jobs = 0
        for i in range(len(job_queue_2d)):
            job_duration = job_queue_2d[i][0]
            if job_duration > 0:  # If there's a job to process
                job_launched = False
                for j in range(len(self.state['nodes'])):
                    node_state = self.state['nodes'][j]
                    if node_state == 0:  # If node is free
                        self.state['nodes'][j] = job_duration  # Book the node for the duration of the job
                        job_queue_2d[i][0] = 0  # Set the duration of the job to 0, marking it as processed
                        job_queue_2d[i][1] = 0  # ... and the age
                        num_processed_jobs += 1
                        job_launched = True
                        break

                if not job_launched:
                    # Increment the age of the job if it wasn't processed
                    job_queue_2d[i][1] += 1

        num_used_nodes = np.sum(self.state['nodes'] > 0)
        num_on_nodes = np.sum(self.state['nodes'] > -1)
        num_off_nodes = np.sum(self.state['nodes'] == -1)
        num_idle_nodes = num_on_nodes - num_used_nodes
        num_unprocessed_jobs = np.sum(job_queue_2d[:, 0] > 0)
        average_future_price = np.mean(self.state['predicted_prices'])

        # update stats
        self.on_nodes.append(num_on_nodes)
        self.used_nodes.append(num_used_nodes)
        self.idle_nodes.append(num_idle_nodes)
        self.job_queue_sizes.append(num_unprocessed_jobs)
        self.prices.append(current_price)

        # calculate reward
        reward = self.calculate_reward(num_used_nodes, num_idle_nodes, current_price, average_future_price, num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d)
        self.episode_reward = self.episode_reward + reward

        # print stats
        self.env_print(f"num_on_nodes: {num_on_nodes}, num_off_nodes: {num_off_nodes}, num_used_nodes: {num_used_nodes}, num_idle_nodes: {num_idle_nodes}, num_node_changes: {num_node_changes}")
        self.env_print(f"num_processed_jobs: {num_processed_jobs}, num_unprocessed_jobs: {num_unprocessed_jobs}")
        self.env_print(f"$$ current_price: {current_price}")
        self.env_print(f"$$ average_future_price: {average_future_price}")
        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        self.env_print("job_queue: ", ' '.join(['[{}]'.format(' '.join(map(str, pair))) for pair in job_queue_2d if not np.array_equal(pair, np.array([0, 0]))]))
        self.env_print(f"total reward: {reward:.15f}")
        self.env_print(f"episode reward: {self.episode_reward:.15f}\n")

        if self.plot_rewards:
            self.plot_reward(num_used_nodes, num_idle_nodes, current_price, num_off_nodes, average_future_price, num_processed_jobs, num_node_changes, job_queue_2d)

        truncated = False
        terminated = False
        self.hour += 1
        if self.hour >= EPISODE_HOURS:
            self.week += (EPISODE_HOURS // WEEK_HOURS)
            self.episode += 1

            # TODO: sparse rewards?

            if self.render_mode == 'human':
                self.plot(EPISODE_HOURS, self.on_nodes, self.used_nodes, self.job_queue_sizes, self.prices, True, self.current_step)

            terminated = True

        # flatten job_queue again
        self.state['job_queue'] = job_queue_2d.flatten()

        if self.render_mode == 'human':
            # go slow to be able to read stuff in human mode
            if not self.quick_plot:
                time.sleep(1)

        return self.state, reward, terminated, truncated, {}

    def calculate_reward(self, num_used_nodes, num_idle_nodes, current_price, average_future_price, num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d):
        # possible rewards:
        # - cost savings (due to disabled nodes)
        # - reduced conventional energy usage
        # - cost of systems doing nothing (should not waste available resources)
        # - job queue advancement

        # 0. Efficiency. Reward calculation based on Workload (W) / Cost (C)
        efficiency_reward_norm = self.reward_efficiency_normalized(num_used_nodes, num_idle_nodes, current_price)

        # 1. increase reward for each turned off node, more if the current price is higher than average
        # turned_off_reward = self.reward_turned_off(num_off_nodes, average_future_price, current_price)

        # 2. increase reward if jobs were scheduled in this step and the current price is below average
        price_reward = self.reward_price(current_price, average_future_price, num_processed_jobs)

        # 3. penalize delayed jobs, more if they are older. but only if there are turned off nodes
        # delayed_jobs_penalty = self.penalty_delayed_jobs(num_off_nodes, job_queue_2d)

        # 4. penalty to avoid too frequent node state changes
        # node_change_penalty = self.penalty_node_changes(num_node_changes)

        # 5. penalty for idling nodes
        idle_penalty_norm = self.penalty_idle_normalized(num_idle_nodes)

        reward = (
            1.0 * efficiency_reward_norm
            # + 0.0 * turned_off_reward
            + 1.0 * price_reward
            # + 0.0 * delayed_jobs_penalty
            # + 0.0 * node_change_penalty
            + 1.0 * idle_penalty_norm
        )

        return reward

    def reward_efficiency(self, num_used_nodes, num_idle_nodes, price):
        # TODO: Consider incorporating the job queue size or the efficiency of node usage.
        workload = num_used_nodes
        idle_cost = COST_IDLE * price * num_idle_nodes
        usage_cost = COST_USED * price * workload
        total_cost = idle_cost + usage_cost
        efficiency_reward = workload / (total_cost + 1e-6)
        # self.env_print(f"$$ workload: {workload}, cost: {COST_IDLE * price * num_idle_nodes + COST_USED * price * num_used_nodes}, efficiency_reward (w/c): {efficiency_reward:.15f}")
        return efficiency_reward

    def reward_turned_off(self, num_off_nodes, average_future_price, current_price):
        turned_off_reward = REWARD_TURN_OFF_NODE * num_off_nodes * (1 / average_future_price * current_price)
        # self.env_print(f"$$ turned_off_reward: {turned_off_reward} ({REWARD_TURN_OFF_NODE} * {num_off_nodes} * (1 / {average_future_price} * {current_price}))")
        return turned_off_reward

    def reward_price(self, current_price, average_future_price, num_processed_jobs):
        # TODO: consider scaling the reward based on how much below average the current price is
        price_reward = 0
        price_reward = (average_future_price - current_price) * num_processed_jobs
        # if current_price < average_future_price:
            # price_reward = REWARD_PROCESSED_JOB * num_processed_jobs
        self.env_print(f"$$ processed during favorable price: {price_reward}")
        return price_reward

    def penalty_delayed_jobs(self, num_off_nodes, job_queue_2d):
        delayed_penalty = 0
        if num_off_nodes > 0:
            for job in job_queue_2d:
                job_duration, job_age = job
                if job_duration > 0:
                    delayed_penalty += PENALTY_WAITING_JOB * job_age # Penalize for each hour a job is delayed
        # self.env_print(f"$$ delayed_penalty: {delayed_penalty}")
        return delayed_penalty

    def penalty_node_changes(self, num_node_changes):
        node_change_penalty = PENALTY_NODE_CHANGE * num_node_changes
        # self.env_print(f"$$ node change penalty: {node_change_penalty}")
        return node_change_penalty

    def penalty_idle(self, num_idle_nodes):
        idle_penalty = PENALTY_IDLE_NODE * num_idle_nodes
        # self.env_print(f"$$ idle nodes penalty: {idle_penalty}")
        return idle_penalty

    def reward_efficiency_normalized(self, num_used_nodes, num_idle_nodes, current_price):
        # normalization: logarithmic|min-max|z-score|exponential moving average|
        current_reward = self.reward_efficiency(num_used_nodes, num_idle_nodes, current_price)
        # self.env_print(f"$$ current_reward: {current_reward}")
        min_reward = self.reward_efficiency(0, MAX_NODES, MAX_PRICE)
        max_reward = self.reward_efficiency(MAX_NODES, 0, MIN_PRICE)
        normalized_reward = normalize(current_reward, min_reward, max_reward)
        # self.env_print(f"$$ normalized_reward: {normalized_reward}")
        # Clip the value to ensure it's between 0 and 1
        normalized_reward = np.clip(normalized_reward, 0, 1)
        # self.env_print(f"$$ CLIPPED normalized_reward: {normalized_reward}")
        return normalized_reward

    def penalty_idle_normalized(self, num_idle_nodes):
        current_penalty = self.penalty_idle(num_idle_nodes)
        # self.env_print(f"$$ current_penalty: {current_penalty} (num_idle_nodes: {num_idle_nodes})")
        min_penalty = self.penalty_idle(0)
        max_penalty = self.penalty_idle(MAX_NODES)
        normalized_penalty = - normalize(current_penalty, min_penalty, max_penalty)
        # self.env_print(f"$$ normalized_penalty: {normalized_penalty}")
        # Clip the value to ensure it's between 0 and 1
        normalized_penalty = np.clip(normalized_penalty, -1, 0)
        # self.env_print(f"$$ CLIPPED normalized_penalty: {normalized_penalty}")
        return normalized_penalty

    def plot(self, num_hours, on_nodes, used_nodes, job_queue_sizes, prices, use_lines, steps):
        hours = np.arange(num_hours)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Electricity Price ($/MWh)', color=color)
        ax1.plot(hours, prices, color=color, label='Electricity Price ($/MWh)')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Count', color=color)

        if use_lines:
            ax2.plot(hours, on_nodes, color='orange', label='Online Nodes')
            ax2.plot(hours, used_nodes, color='green', label='Used Nodes')
            ax2.plot(hours, job_queue_sizes, color='red', label='Job Queue Size')
        else:
            ax2.bar(hours - 0.3, on_nodes, width=0.3, color='orange', label='Online Nodes')
            ax2.bar(hours, used_nodes, width=0.3, color='green', label='Used Nodes')
            ax2.bar(hours + 0.3, job_queue_sizes, width=0.3, color='red', label='Job Queue Size')

        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, MAX_NODES)

        plt.title(f"Electricity Price and Compute Cluster Usage Over Time. model @ step {steps}")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.show()

    def plot_reward(self, num_used_nodes, num_idle_nodes, current_price, num_off_nodes, average_future_price, num_processed_jobs, num_node_changes, job_queue_2d):
        used_nodes, idle_nodes, rewards = [], [], []

        for i in range(MAX_NODES + 1):
            for j in range(MAX_NODES + 1 - i):
                reward = self.calculate_reward(i, j, current_price, average_future_price, num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d)
                used_nodes.append(i)
                idle_nodes.append(j)
                rewards.append(reward)

        plt.figure(figsize=(14, 12))

        scatter = plt.scatter(used_nodes, idle_nodes, c=rewards, cmap='viridis', s=50)
        plt.colorbar(scatter, label='Reward')

        plt.xlabel('Number of Used Nodes')
        plt.ylabel('Number of Idle Nodes')

        title = f"step: {self.current_step}, episode: {self.episode}\ncurrent_price: {current_price:.2f}, average_future_price: {average_future_price:.2f}\nnum_processed_jobs: {num_processed_jobs}, num_node_changes: {num_node_changes}, num_off_nodes: {num_off_nodes}"
        plt.title(title, fontsize=10)

        plt.plot([0, MAX_NODES], [MAX_NODES, 0], 'r--', linewidth=2, label='Max Nodes Constraint')
        plt.plot([0, MAX_NODES - num_off_nodes], [MAX_NODES - num_off_nodes, 0], 'b--', linewidth=2, label='Online/Offline Separator')

        current_reward = self.calculate_reward(num_used_nodes, num_idle_nodes, current_price, average_future_price, MAX_NODES - num_used_nodes - num_idle_nodes, num_processed_jobs, num_node_changes)
        plt.scatter(num_used_nodes, num_idle_nodes, color='red', s=100, zorder=5, label=f'Current Reward: {current_reward:.2f}')

        plt.legend()
        plt.tight_layout()
        plt.show()

def normalize(current, minimum, maximum):
    return (current - minimum) / (maximum - minimum)
