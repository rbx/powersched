import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt

MAX_NODES = 100  # Maximum number of nodes
MAX_QUEUE_SIZE = 20  # Maximum number of jobs in the queue
MAX_CHANGE = 100
MAX_JOB_DURATION = 24 # job runs at most 24h (example)
MAX_JOB_AGE = 168 # job waits maximum a week
ELECTRICITY_PRICE_BASE = 20

class ComputeClusterEnv(gym.Env):
    """An toy environment for scheduling compute jobs based on electricity price predictions."""

    metadata = { 'render.modes': ['human', 'none'] }

    def render(self, mode='human'):
        self.render_mode = mode

    def env_print(self, *args):
        """Prints only if the render mode is 'human'."""
        if self.render_mode == 'human':
            print(*args)

    def __init__(self, render_mode='none'):
        super().__init__()

        self.render_mode = render_mode
        self.hour = 0
        self.week = 0

        self.on_nodes = []
        self.used_nodes = []
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
                low=0.0, # TODO: can be negative
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

        # Iteratively set the price for each subsequent hour
        for i in range(1, 24):
            # put in a simple day/night pattern with a +/- 20% variation
            initial_predicted_prices[i] = ELECTRICITY_PRICE_BASE * (1 + 0.2 * np.sin(i / 24 * 2 * np.pi))
            # Ensure prices do not go negative, temporary
            initial_predicted_prices[i] = max(1.0, initial_predicted_prices[i])

        self.hour = 0
        self.weekly_savings = 0

        self.on_nodes = []
        self.used_nodes = []
        self.job_queue_sizes = []
        self.prices = []

        self.state = {
            'nodes': initial_nodes_state,
            'job_queue': initial_job_queue,
            'predicted_prices': initial_predicted_prices,
        }

        return self.state, {}

    def step(self, action):
        self.env_print(f"week: {self.week}, hour: {self.hour}")

        new_price = ELECTRICITY_PRICE_BASE * (1 + 0.2 * np.sin((self.hour % 24) / 24 * 2 * np.pi))
        current_price = self.state['predicted_prices'][0]
        self.state['predicted_prices'] = np.roll(self.state['predicted_prices'], -1)
        self.state['predicted_prices'][-1] = new_price

        self.env_print("predicted_prices: ", np.array2string(self.state['predicted_prices'], separator=" ", max_line_width=np.inf, formatter={'float_kind': lambda x: "{:05.2f}".format(x)}))

        # reshape the 1d job_queue array into 2d for cleaner code
        # print(f"shape: {self.state['job_queue'].shape}")
        # print(f"job_queue: {self.state['job_queue']}")
        job_queue_2d = self.state['job_queue'].reshape(-1, 2)
        # print(f"shape: {job_queue_2d.shape}")
        # print(f"job_queue: {job_queue_2d}")

        # Update job queue with 0-1 new jobs. If queue is full, do nothing
        new_jobs_count = np.random.randint(0, 2)
        new_jobs_durations = np.random.randint(1, MAX_JOB_DURATION + 1, size=new_jobs_count)
        if new_jobs_count > 0:
            for i, new_job_duration in enumerate(new_jobs_durations):
                for j in range(len(job_queue_2d)):
                    if job_queue_2d[j][0] == 0:
                        job_queue_2d[j] = [new_job_duration, 0]
                        break

        self.env_print(f"new_jobs_durations: {new_jobs_durations}")
        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        self.env_print("job_queue: ", ' '.join(['[{}]'.format(' '.join(map(str, pair))) for pair in job_queue_2d]))

        action_type, action_magnitude = action # Unpack the action array
        action_magnitude += 1

        # Adjust nodes based on action
        if action_type == 0: # Decrease number of available nodes
            self.env_print(f">>> turning OFF up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(self.state['nodes'])):
                if self.state['nodes'][i] == 0: # Find the first available node and turn it off
                    self.state['nodes'][i] = -1
                    nodes_modified += 1
                    if nodes_modified == action_magnitude:  # Stop if enough nodes have been modified
                        break
        elif action_type == 1:
            self.env_print(f">>> Not touching any nodes")
            pass # maintain node count = do nothing
        elif action_type == 2: # Increase number of available nodes
            self.env_print(f">>> turning ON up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(self.state['nodes'])):
                if self.state['nodes'][i] == -1: # Find the first off node and make it available
                    self.state['nodes'][i] = 0
                    nodes_modified += 1
                    if nodes_modified == action_magnitude:  # Stop if enough nodes have been modified
                        break

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

        # Decrementing booked time for nodes
        for i, node_state in enumerate(self.state['nodes']):
            if node_state > 0:  # If node is booked
                self.state['nodes'][i] -= 1  # Decrement the booked time

        num_used_nodes = np.sum(self.state['nodes'] > 0)
        num_on_nodes = np.sum(self.state['nodes'] > -1)
        num_off_nodes = np.sum(self.state['nodes'] == -1)
        num_unprocessed_jobs = np.sum(job_queue_2d[:, 0] > 0)

        # update stats
        self.on_nodes.append(num_on_nodes)
        self.used_nodes.append(num_used_nodes)
        self.job_queue_sizes.append(num_unprocessed_jobs)
        self.prices.append(current_price)

        self.env_print(f"num_on_nodes: {num_on_nodes}, num_off_nodes: {num_off_nodes}, num_used_nodes: {num_used_nodes}")
        self.env_print(f"num_processed_jobs: {num_processed_jobs}, num_unprocessed_jobs: {num_unprocessed_jobs}")

        # rewards:
        # - cost savings (due to disabled nodes)
        # - reduced conventional energy usage
        # - cost of systems doing nothing (should not waste available resources)
        # - job queue advancement

        # Reward components
        REWARD_TURN_OFF_NODE = 0.1  # Reward for each node turned off
        PENALTY_WAITING_JOB = -0.2  # Additional penalty for each hour a job is delayed
        REWARD_PROCESSED_JOB = 5  # Reward for processing jobs under favorable prices

        average_future_price = np.mean(self.state['predicted_prices'])
        self.env_print(f"$$ current_price: {current_price}")
        self.env_print(f"$$ average_future_price: {average_future_price}")

        reward = 0
        # 1. increase reward for each turned off node, more if the current price is higher than average
        turned_off_reward = REWARD_TURN_OFF_NODE * num_off_nodes * (1 / average_future_price * current_price)
        reward += turned_off_reward
        self.env_print(f"$$ turned_off_reward: {turned_off_reward} ({REWARD_TURN_OFF_NODE} * {num_off_nodes} * (1 / {average_future_price} * {current_price}))")

        # 2. decrease reward for delayed jobs, greater if they are older. but only if there are turned off nodes
        delayed_penalty = 0
        if num_off_nodes > 0:
            for job in job_queue_2d:
                job_duration, job_age = job
                if job_duration > 0:
                    delayed_penalty += PENALTY_WAITING_JOB * job_age  # Penalize for each hour a job is delayed
        reward += delayed_penalty
        self.env_print(f"$$ delayed_penalty: {delayed_penalty}")

        # 3. increase reward if jobs were scheduled in this step and the current price is below average
        if current_price < average_future_price:
            processed_during_good_price = REWARD_PROCESSED_JOB * num_processed_jobs
            reward += processed_during_good_price
            self.env_print(f"$$ processed during favorable price: {processed_during_good_price}")

        current_daily_cost = num_on_nodes * current_price
        maximum_daily_cost = MAX_NODES * current_price
        current_saving = maximum_daily_cost - current_daily_cost
        self.weekly_savings += current_saving
        self.env_print(f"$$ current_daily_cost: {current_daily_cost}, current_saving: {current_saving}")
        self.env_print(f"$$ maximum_daily_cost: {maximum_daily_cost}, weekly_savings: {self.weekly_savings}")

        truncated = False
        terminated = False
        self.hour += 1
        if self.hour >= 168:
            # TODO: include some penalty for old jobs
            weekly_reward = self.weekly_savings / 10000
            reward += weekly_reward
            self.env_print(f"$$$$$ weekly_reward: {weekly_reward}")
            self.week += 1

            if self.render_mode == 'human':
                plot(168, self.on_nodes, self.used_nodes, self.job_queue_sizes, self.prices)

            terminated = True

        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        self.env_print("job_queue: ", ' '.join(['[{}]'.format(' '.join(map(str, pair))) for pair in job_queue_2d]))
        self.env_print(f"total reward: {reward}\n")

        # flatten job_queue again
        self.state['job_queue'] = job_queue_2d.flatten()
        # print(f"shape: {self.state['job_queue'].shape}")
        # print(f"job_queue: {self.state['job_queue']}")

        if self.render_mode == 'human':
            # go slow to be able to read stuff in human mode
            time.sleep(1)

        return self.state, reward, terminated, truncated, {}


def plot(num_hours, on_nodes, used_nodes, job_queue_sizes, prices):
    hours = np.arange(num_hours)

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Configure the first y-axis (left side) for electricity prices
    color = 'tab:blue'
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Electricity Price ($/MWh)', color=color)
    ax1.plot(hours, prices, color=color, label='Electricity Price ($/MWh)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create the second y-axis (right side) for the node counts and job queue sizes
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Count', color=color)
    ax2.bar(hours - 0.3, on_nodes, width=0.3, color='orange', label='Online Nodes')
    ax2.bar(hours, used_nodes, width=0.3, color='green', label='Used Nodes')
    ax2.bar(hours + 0.3, job_queue_sizes, width=0.3, color='red', label='Job Queue Size')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Electricity Price and Compute Cluster Usage Over Time')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()
