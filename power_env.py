import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

MAX_NODES = 100  # Maximum number of nodes
MAX_QUEUE_SIZE = 20  # Maximum number of jobs in the queue
MAX_CHANGE = 10

class ComputeClusterEnv(gym.Env):
    """An toy environment for scheduling compute jobs based on electricity price predictions."""

    metadata = { 'render.modes': ['human'] }

    def __init__(self):
        super().__init__()

        self.hours_passed = 0
        self.weeks_passed = 0

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
                high=24, # 24h max
                shape=(MAX_NODES,),  # Correct shape to (100,)
                dtype=np.int32
            ),
            # job queue: [job duration]
            'job_queue': spaces.Box(
                low=0,
                high=10,
                shape=(MAX_QUEUE_SIZE,), # Duration of each job in the queue
                dtype=np.int32
            ),
            # predicted prices for the next 24h
            'predicted_prices': spaces.Box(
                low=0.0, # TODO: can be negative
                high=np.inf, # Assuming there's no maximum price
                shape=(24,), # Prices for the next 24 hours
                dtype=np.float32
            ),
        })


    def reset(self, seed = None, options = None):
        # Initialize all nodes to be 'online but free' (0)
        initial_nodes_state = np.zeros(MAX_NODES, dtype=np.int32)

        # Initialize job queue to be empty
        initial_job_queue = np.zeros(MAX_QUEUE_SIZE, dtype=np.int32)

        # Initialize predicted prices array
        initial_predicted_prices = np.zeros(24, dtype=np.float32)  # Pre-allocate array for 24 hours

        # Set the first hour's price randomly
        initial_predicted_prices[0] = np.random.uniform(low=0.0, high=100.0)

        # Iteratively set the price for each subsequent hour
        for i in range(1, 24):
            price_change = np.random.uniform(low=-10, high=10)
            initial_predicted_prices[i] = initial_predicted_prices[i-1] + price_change
            # Ensure prices do not go negative, temporary
            initial_predicted_prices[i] = max(0.0, initial_predicted_prices[i])

        # Convert to float32 if not already
        initial_predicted_prices = initial_predicted_prices.astype(np.float32)

        self.hours_passed = 0

        self.weekly_savings = 0

        self.state = {
            'nodes': initial_nodes_state,
            'job_queue': initial_job_queue,
            'predicted_prices': initial_predicted_prices,
        }

        return self.state, {}

    def step(self, action):
        print(f"weeks_passed: {self.weeks_passed}, hours_passed: {self.hours_passed}")

        new_price = self.state['predicted_prices'][0] + np.random.uniform(low=-10.0, high=10.0)
        current_price = self.state['predicted_prices'][0]
        self.state['predicted_prices'] = np.roll(self.state['predicted_prices'], -1)
        self.state['predicted_prices'][-1] = new_price

        print("predicted_prices: ", np.array2string(self.state['predicted_prices'], separator=" ", max_line_width=np.inf, formatter={'float_kind': lambda x: "{:05.2f}".format(x)}))

        # Update job queue with 0-1 new jobs. If queue is full, do nothing
        new_jobs_count = np.random.randint(0, 2)
        new_jobs_durations = np.random.randint(1, 11, size=new_jobs_count)
        if new_jobs_count > 0:
            for i, new_job_duration in enumerate(new_jobs_durations):
                for j, queue_job_duration in enumerate(self.state['job_queue']):
                    if queue_job_duration == 0:
                        self.state['job_queue'][j] = new_jobs_durations[i]
                        break

        print(f"new_jobs_durations: {new_jobs_durations}")
        print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        print("job_queue: ", np.array2string(self.state['job_queue'], separator=" ", max_line_width=np.inf))

        action_type, action_magnitude = action # Unpack the action array
        action_magnitude += 1
        print(f"action_type: {action_type}, action_magnitude: {action_magnitude}")

        # Adjust nodes based on action
        if action_type == 0: # Decrease number of available nodes
            print(f">>> turning OFF up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(self.state['nodes'])):
                if self.state['nodes'][i] == 0: # Find the first available node and turn it off
                    self.state['nodes'][i] = -1
                    nodes_modified += 1
                    if nodes_modified == action_magnitude:  # Stop if enough nodes have been modified
                        break
        elif action_type == 1:
            print(f">>> Not touching any nodes")
            pass # maintain node count = do nothing
        elif action_type == 2: # Increase number of available nodes
            print(f">>> turning ON up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(self.state['nodes'])):
                if self.state['nodes'][i] == -1: # Find the first off node and make it available
                    self.state['nodes'][i] = 0
                    nodes_modified += 1
                    if nodes_modified == action_magnitude:  # Stop if enough nodes have been modified
                        break

        # Processing jobs and updating node states
        for i, job_duration in enumerate(self.state['job_queue']):
            if job_duration > 0:  # If there's a job to process
                for j, node_state in enumerate(self.state['nodes']):
                    if node_state == 0:  # If node is free
                        self.state['nodes'][j] = job_duration  # Book the node for the duration of the job
                        self.state['job_queue'][i] = 0  # Mark the job as processed
                        break

        # Decrementing booked time for nodes
        for i, node_state in enumerate(self.state['nodes']):
            if node_state > 0:  # If node is booked
                self.state['nodes'][i] -= 1  # Decrement the booked time

        # rewards:
        # - cost savings (due to disabled nodes)
        # - reduced conventional energy usage
        # - cost of systems doing nothing (should not waste available resources)
        # - job queue advancement

        # Initialize reward components
        REWARD_TURN_OFF_NODE = 0.1  # Reward for each node turned off
        PENALTY_UNPROCESSED_JOB = -10  # Penalty for each unprocessed job in the queue
        BONUS_PROCESSED_JOB = 1  # Bonus for each processed job

        # Calculate the number of off nodes for the reward
        num_on_nodes = np.sum(self.state['nodes'] > -1)
        num_off_nodes = np.sum(self.state['nodes'] == -1)

        # Calculate the number of unprocessed jobs for the penalty
        num_unprocessed_jobs = np.sum(self.state['job_queue'] > 0)

        # Calculate the number of processed jobs for the bonus
        # Assuming we track processed jobs within this step
        num_processed_jobs = new_jobs_count - num_unprocessed_jobs

        print(f"num_off_nodes: {num_on_nodes}")
        print(f"num_off_nodes: {num_off_nodes}")
        print(f"num_unprocessed_jobs: {num_unprocessed_jobs}")
        print(f"num_processed_jobs: {num_processed_jobs}")

        # Calculate the reward
        reward = 0
        reward += REWARD_TURN_OFF_NODE * num_off_nodes# * current_price_price
        if num_off_nodes > 0:
            reward += PENALTY_UNPROCESSED_JOB * num_unprocessed_jobs
        reward += BONUS_PROCESSED_JOB * num_processed_jobs

        current_daily_cost = num_on_nodes * current_price
        maximum_daily_cost = MAX_NODES * current_price
        current_saving = maximum_daily_cost - current_daily_cost
        self.weekly_savings += current_saving
        print(f"$$ current_daily_cost: {current_daily_cost}")
        print(f"$$ maximum_daily_cost: {maximum_daily_cost}")
        print(f"$$ current_saving: {current_saving}")
        print(f"$$ weekly_savings: {self.weekly_savings}")

        truncated = False
        terminated = False
        self.hours_passed += 1
        if self.hours_passed >= 168:
            reward += self.weekly_savings / 10
            self.weeks_passed += 1
            terminated = True

        print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        print("job_queue: ", np.array2string(self.state['job_queue'], separator=" ", max_line_width=np.inf))
        print(f"reward: {reward} ({REWARD_TURN_OFF_NODE * num_off_nodes} + {PENALTY_UNPROCESSED_JOB * num_unprocessed_jobs} + {BONUS_PROCESSED_JOB * num_processed_jobs})\n")

        time.sleep(1)

        return self.state, reward, terminated, truncated, {}
