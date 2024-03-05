import gymnasium as gym
from gymnasium import spaces
import numpy as np

MAX_NODES = 100  # Maximum number of nodes
MAX_QUEUE_SIZE = 10  # Maximum number of jobs in the queue

class ComputeClusterEnv(gym.Env):
    """An toy environment for scheduling compute jobs based on electricity price predictions."""

    metadata = { 'render.modes': ['human'] }

    def __init__(self):
        super().__init__()

        self.hours_passed = 0

        # actions: - change number of available nodes: 0: decrease, 1: maintain, 2: increase
        self.action_space = spaces.Discrete(3)

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

        self.state = {
            'nodes': initial_nodes_state,
            'job_queue': initial_job_queue,
            'predicted_prices': initial_predicted_prices,
        }

        return self.state, {}

    def step(self, action):
        self.hours_passed += 1

        new_price = self.state['predicted_prices'][0] + np.random.uniform(low=-10.0, high=10.0)
        self.state['predicted_prices'] = np.roll(self.state['predicted_prices'], -1)
        self.state['predicted_prices'][-1] = new_price

        # Update job queue with 0-1 new jobs. If queue is full, do nothing
        new_jobs_count = np.random.randint(0, 1)
        new_jobs_duration = np.random.randint(1, 10, size=new_jobs_count)
        if new_jobs_count > 0:
            for i, job_duration in enumerate(self.state['job_queue']):
                if job_duration == 0:
                    self.state['job_queue'][i] = new_jobs_duration

        # Adjust nodes based on action
        if action == 0: # Decrease number of available nodes
            for i in range(len(self.state['nodes'])):
                if self.state['nodes'][i] == 0: # Find the first available node and turn it off
                    self.state['nodes'][i] = -1
                    break
        if action == 1:
            pass # maintain node count = do nothing
        elif action == 2: # Increase number of available nodes
            for i in range(len(self.state['nodes'])):
                if self.state['nodes'][i] == -1: # Find the first off node and make it available
                    self.state['nodes'][i] = 0
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
        PENALTY_UNPROCESSED_JOB = -1  # Penalty for each unprocessed job in the queue
        BONUS_PROCESSED_JOB = 1  # Bonus for each processed job

        # Calculate the number of off nodes for the reward
        num_off_nodes = np.sum(self.state['nodes'] == -1)

        # Calculate the number of unprocessed jobs for the penalty
        num_unprocessed_jobs = np.sum(self.state['job_queue'] > 0)

        # Calculate the number of processed jobs for the bonus
        # Assuming we track processed jobs within this step
        num_processed_jobs = new_jobs_count - num_unprocessed_jobs

        # Calculate the reward
        reward = 0
        reward += REWARD_TURN_OFF_NODE * num_off_nodes * new_price
        reward += PENALTY_UNPROCESSED_JOB * num_unprocessed_jobs
        reward += BONUS_PROCESSED_JOB * num_processed_jobs

        truncated = False
        terminated = False
        if self.hours_passed > 168:
            terminated = True

        return self.state, reward, terminated, truncated, {}
