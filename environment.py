import time

from gymnasium import spaces
import gymnasium as gym
import numpy as np
from colorama import init, Fore

from prices import Prices
from weights import Weights
from plot import plot, plot_reward

init()  # Initialize colorama

WEEK_HOURS = 168

MAX_NODES = 100  # Maximum number of nodes
MAX_QUEUE_SIZE = 100  # Maximum number of jobs in the queue
MAX_CHANGE = 100
MAX_JOB_DURATION = 1 # maximum job runtime
MAX_JOB_AGE = WEEK_HOURS # job waits maximum a week
MAX_NEW_JOBS_PER_HOUR = 5

COST_IDLE = 150 # Watts
COST_USED = 450 # Watts

COST_IDLE_MW = COST_IDLE / 1000000 # MW
COST_USED_MW = COST_USED / 1000000 # MW

EPISODE_HOURS = WEEK_HOURS * 2

# possible rewards:
# - cost savings (due to disabled nodes)
# - reduced conventional energy usage
# - cost of systems doing nothing (should not waste available resources)
# - job queue advancement
# Reward components
# REWARD_TURN_OFF_NODE = 0.1 # Reward for each node turned off
# REWARD_PROCESSED_JOB = 1   # Reward for processing jobs under favorable prices
# PENALTY_NODE_CHANGE = -0.05 # Penalty for changing node state
PENALTY_IDLE_NODE = -0.1 # Penalty for idling nodes
PENALTY_WAITING_JOB = -0.1  # Penalty for each hour a job is delayed

# TODO:
# - should the observation space be normalized too?

class PlottingComplete(Exception):
    """Raised when plotting is complete and the application should terminate."""
    pass

class ComputeClusterEnv(gym.Env):
    """An toy environment for scheduling compute jobs based on electricity price predictions."""

    metadata = { 'render.modes': ['human', 'none'] }

    def render(self, mode='human'):
        self.render_mode = mode

    def set_progress(self, iterations):
        self.current_step = iterations * self.steps_per_iteration
        self.current_episode = self.current_step // EPISODE_HOURS
        self.current_week = self.current_step // WEEK_HOURS
        self.next_plot_save = iterations * self.steps_per_iteration + EPISODE_HOURS

    def env_print(self, *args):
        """Prints only if the render mode is 'human'."""
        if self.render_mode == 'human':
            print(*args)

    def __init__(self, weights: Weights, session, render_mode, quick_plot, external_prices, plot_rewards, plots_dir, plot_once, plot_eff_reward, plot_price_reward, plot_idle_penalty, plot_job_age_penalty, steps_per_iteration):
        super().__init__()

        self.weights = weights
        self.session = session
        self.render_mode = render_mode
        self.quick_plot = quick_plot
        self.plot_once = plot_once
        self.external_prices = external_prices
        self.plot_rewards = plot_rewards
        self.plots_dir = plots_dir
        self.plot_eff_reward = plot_eff_reward
        self.plot_price_reward = plot_price_reward
        self.plot_idle_penalty = plot_idle_penalty
        self.plot_job_age_penalty = plot_job_age_penalty
        self.steps_per_iteration = steps_per_iteration
        self.next_plot_save = self.steps_per_iteration

        self.prices = Prices(self.external_prices)

        self.current_step = 0
        self.current_episode = 0
        self.current_week = 0

        self.reset_state()

        print(f"weights: {self.weights}")
        print(f"prices.MAX_PRICE: {self.prices.MAX_PRICE:.2f}, prices.MIN_PRICE: {self.prices.MIN_PRICE:.2f}")
        print(f"Price Statistics: {self.prices.get_price_stats()}")
        # self.prices.plot_price_histogram(use_original=False)

        min_cost = self.power_cost(0, MAX_NODES, self.prices.MAX_PRICE)
        max_cost = self.power_cost(MAX_NODES, 0, self.prices.MIN_PRICE)

        self.min_efficiency_reward = 0  # Worst case: nodes running but no work being done
        efficiency_with_work = MAX_NODES / (max_cost + 1e-6)
        self.max_efficiency_reward = max(1.0, efficiency_with_work)
        # self.min_efficiency_reward = self.reward_efficiency(0, min_cost)
        # self.max_efficiency_reward = self.reward_efficiency(MAX_NODES, max_cost)

        self.min_price_reward = 0
        # self.min_price_reward = self.reward_price(self.prices.MAX_PRICE, self.prices.MIN_PRICE, MAX_NEW_JOBS_PER_HOUR)  # Worst case: highest current price, lowest future price, max jobs processed
        self.max_price_reward = self.reward_price(self.prices.MIN_PRICE, self.prices.MAX_PRICE, MAX_NEW_JOBS_PER_HOUR)  # Best case: lowest current price, highest future price, max jobs processed

        self.min_idle_penalty = self.penalty_idle(0)
        self.max_idle_penalty = self.penalty_idle(MAX_NODES)

        self.min_job_age_penalty = -0.0
        self.max_job_age_penalty = PENALTY_WAITING_JOB * MAX_JOB_AGE * MAX_QUEUE_SIZE

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
        self.reset_state()

        self.state = {
            # Initialize all nodes to be 'online but free' (0)
            'nodes': np.zeros(MAX_NODES, dtype=np.int32),
            # Initialize job queue to be empty
            'job_queue': np.zeros((MAX_QUEUE_SIZE * 2), dtype=np.int32),
            # Initialize predicted prices array
            'predicted_prices': self.prices.get_predicted_prices(),
        }

        self.baseline_state = {
            'nodes': np.zeros(MAX_NODES, dtype=np.int32),
            'job_queue': np.zeros((MAX_QUEUE_SIZE * 2), dtype=np.int32),
        }

        return self.state, {}

    def reset_state(self):
        self.current_hour = 0
        self.episode_reward = 0

        self.on_nodes = []
        self.used_nodes = []
        self.job_queue_sizes = []
        self.price_stats = []

        self.eff_rewards = []
        self.price_rewards = []
        self.idle_penalties = []
        self.job_age_penalties = []

        self.total_cost = 0
        self.baseline_cost = 0
        self.baseline_cost_off = 0

    def step(self, action):
        self.env_print(Fore.GREEN + f"\nepisode: {self.current_episode}, week: {self.current_week}, step: {self.current_step}, hour: {self.current_hour}")
        self.env_print(Fore.RESET)
        self.current_step += 1

        self.state['predicted_prices'] = self.prices.get_predicted_prices()
        current_price = self.state['predicted_prices'][0]
        self.env_print("predicted_prices: ", np.array2string(self.state['predicted_prices'], separator=" ", max_line_width=np.inf, formatter={'float_kind': lambda x: "{:05.2f}".format(x)}))

        # reshape the 1d job_queue array into 2d for cleaner code
        job_queue_2d = self.state['job_queue'].reshape(-1, 2)

        # Decrement booked time for nodes and complete running jobs
        self.process_ongoing_jobs(self.state['nodes'])

        # Update job queue with new jobs. If queue is full, do nothing
        new_jobs_count = np.random.randint(0, MAX_NEW_JOBS_PER_HOUR + 1)
        new_jobs_durations = np.random.randint(1, MAX_JOB_DURATION + 1, size=new_jobs_count)
        self.add_new_jobs(job_queue_2d, new_jobs_count, new_jobs_durations)

        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        self.env_print("job_queue: ", ' '.join(['[{}]'.format(' '.join(map(str, pair))) for pair in job_queue_2d if not np.array_equal(pair, np.array([0, 0]))]))
        self.env_print(f"new_jobs_durations: {new_jobs_durations}")

        action_type, action_magnitude = action # Unpack the action array
        action_magnitude += 1

        num_node_changes = self.adjust_nodes(action_type, action_magnitude, self.state['nodes'])

        # assign jobs to available nodes
        num_processed_jobs = self.assign_jobs_to_available_nodes(job_queue_2d, self.state['nodes'])

        num_used_nodes = np.sum(self.state['nodes'] > 0)
        num_on_nodes = np.sum(self.state['nodes'] > -1)
        num_off_nodes = np.sum(self.state['nodes'] == -1)
        num_idle_nodes = num_on_nodes - num_used_nodes
        num_unprocessed_jobs = np.sum(job_queue_2d[:, 0] > 0)
        average_future_price = np.mean(self.state['predicted_prices'])

        # update stats
        self.on_nodes.append(num_on_nodes)
        self.used_nodes.append(num_used_nodes)
        self.job_queue_sizes.append(num_unprocessed_jobs)
        self.price_stats.append(current_price)

        # baseline
        baseline_cost, baseline_cost_off = self.baseline_step(current_price, new_jobs_count, new_jobs_durations)
        self.baseline_cost += baseline_cost
        self.baseline_cost_off += baseline_cost_off

        # calculate reward
        reward, step_cost = self.calculate_reward(num_used_nodes, num_idle_nodes, current_price, average_future_price, num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d, num_unprocessed_jobs)
        self.episode_reward = self.episode_reward + reward
        self.total_cost += step_cost
        self.env_print(f"> step cost: €{step_cost:.4f}")

        # print stats
        self.env_print(f"nodes: ON: {num_on_nodes}, OFF: {num_off_nodes}, used: {num_used_nodes}, IDLE: {num_idle_nodes}. node changes: {num_node_changes}")
        self.env_print("nodes: ", np.array2string(self.state['nodes'], separator=" ", max_line_width=np.inf))
        self.env_print(f"price: current: {current_price}, average future: {average_future_price:.4f}")
        self.env_print(f"processed jobs: {num_processed_jobs}, unprocessed jobs: {num_unprocessed_jobs}")
        self.env_print("job queue: ", ' '.join(['[{}]'.format(' '.join(map(str, pair))) for pair in job_queue_2d if not np.array_equal(pair, np.array([0, 0]))]))
        self.env_print(f"step reward: {reward:.4f}, episode reward: {self.episode_reward:.4f}")

        if self.plot_rewards:
            plot_reward(self, num_used_nodes, num_idle_nodes, current_price, num_off_nodes, average_future_price, num_processed_jobs, num_node_changes, job_queue_2d, MAX_NODES)

        truncated = False
        terminated = False
        self.current_hour += 1
        if self.current_hour >= EPISODE_HOURS:
            self.current_week += (EPISODE_HOURS // WEEK_HOURS)
            self.current_episode += 1

            # self.env_print(f"\nepisode: {self.current_episode}, week: {self.current_week}, step: {self.current_step}, hour: {self.current_hour}\nepisode reward: {self.episode_reward:.4f}\n")

            # # sparse reward
            # if self.total_cost < self.baseline_cost_off:
            #     cost_improvement = self.baseline_cost_off - self.total_cost
            #     # Scale the reward to be roughly 10% of total episode reward when cost savings are significant
            #     baseline_reward = 0.1 * (cost_improvement / self.baseline_cost_off) * EPISODE_HOURS
            #     self.env_print(f"$$$BASELINE: {baseline_reward:.4f} (cost savings: €{cost_improvement:.2f})")
            #     reward += baseline_reward
            #     self.env_print(f"TOTAL (dense + sparse) reward: {reward:.4f}")

            if self.render_mode == 'human':
                print(f"total_cost: {self.total_cost}")
                print(f"baseline_cost: {self.baseline_cost:.4f}")
                print(f"baseline_cost_off: {self.baseline_cost_off:.4f}")
                plot(self, EPISODE_HOURS, MAX_NODES, False, True, self.current_step)
                if self.plot_once:
                    raise PlottingComplete
            else:
                if self.current_step > self.next_plot_save:
                    plot(self, EPISODE_HOURS, MAX_NODES, True, False, self.current_step)
                    self.next_plot_save += self.steps_per_iteration
                    print(self.next_plot_save)

            terminated = True

        # flatten job_queue again
        self.state['job_queue'] = job_queue_2d.flatten()

        if self.render_mode == 'human':
            # go slow to be able to read stuff in human mode
            if not self.quick_plot:
                time.sleep(1)

        return self.state, reward, terminated, truncated, {}

    def process_ongoing_jobs(self, nodes):
        for i, node_state in enumerate(nodes):
            if node_state > 0: # If node is booked
                nodes[i] -= 1 # Decrement the booked time

    def add_new_jobs(self, job_queue_2d, new_jobs_count, new_jobs_durations):
        if new_jobs_count > 0:
            for i, new_job_duration in enumerate(new_jobs_durations):
                for j in range(len(job_queue_2d)):
                    if job_queue_2d[j][0] == 0:
                        job_queue_2d[j] = [new_job_duration, 0]
                        break

    def adjust_nodes(self, action_type, action_magnitude, nodes):
        num_node_changes = 0

        # Adjust nodes based on action
        if action_type == 0: # Decrease number of available nodes
            self.env_print(f"   >>> turning OFF up to {action_magnitude} nodes")
            nodes_modified = 0
            for i in range(len(nodes)):
                if nodes[i] == 0: # Find the first available node and turn it off
                    nodes[i] = -1
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
            for i in range(len(nodes)):
                if nodes[i] == -1: # Find the first off node and make it available
                    nodes[i] = 0
                    nodes_modified += 1
                    num_node_changes += 1
                    if nodes_modified == action_magnitude:  # Stop if enough nodes have been modified
                        break

        return num_node_changes

    def assign_jobs_to_available_nodes(self, job_queue_2d, nodes):
        num_processed_jobs = 0

        for i in range(len(job_queue_2d)):
            job_duration = job_queue_2d[i][0]
            if job_duration > 0:  # If there's a job to process
                job_launched = False
                for j in range(len(nodes)):
                    node_state = nodes[j]
                    if node_state == 0:  # If node is free
                        nodes[j] = job_duration  # Book the node for the duration of the job
                        job_queue_2d[i][0] = 0  # Set the duration of the job to 0, marking it as processed
                        job_queue_2d[i][1] = 0  # ... and the age
                        job_launched = True
                        num_processed_jobs += 1
                        break

                if not job_launched:
                    # Increment the age of the job if it wasn't processed
                    job_queue_2d[i][1] += 1

        return num_processed_jobs

    def baseline_step(self, current_price, new_jobs_count, new_jobs_durations):
        job_queue_2d = self.baseline_state['job_queue'].reshape(-1, 2)

        self.process_ongoing_jobs(self.baseline_state['nodes'])

        self.add_new_jobs(job_queue_2d, new_jobs_count, new_jobs_durations)

        self.assign_jobs_to_available_nodes(job_queue_2d, self.baseline_state['nodes'])

        num_used_nodes = np.sum(self.baseline_state['nodes'] > 0)
        num_on_nodes = np.sum(self.baseline_state['nodes'] > -1)
        num_idle_nodes = num_on_nodes - num_used_nodes

        self.baseline_state['job_queue'] = job_queue_2d.flatten()

        baseline_cost = self.power_cost(num_used_nodes, num_idle_nodes, current_price)
        self.env_print(f"> baseline_cost: €{baseline_cost:.4f} | used nodes: {num_used_nodes}, idle nodes: {num_idle_nodes}")
        baseline_cost_off = self.power_cost(num_used_nodes, 0, current_price)
        self.env_print(f"> baseline_cost_off: €{baseline_cost_off:.4f} | used nodes: {num_used_nodes}, idle nodes: 0")
        return baseline_cost, baseline_cost_off

    def calculate_reward(self, num_used_nodes, num_idle_nodes, current_price, average_future_price, num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d, num_unprocessed_jobs):
        # 0. Efficiency. Reward calculation based on Workload (used nodes) (W) / Cost (C)
        total_cost = self.power_cost(num_used_nodes, num_idle_nodes, current_price)
        efficiency_reward_norm = self.reward_efficiency_normalized(num_used_nodes, num_idle_nodes, num_unprocessed_jobs, total_cost)

        # 1. increase reward for each turned off node, more if the current price is higher than average
        # turned_off_reward = self.reward_turned_off(num_off_nodes, average_future_price, current_price)

        # 2. increase reward if jobs were scheduled in this step and the current price is below average
        price_reward_norm = self.reward_price_normalized(current_price, average_future_price, num_processed_jobs)

        # 3. penalize delayed jobs, more if they are older. but only if there are turned off nodes
        job_age_penalty_norm = self.penalty_job_age_normalized(num_off_nodes, job_queue_2d)

        # 4. penalty to avoid too frequent node state changes
        # node_change_penalty = self.penalty_node_changes(num_node_changes)

        # 5. penalty for idling nodes
        idle_penalty_norm = self.penalty_idle_normalized(num_idle_nodes)

        efficiency_reward_weighted = self.weights.efficiency_weight * efficiency_reward_norm
        self.env_print(f"$$$EFF: {efficiency_reward_weighted:.4f} = {efficiency_reward_norm:.4f} x {self.weights.efficiency_weight}")
        price_reward_weighted = self.weights.price_weight * price_reward_norm
        self.env_print(f"$$$PRICE: {price_reward_weighted:.4f} = {price_reward_norm:.4f} x {self.weights.price_weight}")
        idle_penalty_weighted = self.weights.idle_weight * idle_penalty_norm
        self.env_print(f"$$$IDLE: {idle_penalty_weighted:.4f} = {idle_penalty_norm:.4f} x {self.weights.idle_weight}")
        job_age_penalty_weighted = self.weights.job_age_weight * job_age_penalty_norm
        self.env_print(f"$$$AGE: {job_age_penalty_weighted:.4f} = {job_age_penalty_norm:.4f} x {self.weights.job_age_weight}")

        self.eff_rewards.append(efficiency_reward_norm * 100)
        self.price_rewards.append(price_reward_norm * 100)
        self.idle_penalties.append(idle_penalty_norm * 100)
        self.job_age_penalties.append(job_age_penalty_norm * 100)

        reward = (
            efficiency_reward_weighted
            # + 0.0 * turned_off_reward
            + price_reward_weighted
            + idle_penalty_weighted
            + job_age_penalty_weighted
        )

        return reward, total_cost

    def power_cost(self, num_used_nodes, num_idle_nodes, current_price):
        idle_cost = COST_IDLE_MW * current_price * num_idle_nodes
        usage_cost = COST_USED_MW * current_price * num_used_nodes
        total_cost = idle_cost + usage_cost
        # self.env_print(f"$$EFF total_cost: {total_cost} = idle_cost: {idle_cost} + usage_cost: {usage_cost}")
        return total_cost

    def reward_efficiency(self, num_used_nodes, total_cost):
        return num_used_nodes / (total_cost + 1e-6)

    def reward_efficiency_normalized(self, num_used_nodes, num_idle_nodes, num_unprocessed_jobs, total_cost):
        current_reward = 0
        if num_used_nodes + num_idle_nodes == 0:
            if num_unprocessed_jobs == 0:
                current_reward = 1
                self.env_print(f"$$E efficiency_reward: {current_reward:.4f} (nothing is used and no outstanding jobs)")
            else:
                current_reward = np.clip(1.0 / np.log1p(num_unprocessed_jobs), a_min=None, a_max=1.0)
                self.env_print(f"$$E efficiency_reward: {current_reward:.4f} (nothing is used and {num_unprocessed_jobs} outstanding jobs)")
        else:
            current_reward = self.reward_efficiency(num_used_nodes, total_cost)
            self.env_print(f"$$E efficiency_reward (w/c): {current_reward:.4f} (= {num_used_nodes} / (€{total_cost:.2f} + 1e-6)), num_used_nodes: {num_used_nodes}")
            current_reward = normalize(current_reward, self.min_efficiency_reward, self.max_efficiency_reward)
            self.env_print(f"$E normalized_reward: {current_reward:.4f} | min_efficiency_reward: {self.min_efficiency_reward:.4f}, max_efficiency_reward: {self.max_efficiency_reward:.4f}")
        return current_reward

    # def reward_turned_off(self, num_off_nodes, average_future_price, current_price):
    #     turned_off_reward = REWARD_TURN_OFF_NODE * num_off_nodes * (1 / average_future_price * current_price)
    #     # self.env_print(f"$$ turned_off_reward: {turned_off_reward} ({REWARD_TURN_OFF_NODE} * {num_off_nodes} * (1 / {average_future_price} * {current_price}))")
    #     return turned_off_reward

    def reward_price(self, current_price, average_future_price, num_processed_jobs):
        history_avg, future_avg = self.prices.get_price_context()

        if history_avg is not None:
            # We have some history - use both past and future
            context_avg = (history_avg + future_avg) / 2
            price_diff = context_avg - current_price
        else:
            # No history yet - fall back to just using future prices
            price_diff = average_future_price - current_price

        price_reward = price_diff * num_processed_jobs
        # if current_price < average_future_price:
            # price_reward = REWARD_PROCESSED_JOB * num_processed_jobs
        return price_reward

    def reward_price_normalized(self, current_price, average_future_price, num_processed_jobs):
        current_reward = self.reward_price(current_price, average_future_price, num_processed_jobs)
        self.env_print(f"$$P: price_reward: {current_reward:.4f} | current_price: €{current_price:.2f}, average_future_price: €{average_future_price:.2f}, num_processed_jobs: {num_processed_jobs}")
        if num_processed_jobs == 0:
            return 0
        normalized_reward = normalize(current_reward, self.min_price_reward, self.max_price_reward)
        self.env_print(f"$P: normalized_price_reward: {normalized_reward:.4f} | min_price_reward: {self.min_price_reward:.2f}, max_price_reward: {self.max_price_reward:.2f}")
        # Clip the value to ensure it's between 0 and 1
        # normalized_reward = np.clip(normalized_reward, 0, 1)
        return normalized_reward

    # def penalty_node_changes(self, num_node_changes):
    #     node_change_penalty = PENALTY_NODE_CHANGE * num_node_changes
    #     # self.env_print(f"$$ node change penalty: {node_change_penalty}")
    #     return node_change_penalty

    def penalty_idle(self, num_idle_nodes):
        idle_penalty = PENALTY_IDLE_NODE * num_idle_nodes
        return idle_penalty

    def penalty_idle_normalized(self, num_idle_nodes):
        current_penalty = self.penalty_idle(num_idle_nodes)
        self.env_print(f"$$I current_penalty: {current_penalty:.4f} (num_idle_nodes: {num_idle_nodes})")
        normalized_penalty = - normalize(current_penalty, self.min_idle_penalty, self.max_idle_penalty)
        self.env_print(f"$I normalized_penalty: {normalized_penalty:.4f} | min_penalty: {self.min_idle_penalty}, max_penalty: {self.max_idle_penalty}")
        # Clip the value to ensure it's between 0 and 1
        normalized_penalty = np.clip(normalized_penalty, -1, 0)
        self.env_print(f"$I CLIPPED normalized_penalty: {normalized_penalty}")
        return normalized_penalty

    def penalty_job_age(self, num_off_nodes, job_queue_2d):
        job_age_penalty = 0
        if num_off_nodes > 0:
            for job in job_queue_2d:
                job_duration, job_age = job
                if job_duration > 0:
                    job_age_penalty += PENALTY_WAITING_JOB * job_age
        return job_age_penalty

    def penalty_job_age_normalized(self, num_off_nodes, job_queue_2d):
        current_penalty = self.penalty_job_age(num_off_nodes, job_queue_2d)
        self.env_print(f"$$D current_penalty: {current_penalty:.4f}")
        normalized_penalty = - normalize(current_penalty, self.min_job_age_penalty, self.max_job_age_penalty)
        self.env_print(f"$D normalized_penalty: {normalized_penalty:.4f} | min_penalty: {self.min_job_age_penalty}, max_penalty: {self.max_job_age_penalty}")
        normalized_penalty = np.clip(normalized_penalty, -1, 0)
        self.env_print(f"$D CLIPPED normalized_penalty: {normalized_penalty}")
        return normalized_penalty

def normalize(current, minimum, maximum):
    return (current - minimum) / (maximum - minimum)
