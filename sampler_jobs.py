import re
import datetime
import math
import random
from collections import defaultdict

class DurationSampler:
    def __init__(self):
        self.jobs = {}
        self.keys = []
        self.current_position = 0

    def parse_jobs(self, filepath, bin_minutes=60):
        if filepath:
            with open(filepath, 'r') as f:
                data_text = f.read()
        else:
            print("No jobs file path provided.")
            return None

        lines = data_text.strip().split('\n')
        job_lines = [line for line in lines[2:] if line.strip() and not line.strip().startswith('-')]
        jobs = defaultdict(list)

        for line in job_lines:
            parts = re.split(r'\s+', line.strip())

            submit_time = parts[3]
            elapsed_raw = int(parts[-3])
            ncpus = int(parts[-2])
            nnodes = int(parts[-1])

            # Parse submit time
            submit_datetime = datetime.datetime.strptime(submit_time, "%Y-%m-%dT%H:%M:%S")

            # Calculate bin start time based on bin_minutes
            minutes_since_epoch = int(submit_datetime.timestamp() / 60)
            bin_start_minutes = (minutes_since_epoch // bin_minutes) * bin_minutes
            bin_start_datetime = datetime.datetime.fromtimestamp(bin_start_minutes * 60)

            # Format bin key based on bin duration
            if bin_minutes >= 1440:  # Daily or longer (≥ 24 hours)
                key = bin_start_datetime.strftime("%Y-%m-%d")
            elif bin_minutes >= 60:  # Hourly or longer
                key = bin_start_datetime.strftime("%Y-%m-%d %H:00")
            else:  # Less than an hour
                key = bin_start_datetime.strftime("%Y-%m-%d %H:%M")

            # Calculate job metrics
            cores_per_node = ncpus // nnodes
            duration_minutes = max(1, math.ceil(elapsed_raw / 60))

            jobs[key].append({"nnodes": nnodes, "cores_per_node": cores_per_node, "duration_minutes": duration_minutes})

        # Store data internally
        self.jobs = dict(sorted(jobs.items()))
        self.keys = list(self.jobs.keys())
        self.current_position = 0

        return self

    def get_all_jobs(self):
        return self.jobs

    def sample(self, n=1, wrap=True):
        if not self.keys:
            return {}

        result = {}
        bins_sampled = 0

        while bins_sampled < n:
            if self.current_position >= len(self.keys):
                if wrap:
                    self.current_position = 0
                else:
                    break

            current_key = self.keys[self.current_position]
            result[current_key] = self.jobs[current_key]
            bins_sampled += 1
            self.current_position += 1

        return result

    def reset_position(self):
        """Reset the current sampling position to the beginning."""
        self.current_position = 0
        return self

    def sample_random(self, n=1):
        """Sample n random time bins from the parsed data."""
        if not self.keys:
            return {}

        selected_keys = random.sample(self.keys, min(n, len(self.keys)))
        return {key: self.jobs[key] for key in selected_keys}

jobs_sampler = DurationSampler()
