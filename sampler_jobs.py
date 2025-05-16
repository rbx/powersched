import re
import datetime
import math
from collections import defaultdict

class DurationSampler:
    def get_jobs(self, filepath, bin_minutes=60):
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

        return dict(sorted(jobs.items()))

jobs_sampler = DurationSampler()