from sampler_jobs import jobs_sampler
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze job durations from a log file.')
    parser.add_argument('--file-path', default="", help='Path to the job duration log file')
    parser.add_argument('--bin-minutes', type=int, default=60, help='Bin duration in minutes (default: 60)')
    args = parser.parse_args()

    result = jobs_sampler.get_jobs(args.file_path, args.bin_minutes)
    if result is None:
        print("No jobs data found.")
        sys.exit(1)

    for hour, jobs in result.items():
        print(f"hour: {hour}, jobs: {len(jobs)}")
        # for i, job in enumerate(jobs):
            # print(f"  Job {i+1}: Nodes={job['nnodes']}, Cores per node={job['cores_per_node']}, Duration={job['duration_hours']} hours")
