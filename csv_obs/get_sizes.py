import csv
import argparse
from collections import Counter


def main(args):

    step_count = 0
    info_state_counter = Counter()

    with open(args.traj, 'r') as f:
        reader = csv.DictReader(f)
        for obs_dict in reader:
            step_count += 1
            info_state = obs_dict['info_state']
            next_info_state = obs_dict['next_info_state']
            info_state_counter.update([info_state])
            info_state_counter.update([next_info_state])
            if step_count % 100_000 == 0:
                print('Processing step', step_count, end='\r')
        print(' ' * 100, end='\r')

    # delete dummy info state
    del info_state_counter['000000000000000000000000000000']
    print(f"{len(info_state_counter)} info states")
    print(f"{step_count} trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True)
    args = parser.parse_args()
    main(args)
