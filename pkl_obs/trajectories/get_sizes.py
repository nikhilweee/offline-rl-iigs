import pickle
import argparse
from collections import Counter


def main(args):
    with open(args.traj, "rb") as f:
        trajectories = pickle.load(f)

    state_counter = Counter()
    info_state_counter = Counter()
    step_counter = 0

    for traj in trajectories:
        for step in traj:
            step_counter += 1
            state, action, _, _ = step
            state_string = str(state.history())
            state_counter.update([state_string])
            next_state = state.child(action)
            next_state_string = str(next_state.history())
            state_counter.update([next_state_string])
            if state.is_chance_node():
                continue
            info_string = state.information_state_string()
            info_state_counter.update([info_string])

    print(f"{len(state_counter)} states, {len(info_state_counter)} info states")
    print(f"{len(trajectories)} trajectories, {step_counter} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traj", help="path to trajectory pkl file")
    args = parser.parse_args()
    main(args)
