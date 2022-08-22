import pickle
import argparse
from collections import Counter

def main(args):
    with open(f'trajectories/traj-mixed-{args.num}.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    state_counter = Counter()
    
    for traj in trajectories:
        for step in traj:
            state, _, _, _ = step
            if state.is_chance_node():
                continue
            info_string = state.information_state_string()
            state_counter.update([info_string])

    states, counts = zip(*state_counter.most_common())
    print(counts)
    print(len(counts))
    print(sum(counts))
    for state in states[:30]:
        print(state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=936)
    args = parser.parse_args()
    main(args)