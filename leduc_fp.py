import sys
import pickle
import argparse
import pyspiel
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms.fictitious_play import XFPSolver, _policy_dict_at_state

class SampleXFPSolver(XFPSolver):

    def sample_episode_trajectory(self, state, trajectory):
        if state.is_terminal():
            return trajectory
        elif state.is_chance_node():
            outcomes = []
            probs = []
            for action, prob in state.chance_outcomes():
                outcomes.append(action)
                probs.append(prob)
            outcome = np.random.choice(outcomes, p=probs)
            state.apply_action(outcome)
            return self.sample_episode_trajectory(state, trajectory)
        else:
            player = state.current_player()
            state_policy = _policy_dict_at_state(self._policies[player], state)
            actions = []
            probs = []
            for action in state_policy:
                actions.append(action)
                probs.append(state_policy[action])
            action = np.random.choice(actions, p=probs)
            trajectory.append([
                np.array(state.information_state_tensor(), dtype=np.float32),
                np.array(state.legal_actions_mask(), dtype=np.int64),
                np.array(action, dtype=np.int64)
            ])
            state.apply_action(action)
            return self.sample_episode_trajectory(state, trajectory)

    def sample_episode_trajectories(self, num):
        print(f"Collecting {num} Trajectories")
        trajectories = []
        for _ in range(num):
            state = self._game.new_initial_state()
            trajectory = self.sample_episode_trajectory(state, [])
            trajectories.extend(trajectory)
        return trajectories


def main(args):
    game = pyspiel.load_game('leduc_poker', {"players": 2})
    writer = SummaryWriter(f'runs/fp/{args.mode}')
    xfp_solver = SampleXFPSolver(game)
    # train a fictitous play policy
    trajectories = []
    if args.mode == 'mixed-exp':
        samples_gen = (int(x) for x in torch.softmax(torch.arange(10, 0, -0.01).float(), -1) * args.num_episodes)
    if args.mode == 'mixed-lin':
        samples_gen = (int(x) for x in torch.ones(1000) * args.num_episodes / 1000)

    for i in range(1000):
        xfp_solver.iteration()
        conv = exploitability.exploitability(game, xfp_solver.average_policy())
        writer.add_scalar('conv', conv, i + 1)
        if args.mode in ['mixed-exp', 'mixed-lin']:
            trajectories.extend(xfp_solver.sample_episode_trajectories(next(samples_gen)))
        print("Iteration: {} Conv: {}".format(i + 1, conv))
    # sample episodes
    if args.mode == 'expert':
      trajectories.extend(xfp_solver.sample_episode_trajectories(args.num_episodes))
    # save trajectories
    with open(f'trajectories-{args.mode}-1M.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
        print(f"Saved trajectories to {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['expert', 'mixed-lin', 'mixed-exp'], required=True)
    parser.add_argument('--num_episodes', default=1_000_000)
    args = parser.parse_args()
    main(args)
