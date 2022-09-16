import argparse
import pyspiel
import torch
import numpy as np
from observation import Observation, ObservationBuffer
from torch.utils.tensorboard import SummaryWriter
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms.fictitious_play import XFPSolver, _policy_dict_at_state


class SampleXFPSolver(XFPSolver):

    def sample_episode_trajectory(self, state, trajectory):
        if state.is_terminal():
            trajectory[-1].next_info_state = torch.zeros(30)
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
            info_state = state.information_state_tensor()
            action_mask = state.legal_actions_mask()
            reward = state.player_reward(player)
            state.apply_action(action)

            if len(trajectory) > 0 and trajectory[-1].next_info_state is None:
                trajectory[-1].next_info_state = info_state
            if not state.is_chance_node() and not state.is_terminal():
                next_info_state = state.information_state_tensor()
            else:
                next_info_state = None

            trajectory.append(Observation(
                info_state, action, action_mask, reward, next_info_state
            ))
            return self.sample_episode_trajectory(state, trajectory)

    def sample_episode_trajectories(self, num):
        print(f"Collecting {num} Episodes")
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
    trajectories = ObservationBuffer()
    if args.mode == 'mixed-exp':
        # 1000 iters: 9.1 for 1M, 6.4 for 100K, 3.6 for 10K
        # 100 iters: 11.7 for 1M, 9.1 for 100K, 6.4 for 10K
        range_start = 9.1
        range_step = - range_start / args.num_iterations
        gen_tensor = torch.arange(range_start, 0, range_step).float()
        gen_tensor = torch.softmax(gen_tensor, -1) * args.num_episodes
        samples_gen = (int(x) for x in gen_tensor)
    if args.mode == 'mixed-const':
        samples_per_iter = args.num_episodes / args.num_iterations
        gen_tensor = torch.ones(args.num_iterations) * samples_per_iter
        samples_gen = (int(x) for x in gen_tensor)

    for i in range(args.num_iterations):
        xfp_solver.iteration()
        conv = exploitability.exploitability(game, xfp_solver.average_policy())
        writer.add_scalar('conv', conv, i + 1)
        if args.mode in ['mixed-exp', 'mixed-const']:
            num_samples = next(samples_gen)
            if num_samples < 1:
                break
            trajectories.extend(xfp_solver.sample_episode_trajectories(num_samples))
        print("Iteration: {} Conv: {}".format(i + 1, conv))
    # sample episodes
    if args.mode == 'expert':
        trajectories.extend(xfp_solver.sample_episode_trajectories(args.num_episodes))
    # save trajectories
    save_path = args.traj if args.traj else f'trajectories/traj-{args.mode}-{args.num_episodes:.0e}.csv'
    trajectories.to_csv(save_path)
    print(f"Saved {len(trajectories)} episodes to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['expert', 'mixed-const', 'mixed-exp'], required=True)
    parser.add_argument('--traj', default=None)
    parser.add_argument('--num_episodes', type=int, default=10_000)
    parser.add_argument('--num_iterations', type=int, default=1_000)
    args = parser.parse_args()
    main(args)
