# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example use of the CFR algorithm on Leduc Poker."""

import pickle
import logging
import argparse
import pyspiel
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from open_spiel.python.algorithms.cfr import _CFRSolver as CFRSolver
from open_spiel.python.algorithms import exploitability

logging.basicConfig(
    format=('%(asctime)s.%(msecs)03d %(levelname)s '
            '%(module)s L%(lineno)03d | %(message)s'),
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__file__)


class SampleCFRSolver(CFRSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_states = []

    def get_state_rewards(self, state):
        if not state.is_chance_node():
            rewards = state.rewards()
        else:
            rewards = [0., 0.]
        return rewards

    def sample_traj(self, state, traj):
        """Sample a single traj."""

        if not state.is_chance_node():
            info_tensor = state.information_state_tensor()
            if info_tensor not in self.info_states:
                self.info_states.append(info_tensor)

        if state.is_terminal():
            return traj

        elif state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            action = np.random.choice(actions, p=probs)
            next_state = state.child(action)
            rewards = self.get_state_rewards(next_state)
            traj.append((state, action, next_state, [0.0, 0.0]))
            return self.sample_traj(next_state, traj)

        else:
            avg_policy = self.average_policy()
            action_probs = avg_policy.action_probabilities(state)
            actions, probs = zip(*action_probs.items())
            action = np.random.choice(actions, p=probs)
            next_state = state.child(action)
            rewards = self.get_state_rewards(next_state)
            traj.append((state, action, next_state, rewards))
            return self.sample_traj(next_state, traj)

    def sample_trajs(self, num, unique=False):
        """Sample multiple trajectories."""

        trajs = []
        for _ in range(num):
            state = self._game.new_initial_state()
            start_len = len(self.info_states)
            traj = self.sample_traj(state, [])
            end_len = len(self.info_states)
            if not unique or end_len > start_len:
                trajs.append(traj)
        num_steps = sum(len(ep) for ep in trajs)
        logger.info(f"Collected {len(trajs)} episodes, {num_steps} steps "
                    f"{len(self.info_states)} unique info states")
        return trajs


def main(args):
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    cfr_solver = SampleCFRSolver(game, regret_matching_plus=False,
                                 alternating_updates=True, linear_averaging=False)

    trajectories = []
    logger.info(f'Collecting {args.coverage} info states')
    while len(cfr_solver.info_states) < args.coverage:
        trajectories.extend(cfr_solver.sample_trajs(100, unique=True))

    writer = SummaryWriter(f'runs/cfr/{args.label}')

    logger.info(f'Running CFR')
    for i in range(args.iterations):
        cfr_solver.evaluate_and_update_policy()
        if (i+1) % args.eval_freq == 0:
            conv = exploitability.exploitability(
                game, cfr_solver.average_policy())
            logger.info(f"Iteration {i+1} Exploitability {conv}")
            writer.add_scalar("conv", conv, i+1)
            if args.mode == 'mixed':
                trajectories.extend(cfr_solver.sample_trajs(1000//(i+1)))

    if args.mode == 'expert':
        trajectories = cfr_solver.sample_trajs(714)

    num_trajs = len(trajectories)
    num_info_states = len(cfr_solver.info_states)
    num_steps = sum(len(ep) for ep in trajectories)
    logger.info(f'Collected {num_trajs} episodes, {num_steps} steps in total')

    with open(f'trajectories/traj-{args.mode}-{num_info_states}.pkl', 'wb') as f:
        logger.info(f'Saved trajectories to {f.name}')
        pickle.dump(trajectories, f)

    return num_info_states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--coverage', type=int, default=0)
    parser.add_argument('--mode', default='mixed', choices=['mixed', 'expert'])
    parser.add_argument('--label', default='collect')
    parser.add_argument('--eval_freq', default=5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
