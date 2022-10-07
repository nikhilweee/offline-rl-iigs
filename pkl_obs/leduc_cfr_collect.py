"""
Collect an offline dataset for Leduc Poker.
"""

import pickle
import logging
import argparse
import pyspiel
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from open_spiel.python.algorithms.cfr import _CFRSolver as CFRSolver
from open_spiel.python.algorithms import exploitability

logging.basicConfig(
    format=(
        "%(asctime)s.%(msecs)03d %(levelname)s "
        "%(module)s L%(lineno)03d | %(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__file__)


class SampleCFRSolver(CFRSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_states = []
        self.states = []
        self.collected_info_states = []
        self.collected_states = []

    def get_state_rewards(self, state):
        if not state.is_chance_node():
            rewards = state.rewards()
        else:
            rewards = [0.0, 0.0]
        return rewards

    def sample_traj(self, state, traj):
        """Sample a single traj."""

        # To reach a state, only the most recent action is sufficient.
        # We can increment the counter without adding to the trajectory.
        state_string = str(state.history())
        if state_string not in self.states:
            self.states.append(state_string)

        if state.is_terminal():
            return traj

        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            action = np.random.choice(actions, p=probs)
            next_state = state.child(action)
            rewards = self.get_state_rewards(next_state)
            traj.append((state, action, next_state, [0.0, 0.0]))
            return self.sample_traj(next_state, traj)

        else:
            info_string = state.information_state_string()
            if info_string not in self.info_states:
                self.info_states.append(info_string)
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
        if unique == 'states':
            container = self.states
        elif unique == 'info_states':
            container = self.info_states
        else:
            container = []

        for _ in range(num):
            state = self._game.new_initial_state()
            start_len = len(container)
            traj = self.sample_traj(state, [])
            end_len = len(container)
            if not unique or end_len > start_len:
                trajs.append(traj)
                self.collected_states = self.states
                self.collected_info_states = self.info_states
            else:
                self.info_states = self.collected_info_states
                self.states = self.collected_states
        num_steps = sum(len(ep) for ep in trajs)
        logger.info(
            f"Collected {len(trajs)} episodes, {num_steps} transitions. "
            f"Total {len(self.states)} states, {len(self.info_states)} info states"
        )
        return trajs


def main(args):
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    cfr_solver = SampleCFRSolver(
        game,
        regret_matching_plus=False,
        alternating_updates=True,
        linear_averaging=False,
    )

    trajectories = []

    logger.info(f"Collecting {args.min_info_states} info states")
    while len(cfr_solver.info_states) < args.min_info_states:
        trajectories.extend(cfr_solver.sample_trajs(100, unique='info_states'))

    logger.info(f"Collecting {args.min_states} states")
    while len(cfr_solver.states) < args.min_states:
        trajectories.extend(cfr_solver.sample_trajs(100, unique='states'))

    writer = SummaryWriter(f"runs/cfr/{args.label}")

    logger.info(f"Running CFR")
    for i in range(args.iterations + 1):
        if i % args.eval_freq == 0:
            conv = exploitability.exploitability(
                game, cfr_solver.average_policy()
            )
            logger.info(f"Iteration {i} Exploitability {conv}")
            writer.add_scalar("conv", conv, i)
            if args.mode == "mixed" and i > 0:
                num_trajs = (args.iterations * 1) // i
                trajectories.extend(
                    cfr_solver.sample_trajs(num_trajs)
                )
        cfr_solver.evaluate_and_update_policy()

    if args.mode == "expert":
        # trajectories = cfr_solver.sample_trajs(714)
        trajectories = cfr_solver.sample_trajs(5132)

    num_trajs = len(trajectories)
    num_info_states = len(cfr_solver.info_states)
    num_states = len(cfr_solver.states)
    num_steps = sum(len(ep) for ep in trajectories)
    logger.info(f"Collected {num_trajs} episodes, {num_steps} steps in total")

    with open(
        f"trajectories/traj-{args.mode}-{num_info_states}-"
        f"{num_states}-{num_steps}.pkl", "wb"
    ) as f:
        logger.info(f"Saved trajectories to {f.name}")
        pickle.dump(trajectories, f)

    return num_info_states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--min_info_states", type=int, default=0)
    parser.add_argument("--min_states", type=int, default=0)
    parser.add_argument("--mode", default="mixed", choices=["mixed", "expert"])
    parser.add_argument("--label", default="collect")
    parser.add_argument("--eval_freq", default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
