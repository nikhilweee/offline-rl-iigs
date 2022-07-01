"""Implements Tabular CRR from https://arxiv.org/abs/2006.15134"""

import logging
import argparse
import numpy as np
import pyspiel
from torch.utils.tensorboard import SummaryWriter
from crr_utils import DataTree, NestedDict
from observation import ObservationBuffer
from open_spiel.python.algorithms import exploitability

logging.basicConfig(format="%(asctime)s [%(levelname).1s]: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Policy:

    def __init__(self, source_policy):
        self.sp = source_policy
        self.pdict = NestedDict()

    def __call__(self, state, action_mask=None):
        if state not in self.pdict:
            self.pdict[state] = self.sp(state)
        probs = self.pdict[state]
        assert np.allclose(sum(probs.values()), 1)
        return self.reweight_actions(state, probs, action_mask)

    def update(self, state, probs):
        self.pdict[state] = probs

    def reweight_actions(self, state, probs, action_mask):
        if not action_mask:
            return probs
        prob_sum = 0
        for idx, mask in enumerate(action_mask):
            if mask == '0' or idx not in probs:
                probs[idx] = 0
            prob_sum += probs[idx]
        for action in probs.keys():
            probs[action] /= prob_sum
        assert np.allclose(sum(probs.values()), 1)
        return probs

    def action_probabilities(self, game_state):
        info_state = game_state.information_state_tensor()
        legal_mask = game_state.legal_actions_mask()
        info_state = ''.join([str(int(x)) for x in info_state])
        legal_mask = ''.join([str(int(x)) for x in legal_mask])
        probs = self(info_state, legal_mask)
        return probs


class CRR:

    def __init__(self, trajectories):
        self.gamma = 0.9
        self.tree = DataTree()
        self.vdict = NestedDict()
        self.qdict = NestedDict()
        self.policy = Policy(self.empirical_policy)
        self.calculate_counts(trajectories)

    def calculate_counts(self, trajectories):
        logger.info(f"Updating counts for {len(trajectories.samples)} samples")
        for obs in trajectories.samples:
            self.tree.add_obs(obs.to_dict())

    def empirical_policy(self, state):
        if state not in self.tree:
            return {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
        probs = {}
        for action in self.tree[state]:
            probs[action] = self.tree[state][action].count / self.tree[
                state].count
        assert np.allclose(sum(probs.values()), 1)
        return probs

    def empirical_transition(self, state, action):
        if state not in self.tree or action not in self.tree[state]:
            # define a new terminal state
            term_state = '000000000000000000000000000000'
            self.tree.update([state, action, term_state])
            self.tree[state][action].reward = 1
        probs = {}
        for next_state in self.tree[state][action]:
            probs[next_state] = self.tree[state][action][
                next_state].count / self.tree[state][action].count
        assert np.allclose(sum(probs.values()), 1)
        return probs

    def state_prob(self, state):
        return self.tree[state].count / self.tree.count

    def qvalue(self, policy, state, action):
        term_state = '000000000000000000000000000000'
        if state == term_state:
            self.qdict[state][action] = 1
        if state in self.qdict and action in self.qdict[state]:
            return self.qdict[state][action]
        if state not in self.tree or action not in self.tree[state]:
            self.tree.update([state, action, term_state])
            self.tree[state][action].reward = 1
        qvalue = self.tree[state][action].reward
        state_probs = self.empirical_transition(state, action)
        for next_state, state_prob in state_probs.items():
            action_probs = policy(next_state)
            for next_action, action_prob in action_probs.items():
                qvalue += state_prob * action_prob * \
                    self.gamma * self.qvalue(policy, next_state, next_action)
        self.qdict[state][action] = qvalue
        return self.qdict[state][action]

    def vvalue(self, policy, state):
        if state in self.vdict:
            return self.vdict[state]
        vvalue = 0
        action_probs = policy(state)
        for action, action_prob in action_probs.items():
            vvalue += action_prob * self.qvalue(policy, state, action)
        self.vdict[state] = vvalue
        return self.vdict[state]

    def reset_qvdict(self):
        self.qdict = NestedDict()
        self.vdict = NestedDict()


def main(args):
    logger.info(f'Reading dataset')
    trajectories = ObservationBuffer.from_csv(args.traj, limit=1_000_000)
    writer = SummaryWriter(f'runs/crr/{args.mode}/{args.suffix}')
    crr = CRR(trajectories)
    # Filter unique states
    visited_states = []
    unique_samples = []
    for obs in trajectories.samples:
        obs_dict = obs.to_dict()
        state = obs_dict['info_state']
        if state in visited_states:
            continue
        visited_states.append(state)
        unique_samples.append(obs)
    # Start running
    logger.info(f'Starting training')
    for epoch in range(args.epochs):
        for idx, obs in enumerate(unique_samples):
            obs_dict = obs.to_dict()
            state = obs_dict['info_state']
            action_mask = obs_dict['action_mask']
            indicators = np.zeros(3)
            # Calculate indicator values for actions
            for action in range(3):
                if args.mode == 'bin':
                    if crr.qvalue(crr.policy, state, action) >= crr.vvalue(crr.policy, state):
                        indicators[action] = crr.empirical_policy(state)[action]
                if args.mode == 'exp':
                    indicators[action] = crr.qvalue(crr.policy, state, action) - crr.vvalue(crr.policy, state)
                    # Beta is 1 for now. Z is the normalization constant.
                    indicators[action] = np.exp(indicators[action] / 1) * crr.empirical_policy(state)[action]
            if args.mode == 'bin':
                argsort = indicators.argsort()
                while action_mask[argsort[-1]] == '0':
                    argsort = argsort[:-1]
                probs = {action: 0.0 for action in range(3)}
                probs[argsort[-1]] = 1
            if args.mode == 'exp':
                indicators = indicators / np.sum(indicators)
                probs = {action: indicators[action] for action in range(3)}
            # Update policy, reset Q and V dicts
            crr.policy.update(state, probs)
            crr.reset_qvdict()
            # Store summary
            if (idx + 1) % 100 == 0 or (idx + 1) == len(unique_samples):
                game = pyspiel.load_game("leduc_poker", {"players": 2})
                conv = exploitability.exploitability(game, crr.policy)
                writer.step = epoch * len(unique_samples) + (idx + 1)
                writer.add_scalar("conv", conv, writer.step)
                logger.info(
                    f'Ep: {epoch+1:2d} '
                    f'Obs: {idx+1:6d}/{len(unique_samples)} '
                    f'Exp: {conv:.04f}'
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', default=None)
    parser.add_argument('--suffix', default='test')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--mode', choices=['bin', 'exp'], required=True)
    args = parser.parse_args()
    main(args)
