"""Implements Tabular CRR from https://arxiv.org/abs/2006.15134"""

import logging
import argparse
import numpy as np
import pyspiel
from crr_utils import DataTree, NestedDict, possible_policies
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
        return self.reweight_actions(probs, action_mask)

    def update(self, state, probs):
        self.pdict[state] = probs

    def reweight_actions(self, probs, action_mask):
        if not action_mask:
            return probs
        prob_sum = 0
        for idx, mask in enumerate(action_mask):
            if idx in probs.keys():
                if mask == '0':
                    del probs[idx]
                else:
                    prob_sum += probs[idx]
        if prob_sum == 0:
            prob_sum += 1e-10
        for action in probs.keys():
            probs[action] /= prob_sum
        return probs

    def action_probabilities(self, game_state):
        info_state = game_state.information_state_tensor()
        legal_mask = game_state.legal_actions_mask()
        info_state = ''.join([str(int(x)) for x in info_state])
        legal_mask = ''.join([str(int(x)) for x in legal_mask])
        probs = self(info_state, legal_mask)
        assert sum(list(probs.values())) - 1 < 1e-10
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
        for obs in trajectories.samples:
            self.tree.add_obs(obs.to_dict())

    def empirical_policy(self, state):
        if state not in self.tree:
            return {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
        probs = {}
        for action in self.tree[state]:
            probs[action] = self.tree[state][action].count / self.tree[
                state].count
        assert sum(probs.values()) - 1 < 1e-10
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
        assert sum(probs.values()) - 1 < 1e-10
        return probs

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

    def reset_qdict(self):
        self.qdict = NestedDict()


def main(args):
    logger.info(f'Loading Dataset')
    trajectories = ObservationBuffer.from_csv(args.traj)
    crr = CRR(trajectories)
    for idx, obs in enumerate(trajectories.samples):
        obs_dict = obs.to_dict()
        state = obs_dict['info_state']
        action_mask = obs_dict['action_mask']
        expectations = {}
        for action_probs in possible_policies(action_mask):
            policy = Policy(crr.policy)
            policy.update(state, action_probs)
            crr.reset_qdict()
            expectation = 0
            for action in action_probs.keys():
                if crr.qvalue(policy, state, action) >= crr.vvalue(
                        policy, state):
                    expectation += crr.empirical_policy(
                        state)[action] * np.log(action_probs[action])
            expectations[policy] = expectation
        next_policy = max(expectations,
                          key=lambda policy: expectations[policy])
        crr.policy = next_policy
        game = pyspiel.load_game("leduc_poker", {"players": 2})
        conv = exploitability.exploitability(game, next_policy)
        logger.info(
            f'Observation {idx+1:04d}/{len(trajectories.samples)} Exploitability {conv}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', required=True)
    parser.add_argument('--suffix', default='test')
    args = parser.parse_args()
    main(args)
