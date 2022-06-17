"""Implements Tabular CRR from https://arxiv.org/abs/2006.15134"""

import logging
import argparse
from observation import ObservationBuffer

logging.basicConfig(format="%(asctime)s [%(levelname).1s]: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Node:
    """Tree data structure to hold offline dataset."""

    def __init__(self):
        self.count = 0
        self.children = {}

    def __getitem__(self, key):
        """To facilitate self[key] syntax sugar."""
        return self.children[key]

    def __setitem__(self, key, value):
        """self[key] = value."""
        self.children[key] = value

    def __contains__(self, key):
        """if key in self"""
        return key in self.children

    def __iter__(self):
        """for key in self"""
        return iter(self.children)

    def update(self, key_list):
        """Add a root to leaf path, incrementing counts along the way."""
        node = self
        node.count += 1
        for key in key_list:
            if key not in node:
                node[key] = Node()
            node = node[key]
            node.count += 1

    def add_obs(self, obs):
        """Add root to leaf path from an observation."""
        state = obs['info_state']
        action = obs['action']
        reward = float(obs['reward'])
        next_state = obs['next_info_state']

        node = self
        node.count += 1
        if state not in node:
            node[state] = Node()
        node = node[state]
        node.count += 1
        if action not in node:
            node[action] = Node()
            node[action].reward = 0
        avg_reward = (node[action].reward *
            node[action].count + reward) / \
                (node[action].count + 1)
        node[action].reward = avg_reward
        node[action].count += 1
        node = node[action]
        if next_state not in node:
            node[next_state] = Node()
        node[next_state].count += 1

    def __repr__(self):
        return repr(self.__dict__)


class CRR:

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.gamma = 0.9
        self.dataset = Node()
        self.calculate_counts()

    def calculate_counts(self):
        for obs in self.trajectories.samples:
            self.dataset.add_obs(obs.to_dict())

    def empirical_policy(self, state):
        if state not in self.dataset:
            return {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
        probs = {}
        for action in self.dataset[state]:
            probs[action] = self.dataset[state][action].count / self.dataset[
                state].count
        assert sum(probs.values()) - 1 < 1e-10
        return probs

    def empirical_transition(self, state, action):
        if state not in self.dataset or action not in self.dataset[state]:
            # define a new terminal state
            term_state = '000000000000000000000000000000'
            self.dataset.update([state, action, term_state])
            self.dataset[state][action].reward = 1
        probs = {}
        for next_state in self.dataset[state][action]:
            probs[next_state] = self.dataset[state][action][
                next_state].count / self.dataset[state][action].count
        assert sum(probs.values()) - 1 < 1e-10
        return probs

    def qvalue(self, state, action):
        if state == '000000000000000000000000000000':
            self.dataset.update([state, action])
            self.dataset[state][action].q = 1
        if hasattr(self.dataset[state][action], 'q'):
            return self.dataset[state][action].q
        qvalue = self.dataset[state][action].reward
        state_probs = self.empirical_transition(state, action)
        for next_state, state_prob in state_probs.items():
            action_probs = self.empirical_policy(next_state)
            for next_action, action_prob in action_probs.items():
                qvalue += state_prob * action_prob * \
                    self.gamma * self.qvalue(next_state, next_action)
        self.dataset[state][action].q = qvalue
        return self.dataset[state][action].q


def main(args):
    logger.info(f'Loading Dataset')
    trajectories = ObservationBuffer.from_csv(args.traj)
    crr = CRR(trajectories)
    for obs in crr.trajectories.samples:
        od = obs.to_dict()
        print(od['info_state'], od['action'])
        print(crr.qvalue(od['info_state'], od['action']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', required=True)
    parser.add_argument('--suffix', default='test')
    args = parser.parse_args()
    main(args)
