import numpy as np

def simplex_3d():
    range = np.arange(0, 1, 0.01)
    m = np.stack(
        np.meshgrid(range, range), -1
    ).reshape(-1, 2)
    m = m[m[:, 0] + m[:, 1] <= 1]
    s = 1 - m.sum(-1, keepdims=True)
    c = np.concatenate([m, s], -1)
    return c

def possible_policies(action_mask):
    simplex = simplex_3d()
    for idx in range(3):
        if action_mask[idx] == '0':
            simplex = simplex[simplex[:, idx] == 0]
    simplex[simplex == 0] = 1e-10
    return simplex

class DataNode:
    def __init__(self):
        self.count = 0
        self.children = {}

    def __getitem__(self, key):
        """self[key]"""
        return self.children[key]

    def __setitem__(self, key, value):
        """self[key] = value"""
        self.children[key] = value

    def __contains__(self, key):
        """if key in self"""
        return key in self.children

    def __iter__(self):
        """for key in self"""
        return iter(self.children)

    def __repr__(self):
        """to help with debugging."""
        repr_dict = vars(self)
        # repr_dict.pop('children', None)
        return repr(repr_dict)


class DataTree(DataNode):
    """Tree data structure to hold offline dataset."""

    def __init__(self):
        super().__init__()

    def update(self, key_list):
        """Add a root to leaf path, incrementing counts along the way."""
        node = self
        node.count += 1
        for key in key_list:
            if key not in node:
                node[key] = DataNode()
            node = node[key]
            node.count += 1

    def add_obs(self, obs):
        """Add root to leaf path from an observation."""
        state = obs['info_state']
        action = obs['action']
        reward = obs['reward']
        next_state = obs['next_info_state']

        node = self
        node.count += 1
        if state not in node:
            node[state] = DataNode()
        node = node[state]
        node.count += 1
        if action not in node:
            node[action] = DataNode()
            node[action].reward = 0
        avg_reward = (node[action].reward *
            node[action].count + reward) / \
                (node[action].count + 1)
        node[action].reward = avg_reward
        node[action].count += 1
        node = node[action]
        if next_state not in node:
            node[next_state] = DataNode()
        node[next_state].count += 1


class NestedDict(dict):

    def __getitem__(self, key):
        return self.setdefault(key, NestedDict())
