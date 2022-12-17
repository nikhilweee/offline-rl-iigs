"""Collect a dataset of state, action, and next state"""

import pickle
import pyspiel
import numpy as np

class Sampler:
    
    def __init__(self):
        self.states = {}

    def sample_traj(self, state):

        if state.is_terminal():
            return

        state_ser = state.serialize()
        hist = list(map(str, state.history()))
        hist_str = '\n'.join(hist) + '\n'
        assert state_ser == hist_str
        if state_ser not in self.states:
            self.states[state_ser] = {}

        actions = state.legal_actions()
        action = np.random.choice(actions)

        next_state = state.child(action)
        next_state_ser = next_state.serialize()
        if next_state_ser not in self.states[state_ser]:
            self.states[state_ser][action] = next_state_ser
        return self.sample_traj(next_state)


def main():
    game = pyspiel.load_game('leduc_poker', {'players': 2})
    initial_state = game.new_initial_state()
    sampler = Sampler()
    for idx in range(1_000_000):
        sampler.sample_traj(initial_state)
        if (idx + 1) % 1_000 != 0:
            continue
        print(f'iter {idx+1} states {len(sampler.states)}')
        with open('trajectories/traj-3937.pkl', 'wb') as f:
            pickle.dump(sampler.states, f)
        if len(sampler.states) == 3937:
            break

if __name__ == '__main__':
    main()
