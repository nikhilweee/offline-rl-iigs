"""
Run CFR on an offline dataset.
"""

import argparse
import copy
import logging
import pickle
from collections import defaultdict

import numpy as np
import pyspiel
from env import MLP, RNN, Predictor
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, exploitability
from torch.utils.tensorboard import SummaryWriter

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


class EmpiricalPolicy(policy.TabularPolicy):
    def __init__(self, game, players=None, states=None, actions=None):
        """Initializes an empirical policy for all players in the game."""
        players = sorted(players or range(game.num_players()))
        super().__init__(game, players)
        # unless otherwise specified, states here refer to info states
        self.game_type = game.get_type()
        self.state_lookup = {}
        self.states_per_player = [[] for _ in range(game.num_players())]
        self.states = []

        self.update_empirical_policy(states, actions)

    def update_empirical_policy(self, states, actions):

        legal_actions_list = []
        state_in_list = []
        empirical_actions_list = []

        for player in self.player_ids:
            for hist, state in sorted(states.items(), key=lambda pair: pair[0]):
                if (
                    not state.is_simultaneous_node()
                    and player != state.current_player()
                ):
                    continue
                legal_actions = state.legal_actions_mask(player)
                if not any(legal_actions):
                    continue
                # info_key is an info state and hist is a state
                info_key = self._state_key(state, player)
                # skip adding a duplicate info state node.
                if info_key in self.state_lookup:
                    state_index = self.state_lookup[info_key]
                    action_counts = empirical_actions_list[state_index]
                    for action in actions[hist]:
                        action_counts[action] += 1
                else:
                    state_index = len(legal_actions_list)
                    self.state_lookup[info_key] = state_index
                    legal_actions_list.append(legal_actions)
                    self.states_per_player[player].append(info_key)
                    self.states.append(state)
                    action_counts = np.zeros(len(legal_actions))
                    for action in actions[hist]:
                        action_counts[action] += 1
                    empirical_actions_list.append(action_counts)
                    if self.game_type.provides_information_state_tensor:
                        state_in_list.append(
                            state.information_state_tensor(player)
                        )
                    elif self.game_type.provides_observation_tensor:
                        state_in_list.append(state.observation_tensor(player))

        self.state_in = None
        if state_in_list:
            self.state_in = np.array(state_in_list)

        self.legal_actions_mask = np.array(legal_actions_list)
        self.empirical_actions = np.array(empirical_actions_list)

        self.action_probability_array = self.empirical_actions / np.sum(
            self.empirical_actions, axis=-1, keepdims=True
        )
        logger.info(
            f"empirical_action_prob_array: {self.action_probability_array.shape}"
        )


class UniformPolicy(policy.TabularPolicy):
    def __init__(self, parent_policy):
        """Initializes a hybrid policy for all players in the game."""

        self._parent_policy = parent_policy
        # unless otherwise specified, states here refer to info states
        self.game = self._parent_policy.game
        self.game_type = self._parent_policy.game_type
        self.player_ids = self._parent_policy.player_ids
        self.state_lookup = self._parent_policy.state_lookup
        self.states_per_player = self._parent_policy.states_per_player
        self.states = self._parent_policy.states

        self.update_hybrid_policy()

    def update_hybrid_policy(self):
        # Get all states in the game at which players have to make decisions unless
        # they are explicitly specified.
        states = policy.get_all_states.get_all_states(
            self.game,
            depth_limit=-1,
            include_terminals=False,
            include_chance_states=False,
            include_mean_field_states=False,
            to_string=lambda s: s.history_str(),
        )

        # Assemble legal actions for every valid (state, player) pair, keyed by
        # information state string.
        legal_actions_list = self._parent_policy.legal_actions_mask.tolist()
        state_in_list = []
        if self._parent_policy.state_in is not None:
            state_in_list = self._parent_policy.state_in.tolist()
        for player in self.player_ids:
            # States are ordered by their history.
            for _, state in sorted(states.items(), key=lambda pair: pair[0]):
                if (
                    not state.is_simultaneous_node()
                    and player != state.current_player()
                ):
                    continue
                legal_actions = state.legal_actions_mask(player)
                if not any(legal_actions):
                    continue
                key = self._state_key(state, player)
                if key not in self.state_lookup:
                    state_index = len(legal_actions_list)
                    self.state_lookup[key] = state_index
                    legal_actions_list.append(legal_actions)
                    self.states_per_player[player].append(key)
                    self.states.append(state)
                    if self.game_type.provides_information_state_tensor:
                        state_in_list.append(
                            state.information_state_tensor(player)
                        )
                    elif self.game_type.provides_observation_tensor:
                        state_in_list.append(state.observation_tensor(player))

        # Put legal action masks in a numpy array and
        # create the uniform random policy.

        self.state_in = None
        if state_in_list:
            self.state_in = np.array(state_in_list)
        self.legal_actions_mask = np.array(legal_actions_list)
        self.action_probability_array = self.legal_actions_mask / np.sum(
            self.legal_actions_mask, axis=-1, keepdims=True
        )
        logger.info(
            f"uniform_action_prob_array: {self.action_probability_array.shape}"
        )


class OfflineCFRSolver(cfr.CFRSolver):
    def __init__(self, game, trajs, model, env_ckpt):
        # cannot call super().__init__(game) because that creates a tabular
        # policy using the actual game states and doesn't use collected trajs
        self._game = game
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()

        # states, actions = self._extract_trajs(trajs)
        self._current_policy = EmpiricalPolicy(game, states={}, actions={})
        self._average_policy = UniformPolicy(self._current_policy)
        self._current_policy = copy.copy(self._average_policy)

        # Make a dict of ALL info state nodes.
        # Each info state has a list of legal actions
        # and its index in the tabular policy array.
        self._info_state_nodes = {}
        self._initialize_info_state_nodes()
        logger.info(f"info_state_nodes: {len(self._info_state_nodes)}")

        self._iteration = 0
        self._linear_averaging = False
        self._alternating_updates = True
        self._regret_matching_plus = False

        if model == "rnn":
            pred_model = RNN()
        if model == "mlp":
            pred_model = MLP()

        self.predictor = Predictor(pred_model, env_ckpt, game)

    def _initialize_info_state_nodes(self):
        for info_state, index in self._average_policy.state_lookup.items():
            legal_mask = self._average_policy.legal_actions_mask[index]
            legal_actions = [x for x in [0, 1, 2] if legal_mask[x] == 1]
            info_state_node = cfr._InfoStateNode(
                legal_actions=legal_actions, index_in_tabular_policy=index
            )
            self._info_state_nodes[info_state] = info_state_node

    def _extract_trajs(self, trajs):
        """Make a dict of states encountered so far."""
        self.action_freqs = {}
        state_dict = {}
        action_dict = defaultdict(list)
        for traj in trajs:
            for step in traj:
                state, action, _, _ = step
                # TODO: Check if history_str() can be used
                key = str(state.history())
                state_dict[key] = state
                action_dict[key].append(action)
                # Update action frequencies. This is used for CFR
                if key not in self.action_freqs:
                    self.action_freqs[key] = {action: 1}
                elif action not in self.action_freqs[key]:
                    self.action_freqs[key][action] = 1
                else:
                    self.action_freqs[key][action] += 1

        self.action_probs = {}
        for key in self.action_freqs:
            freq_sum = sum(self.action_freqs[key].values())
            action_prob_dict = {}
            for action, freq in self.action_freqs[key].items():
                action_prob_dict[action] = freq / freq_sum
            self.action_probs[key] = action_prob_dict

        self.missing_state_count = 0
        self.found_state_count = 0
        # self._reset_action_probs(self._root_node)

        logger.info(
            f"Loaded {len(self.action_probs)} non-terminal states from the dataset."
        )

        # This state_dict does not include terminal states
        return state_dict, action_dict

    def _compute_counterfactual_regret_for_player(
        self, state, policies, reach_probabilities, player
    ):

        if state.is_terminal():
            return np.asarray(state.returns())

        if state.is_chance_node():
            state_value = 0.0
            for action, action_prob in state.chance_outcomes():
                assert action_prob > 0
                # new_state = state.child(action)
                new_state = self.predictor.next_state(state, action)
                new_reach_probabilities = reach_probabilities.copy()
                new_reach_probabilities[-1] *= action_prob
                state_value += (
                    action_prob
                    * self._compute_counterfactual_regret_for_player(
                        new_state, policies, new_reach_probabilities, player
                    )
                )
            return state_value

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)

        # No need to continue on this history branch as no update will be performed
        # for any player.
        # The value we return here is not used in practice. If the conditional
        # statement is True, then the last taken action has probability 0 of
        # occurring, so the returned value is not impacting the parent node value.
        if all(reach_probabilities[:-1] == 0):
            return np.zeros(self._num_players)

        state_value = np.zeros(self._num_players)

        # The utilities of the children states are computed recursively. As the
        # regrets are added to the information state regrets for each state in that
        # information state, the recursive call can only be made once per child
        # state. Therefore, the utilities are cached.
        children_utilities = {}

        info_state_node = self._info_state_nodes[info_state]
        if policies is None:
            info_state_policy = self._get_infostate_policy(info_state)
        else:
            info_state_policy = policies[current_player](info_state)

        assert state.legal_actions() == info_state_node.legal_actions

        for action in state.legal_actions():
            action_prob = info_state_policy.get(action, 0.0)
            # new_state = state.child(action)
            new_state = self.predictor.next_state(state, action)
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[current_player] *= action_prob
            child_utility = self._compute_counterfactual_regret_for_player(
                new_state,
                policies=policies,
                reach_probabilities=new_reach_probabilities,
                player=player,
            )

            state_value += action_prob * child_utility
            children_utilities[action] = child_utility

        simulatenous_updates = player is None
        if not simulatenous_updates and current_player != player:
            return state_value

        reach_prob = reach_probabilities[current_player]
        counterfactual_reach_prob = np.prod(
            reach_probabilities[:current_player]
        ) * np.prod(reach_probabilities[current_player + 1 :])
        state_value_for_player = state_value[current_player]

        for action, action_prob in info_state_policy.items():
            cfr_regret = counterfactual_reach_prob * (
                children_utilities[action][current_player]
                - state_value_for_player
            )

            info_state_node.cumulative_regret[action] += cfr_regret
            if self._linear_averaging:
                info_state_node.cumulative_policy[action] += (
                    self._iteration * reach_prob * action_prob
                )
            else:
                info_state_node.cumulative_policy[action] += (
                    reach_prob * action_prob
                )

        return state_value


def main(args):
    logger.info(f"Reading dataset: {args.traj}")
    with open(args.traj, "rb") as f:
        trajs = pickle.load(f)
    logger.info(f"Loaded dataset : {args.traj}")

    game = pyspiel.load_game("leduc_poker", {"players": 2})
    cfr_solver = OfflineCFRSolver(game, trajs, args.model, args.env_ckpt)
    writer = SummaryWriter(f"runs/cfr_env/{args.model}/{args.label}")

    for i in range(args.iterations + 1):
        if i % args.print_freq == 0:
            conv = exploitability.exploitability(
                game, cfr_solver.average_policy()
            )
            logger.info(f"Iteration {i} Exploitability {conv}")
            writer.add_scalar("conv", conv, i)
        cfr_solver.evaluate_and_update_policy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj", default="trajectories/traj-010-824-4463-69032.pkl"
    )
    parser.add_argument(
        "--env_ckpt", default="runs/env/rnn/model_epoch_000100_loss_1_0440.pt"
    )
    parser.add_argument("--label", default="default")
    parser.add_argument("--model", choices=["mlp", "rnn"], default="rnn")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--print_freq", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
