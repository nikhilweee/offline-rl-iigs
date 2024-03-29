"""
Build on Offline CFR and introduce a value function for unexplored nodes.
"""


import argparse
import logging
import pickle
import pyspiel
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


import numpy as np

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, exploitability

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


class OfflineEmpiricalPolicy(policy.TabularPolicy):
    def __init__(self, game, players=None, states=None, actions=None):
        """Initializes an empirical policy for all players in the game."""
        players = sorted(players or range(game.num_players()))
        super().__init__(game, players)

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
                # Skip adding a duplicate info state node.
                if key in self.state_lookup:
                    state_index = self.state_lookup[key]
                    action_counts = empirical_actions_list[state_index]
                    for action in actions[str(state.history())]:
                        action_counts[action] += 1
                else:
                    state_index = len(legal_actions_list)
                    self.state_lookup[key] = state_index
                    legal_actions_list.append(legal_actions)
                    self.states_per_player[player].append(key)
                    self.states.append(state)
                    action_counts = np.zeros(len(legal_actions))
                    for action in actions[str(state.history())]:
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


class OfflineHybridPolicy(policy.TabularPolicy):
    def __init__(self, parent_policy):
        """Initializes a hybrid policy for all players in the game."""

        self._parent_policy = parent_policy

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
                    state.is_simultaneous_node()
                    or player == state.current_player()
                ):
                    legal_actions = state.legal_actions_mask(player)
                    if any(legal_actions):
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
                                state_in_list.append(
                                    state.observation_tensor(player)
                                )

        # Put legal action masks in a numpy array and create the uniform random
        # policy.
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
    def __init__(self, game, trajs):
        self.mode = "train"
        self.dataset = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = MLP(input_size=30, output_size=1, hidden_size=256, device=device)

        self._game = game
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()

        states, actions = self._process_trajs(trajs)
        self._current_policy = OfflineEmpiricalPolicy(
            game, states=states, actions=actions
        )
        self._average_policy = OfflineHybridPolicy(self._current_policy)

        # Make a dict of ALL info state nodes.
        # Each info state has a list of legal actions
        # and its index in the tabular policy array.
        self._info_state_nodes = {}
        self._initialize_info_state_nodes(self._root_node)
        logger.info(f"info_state_nodes: {len(self._info_state_nodes)}")

        self._iteration = 0
        self._linear_averaging = False
        self._alternating_updates = True
        self._regret_matching_plus = False

    def _get_infostate_policy(self, info_state_str):
        """Returns an {action: prob} dictionary for the policy on `info_state`."""
        info_state_node = self._info_state_nodes[info_state_str]
        prob_vec = self._current_policy.action_probability_array[
            info_state_node.index_in_tabular_policy
        ]
        return {
            action: prob_vec[action] for action in info_state_node.legal_actions
        }

    def _initialize_info_state_nodes(self, state):

        if state.is_terminal():
            return

        state_key = str(state.history())
        if state.is_chance_node():
            for action in self.action_probs[state_key].keys():
                self._initialize_info_state_nodes(state.child(action))
            return

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)

        info_state_node = self._info_state_nodes.get(info_state)
        if info_state_node is None:
            legal_actions = state.legal_actions(current_player)
            info_state_node = cfr._InfoStateNode(
                legal_actions=legal_actions,
                index_in_tabular_policy=self._current_policy.state_lookup[
                    info_state
                ],
            )
            self._info_state_nodes[info_state] = info_state_node

        for action in self.action_probs[state_key].keys():
            self._initialize_info_state_nodes(state.child(action))

    def _log_estimate(self, action_prob, estimate):
        if not estimate:
            self.missing_state_count += 1
            logger.info(
                f"estimate: {estimate} actual: "
                f"{action_prob:06.4f} diff: {estimate}"
            )
        else:
            self.found_state_count += 1
            diff = estimate - action_prob
            logger.info(
                f"estimate: {estimate:06.4f} actual: "
                f"{action_prob:06.4f} diff: {diff: 06.4f}"
            )

    def _set_action_prob(self, state_key, action, prob):
        if state_key not in self.action_probs:
            self.action_probs[state_key] = {action: prob}
        else:
            self.action_probs[state_key][action] = prob

    def _reset_action_probs(self, state):
        key = str(state.history())
        if state.is_terminal():
            if key in self.action_probs:
                self.found_state_count += 1
        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                estimate = self.action_probs.get(key, {}).get(action, None)
                self._log_estimate(action_prob, estimate)
                self._set_action_prob(key, action, action_prob)
                new_state = state.child(action)
                self._reset_action_probs(new_state)
        if state.is_player_node():
            action_sum = len(state.legal_actions())
            for action in state.legal_actions():
                action_prob = 1 / action_sum
                estimate = self.action_probs.get(key, {}).get(action, None)
                self._log_estimate(action_prob, estimate)
                self._set_action_prob(key, action, action_prob)
                new_state = state.child(action)
                self._reset_action_probs(new_state)
        if state.is_initial_state():
            self.found_state_count += 1
            logger.info(
                f"{self.missing_state_count} states missing, "
                f"{self.found_state_count} found."
            )
        return

    def _process_trajs(self, trajs):
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
                if action not in self.action_freqs[key]:
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
        self._reset_action_probs(self._root_node)

        logger.info(
            f"Loaded {len(self.action_probs)} non-terminal states from the dataset."
        )

        return state_dict, action_dict

    def _compute_counterfactual_regret_for_player(
        self, state, policies, reach_probabilities, player
    ):

        if state.is_terminal():
            return np.asarray(state.returns())

        state_key = str(state.history())

        if state.is_chance_node():
            state_value = 0.0

            # all_actions, _ = zip(*state.chance_outcomes())
            # tree_actions = self.action_probs[state_key].keys()
            # missing_actions = set(all_actions) - set(tree_actions)
            # if missing_actions:
            #     logger.info(f'{state_key}: {missing_actions}')

            # TODO: Change these lines to switch to online
            # for action, action_prob in state.chance_outcomes():
            for action, action_prob in self.action_probs[state_key].items():
                assert action_prob > 0
                new_state = state.child(action)
                new_reach_probabilities = reach_probabilities.copy()
                new_reach_probabilities[-1] *= action_prob
                state_value += (
                    action_prob
                    * self._compute_counterfactual_regret_for_player(
                        new_state, policies, new_reach_probabilities, player
                    )
                )
            return self._train_value_network(state, state_value)

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)

        # Comments stripped for brevity
        if all(reach_probabilities[:-1] == 0):
            return np.zeros(self._num_players)

        state_value = np.zeros(self._num_players)

        # Commentes stripped for brevity
        children_utilities = {}

        info_state_node = self._info_state_nodes[info_state]
        if policies is None:
            info_state_policy = self._get_infostate_policy(info_state)
        else:
            info_state_policy = policies[current_player](info_state)

        # all_actions = state.legal_actions()
        # tree_actions = self.action_probs[state_key].keys()
        # missing_actions = set(all_actions) - set(tree_actions)
        # if missing_actions:
        #     logger.info(f'{state_key}: {missing_actions}')

        # TODO: Change these lines to switch to online
        # for action in state.legal_actions():
        for action in self.action_probs[state_key].keys():
            action_prob = info_state_policy.get(action, 0.0)
            new_state = state.child(action)
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
            return self._train_value_network(state, state_value)

        reach_prob = reach_probabilities[current_player]
        counterfactual_reach_prob = np.prod(
            reach_probabilities[:current_player]
        ) * np.prod(reach_probabilities[current_player + 1 :])
        state_value_for_player = state_value[current_player]

        # all_actions = info_state_policy.keys()
        # tree_actions = self.action_probs[state_key].keys()
        # missing_actions = set(all_actions) - set(tree_actions)
        # if missing_actions:
        #     logger.info(f'{state_key}: {missing_actions}')

        # TODO: Change these lines to switch to online
        # this is essentially state.legal_actions()
        # for action in info_state_policy.keys():
        for action in self.action_probs[state_key].keys():
            action_prob = info_state_policy.get(action)
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

        return self._train_value_network(state, state_value)

    def _train_value_network(self, state, value):
        if self.mode == "train" and len(state.history()) > 3:
            key = torch.tensor(state.information_state_string())
            self.dataset.append((key, value))
        if self.mode == "eval":
            key = torch.tensor(state.information_state_string())
            value = self.network(key)
        return value


class MLP(nn.Module):
    def __init__(
        self, input_size=30, output_size=3, hidden_size=256, device="cpu"
    ):
        super().__init__()
        logger.info(f"Using Device: {device}")
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, state):
        out = self.linear1(state)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out

    def action_probabilities(self, state):
        info_state = state.information_state_tensor()
        legal_mask = state.legal_actions_mask()
        info_state = torch.tensor(info_state, device=self.device)
        legal_mask = torch.tensor(legal_mask, device=self.device)
        illegal_mask = (1 - legal_mask).bool()
        out = self(info_state)
        out = out.masked_fill(illegal_mask, -float("inf"))
        out = nn.functional.softmax(out, dim=-1)
        probs = {idx: prob for idx, prob in enumerate(out) if prob.isfinite()}
        return probs

    def train(self, dataset):

        data_loader = DataLoader(dataset, batch_size=1024 * 100, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-5)

        logger.info(f"Starting Training")
        for epoch in range(args.epochs):
            running_loss = 0.0
            for idx, batch in enumerate(data_loader):
                state, value = batch
                state = state.to(self.device)
                value = value.to(self.device)

                pred = self(state)
                loss = criterion(pred, value)
                loss.backward()

                writer.step = epoch * len(data_loader) + (idx + 1)
                # log_gradients(model.named_parameters(), writer)

                optimizer.step()
                running_loss += loss.item()
                if len(data_loader) >= 100:
                    print_interval = len(data_loader) // 50
                else:
                    print_interval = 1
                if (idx + 1) % print_interval == 0:
                    avg_running_loss = running_loss / print_interval
                    writer.add_scalar("loss", avg_running_loss, writer.step)
                    game = pyspiel.load_game("leduc_poker", {"players": 2})
                    conv = exploitability.exploitability(game, model)
                    writer.add_scalar("conv", conv, writer.step)
                    logger.info(
                        f"epoch: {epoch + 1:02d} batch: {idx + 1:03d} loss: {avg_running_loss:.04f} conv: {conv:.04f}"
                    )
                    running_loss = 0.0


def main(args):
    logger.info(f"Reading dataset: {args.traj}")
    with open(args.traj, "rb") as f:
        trajs = pickle.load(f)
    logger.info(f"Loaded dataset : {args.traj}")

    game = pyspiel.load_game("leduc_poker", {"players": 2})
    cfr_solver = OfflineCFRSolver(game, trajs)
    writer = SummaryWriter(f"runs/cfr/{args.label}")

    for i in range(args.iterations + 1):
        if i % args.print_freq == 0:
            cfr_solver.mode == "eval"
            cfr_solver.evaluate_and_update_policy()
            conv = exploitability.exploitability(
                game, cfr_solver.average_policy()
            )
            logger.info(f"Iteration {i} Exploitability {conv}")
            writer.add_scalar("conv", conv, i)
            cfr_solver.mode == "train"
        cfr_solver.evaluate_and_update_policy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", default="trajectories/traj-mixed.pkl")
    parser.add_argument("--label", default="offline")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--print_freq", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
