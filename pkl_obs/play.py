import pyspiel

game = pyspiel.load_game("leduc_poker", {"players": 2})
state = game.new_initial_state()
state.apply_action(0)
state.apply_action(1)
tensor = state.information_state_tensor()
