import numpy as np
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(sign=' ', precision=4, suppress=True)
writer = SummaryWriter('runs/rps/')

def get_strategy(cum_regrets):
    strategy = np.where(cum_regrets>0, cum_regrets, 0)
    if strategy.sum() == 0:
        return np.ones(3) / 3
    else:
        return strategy / strategy.sum()

def utility_matrix():
    # we are the row, they are the column
    matrix = np.array(
        [[0, -1, 1],
         [1, 0, -1],
         [-1, 1, 0]]
    )
    return matrix

def get_regrets(one_action, two_action):
    utility = utility_matrix()
    regrets = np.zeros(3)
    for x in range(3):
        regrets[x] += utility[x, two_action] - utility[one_action, two_action]
    return regrets

def get_action(strategy):
    return np.random.choice([0, 1, 2], p=strategy)

def main():
    # two_strategy = np.ones(3) * 1/3
    # one_strategy = np.ones(3) * 1/3
    one_cum_regrets = np.zeros(3)
    one_cum_strategy = np.zeros(3)
    two_cum_regrets = np.zeros(3)
    two_cum_strategy = np.zeros(3)
    for idx in range(1_000_000):
        two_strategy = get_strategy(two_cum_regrets)
        two_cum_strategy += two_strategy
        two_action = get_action(two_strategy)
        one_strategy = get_strategy(one_cum_regrets)
        one_cum_strategy += one_strategy
        one_action = get_action(one_strategy)
        one_regrets = get_regrets(one_action, two_action)
        two_regrets = get_regrets(two_action, one_action)
        one_cum_regrets += one_regrets
        two_cum_regrets += two_regrets
        if idx % 10_000 == 0:
            one_avg_strategy = one_cum_strategy / one_cum_strategy.sum()
            one_avg_regret = one_cum_regrets / (idx + 1)
            print(idx + 1, 'T:', two_action, 'S:', one_strategy, 'U:', one_action, 'R:', regrets, 'CR:', cum_regrets, 'CS:', one_avg_strategy)
            writer.add_scalars('average_strategy', {'rock': one_avg_strategy[0], 'paper': one_avg_strategy[1], 'scissor': one_avg_strategy[2]}, idx + 1)
            writer.add_scalars('current_strategy', {'rock': one_strategy[0], 'paper': one_strategy[1], 'scissor': one_strategy[2]}, idx + 1)
            writer.add_scalars('total_regret', {'rock': one_cum_regrets[0], 'paper': one_cum_regrets[1], 'scissor': one_cum_regrets[2]}, idx + 1)
            writer.add_scalars('average_regret', {'rock': one_avg_regret[0], 'paper': one_avg_regret[1], 'scissor': one_avg_regret[2]}, idx + 1)

if __name__ == '__main__':
    main()