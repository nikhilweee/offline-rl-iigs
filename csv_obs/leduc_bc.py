import pyspiel
import torch
import logging
import argparse
from observation import ObservationBuffer
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from open_spiel.python.algorithms import exploitability


logging.basicConfig(format="%(asctime)s [%(levelname).1s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class MLP(nn.Module):

    def __init__(self, input_size=30, output_size=3, hidden_size=256, device='cpu'):
        super().__init__()
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
    
    def action_probabilities(self, game_state):
        info_state = game_state.information_state_tensor()
        legal_mask = game_state.legal_actions_mask()
        info_state = torch.tensor(info_state, device=self.device)
        legal_mask = torch.tensor(legal_mask, device=self.device)
        illegal_mask = (1 - legal_mask).bool()
        out =  self(info_state)
        out = out.masked_fill(illegal_mask, -float('inf'))
        out = nn.functional.softmax(out, dim=-1)
        probs = {idx: prob for idx, prob in enumerate(out) if prob.isfinite()}
        return probs


def log_gradients(named_params, writer):
    learnable = {name: param.grad.detach().norm()
        for name, param in named_params if param.grad is not None}
    
    def filter_norm(query=None):
        if query is not None:
            results = [value for key, value in learnable.items() if query in key]
        else:
            results = list(learnable.values())
        if not results:
            results = [torch.tensor(0.0)]
        return torch.norm(torch.stack(results))

    writer.add_scalar('grad_norm/all', filter_norm(), writer.step)
    writer.add_scalar('grad_norm/linear1', filter_norm('linear1'), writer.step)
    writer.add_scalar('grad_norm/linear2', filter_norm('linear2'), writer.step)
    writer.add_scalar('grad_norm/linear3', filter_norm('linear3'), writer.step)


def main(args):
    logger.info(f'Loading Trajectories')
    trajectories = ObservationBuffer.from_csv(args.traj)
    data_loader = DataLoader(trajectories, batch_size=1024, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using Device: {device}')
    writer = SummaryWriter(f'runs/bc/{args.label}')
    model = MLP(input_size=30, output_size=3, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger.info(f'Starting Training')
    for epoch in range(args.epochs):
        running_loss = 0.0
        for idx, batch in enumerate(data_loader):
            state, action, _, _, _ = batch
            state = state.to(device)
            action = action.to(device)

            pred = model(state)
            loss = criterion(pred, action)
            loss.backward()

            writer.step = epoch * len(data_loader) + (idx + 1)
            # log_gradients(model.named_parameters(), writer)

            optimizer.step()
            running_loss += loss.item()

            if (idx + 1) % (len(data_loader) // 50) == 0:
                avg_running_loss = running_loss / (len(data_loader) // 50)
                writer.add_scalar("loss", avg_running_loss, writer.step)
                game = pyspiel.load_game("leduc_poker", {"players": 2})
                conv = exploitability.exploitability(game, model)
                writer.add_scalar("conv", conv, writer.step)
                logger.info(f'epoch: {epoch + 1:02d} batch: {idx + 1:06d} loss: {avg_running_loss:.04f} conv: {conv:.04f}')
                running_loss = 0.0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', default=None)
    parser.add_argument('--label', default='test')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()
    main(args)
