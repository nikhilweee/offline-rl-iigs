"""
Run BC on an offline dataset for Leduc Poker.
"""

import pyspiel
import torch
import pickle
import logging
import argparse
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from open_spiel.python.algorithms import exploitability


logging.basicConfig(
    format="%(asctime)s [%(levelname).1s]: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(
        self, input_size=30, output_size=3, hidden_size=256, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.to(device)

    def forward(self, state):
        return self.layers(state)

    def action_probabilities(self, state):
        info_state = state.information_state_tensor()
        legal_mask = state.legal_actions_mask()
        info_state = torch.tensor(info_state, device=self.device)
        legal_mask = torch.tensor(legal_mask, device=self.device)
        illegal_mask = (1 - legal_mask).bool()
        out = self(info_state)
        out = out.masked_fill(illegal_mask, -float("inf"))
        out = nn.functional.softmax(out, dim=-1)
        probs = {
            idx: prob.item() for idx, prob in enumerate(out) if prob.isfinite()
        }
        return probs


def load_dataset(traj_path):
    dataset = []
    with open(traj_path, "rb") as f:
        trajectories = pickle.load(f)
    for episode in trajectories:
        for step in episode:
            state, action, _, _ = step
            if not state.is_player_node():
                continue
            info_state = state.information_state_tensor()
            dataset.append((torch.tensor(info_state), action))
    return dataset


def main(args):
    logger.info(f"Reading dataset: {args.traj}")
    dataset = load_dataset(args.traj)
    logger.info(f"Loaded dataset : {args.traj}")

    data_loader = DataLoader(dataset, batch_size=10240, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {device}")
    writer = SummaryWriter(f"runs/bc/{args.label}")
    model = MLP(input_size=30, output_size=3, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger.info(f"Starting Training")
    for epoch in range(args.epochs):
        running_loss = 0.0
        for idx, batch in enumerate(data_loader):
            state, action = batch
            state = state.to(device)
            action = action.to(device)

            pred = model(state)
            loss = criterion(pred, action)
            loss.backward()

            writer.step = epoch * len(data_loader) + (idx + 1)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj", default="trajectories/traj-010-824-4463-69032.pkl"
    )
    parser.add_argument("--label", default="default")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    main(args)
