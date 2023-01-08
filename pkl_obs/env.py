"""
Train an environment model for Leduc Poker.
"""

import os
import sys
import argparse
import contextlib
import logging
import pickle

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format="%(asctime)s [%(levelname).1s]: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# https://stackoverflow.com/a/17954769/
@contextlib.contextmanager
def silence_stderr(to=os.devnull):
    stderr_fd = sys.stderr.fileno()
    orig_fd = os.dup(stderr_fd)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, stderr_fd)
    try:
        yield
    finally:
        os.dup2(orig_fd, stderr_fd)
        os.close(orig_fd)
        os.close(null_fd)


class RNN(nn.Module):
    def __init__(
        self, hidden_size=256, vocab_size=7, output_size=6, device="cpu"
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=6)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, state, slen):
        state = self.embed(state)
        state_packed = pack_padded_sequence(
            state, slen, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(state_packed)
        hidden = hidden.squeeze()
        out = self.linear(hidden)
        next_state = F.softmax(out, dim=-1)
        return next_state


class MLP(nn.Module):
    def __init__(
        self,
        input_size=12,
        embed_size=32,
        output_size=6,
        vocab_size=7,
        hidden_size=256,
        device="cpu",
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=6)
        self.layers = nn.Sequential(
            nn.Linear(input_size * embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.to(device)

    def forward(self, state, slen):
        state = self.embed(state)
        state = state.flatten(start_dim=1)
        next_state = self.layers(state)
        next_state = F.softmax(next_state, dim=-1)
        return next_state


class TrainDataset(Dataset):
    def __init__(self, path, device="cpu"):
        self.dataset = []
        self.device = device
        self.load_dataset(path)

    def load_dataset(self, path):
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
        for episode in trajectories:
            for step in episode:
                state, action, next_state, _ = step
                state_ser = [
                    int(x)
                    for x in state.serialize()
                    .replace("\n", " ")
                    .strip()
                    .split()
                ]
                next_state_ser = [
                    int(x)
                    for x in next_state.serialize()
                    .replace("\n", " ")
                    .strip()
                    .split()
                ]
                state_tensor = torch.tensor(
                    state_ser + [action], dtype=torch.long, device=self.device
                )
                # Use 6 as the padded value since there are only 6 actions (0-5)
                state_tensor = F.pad(
                    state_tensor, (0, 12 - len(state_tensor)), value=6
                )
                state_tensor_len = torch.tensor(
                    len(state_ser) + 1, dtype=torch.long, device="cpu"
                )
                next_state_tensor = torch.tensor(
                    next_state_ser[-1], dtype=torch.long, device=self.device
                )
                self.dataset.append(
                    (state_tensor, state_tensor_len, next_state_tensor)
                )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class EvalDataset(Dataset):
    def __init__(self, path, device="cpu"):
        self.dataset = []
        self.device = device
        self.load_dataset(path)

    def load_dataset(self, path):
        with open(path, "rb") as f:
            states = pickle.load(f)
        for state, choices in states.items():
            state_ser = [
                int(x) for x in state.replace("\n", " ").strip().split()
            ]
            state_tensor_len = torch.tensor(
                len(state_ser) + 1, dtype=torch.long, device="cpu"
            )
            for action, next_state in choices.items():
                state_tensor = torch.tensor(
                    state_ser + [action], dtype=torch.long, device=self.device
                )
                state_tensor = F.pad(
                    state_tensor, (0, 12 - len(state_tensor)), value=6
                )
                next_state_ser = [
                    int(x)
                    for x in next_state.replace("\n", " ").strip().split()
                ]
                next_state_tensor = torch.tensor(
                    next_state_ser[-1], dtype=torch.long, device=self.device
                )
                self.dataset.append(
                    (state_tensor, state_tensor_len, next_state_tensor)
                )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class Predictor:
    def __init__(self, model, ckpt_path, game):
        self.model = model
        self.game = game
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        ckpt_model = ckpt["model"]
        self.model.load_state_dict(ckpt_model)

    def next_state(self, state, action):
        state_ser = [
            int(x) for x in state.serialize().replace("\n", " ").strip().split()
        ]
        state_tensor = torch.tensor(
            state_ser + [action], dtype=torch.long, device="cpu"
        )
        state_tensor = F.pad(state_tensor, (0, 12 - len(state_tensor)), value=6)
        state_tensor_len = torch.tensor(
            len(state_ser) + 1, dtype=torch.long, device="cpu"
        )
        pred = self.model(
            state_tensor.unsqueeze(0), state_tensor_len.unsqueeze(0)
        )

        _, labels = pred.sort(dim=-1, descending=True)

        for label in labels.squeeze():
            next_state_ser = state_ser + [label.item()]
            next_state_str = "\n".join(str(x) for x in next_state_ser) + "\n"
            # assert state.child(action).serialize() == next_state_str
            try:
                with silence_stderr():
                    next_state = self.game.deserialize_state(next_state_str)
                break
            except:
                continue

        return next_state


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Reading Datasets")
    train_dataset = TrainDataset(args.train_traj, device)
    eval_dataset = EvalDataset(args.eval_traj, device)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1024, shuffle=True)

    logger.info(f"Using Device: {device}")
    writer = SummaryWriter(f"runs/env/{args.label}")

    if args.model == "mlp":
        model = MLP(device=device)
    if args.model == "rnn":
        model = RNN(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1000

    def train_epoch(epoch):
        running_loss = 0.0
        for idx, batch in enumerate(train_loader):
            model.zero_grad()
            state, slen, next_state = batch
            pred = model(state, slen)
            loss = criterion(pred, next_state)
            loss.backward()
            optimizer.step()

            running_loss = ((running_loss * idx) + loss.item()) / (idx + 1)
            writer.step = epoch * len(train_loader) + (idx + 1)

            # print statistics
            writer.add_scalar("loss", running_loss, writer.step)
        logger.info(
            f"epoch: {epoch + 1:02d} batch: {idx + 1:03d} loss: {running_loss:.04f}"
        )

        # necessary to access best_loss in outer scope
        nonlocal best_loss

        # save checkpoint
        if (epoch + 1) % 5 == 0 and running_loss < best_loss:
            best_loss = running_loss
            loss_str = f"{running_loss:.04f}".replace(".", "_")
            checkpoint_path = f"runs/env/{args.label}/model_epoch_{epoch + 1:06d}_loss_{loss_str}.pt"
            logger.info(f"saving model: {checkpoint_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "running_loss": running_loss,
                },
                checkpoint_path,
            )
            return checkpoint_path

        return None

    def eval_ckpt(epoch, ckpt_path):
        logger.info(f"loading model: {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        model.eval()

        correct, total = 0, 0
        for _, batch in enumerate(eval_loader):
            state, slen, next_state = batch
            pred = model(state, slen)
            _, label = pred.max(dim=-1)
            correct_batch = sum(label == next_state)
            correct += correct_batch
            total += next_state.numel()
        accuracy = correct / total
        writer.add_scalar("accuracy", accuracy, writer.step)

        logger.info(f"accuracy: {accuracy:.04f}")

    logger.info(f"Starting Training")
    for epoch in range(args.epochs):
        ckpt_path = train_epoch(epoch)
        if not ckpt_path:
            continue
        eval_ckpt(epoch, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_traj",
        default="trajectories/traj-010-824-4463-69032.pkl",
        required=False,
    )
    parser.add_argument(
        "--eval_traj",
        default="trajectories/traj-test-3937.pkl",
        required=False,
    )
    parser.add_argument("--label", default="default")
    parser.add_argument("--model", choices=["mlp", "rnn"], default="rnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    main(args)
