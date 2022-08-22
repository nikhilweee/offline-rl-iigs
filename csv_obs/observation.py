import csv
import torch


class Observation:

    def __init__(self, info_state, action, action_mask, reward, next_info_state):
        self.info_state = info_state
        self.action = action
        self.action_mask = action_mask
        self.reward = reward
        self.next_info_state = next_info_state

    def to_dict(self):
        return {
            'info_state': ''.join([str(int(x)) for x in self.info_state]), 
            'action': int(self.action),
            'action_mask': ''.join([str(int(x)) for x in self.action_mask]),
            'reward': float(self.reward),
            'next_info_state': ''.join([str(int(x)) for x in self.next_info_state])
        }

    def to_list(self):
        return [
            torch.tensor(self.info_state, dtype=torch.float32),
            torch.tensor(self.action),
            torch.tensor(self.action_mask),
            torch.tensor(self.reward, dtype=torch.float32),
            torch.tensor(self.next_info_state, dtype=torch.float32),
        ]
    
    def __repr__(self):
        str_list = [str(value) for value in self.to_dict().values()]
        return " ".join(str_list)

    @staticmethod
    def from_dict(obs_dict):
        info_state = [int(x) for x in obs_dict['info_state']]
        action = int(obs_dict['action'])
        action_mask = [int(x) for x in obs_dict['action_mask']]
        reward = float(obs_dict['reward'])
        next_info_state = [int(x) for x in obs_dict['next_info_state']]
        return Observation(info_state, action, action_mask, reward, next_info_state)


class ObservationBuffer:

    def __init__(self):
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].to_list()

    def append(self, obs):
        self.samples.append(obs)

    def extend(self, obs_list):
        self.samples.extend(obs_list)

    def to_csv(self, path):
        with open(path, 'w') as f:
            fieldnames = self.samples[0].to_dict().keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for obs in self.samples:
                writer.writerow(obs.to_dict())

    @staticmethod
    def from_csv(path, limit=None):
        buffer = ObservationBuffer()
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for idx, obs_dict in enumerate(reader):
                obs = Observation.from_dict(obs_dict)
                buffer.append(obs)
                if limit is not None and (idx + 1) == limit:
                        break
        return buffer
