import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


################################################################################################################
# Prepare patient trajectories 
#
################################################################################################################
class PrepareMimic:
    def __init__(self, data_file, minibatch_size, context_dim=0, state_dim=40, num_actions=25):
        """
        Prepare a single dataset (train, validation, or test).
        """
        self.device = torch.device('cpu')
        self.data_file = data_file
        self.minibatch_size = minibatch_size
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.context_dim = context_dim

        # Load raw data
        self.demog, self.states, self.interventions, self.lengths, self.times, self.acuities, self.rewards = torch.load(self.data_file)
        idx = torch.arange(self.demog.shape[0])
        
        # Create dataset and dataloader
        self.dataset = TensorDataset(self.demog, self.states, self.interventions, self.lengths, self.times, self.acuities, self.rewards, idx)
        self.loader = DataLoader(self.dataset, batch_size=self.minibatch_size, shuffle=False)

    def create_dataset(self):
        """
        Process raw data into trajectories.
        """
        trajectories = []

        with torch.no_grad():
            for dem, obs, acu, _, timesteps, scores, rewards, _ in self.loader:
                dem = dem.to(self.device)
                obs = obs.to(self.device)
                acu = acu.to(self.device)
                scores = scores.to(self.device)
                rewards = rewards.to(self.device)

                max_length = int(timesteps.max().item())

                # Truncate to max_length
                obs = obs[:, :max_length, :]
                dem = dem[:, :max_length, :]
                acu = acu[:, :max_length, :]
                scores = scores[:, :max_length, :]
                rewards = rewards[:, :max_length]

                # Loop over all transitions
                for i_trans in range(obs.shape[0]):
                    trajectory = {
                        'observations': obs[i_trans].cpu().numpy(),
                        'dem_observations': torch.cat((obs[i_trans], dem[i_trans]), dim=-1).cpu().numpy(),
                        'actions': acu[i_trans].argmax(dim=-1).cpu().numpy(),
                        'rewards': rewards[i_trans].cpu().numpy(),
                        'acuities': scores[i_trans].cpu().numpy(),
                    }
                    trajectories.append(trajectory)

        return trajectories

    def save_dataset(self, output_file):
        """
        Save processed trajectories to a .pickle file.
        """
        trajectories = self.create_dataset()
        with open(output_file, 'wb') as f:
            pickle.dump(trajectories, f)


# Example usage
if __name__ == "__main__":
    # Paths to data
    train_file = "data/train_set_tuples"
    val_file = "data/val_set_tuples"
    test_file = "data/test_set_tuples"

    # Output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare and save train dataset
    train_processor = PrepareMimic(train_file, minibatch_size=32)
    train_processor.save_dataset(os.path.join(output_dir, "train_Phys45.pickle"))

    # Prepare and save validation dataset
    val_processor = PrepareMimic(val_file, minibatch_size=32)
    val_processor.save_dataset(os.path.join(output_dir, "val_Phys45.pickle"))

    # Prepare and save test dataset
    test_processor = PrepareMimic(test_file, minibatch_size=32)
    test_processor.save_dataset(os.path.join(output_dir, "test_Phys45.pickle"))
