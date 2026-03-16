import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

SEQ_LEN = 450
CHANELL = range(0, 8)
# CHANELL = [0, 1, 2, 4, 5, 6, 7]
# CHANELL = [0, 1, 2, 4, 5, 6]
# CHANELL = [0, 2]
# CHANELL = [0]
# CHANELL = [1, 2, 5, 6]

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file
        self.transform = transform

        # Open the file once to read dataset length
        with h5py.File(self.h5_file, 'r') as f:
            self.X = f['data'][:]  # features
            self.y1 = f['angle_label'][:]  # labels for task 1
            self.y2 = f['distance_label'][:]  # labels for task 2
            # self.y1 = f['cartesian_x'][:]  # labels for task 1
            # self.y2 = f['cartesian_y'][:]  # labels for task 2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx][np.ix_(range(SEQ_LEN), CHANELL)]
        y1 = self.y1[idx]
        y2 = self.y2[idx]

        if self.transform:
            x = self.transform(x)

        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        amplitude = 0.4
        # noise = (torch.rand_like(x) - 0.5) * 2 * amplitude
        # noise = torch.randn_like(x) * amplitude
        # x = x + noise
        # y1 = torch.tensor(y1, dtype=torch.int64)
        # y2 = torch.tensor(y2, dtype=torch.int64)
        y1 = torch.tensor(y1, dtype=torch.float32)
        y2 = torch.tensor(y2, dtype=torch.float32)

        return x, y1, y2
    
if __name__ == "__main__":
    dataset = HDF5Dataset('train_data_reg.h5')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch_idx, (data, target1, target2) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        print(f"Target shape 1: {target1.shape}")
        print(f"Target shape 2: {target2.shape}")
        break  # Just to test the first batch