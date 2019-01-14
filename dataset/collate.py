import torch
import numpy as np
from torch.utils.data import DataLoader

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_dim = max(map(lambda x: x[0].shape[self.dim], batch))

        # pad according to max_len
        new_batch = []
        for x, y, z in batch:

            nx = np.pad(x, ((0, max_dim-x.shape[0]), (0, 0), (0, 0), (0, 0)), 'constant')
            ny = np.pad(y, ((0, max_dim-y.shape[0])), 'constant', constant_values=-1)
            nz = np.pad(z, ((0, max_dim-z.shape[0]), (0, 0)), 'constant', constant_values=-1)

            new_batch.append((nx, ny, nz))

        # stack all
        im_in = np.stack([x[0] for x in new_batch], axis=0)
        gt = np.stack([x[1] for x in new_batch], axis=0)
        b_in = np.stack([x[2] for x in new_batch], axis=0)
        
        # convert to torch tensors
        im_in = torch.from_numpy(im_in)
        gt = torch.from_numpy(gt)
        b_in = torch.from_numpy(b_in)

        return im_in, gt, b_in

    def __call__(self, batch):
        return self.pad_collate(batch)