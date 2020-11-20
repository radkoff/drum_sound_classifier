from functools import partial
import operator

import numpy as np
import torch
from torch.utils.data import DataLoader


def collate(batch, desired_len):
    '''
    Pads the shorter inputs of a batch so they all have the same shape.
    :param max_length: Don't let inputs go beyond this size.
    :return: Batched inputs, a tensor of floats of shape [batch_size, 1, n_mels, length]
    '''
    # Inputs are [1 x n_mels x n_frames]
    n_mels = batch[0][0].shape[1]
    tensor = torch.zeros((len(batch), 1, n_mels, desired_len), dtype=torch.float, requires_grad=False)

    for batch_i, (instance, target) in enumerate(batch):
        replace_len = min(instance.shape[2], desired_len)
        trimmed_instance = instance[:,:,:replace_len]
        tensor.data[batch_i,:,:,:replace_len] = torch.FloatTensor(trimmed_instance)

    return tensor, torch.LongTensor(list(map(operator.itemgetter(1), batch)))


def load(seq_dataset, batch_size, is_train, desired_len, num_workers=8):
    return DataLoader(seq_dataset, batch_size=batch_size, shuffle=is_train,
                      collate_fn=partial(collate, desired_len=desired_len),
                      num_workers=num_workers, pin_memory=True)
