from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, batch_size=64, train_ratio=None,
    val_ratio=0.1, test_ratio=0.1, return_test=False, num_workers=1,
    pin_memory=False, **kwargs):

    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using  all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=train_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)

torch.manual_seed(137)

def get_train_val_test_split(dataset, batch_size=1,
                        num_workers=1, pin_memory=False,
                        train_ratio=0.6, val_ratio=0.2):
    total_size = len(dataset)
    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_set, val_set, test_set = random_split(dataset,
                            [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=val_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=test_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader