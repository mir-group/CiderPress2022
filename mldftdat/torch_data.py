import numpy as np 
from torch.utils.data import Dataset, DataLoader
import os, subprocess
import json
from monty.json import MontyDecoder
from pyscf import lib
from workflow_utils import load_analyzer_data

dat_name = 'ML_DATA.json'


class ElectronDataset(Dataset):

    def __init__(self, dataset_directory):

        self.samples = get_mol_ids(dataset_directory)

    def __getitem__(self, idx):
        return load_analyzer_data(self.samples[idx])
