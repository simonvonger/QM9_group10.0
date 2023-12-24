import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from DataImport import DataSetQM9


class DataLoaderQM9(DataLoader):
    def __init__(self, datapath: str = "data", batch_size: int = 50, r_cut: float = 5., self_edge: bool = False,
                 test_split: float = 0.1, val_split: float = 0.2, nworkers: int = 2, device: torch.device = "cpu"):
        self.r_cut = r_cut
        self.Dataset = DataSetQM9(path=datapath, r_cut=r_cut, self_edge=self_edge)
        self.length = len(self.Dataset)
        self.train_sampler = SubsetRandomSampler(np.array(range(self.length)))
        self.valid_sampler = self.test_sampler = None
        self.device = device

        if test_split:
            self.test_sampler, self.train_sampler = self._split(test_split)
        if val_split:
            self.valid_sampler, self.train_sampler = self._split(val_split)

        self.init_kwargs = {'batch_size': batch_size, 'num_workers': nworkers}
        super().__init__(self.Dataset, sampler=self.train_sampler, collate_fn=self.collate_fn, **self.init_kwargs)

    def collate_fn(self, data, pin_memory = True):

        batch_dict = {k: [dic[k] for dic in data] for k in data[0].keys()} 

        if pin_memory:
            pin = lambda x: x.pin_memory()
        else:
            pin = lambda x: x

        n_atoms = torch.tensor(batch_dict["n_atom"])
        
        ids = torch.repeat_interleave(torch.tensor(range(len(batch_dict['n_atom']))), n_atoms)

        edges_coord = torch.cumsum(torch.cat((torch.tensor([0]), n_atoms[:-1])), dim=0)
        neighbours = torch.tensor([local_neigh.shape[0] for local_neigh in batch_dict['edges']])
        edges_coord = torch.cat([torch.repeat_interleave(edges_coord, neighbours).unsqueeze(dim=1), torch.repeat_interleave(edges_coord, neighbours).unsqueeze(dim=1)], dim=1)
        edges_coord += torch.cat(batch_dict['edges'])

        return {
            'z': torch.cat(batch_dict['z']).to(self.device),
            'xyz': torch.cat(batch_dict['xyz']).to(self.device),
            'edges': edges_coord.to(self.device),
            'r_ij': torch.cat(batch_dict['r_ij']).to(self.device),
            'r_ij_normalized': torch.cat(batch_dict['r_ij_normalized']).to(self.device),
            'graph_idx': ids.to(self.device),
            'targets': torch.cat(batch_dict['targets']).to(self.device)
        }

    def _split(self, validation_split: float):
 
        train_idx = np.array(range(self.length))

        split_idx = np.random.choice(train_idx, int(self.length*validation_split), replace=False)
        
        train_idx = np.delete(train_idx, split_idx)
        split_idx = np.random.choice(
            train_idx,
            int(self.length * validation_split),
            replace=False
        )

        train_idx = np.setdiff1d(train_idx, split_idx)

        return SubsetRandomSampler(split_idx), SubsetRandomSampler(train_idx)

    def get_val(self) -> DataLoader:
        if self.valid_sampler is None:
            return DataLoader(self.Dataset, collate_fn=self.collate_fn, **self.init_kwargs, pin_memory=True)
        else:
            return DataLoader(self.Dataset, sampler=self.valid_sampler, collate_fn=self.collate_fn, **self.init_kwargs,
                              pin_memory=True)

    def get_test(self) -> DataLoader:
        if self.test_sampler is None:
            return DataLoader(self.Dataset, collate_fn=self.collate_fn, **self.init_kwargs, pin_memory=True)
        else:
            return DataLoader(self.Dataset, sampler=self.test_sampler, collate_fn=self.collate_fn, **self.init_kwargs,
                              pin_memory=True)
