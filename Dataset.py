import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9


class DataSetQM9(Dataset):
    
    def __init__(self,r_cut:float,path:str,self_edge:bool=False, device: torch.device = "cpu" ):
        super(DataSetQM9,self).__init__()
        self.data = QM9(root=path)
        self.r_cut=r_cut
        self.self_edge=self_edge
        self.device = device
    def edges_atoms(self,pos) -> (torch.Tensor,torch.Tensor,torch.Tensor):
        n_atoms=pos.shape[0]
        edges, r, r_ij_normalized = [], [], []

        for i in range(n_atoms):
            for j in range(i+1):
                if i == j and self.self_edge:
                    edges.append([i,j])
                diff = pos[j]-pos[i]
                r_ij = torch.linalg.norm(diff)
                #Filters r_ij less than r_cut
                if r_ij <= self.r_cut and i!=j:
                    edges.extend([i,j], [j,i])
                    r.extend([r_ij.item()] *2)
                    r_ij_normalized.extend([(diff/r_ij).tolist(),(-diff/r_ij).tolist()])
        return torch.tensor(edges), torch.tensor(r).unsqueeze(dim=-1), torch.tensor(r_ij_normalized)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx)-> torch.Tensor:
        edges, r, r_ij_normalized = self.edges_atoms(self.data[idx]['pos'])
        edges = edges.to(self.device)
        r = r.to(self.device)
        r_ij_normalized = r_ij_normalized.to(self.device)
        molecule = self.data[idx].clone().detach()

        return {'z':molecule['z'],'xyz' : molecule['pos'],'edges': edges, 'r_ij': r, 'r_ij_normalized': r_ij_normalized, 'targets': molecule['y'],'n_atom': molecule['z'].shape[0]}
        
    
    


