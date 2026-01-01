import torch
from torch.utils.data import Dataset
import h5py


class ExtEvtsInitConds(Dataset):
    def __init__(self, main_path, datafile, num_sims, device='cuda'):

        self.main_path = main_path
        self.datafile = datafile
        self.num_sims = num_sims
        self.device = device

        self.u_hat_0_list = []

        self.setup_init_lists_datafile()
        

    def setup_init_lists_datafile(self):
        print("Loading data (initial conditions) ...")
        
        with h5py.File(self.datafile, "r") as f:
            u_hat = f[f"velocity_train"][:self.num_sims]          

        u_hat = torch.tensor(u_hat, device=self.device)

        self.u_hat_0_list = [u_hat[b].clone() for b in range(self.num_sims)]


    def __getitem__(self, index):
        return self.u_hat_0_list[index].detach().clone()
    
    def __len__(self):
        return len(self.u_hat_0_list)   