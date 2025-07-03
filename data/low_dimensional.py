import torch
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, target, input, standardize=True, std_params=None, dim_cat=0):
        # names (target, input) taken from conditional image generation, to make the roles more clear (target=target (y), input=conditioning (x))
        super().__init__()
        self.orig_target = self.target = target
        self.orig_input = self.input = input

        self.dim_cat = dim_cat

        if standardize:
            self._standardize(std_params=std_params)
            
    def _standardize(self, std_params=None):
        if std_params is None:
            if self.dim_cat == 0:
                self.input_mean = self.input.mean(dim=0, keepdim=True)
                self.input_std = self.input.std(dim=0, keepdim=True)
            else:
                input_mean_num = self.input[:,:-self.dim_cat].mean(dim=0, keepdim=True)
                input_mean_cat = torch.zeros((1, self.dim_cat))
                self.input_mean = torch.concat([input_mean_num, input_mean_cat], axis=1)

                input_std_num = self.input[:,:-self.dim_cat].std(dim=0, keepdim=True)
                input_std_cat = torch.ones((1, self.dim_cat))
                self.input_std = torch.concat([input_std_num, input_std_cat], axis=1)

            self.target_mean = self.target.mean(dim=0, keepdim=True)
            self.target_std = self.target.std(dim=0, keepdim=True)
        else:
            self.target_mean = std_params["target"]["mean"]
            self.target_std = std_params["target"]["std"]
            self.input_mean = std_params["input"]["mean"]
            self.input_std = std_params["input"]["std"]

        self.target = (self.target - self.target_mean) / self.target_std
        self.input = (self.input - self.input_mean) / self.input_std
        
    def destandardize_output(self, x):
        if hasattr(self, "target_mean") and hasattr(self, "target_std"):
            return x * self.target_std.to(x.device) + self.target_mean.to(x.device)
        else:
            return x 


    def get_std_params(self):
        return {
            "target": {
                "mean": self.target_mean,
                "std": self.target_std,
            },
            "input": {
                "mean": self.input_mean,
                "std": self.input_std,
            }
        }

    def __getitem__(self, index):
        return self.target[index].unsqueeze(0), self.input[index].unsqueeze(0)
        
    def __len__(self):
        return len(self.input)