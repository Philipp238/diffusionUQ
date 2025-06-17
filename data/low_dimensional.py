import torch
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, images, labels, standardize=True, std_params=None, dim_cat=0):
        # names (images, labels) taken from conditional image generation, to make the roles more clear (images=target (y), labels=conditioning (x))
        super().__init__()
        self.orig_images = self.images = images
        self.orig_labels = self.labels = labels

        self.dim_cat = dim_cat

        if standardize:
            self._standardize(std_params=std_params)
            
    def _standardize(self, std_params=None):
        if std_params is None:
            if self.dim_cat == 0:
                self.labels_mean = self.labels.mean(dim=0, keepdim=True)
                self.labels_std = self.labels.std(dim=0, keepdim=True)
            else:
                labels_mean_num = self.labels[:,:-self.dim_cat].mean(dim=0, keepdim=True)
                labels_mean_cat = torch.zeros((1, self.dim_cat))
                self.labels_mean = torch.concat([labels_mean_num, labels_mean_cat], axis=1)

                labels_std_num = self.labels[:,:-self.dim_cat].std(dim=0, keepdim=True)
                labels_std_cat = torch.ones((1, self.dim_cat))
                self.labels_std = torch.concat([labels_std_num, labels_std_cat], axis=1)

            self.images_mean = self.images.mean(dim=0, keepdim=True)
            self.images_std = self.images.std(dim=0, keepdim=True)
        else:
            self.images_mean = std_params["images"]["mean"]
            self.images_std = std_params["images"]["std"]
            self.labels_mean = std_params["labels"]["mean"]
            self.labels_std = std_params["labels"]["std"]

        self.images = (self.images - self.images_mean) / self.images_std
        self.labels = (self.labels - self.labels_mean) / self.labels_std
        
    def destandardize_image(self, x):
        if hasattr(self, "images_mean") and hasattr(self, "images_std"):
            return x * self.images_std.to(x.device) + self.images_mean.to(x.device)
        else:
            return x 


    def get_std_params(self):
        return {
            "images": {
                "mean": self.images_mean,
                "std": self.images_std,
            },
            "labels": {
                "mean": self.labels_mean,
                "std": self.labels_std,
            }
        }

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
        
    def __len__(self):
        return len(self.labels)