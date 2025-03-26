from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, images, labels, standardize=True):
        # names (images, labels) taken from conditional image generation, to make the roles more clear (images=target (y), labels=conditioning (x))
        super().__init__()
        if standardize:
            self.images_means = images.mean(dim=0, keepdim=True)
            self.images_stds = images.std(dim=0, keepdim=True)
            images = (images - self.images_means) / self.images_stds
            
            self.labels_mean = labels.mean()
            self.labels_std = labels.std()
            labels = (labels - self.labels_mean) / self.labels_std
            
        self.images = images
        self.labels = labels
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
        
    def __len__(self):
        return len(self.labels)