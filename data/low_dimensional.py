from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, images, labels, standardize=True):
        # names (images, labels) taken from conditional image generation, to make the roles more clear (images=target (y), labels=conditioning (x))
        super().__init__()
        self.orig_images = self.images = images
        self.orig_labels = self.labels = labels

        if standardize:
            self._standardize()
            
    def _standardize(self):
        self.images_means = self.images.mean(dim=0, keepdim=True)
        self.images_stds = self.images.std(dim=0, keepdim=True)
        self.images = (self.images - self.images_means) / self.images_stds
        
        self.labels_mean = self.labels.mean()
        self.labels_std = self.labels.std()
        self.labels = (self.labels - self.labels_mean) / self.labels_std
        
    def destandardize_image(self, x):
        if hasattr(self, "images_means") and hasattr(self, "images_stds"):
            return x * self.images_stds.to(x.device) + self.images_means.to(x.device)
        else:
            return x 


    def __getitem__(self, index):
        return self.images[index], self.labels[index]
        
    def __len__(self):
        return len(self.labels)