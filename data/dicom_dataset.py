import os
import pydicom
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DicomDataset(Dataset):
    def __init__(self, dicom_dir, transform=None, normalize='minmax'):
        """
        Args:
            dicom_dir (str): Directory with all the DICOM files.
            transform (callable, optional): Optional transform to be applied on a sample.
            normalize (str): Type of normalization: 'minmax' or 'zscore'
        """
        self.dicom_dir = dicom_dir
        self.file_list = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        dcm_path = os.path.join(self.dicom_dir, self.file_list[idx])
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)

        # Normalization
        if self.normalize == 'minmax':
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
        elif self.normalize == 'zscore':
            img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        else:
            raise ValueError("normalize must be 'minmax' or 'zscore'")

        # Convert to PIL image (grayscale mode)
        img = Image.fromarray((img * 255).astype(np.uint8)).convert('L')

        # Define default transform if not provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # Converts to [0,1] and adds channel dimension
            ])

        img_tensor = self.transform(img)  # Shape: [1, 224, 224]

        return img_tensor
