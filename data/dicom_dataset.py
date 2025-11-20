"""
Class and get function implementations based on Rajveer Jadhav code, named muichimon on Kaggle and github.
Linked to his Github : https://github.com/muichi-mon
"""
import numpy as np
import pandas as pd
import torch
import os
import pydicom
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from process import get_aneurysm_present, get_modality, normalization
from sklearn.model_selection import train_test_split

class AneurysmDataset(Dataset):
    def __init__(self,
                 series_dir: str,
                 train_df: pd.DataFrame,
                 val_size: float=0.2,
                 transform=None, 
                 normalize: str='minmax'):
        """
        Params
        -------
        series_dir : str
            Directory with all the training-set series instances folders.
        transform : Optional
            Transform to be applied on a sample.
        normalize : str
            Type of normalization, 'minmax' or 'zscore'
        """
        self.df = train_df

        # path of the series directory
        # .../series/
        self.series_dir = Path(series_dir)

        train_dirs, val_dirs = train_test_split(train_df, test_size=val_size, shuffle=True, random_state=0)
        
        # list of paths of the series instance
        # .../series/SeriesInstanceUID/
        self.series_instance_list = sorted([s for s in self.series_dir.iterdir() if s.is_dir()])  
       
        self.transform = transform
        self.normalize = normalize


    def __len__(self):
        return len(self.series_dir)

    def __getitem__(self, id):
        path_series_id = self.series_instance_list[id]
        series_uid = path_series_id.name

        # Load DICOM slices
        dicom_files = list(path_series_id.glob('*.dcm'))
        dicoms = [pydicom.dcmread(file) for file in dicom_files]
        dicoms.sort(key=lambda dcm: int(dcm.InstanceNumber))

        slices = [dcm.pixel_array.astype(np.float32) for dcm in dicoms]

        # Slices stacked into 3D volume
        volume = np.stack(slices, axis=-1)  # [H, W, D]
        volume = torch.from_numpy(volume).unsqueeze(0).float()  # [1, H, W, D]

        label = get_aneurysm_present(df=self.df, Series_UID=series_uid)
        modality = get_modality(df=self.df, Series_UID=series_uid)

        return {
            "image": volume,
            "label": label,
            "modality": modality,
        }
