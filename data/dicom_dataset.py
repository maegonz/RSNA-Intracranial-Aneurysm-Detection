import numpy as np
import pandas as pd
import nibabel as nb
import torch
import pydicom
from pathlib import Path
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from process import get_aneurysm_present, get_modality
from sklearn.model_selection import train_test_split


class AneurysmDataset(Dataset):
    def __init__(self,
                 series_dir: str,
                 seg_dir: str,
                 dataframe: pd.DataFrame,
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
        self.df = dataframe
        self.normalize = normalize
        self.transform = transform

        # path of the series directory
        # .../series/
        self.series_dir = Path(series_dir)

        # list of paths of the series instance
        # .../series/SeriesInstanceUID/
        self.series_instance_list = sorted([s for s in self.series_dir.iterdir() if s.is_dir()])

        # path of segmentations directory
        # ../segmentation/
        self.seg_dir = Path(seg_dir)

        # list of paths of the segmentation files
        # ../segmentation/SeriesInstanceUID
        self.seg_list = sorted([p for p in self.seg_dir.iterdir() if p.suffix=='.nii' and not p.name.endswith('_cowseg.nii')])
        self.seg_dict = {p.name: p for p in self.seg_list}

    def __len__(self):
        return len(self.series_instance_list)

    def __getitem__(self, id: int):
        path_series_id = self.series_instance_list[id]
        # Series UID (e.g, '1.2.826.0.1.3680043.8.498.28151846385510404823380448236003102416')
        series_uid = path_series_id.name
        
        path_seg_id = self.seg_dict.get(f"{series_uid}.nii", False)

        volume = self._load_volume(path_series_id)
        
        if path_seg_id:
            mask = self._load_mask(path_series_id)
        else:
            mask = torch.zeros(volume.shape, dtype=volume.dtype)
        
        label = get_aneurysm_present(df=self.df, Series_UID=series_uid)
        modality = get_modality(df=self.df, Series_UID=series_uid)

        return {
            "volume": volume,
            "label": label,
            "mask": mask,
            "modality": modality,
            "uid": series_uid
        }
    
    def _load_volume(self, path_series_id: Path):
        """
        Loads and stacks all DICOM slices into a 3D tensor.
        """
        # Load DICOM slices
        dicom_files = list(path_series_id.glob('*.dcm'))
        dicoms = [pydicom.dcmread(file) for file in dicom_files]
        dicoms.sort(key=lambda dcm: int(dcm.InstanceNumber))

        # Slices stacked into 3D volume
        slices = [dcm.pixel_array.astype(np.float32) for dcm in dicoms]
        volume = np.stack(slices, axis=-1)  # [H, W, D]

        volume = self._transform_volume(volume)  # [1, 1, D, H, W]
        return volume

    def _transform_volume(self, volume):
        # Volume normalization
        if self.normalize == "minmax":
            vmin, vmax = volume.min(), volume.max()
            # transformation such that the min is 0 and the max is 1
            if vmax > vmin:
                volume = (volume - vmin) / (vmax - vmin)
        elif self.normalize == "zscore":
            # transformation such that the mean is 0 abd the standard deviation is 1
            mean, std = volume.mean(), volume.std()
            if std > 0:
                volume = (volume - mean) / std

        volume = torch.from_numpy(volume).unsqueeze(0).float()  # [1, H, W, D]
        volume = volume.unsqueeze(0)  # [N, Channels, Height, Width, Depth]
        volume = F.interpolate(
            volume,
            size=(256, 256, 96),  # [D, H, W]
            mode='trilinear',
            align_corners=False
        )
        volume = volume.permute(0, 1, 4, 2, 3)  # [N, Channels, Depth, Height, Width]
        return volume
    
    def _load_mask(self, path_seg_id: Path):
        path_seg_id = str(path_seg_id)

        img = nb.load(path_seg_id + '.nii').get_fdata()
        img = self._transform_volume(img)

        mask = nb.load(path_seg_id + '_cowseg.nii').get_fdata().astype(int)
        mask = self._transform_volume(mask)

        return img, mask
    
    def split(self,
              val_size: float=0.2,
              batch_size: int=1,
              num_workers: int=4,
              shuffle: bool=True):
        """
        Returns
        -------
        train_loader, val_loader
        """
        # indices of every series's path
        indices = list(range(len(self.series_instance_list)))

        # separation of indices into training and validations
        train_id, val_id = train_test_split(
            indices,
            test_size=val_size,
            shuffle=True,
            random_state=0
        )

        train_set = Subset(self, train_id)
        val_set = Subset(self, val_id)

        train_loader = DataLoader(
            train_set, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers
        )

        val_loader = DataLoader(
            val_set, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader