import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import pydicom
import os
import cv2

def normalization(image):
    """
    Normalize an input image to the range [0, 1].

    The function rescales pixel intensities so that the minimum value becomes 0
    and the maximum becomes 1.

    Params
    -------
    image : np.ndarray
        The input image to normalize.

    Returns
    -------
    np.ndarray
        The normalized image with values in the range [0, 1].
    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def resize(image, size: tuple):
    """
    Resize an image to the specified spatial dimensions.

    Params
    -------
    image : np.ndarray
        The input image to resize.
    size : tuple
        The desired output size in the format (width, height).

    Returns
    -------
    np.ndarray
        The resized image.
    """
    image = cv2.resize(image, size)
    return image


def recup(init_path: str, Series_UID: str, SOP_UID: str):
    """
    Return a specific image, identified by SOP_UID, from the series identified by Series_UID.

    Params
    -------
    init_path : str
        The root directory or initial path containing the DICOM data.
    Series_UID : str
        The UID of the series from which the image will be retrieved.
    SOP_UID : str
        The UID of the image within the specified series.

    Returns
    -------
    img : np.ndarray
        The requested image as a numpy array.
    """
    series_path = init_path + Series_UID
    for root, _, files in os.walk(series_path):
        for file in files:
            if file == (SOP_UID + '.dcm'):
                filepath = os.path.join(root, file)
                ds = pydicom.dcmread(filepath, force=True)
                img = ds.pixel_array.astype(np.float32)
                img = normalization(img)
    return img

def coordinates(crd, height, width):
    l_crd = crd.split(',')
    x, y = float(l_crd[0][6:-1]), float(l_crd[1][6:-1])
    return (x,y), (x-(height/2), y-(width/2))

def imshow(img, xy: tuple, height, width):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img, cmap='gray')
    rect = ptc.Rectangle(xy, height=height, width=width, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.show()

def segmentation(init_seg_path: str, Series_UID: str):
    """
    Load an image volume and its corresponding segmentation mask for a given series.

    The function reads a NIfTI image (.nii) and its associated segmentation file
    (suffix: `_cowseg.nii`) from the specified directory. Both volumes are returned
    as NumPy arrays.

    Parameters
    ----------
    init_seg_path : str
        Iinitial path where the segmentation files are stored.
    Series_UID : str
        The unique identifier of the image series to load.

    Returns
    -------
    img : np.ndarray
        The loaded image volume.
    mask : np.ndarray
        The corresponding segmentation mask as an integer array.
    """
    seg_path = init_seg_path + Series_UID
    img = nb.load(seg_path + '.nii').get_fdata()
    mask = nb.load(seg_path + '_cowseg.nii').get_fdata().astype(int)
    print(img.shape)
    return img, mask