import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import pydicom
import os
import cv2

def normalization(image, ):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def resize(image, size: tuple):
    image = cv2.resize(image, size)
    return image

def recup(init_path: str, series_id: str, image_id: str):
    series_path = init_path + series_id
    for root, _, files in os.walk(series_path):
        for file in files:
            if file == (image_id + '.dcm'):
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

def segmentation(init_seg_path: str, series_id: str):
    seg_path = init_seg_path + series_id
    img = nb.load(seg_path + '.nii').get_fdata()
    mask = nb.load(seg_path + '_cowseg.nii').get_fdata().astype(int)
    print(img.shape)
    return img, mask