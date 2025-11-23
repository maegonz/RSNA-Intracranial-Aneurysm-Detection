import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def dice_loss(pred, mask, epsilon=1e-07):
    """
    DICE metric provides a measure of the similarity between the predicted segmentation
    and the ground truth segmentation.

    Params
    -------
    pred

    mask
    
    Returns
    -------
    dice: float
        Float between [0;1]
    """

    pred_copy = pred.clone()

    pred_copy[pred_copy < 0] = 0
    pred_copy[pred_copy > 1] = 1

    intersection = abs(torch.sum(pred_copy* mask))
    union = abs(torch.sum(pred_copy) + torch.sum(mask))
    coeff = (2 * intersection + epsilon) / (union + epsilon)

    return 1 - coeff

def train_step(model,
               volume,
               label,
               mask=None):
    clf_logits, seg_pred = model(volume)
    clf_loss = F.binary_cross_entropy_with_logits(clf_logits, label)

    if mask is not None:
        bce = F.binary_cross_entropy(seg_pred, mask)
        dc = dice_loss(seg_pred, mask)
        seg_loss = bce + dc
    else:
        seg_loss, dc = 0, 0
    
    total_loss = clf_loss + seg_loss
    return total_loss, clf_loss, dc

def train(model,
         train_dl: DataLoader,
         optimizer,
         device,
         epochs,
         val_dl: DataLoader=None):
    """
    Train a PyTorch model
    """

    model = model.to(device)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0
        running_dc = 0

        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for item in loop:
            volume = item['volume'].to(device)
            label = item['label'].to(device)
            mask = item['mask'].to(device)

            loss, clf_loss, dc = train_step(model, volume, label, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dc += dc.item()


        epoch_loss = running_loss / len(train_dl.dataset)
        epoch_dc = running_dc / len(train_dl.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Train DC: {epoch_dc:.4f}")

        if val_dl is not None:
            val_loss = evaluate(model, val_dl, device)
            print(f"Val Loss:   {val_loss:.4f}")

def evaluate(model, 
             data_loader: DataLoader,
             device):
    """
    Evaluate the model on a dataset.
    """

    model.eval()
    total_loss = 0
    # total_dc = 0

    with torch.no_grad():
        for item in data_loader:
            volume = item['volume'].to(device)
            label = item['label'].to(device)
            
            clf_logits, seg_logits = model(volume)
            
            loss = F.binary_cross_entropy_with_logits(clf_logits, label)
            total_loss += loss.item()
            # dc = dice(outputs, mask)
            # total_dc += dc.item()
    
    return total_loss/len(data_loader)