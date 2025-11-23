import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def dice(pred, mask, epsilon=1e-07):
    """
    DICE metric provides a measure of the similarity between the predicted segmentation
    and the ground truth segmentation.
    Params
    -------

    Returns
    -------
    """

    pred_copy = pred.clone()

    pred_copy[pred_copy < 0] = 0
    pred_copy[pred_copy > 1] = 1

    intersection = abs(torch.sum(pred_copy* mask))
    union = abs(torch.sum(pred_copy) + torch.sum(mask))
    coeff = (2 * intersection + epsilon) / (union + epsilon)
    dice = 1 - coeff

    return dice

# def train_step(model,
#                volume,
#                label,
#                mask=None):
    
#     clf_logits, seg_pred = model(volume)

#     clf_loss = F.binary_cross_entropy_with_logits(clf_logits, label.float())

#     if mask is not None:
#         bce = F.binary_cross_entropy(seg_pred, mask)
#         dice = dice(seg_pred, mask)
#         seg_loss = bce + dice

def train(model,
         train_set: DataLoader,
         criterion,
         optimizer,
         device,
         epochs,
         val_set: DataLoader=None):
    """
    Train a PyTorch model
    """

    model = model.to(device)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0
        running_dc = 0

        loop = tqdm(train_set, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for idx, img_mask in enumerate(loop):
            img, mask = img_mask[0].float().to(device), img_mask[1].float.to(device)

            pred = model(img)
            optimizer.zero_grad()

            dc = dice(pred, mask)
            loss = criterion(pred, mask)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dc += dc.item()


        epoch_loss = running_loss / len(train_set.dataset)
        epoch_dc = running_dc / len(train_set.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")

        if val_set is not None:
            val_loss = evaluate(model, val_set, criterion, device)
            print(f"Val Loss:   {val_loss:.4f}")

def evaluate(model, 
             data_set: DataLoader,
             criterion,
             device):
    """
    Evaluate the model on a dataset.
    """

    model.eval()
    total_loss = 0
    # total_dc = 0

    with torch.no_grad():
        for inputs, labels in data_set:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # dc = dice(outputs, mask)
            total_loss += loss.item() * inputs.size(0)
            # total_dc += dc.item()
    
    return total_loss/len(data_set.dataset)