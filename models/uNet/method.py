import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def dice_loss(pred, mask, epsilon=1e-07):
    """
    Computes the Dice loss between predicted segmentation probabilities and
    a binary ground-truth mask.

    Dice score ranges from 0 to 1 and measures the overlap between two sets.
    Dice *loss* is defined as `1 - dice_score`.

    Params
    -------
    pred : torch.Tensor
        Predicted segmentation map with values ideally in [0, 1].
        Shape: (N, C, D, H, W) or similar.
    mask : torch.Tensor
        Ground-truth binary mask with the same shape as `pred`.
    epsilon : float, optional
        Small constant added for numerical stability. Default is 1e-07.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing Dice loss (float in [0, 1]).
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
    """
    Performs a single forward-backward optimization step for a multi-task model
    performing classification and optional segmentation.

    Params
    -------
    model : nn.Module
        Model returning (classification_logits, segmentation_prediction).
    volume : torch.Tensor
        Input 3D volume batch, e.g. shape (N, C, D, H, W).
    label : torch.Tensor
        Binary classification labels for the batch.
    mask : torch.Tensor or None, optional
        Ground-truth segmentation mask. If None, segmentation loss is skipped.

    Returns
    -------
    total_loss : torch.Tensor
        Sum of classification loss and (optional) segmentation loss.
    clf_loss : torch.Tensor
        Classification loss component.
    dc : torch.Tensor or float
        Dice loss component. Zero if mask is None.
    """
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
    Trains a PyTorch model over multiple epochs.

    Params
    -------
    model : nn.Module
        Model to be trained.
    train_dl : DataLoader
        Dataloader providing training batches. Items must contain
        keys: 'volume', 'label', and 'mask'.
    optimizer : torch.optim.Optimizer
        Optimizer responsible for updating model parameters.
    device : str or torch.device
        Computing device ('cpu' or 'cuda').
    epochs : int
        Number of training epochs.
    val_dl : DataLoader or None, optional
        Optional validation dataloader. If provided, validation loss is evaluated
        at the end of each epoch.

    Returns
    -------
    None
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
    Evaluates the model on a dataset using classification loss only.

    Params
    -------
    model : nn.Module
        Model to evaluate.
    data_loader : DataLoader
        Dataloader providing evaluation batches.
    device : str or torch.device
        Device on which tensors should be executed.

    Returns
    -------
    float
        Mean classification loss over the dataset.
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