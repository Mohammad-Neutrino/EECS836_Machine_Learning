import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid

import matplotlib
matplotlib.rcParams.update({'font.size': 10})
font = {'weight' : 'bold', 'size'   : 10}
matplotlib.rc('font', **font)

def compute_iou(preds, targets, threshold = 0.5):
    preds_bin = preds > threshold
    targets_bin = targets > threshold
    intersection = (preds_bin & targets_bin).float().sum()
    union = (preds_bin | targets_bin).float().sum()
    return (intersection / union).item() if union > 0 else 0.0

def compute_f1(preds, targets, threshold = 0.5):
    preds_bin = preds > threshold
    targets_bin = targets > threshold
    tp = (preds_bin & targets_bin).float().sum()
    fp = (preds_bin & ~targets_bin).float().sum()
    fn = (~preds_bin & targets_bin).float().sum()
    return (2 * tp / (2 * tp + fp + fn)).item() if tp + fp + fn > 0 else 0.0

def visualize_predictions(model, dataloader, device, save_dir, epoch):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            preds = sigmoid(model(x)).cpu().numpy()
            inputs = x.cpu().numpy()
            labels = y.cpu().numpy()

            for i in range(min(len(inputs), 3)):  # Show 3 images max
                img = inputs[i][0]
                pred_mask = preds[i][0] > 0.5
                true_mask = labels[i][0] > 0.5

                fig, ax = plt.subplots(1, 3, figsize = (15, 5))
                ax[0].imshow(img, cmap = "gray", aspect = "auto")
                ax[0].set_title("Input Echogram", fontweight = 'bold')

                ax[1].imshow(img, cmap="gray", aspect = "auto")
                ax[1].imshow(true_mask, cmap = "Reds", alpha = 0.5)
                ax[1].set_title("Ground Truth", fontweight = 'bold')

                ax[2].imshow(img, cmap = "gray", aspect = "auto")
                ax[2].imshow(pred_mask, cmap = "Blues", alpha = 0.5)
                ax[2].set_title("Prediction", fontweight = 'bold')

                for a in ax:
                    a.axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"epoch{epoch}_sample{idx}_{i}.png"))
                plt.close()

