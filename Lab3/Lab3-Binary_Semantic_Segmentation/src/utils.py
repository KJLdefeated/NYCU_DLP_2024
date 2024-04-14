import torch

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    with torch.no_grad():
        sum = 0
        pred_mask = pred_mask > 0.5
        pred_mask = pred_mask.float().flatten(start_dim=1)
        for i in range(pred_mask.shape[0]):
            intersection = torch.sum(pred_mask[i] * gt_mask[i])
            sum += 2.0 * intersection / (torch.sum(pred_mask[i]) + torch.sum(gt_mask[i]))
    return (sum / pred_mask.shape[0]).item()