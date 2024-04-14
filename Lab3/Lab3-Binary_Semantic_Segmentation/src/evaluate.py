import torch
from tqdm import tqdm
from utils import dice_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model, data):
    # implement the evaluation function here
    model.eval()
    with torch.no_grad():
        sum = 0
        for sample in tqdm(data):
            image, mask = sample["image"].to(device), sample["mask"].to(device)
            pred_mask = model(image)
            sum += dice_score(pred_mask, mask)
    return sum / len(data)