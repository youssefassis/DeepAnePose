import torch
from torch import nn as nn
from deepanepose.utils import get_logger

logger = get_logger("Loss function")

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        logger.info("CosineSimilarityLoss is used")
        
    def forward(self, prediction, target):
        return  1. - torch.nn.functional.cosine_similarity(prediction, target).mean()

class YOLO_Regression_Cosine_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine_loss = CosineSimilarityLoss()
        self.sigmoid = nn.Sigmoid()
        self.confidence_loss = nn.BCEWithLogitsLoss()
        
        logger.info(f"YOLO_Regression_Loss is used ({self.confidence_loss}, {self.mse}, {self.cosine_loss})")

    def compute_loss(self, prediction, target): 
        batch_size = prediction[0].shape[0]
        mask = target[0][..., 0]==1
        n_pos_cells = mask.sum().float()
        
        # Confidence Loss (the object term only exists when the batch has aneurysm cells)
        confidence_obj_loss = self.confidence_loss(prediction[0][..., 0][mask], target[0][..., 0][mask]) if n_pos_cells > 0 else 0.
        confidence_noobj_loss = self.confidence_loss(prediction[0][..., 0][~mask], target[0][..., 0][~mask])
        confidence_loss = confidence_obj_loss + 0.5 * n_pos_cells * confidence_noobj_loss
        
        # Center & Size Estimation
        regression_loss = 0.
        if n_pos_cells > 0:
            prediction[1] = self.sigmoid(prediction[1][..., 0:3]) # Normalize x y z coords (relative to grid size) ---- Center
            regression_loss = self.mse(prediction[1][..., 0:3][mask], target[1][..., 0:3][mask]) + self.mse(prediction[2][..., 0:3][mask], target[2][..., 0:3][mask])
            
        # Orientation Estimation
        cosine_loss = 0.
        if n_pos_cells > 0:
            vectors1 = (prediction[1][..., 0:3][mask] - (prediction[1][..., 0:3][mask] + prediction[2][..., 0:3][mask]))
            vectors2 = (target[1][..., 0:3][mask] - (target[1][..., 0:3][mask] + target[2][..., 0:3][mask]))
            cosine_loss = self.cosine_loss (vectors1, vectors2)
                        
        return (confidence_loss + 5 * (regression_loss + cosine_loss)) / batch_size
    
    def forward(self, predictions, targets):
        loss = 0.
        for i in range(len(predictions)):
            loss += self.compute_loss(predictions[i], targets[i])
        return loss

def get_loss_criterion(name):
    assert name is not None, 'Could not find any loss function'
    
    if name == "YOLO_Regression_Cosine_Loss":
        return YOLO_Regression_Cosine_Loss()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
