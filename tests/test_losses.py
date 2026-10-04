import math

import torch

from losses import YOLO_Regression_Cosine_Loss


def test_loss_of_a_batch_without_aneurysm_is_finite():
    zeros = [torch.zeros(2, 2, 2, 2, 1), torch.zeros(2, 2, 2, 2, 3), torch.zeros(2, 2, 2, 2, 3)]

    loss = YOLO_Regression_Cosine_Loss()([[t.clone() for t in zeros]], [zeros])

    assert math.isfinite(loss.item())
