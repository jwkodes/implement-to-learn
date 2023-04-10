from pathlib import PureWindowsPath
import torch
import torch.nn as nn
from utils import intersection_over_union


class YOLOLoss(nn.Module):
    """
    Loss function for YOLOv1.
    """

    def __init__(
        self,
        num_splits=7,
        num_boxes=2,
        num_classes=20,
        lambda_coord=5,
        lambda_noobj=0.5,
    ):
        """
        Inputs:
        - num_splits: Grid resolution, divide image into (num_splits x num_splits) cells
        - num_boxes: Number of bounding boxes for each grid cell
        - num_classes: Number of classes
        """
        super().__init__()
        self.S, self.B, self.C = num_splits, num_boxes, num_classes
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj

    def forward(self, pred, ground_truth):
        """
        Inputs:
        - pred: Prediction tensor (Nx[B*5+C]xSxS), order for axis 1 is p,x,y,w,h,...,class1...
        - ground_truth: Ground truth tensor (Nx[5+C]xSxS), order for axis 1 is p,x,y,w,h,...,class1...
        """

        # Permutate
        pred = torch.permute(pred, (0, 2, 3, 1))  # NxSxSx[B*5+C]
        ground_truth = torch.permute(ground_truth, (0, 2, 3, 1))  # NxSxSx[5+C]

        # Extract box information and calculate iou
        pred_boxes = torch.stack(
            torch.split(pred[:, :, :, : self.B * 5], 5, dim=-1)
        )  # BxNxSxSx5
        ground_truth_boxes = ground_truth[:, :, :, :5]  # NxSxSx5
        IoUs = intersection_over_union(
            pred_boxes[..., 1:], ground_truth_boxes[..., 1:]
        ).squeeze(
            -1
        )  # BxNxSxS
        _, chosen_boxes_idx = torch.max(
            IoUs, dim=0
        )  # NxSxS, value is [0,1,...,B] and represents the index of the best box

        # Generate matrix that denotes if j-th bounding box in cell i is responsible for that prediction
        temp = torch.ones_like(IoUs) * (
            torch.Tensor(range(self.B)).view(self.B, 1, 1, 1)
        ).to(
            IoUs.device
        )  # BxNxSxS, basically temp[i,:,:,:]=i
        I_ij_obj = temp == chosen_boxes_idx  # BxNxSxS
        del temp

        # Generate matrix that denotes if object appears in cell i
        I_i_obj = ground_truth[:, :, :, 1]  # NxSxS, either 0 or 1

        # Box loss
        mseloss = nn.MSELoss(reduction="sum")
        box_x_loss = mseloss(
            I_ij_obj * (pred_boxes[:, :, :, :, 1]),
            I_ij_obj * (ground_truth_boxes[:, :, :, 1]),
        )
        box_y_loss = mseloss(
            I_ij_obj * (pred_boxes[:, :, :, :, 2]),
            I_ij_obj * (ground_truth_boxes[:, :, :, 2]),
        )
        box_w_loss = mseloss(
            I_ij_obj * (pred_boxes[:, :, :, :, 3]),
            I_ij_obj * (ground_truth_boxes[:, :, :, 3]),
        )  # By right should square root this, according to the paper
        box_h_loss = mseloss(
            I_ij_obj * (pred_boxes[:, :, :, :, 4]),
            I_ij_obj * (ground_truth_boxes[:, :, :, 4]),
        )  # By right should square root this, according to the paper

        # Object loss
        object_loss = mseloss(
            I_ij_obj * (pred_boxes[:, :, :, :, 0]),
            I_ij_obj * (ground_truth_boxes[:, :, :, 0]),
        )

        # No object loss
        no_object_loss = mseloss(
            (~I_ij_obj) * (pred_boxes[:, :, :, :, 0]),
            (~I_ij_obj) * (ground_truth_boxes[:, :, :, 0]),
        )

        # Class loss
        class_loss = mseloss(
            I_i_obj.unsqueeze(-1) * (pred[:, :, :, self.B * 5 :]),
            I_i_obj.unsqueeze(-1) * (ground_truth[:, :, :, 5:]),
        )

        loss = (
            self.lambda_coord * (box_x_loss + box_y_loss + box_w_loss + box_h_loss)
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
