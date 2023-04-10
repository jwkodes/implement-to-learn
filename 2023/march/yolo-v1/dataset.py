import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import cv2
from yolo_utils import *


class VOCDatasetYOLO(torch.utils.data.Dataset):
    """
    VOC 2012 dataset for YOLOv1.
    """

    def __init__(
        self,
        csv_file,
        data_dir,
        annotation_dir,
        num_splits=7,
        num_boxes=2,
        num_classes=20,
        transform=None,
    ):
        """
        Inputs:
        - csv_file (string): Path to the csv file that maps image file to annotation file
        - data_dir (string): Directory with all the images
        - annotation_dir (string): Directory with all the annotations
        - num_splits: Grid resolution, divide image into (num_splits x num_splits) cells
        - num_boxes: Number of bounding boxes for each grid cell
        - num_classes: Number of classes
        - transform (callable, optional): Optional transform to be applied on a sample
        """

        self.file_mapping = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.S, self.B, self.C = num_splits, num_boxes, num_classes
        self.transform = transform

    def __len__(self):
        return len(self.file_mapping)

    def __getitem__(self, index):
        # Load image
        image_filename = self.file_mapping.iloc[index, 0]
        image = Image.open(os.path.join(self.data_dir, image_filename))
        if image is None:
            raise ValueError(
                f"Image is empty, check filepath -> {os.path.join(self.data_dir,image_filename)}"
            )

        # Load label text file
        label_filename = self.file_mapping.iloc[index, 1]
        boxes = []
        with open(os.path.join(self.annotation_dir, label_filename)) as f:
            for line in f.readlines():
                line = line.split()
                c = int(line[0])
                x = float(line[1])
                y = float(line[2])
                w = float(line[3])
                h = float(line[4])
                boxes.append([x, y, w, h, c])

        # Apply transform
        if self.transform:
            # # torchvision transform
            # # image,boxes=self.transform(image,boxes) # Commented out as I did not define my own custom transform that applies to bounding boxes
            # image = self.transform(image)

            # albumentations transform
            # Convert PIL image to numpy array
            image_np = np.array(image)
            augmented = self.transform(image=image_np, bboxes=boxes)
            image = augmented["image"]
            boxes = augmented["bboxes"]

            # # Debugging, visualise augmented images and bounding boxes
            # image_cv = np.array(image)
            # image_cv = np.moveaxis(image_cv, 0, -1)
            # image_cv = image_cv[:, :, ::-1].copy()
            # overlay_annotation(image_cv, boxes, True, "center", False)
            # cv2.imshow("image_cv", image_cv)
            # cv2.waitKey()

        # Generate ground truth tensor based on bounding boxes loaded previously
        # Ground truth tensor: ([5+C]xSxS), order for axis 0 is p,x,y,w,h,...,class1...
        gt_tensor = torch.zeros((5 + self.C, self.S, self.S))
        for box in boxes:
            # Class id, x_center, y_center, w, h
            # x,y,w,h are all normalised by image dimensions
            x, y, w, h, c = box
            c = int(c)

            # Determine cell that's in charge of this box
            cell_row = int(self.S * y)
            cell_col = int(self.S * x)

            # Convert x,y,w,h wrt to cell (as described in the paper)
            y_cell = self.S * y - cell_row
            x_cell = self.S * x - cell_col
            w_cell = w * self.S
            h_cell = h * self.S

            # Fill up tensor
            if (
                gt_tensor[0, cell_row, cell_col] == 0
            ):  # Do not override if this cell already has an object
                gt_tensor[0, cell_row, cell_col] = 1.0  # p, objectness score
                gt_tensor[1, cell_row, cell_col] = x_cell
                gt_tensor[2, cell_row, cell_col] = y_cell
                gt_tensor[3, cell_row, cell_col] = w_cell
                gt_tensor[4, cell_row, cell_col] = h_cell
                gt_tensor[5 + c, cell_row, cell_col] = 1.0

        return image, gt_tensor, image_filename
