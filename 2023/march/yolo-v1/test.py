import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
from model import YOLOv1
from loss import YOLOLoss
import albumentations as A
import albumentations.pytorch
from dataset import VOCDatasetYOLO

# Inputs
BATCH_SIZE = 4
SAVE_IMAGE_BBOX_OVERLAY = True  # To save image with bbox overlaid
SAVE_IMAGE_BBOX_OVERLAY_DIR = "result/"
SAVE_IMAGE_BBOX_OVERLAY_MAX_NUM = (
    100  # Maximum number of images with bbox overlay to save
)
CHECKPOINT_FILEPATH = "checkpoint/weights-best-val-mAP.pt"
TEST_CSV_FILE = "data/test.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"


def main():

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_FILEPATH)

    # YOLO cell related hyperparameters
    num_cell_splits = checkpoint["S"]
    num_cell_bboxes = checkpoint["B"]
    num_classes = checkpoint["C"]

    model = YOLOv1(
        in_channels=3,
        num_splits=num_cell_splits,
        num_boxes=num_cell_bboxes,
        num_classes=num_classes,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    loss_fn = YOLOLoss()

    # Prepare test dataset
    # transform = checkpoint["transform"]
    albumentations_transform = A.Compose(
        [
            A.Resize(448, 448),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.pytorch.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo"),
    )
    test_dataset = VOCDatasetYOLO(
        TEST_CSV_FILE,
        IMAGE_DIR,
        LABEL_DIR,
        num_splits=num_cell_splits,
        num_boxes=num_cell_bboxes,
        num_classes=num_classes,
        transform=albumentations_transform,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True
    )

    # Test and get bboxes
    pred_bboxes, target_bboxes = utils.get_bboxes(
        test_dataloader,
        model,
        iou_threshold=0.3,
        threshold=0.4,
        device=DEVICE,
        save_image_overlay=SAVE_IMAGE_BBOX_OVERLAY,
        save_image_overlay_dir=SAVE_IMAGE_BBOX_OVERLAY_DIR,
        save_image_overlay_max_num=SAVE_IMAGE_BBOX_OVERLAY_MAX_NUM,
    )

    mAP = utils.mean_average_precision(
        pred_bboxes, target_bboxes, iou_threshold=0.5, box_format="midpoint"
    )

    print(f"Test mAP: {mAP}")
    if SAVE_IMAGE_BBOX_OVERLAY and SAVE_IMAGE_BBOX_OVERLAY_DIR is not None:
        print(
            f"{SAVE_IMAGE_BBOX_OVERLAY_MAX_NUM} images with bounding box overlays should have been saved in {SAVE_IMAGE_BBOX_OVERLAY_DIR}!"
        )


if __name__ == "__main__":
    main()
