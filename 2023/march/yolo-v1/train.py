import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
import os
from loss import YOLOLoss
from dataset import VOCDatasetYOLO
from model import YOLOv1
import albumentations as A
import albumentations.pytorch
from torch.utils.tensorboard import SummaryWriter
import utils

# YOLO cell related hyperparameters
NUM_CELL_SPLITS = 7
NUM_CELL_BBOX = 2
NUM_CLASSES = 20

# Hyperparameters
LEARNING_RATE = (
    1e-6  # Initial learning rate (subsequent lr is controlled by lr scheduler)
)
BATCH_SIZE = 8
NUM_EPOCHS = 2000
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

# Other misc settings
LOAD_CHECKPOINT = False
LOAD_CHECKPOINT_FILEPATH = None
SAVE_CHECKPOINT = True
SAVE_CHECKPOINT_FREQ = 50
CHECKPOINT_DIR = "checkpoint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True
IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
TRAIN_CSV_FILE = (
    "data/train.csv"  # CSV file with the filenames of the images and labels
)
# TRAIN_CSV_FILE = (
#     "data/8examples.csv"  # CSV file with the filenames of the images and labels
# )


def train(train_dataloader, model, optimizer, loss_fn, val=False):
    avg_epoch_loss = 0

    # Set model to test mode if we are testing validation set
    if val:
        model.eval()

    for batch_idx, (image, label_tensor, _) in enumerate(tqdm.tqdm(train_dataloader)):
        image, label_tensor = image.to(DEVICE), label_tensor.to(DEVICE)
        pred = model(image)
        loss = loss_fn(pred, label_tensor)
        avg_epoch_loss += loss.item()

        if not val:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    avg_epoch_loss /= batch_idx + 1

    if val:
        model.train()

    return avg_epoch_loss


# Define the learning rate schedule (from YOLOv1 paper)
def lr_lambda_original(epoch, initial_lr=LEARNING_RATE):
    if epoch < 25:
        # Linearly increase the learning rate from 0.001 to 0.01 for the first 25 epochs
        return ((1e-02 - initial_lr) / 25 * epoch + initial_lr) / initial_lr
    elif epoch < 100:
        return 1e-02 / initial_lr
    elif epoch < 130:
        return 1e-03 / initial_lr
    else:
        return 1e-04 / initial_lr


# Define the learning rate schedule (modified)
def lr_lambda(epoch, initial_lr=LEARNING_RATE):
    if epoch < 200:
        # Linearly increase the learning rate from 0.001 to 0.01 for the first 200 epochs
        return ((1e-05 - initial_lr) / 200 * epoch + initial_lr) / initial_lr
    elif epoch < 400:
        return 1e-05 / initial_lr
    elif epoch < 600:
        return 1e-06 / initial_lr
    else:
        return 1e-07 / initial_lr


def main():
    model = YOLOv1(
        in_channels=3,
        num_splits=NUM_CELL_SPLITS,
        num_boxes=NUM_CELL_BBOX,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)
    # optimizer = optim.Adam(
    #     model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    # )
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )

    # Scheduler
    # # Schedule lr according to YOLOv1 paper
    # # Gradually increase lr from 1e-03 to 1e-02 (linearly for first 25 epochs)
    # # Hold at 1e-02 for 75 epochs
    # # Hold at 1e-03 for 30 epochs
    # # Hold at 1e-04 thereafter
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_original)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    loss_fn = YOLOLoss()

    # Prepare transforms
    # # PyTorch automatically normalises PIL Image to [0 1] when ToTensor() is used
    # torch_vision_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Resize((448, 448)),
    #     ]
    # )
    albumentations_transform = A.Compose(
        [
            A.Resize(448, 448),
            A.RandomCrop(384, 384, p=0.2),
            A.ToGray(p=0.1),
            A.Blur(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.ChannelShuffle(p=0.2),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Cutout(p=0.2),
            A.Resize(448, 448),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.pytorch.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=1024, min_visibility=0.1),
    )
    transform = albumentations_transform

    # Prepare train and val dataset
    train_val_dataset = VOCDatasetYOLO(
        TRAIN_CSV_FILE,
        IMAGE_DIR,
        LABEL_DIR,
        num_splits=NUM_CELL_SPLITS,
        num_boxes=NUM_CELL_BBOX,
        num_classes=NUM_CLASSES,
        transform=transform,
    )
    # Split into train and val
    gen = torch.Generator()
    gen.manual_seed(
        0
    )  # Ensure split is the same every time, in order to not cross contaminate when loading from old checkpoints
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [0.8, 0.2], generator=gen
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    print(
        f"Total train images: {len(train_dataset)}, total val images: {len(val_dataset)}"
    )

    # Prepare tensorboard
    tb_writer = SummaryWriter()

    # Load checkpoint
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(LOAD_CHECKPOINT_FILEPATH)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        start_epoch = checkpoint["epoch"] + 1
        best_val_mAP = checkpoint["val_mAP"]
        print(f"Loaded checkpoint file {LOAD_CHECKPOINT_FILEPATH}")
    else:
        start_epoch = 1
        best_val_mAP = -1.0

    # Train
    for epoch in range(start_epoch, NUM_EPOCHS):
        loss = train(train_dataloader, model, optimizer, loss_fn)

        # Evaluate performance on val set
        # Actually there are repeated computations in train and get_bboxes, TODO: Fix this
        val_loss = train(val_dataloader, model, None, loss_fn, val=True)
        pred_bboxes, target_bboxes = utils.get_bboxes(
            val_dataloader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )
        val_mAP = utils.mean_average_precision(
            pred_bboxes, target_bboxes, iou_threshold=0.5, box_format="midpoint"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "transform": transform,
            "S": NUM_CELL_SPLITS,
            "C": NUM_CLASSES,
            "B": NUM_CELL_BBOX,
            "val_mAP": val_mAP,
        }

        # Update the learning rate based on the schedule
        scheduler.step()

        # Update tensorboard graphs
        tb_writer.add_scalar("Train loss", loss, epoch)
        tb_writer.add_scalar("Val loss", val_loss, epoch)
        tb_writer.add_scalar("Val mAP", val_mAP, epoch)
        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Save model with best val mAP separately
        if val_mAP > best_val_mAP:
            best_val_mAP = val_mAP
            torch.save(
                state,
                os.path.join(CHECKPOINT_DIR, f"weights-best-val-mAP.pt"),
            )
            print(f"Saved checkpoint at epoch {epoch} as best mAP!")

        # Save model and epoch number
        if SAVE_CHECKPOINT and epoch % SAVE_CHECKPOINT_FREQ == 0 and epoch != 0:

            torch.save(
                state,
                os.path.join(CHECKPOINT_DIR, f"weights-epoch{epoch}.pt"),
            )
            print(f"Saved checkpoint at epoch {epoch}")
        print(
            f"Epoch: {epoch}, loss: {loss}, val loss: {val_loss}, val mAP: {val_mAP}, lr: {optimizer.param_groups[0]['lr']}"
        )


if __name__ == "__main__":
    main()
