import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv layer with leaky ReLU.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(
            out_channels
        )  # Batchnorm was invented only after YOLOv1, so this is not following the paper. Added this as it speeds up convergence, and makes training more stable.
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        # Batchnorm (for decent size batches)
        return self.leakyReLU(self.batchnorm(self.conv(x)))

        # # Layernorm (for small batches)
        # x = self.conv(x)
        # _, c, h, w = x.shape
        # self.layernorm = nn.LayerNorm([c, h, w]).to(x.device)
        # return self.leakyReLU(self.layernorm(x))


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, num_splits=7, num_boxes=2, num_classes=20):
        """
        YOLOv1.

        Inputs:
        - num_splits: Grid resolution, divide image into (num_splits x num_splits) cells
        - num_boxes: Number of bounding boxes for each grid cell
        - num_classes: Number of classes
        """
        super().__init__()
        self.in_channels, self.S, self.B, self.C = (
            in_channels,
            num_splits,
            num_boxes,
            num_classes,
        )
        self.conv_backbone = self._create_conv_backbone()
        self.fc_layers = self._create_fc_layers()

    def forward(self, x):
        x = self.conv_backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = torch.reshape(x, (-1, self.B * 5 + self.C, self.S, self.S))
        return x

    def _create_conv_backbone(self):
        layers = []

        # Block 1
        in_channels, out_channels = self.in_channels, 64
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(7, 7), stride=2, padding=3
            )
        ]
        layers += [nn.MaxPool2d((2, 2), 2)]

        # Block 2
        in_channels, out_channels = out_channels, 192
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        ]
        layers += [nn.MaxPool2d((2, 2), 2)]

        # Block 3
        in_channels, out_channels = out_channels, 128
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(1, 1), stride=1, padding="same"
            )
        ]
        in_channels, out_channels = out_channels, 256
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        ]
        in_channels, out_channels = out_channels, 256
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(1, 1), stride=1, padding="same"
            )
        ]
        in_channels, out_channels = out_channels, 512
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        ]
        layers += [nn.MaxPool2d((2, 2), 2)]

        # Block 4
        for _ in range(4):
            in_channels, out_channels = out_channels, 256
            layers += [
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=1,
                    padding="same",
                )
            ]
            in_channels, out_channels = out_channels, 512
            layers += [
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
            ]
        in_channels, out_channels = out_channels, 512
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(1, 1), stride=1, padding="same"
            )
        ]
        in_channels, out_channels = out_channels, 1024
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        ]
        layers += [nn.MaxPool2d((2, 2), 2)]

        # Block 5
        for _ in range(2):
            in_channels, out_channels = out_channels, 512
            layers += [
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=1,
                    padding="same",
                )
            ]
            in_channels, out_channels = out_channels, 1024
            layers += [
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same",
                )
            ]
        in_channels, out_channels = out_channels, 1024
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        ]
        in_channels, out_channels = out_channels, 1024
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1
            )
        ]

        # Block 6
        in_channels, out_channels = out_channels, 1024
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        ]
        in_channels, out_channels = out_channels, 1024
        layers += [
            ConvBlock(
                in_channels, out_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        ]

        return nn.Sequential(*layers)

    def _create_fc_layers(self):
        layers = []

        # Original (In paper)
        layers += [nn.Linear(1024 * self.S * self.S, 4096)]
        layers += [nn.LeakyReLU(0.1)]
        layers += [nn.Dropout(0.5)]
        layers += [nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))]

        # # Modified to reduce ram usage
        # layers += [nn.Linear(1024 * self.S * self.S, 1024)]
        # layers += [nn.LeakyReLU(0.1)]
        # layers += [nn.Dropout(0.5)]
        # layers += [nn.Linear(1024, self.S * self.S * (self.B * 5 + self.C))]

        return nn.Sequential(*layers)
