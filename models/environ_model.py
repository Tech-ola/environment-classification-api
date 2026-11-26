import torch.nn as nn


class EnvironPredictModel(nn.Module):
    def __init__(
        self,
        input_feature: int,
        hidden_units: int,
        output_feature: int
    ):
        super().__init__()

        self.layer_stack_1 = nn.Sequential(
            nn.Conv2d(
                input_feature,
                hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_stack_2 = nn.Sequential(
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_units,
                hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_preperty = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 13 * 13,
                out_features=output_feature
            )
        )

    def forward(self, x):
        x = self.layer_stack_1(x)
        x = self.layer_stack_2(x)
        x = self.linear_preperty(x)
        return x
