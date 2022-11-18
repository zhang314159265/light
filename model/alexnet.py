import torch
from torch import nn, optim

class AlexNetCopied(nn.Module):
    """
    This is copied from torchvision
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = AlexNetCopied(num_classes=num_classes) 
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def get_example_inputs(self, batch_size=3, return_label=False):
        inputs = (torch.rand(batch_size, 3, 224, 224),)
        if not return_label:
            return inputs
        return inputs, torch.randint(0, self.num_classes, (batch_size,))

@torch.enable_grad()
def train(model, num_batches=2):
    model.train()
    ce = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(num_batches):
        inputs, label = model.get_example_inputs(return_label=True)

        pred = model(*inputs)
        loss = ce(pred, label).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

@torch.no_grad()
def predict(model):
    model.eval()
    inputs = model.get_example_inputs()
    pred = model(*inputs)
    print(f"pred {pred}")
    pred_cls = pred.max(dim=1)[1]
    print(f"pred_cls {pred_cls}")

torch.manual_seed(23)
model = AlexNet()
train(model)
predict(model)
