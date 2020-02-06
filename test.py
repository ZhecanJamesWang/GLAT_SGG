import torch

# x 224, 224, 3

class Smallnet():
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc = nn.Linear()
        self.relu = F.relu()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1)
        x = self.fc(x)
        return x


