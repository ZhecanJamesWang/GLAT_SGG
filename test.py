import torch
import torch.nn as nn

# # x 224, 224, 3
#
# class Smallnet():
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(3, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
#         self.fc = nn.Linear()
#         self.relu = F.relu()
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = x.view(-1)
#         x = self.fc(x)
#         return x
#
#

x = torch.randn(2, 2)
x.requires_grad = True

lin0 = nn.Linear(2, 2)
lin1 = nn.Linear(2, 2)
lin2 = nn.Linear(2, 2)
x1 = lin0(x)

# with torch.no_grad():
#     x2 = lin1(x1)

x2 = lin1(x1)

# # x2.requires_grad = False
#
# for param in lin1.parameters():
#     param.requires_grad = False

x3 = lin2(x2)
x3.sum().backward()
print(lin0.weight.grad, lin1.weight.grad, lin2.weight.grad)
