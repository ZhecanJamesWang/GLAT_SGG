"""
Linear Merge Model
"""

import torch
import torch.nn as nn
from torch.autograd import Variable



# class LinearMerge(nn.Module):
#     """
#     Module for Linear Merging Features
#     """
#     # def __init__(self, inputSize=100):
#     #     pass
#         # self.fc1 = nn.Linear(inputSize, 50)
#         # self.fc2 = nn.Linear(int(inputSize/2), int(inputSize/2))
#
#     def __init__(self):
#         super(LinearMerge, self).__init__()
#         self.fc1 = nn.Linear(200, 100)
#         self.fc2 = nn.Linear(100, 100)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, logit_base_predicate, logit_glat_predicate):
#         # x = torch.t(torch.cat((logit_base_predicate.data, logit_glat_predicate), dim=0))
#         x = torch.t(torch.cat((logit_base_predicate, logit_glat_predicate), dim=0))
#         x = self.fc1(Variable(x))
#         x = self.fc2(x).t()
#         x = self.softmax(x)
#         return x
    # def forward(self, logit_base_predicate, logit_glat_predicate):
    #     x = torch.t(torch.cat((logit_base_predicate, logit_glat_predicate), dim=0))
    #     x = self.fc1(x)
    #     x = self.fc2(x).t()
    #     x = self.softmax(x)
    #     return x


class LinearMerge(nn.Module):
    """
    Module for Linear Merging Features
    """
    # def __init__(self, inputSize=100):
    #     pass
        # self.fc1 = nn.Linear(inputSize, 50)
        # self.fc2 = nn.Linear(int(inputSize/2), int(inputSize/2))

    def __init__(self):
        super(LinearMerge, self).__init__()
        self.fc1 = nn.Linear(104, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 80)
        self.fc4 = nn.Linear(80, 52)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
