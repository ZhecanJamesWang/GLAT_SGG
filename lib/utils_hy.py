import numpy as np
import scipy.sparse as sp
import re
import torch
import pdb
import os

class Counter(object):
    def __init__(self, classes=2):
        # self.corr_cumul = 0
        # self.num_cumul = 0
        self.classes = classes
        self.correct = [0] * self.classes
        self.num_pred = [0] * self.classes
        self.num_label = [0] * self.classes

    def add(self, pred, labels):

        if pred.size()[-1] != 1 and len(pred.size()) != 1:
            preds = pred.max(1)[1].type_as(labels)
        else:
            preds = pred

        acc = preds == labels

        for i in range(self.classes):
            self.correct[i] += (preds[acc] == i).sum()
            # self.correct[i] += preds[preds == i].eq(labels[labels == i]).double()
            self.num_pred[i] += len(preds[preds == i])
            self.num_label[i] += len(labels[labels == i])

    def class_acc(self):
        # print("self.correct: ", self.correct)
        # print("np.asarray(self.num_pred): ", np.asarray(self.num_pred))
        return list(self.correct/np.asarray(self.num_pred))

    def overall_acc(self):
        # print("sum(self.correct): ", sum(self.correct))
        # print("sum(self.num_pred)： ", sum(self.num_pred))
        return float(sum(self.correct))/sum(self.num_pred)

    def recall(self):
        return list(self.correct/np.asarray(self.num_label))
