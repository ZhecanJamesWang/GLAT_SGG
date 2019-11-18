import torch
import os
import numpy as np
import datetime

now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M")
model_path = "saved_models/"

if not os.path.exists(model_path):
    os.makedirs(model_path)


def save_model(key, model, prefix, epoch, best_acc):
    diretory = os.path.join(model_path, date)

    if not os.path.exists(diretory):
        os.makedirs(diretory)

    diretory += ("/" + "_".join([prefix, key]) + ".pth")

    ckpt = dict(
        epoch=epoch,
        best_acc=best_acc,
        model=model.state_dict(),
    )

    torch.save(ckpt, diretory)
    print("model saved at: ", diretory)


class Counter(object):
    def __init__(self, prefix=""):
        self.best_values = {}
        self.prefix = prefix
        diretory = os.path.join(model_path, date)

        if not os.path.exists(diretory):
            os.makedirs(diretory)

    def add(self, key, value, model, epoch):
        if key in self.best_values:
            if float(value) > self.best_values[key]:
                self.best_values[key] = value
                save_model(key, model, self.prefix)
        else:
            self.best_values[key] = value
            save_model(key, model, self.prefix, epoch, self.best_values[key])


