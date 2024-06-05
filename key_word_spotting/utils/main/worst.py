from collections import ChainMap
import argparse
import os
import random
import sys

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import copy
import time

import model as mod
from manage_audio import AudioPreprocessor
from trans import Transformation
from _core import Smooth

device = torch.device("cuda")
my_t = 300

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def print_eval(name, scores, labels, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    # print("{} accuracy: {:>5}, loss: {}".format(name, accuracy, loss), end=end)
    return accuracy.item()

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def evaluate(config, model=None, test_loader=None):
    # print("before test_loader")
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_set.audio_files_to_wav()
        # print("test data ", test_set[0])
        test_loader = data.DataLoader(
            test_set, 
            # drop_last=True,
            batch_size= config["batch_size"], # len(test_set),
            num_workers=32,
            collate_fn=test_set.collate_fn)

    if not model:
        model = config["model_class"](config)
        # print(model)
        model.load(config["input_file"])
    # print("loaded model...")
    if not config["no_cuda"]:
        # model= nn.DataParallel(model)
        model.to(device)
    model.eval()
    results = []
    total = 0
    nb_classes = 12
    # smoothed_classifier = Smooth(model, nb_classes, config["sigma"])
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    # print("length of test_loader ",len(test_loader),test_loader)
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        model_in = model_in.to(device)
        labels = labels.to(device)
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)

        # smoothed_classifier.base_classifier.eval()
        # forward_a, tt1 = smoothed_classifier._sample_noise(model_in, 10000, config["batch_size"])

        results.append(print_eval("test", scores, labels) * model_in.size(0))
        _, preds = torch.max(scores, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
        # labels = labels.to("cpu")
        # for i in range(len(forward_a)):
            # if labels[i] != 0:
            #     results += forward_a[i][labels[i]]
            # else:
            #     total -= sum(forward_a[i])
            # for j in range(len(forward_a[0])):
                # confusion_matrix[labels[i], j] += forward_a[i][j]
            confusion_matrix[t.long(), p.long()] += 1
        total += model_in.size(0)
        # total += tt1
    # print("final test accuracy: {}".format(sum(results) / total))
    # return confusion_matrix.diag(), confusion_matrix.sum(1), sum(results) / total
    return confusion_matrix.diag(), confusion_matrix.sum(1), (sum(confusion_matrix.diag()[2:]) / sum(confusion_matrix.sum(1)[2:]))*100


def main(sigma):
    data_folders = ["../../data/recorded_dataset/ATR", "../../data/recorded_dataset/clipon",
                    "../../data/recorded_dataset/maono", "../../data/recorded_dataset/USB",
                    "../../data/recorded_dataset/USBplug"
                    ]
    f = open("logs/log.txt",'w')

    model_name = f"worst_sigma={sigma}.pt"
    print("model name ",model_name)
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", model_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False, n_epochs=50, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_file=output_file, cache_size=32768, momentum=0.9, weight_decay=0.00001,
        sigma = sigma)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)

    for data_folder in data_folders:
        config["data_folder"] = data_folder
        pred, total, res = evaluate(config)
        print(f"Data folder: {data_folder}\n {pred} \n {total} \n {pred/total} \n {res}\n", file=f)
    f.close()

if __name__ == "__main__":
    for sigma in [0.5]:
        main(sigma)


