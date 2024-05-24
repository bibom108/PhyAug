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
from torch.utils.data import DataLoader, RandomSampler
import copy
import time

import _my_model as mod
from _my_manage_audio import AudioPreprocessor
from statsmodels.stats.proportion import proportion_confint

use_tf = False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # specify which GPU(s) to be used
device = torch.device("cuda")

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
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() #/ batch_size
    print("{} accuracy: {:>5}".format(name, accuracy / batch_size))
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
        if use_tf == True:
            tf = mod.get_tf_meetroom_loc2_45cm()
            audio_processor = mod.AudioPreprocessor(n_mels=config["n_mels"], n_dct_filters=config["n_dct_filters"], hop_ms=10, tf = tf)
            test_set.audio_processor = audio_processor
            test_loader = data.DataLoader(
                test_set,
                batch_size= config["batch_size"], # len(test_set),
                num_workers=32,
                collate_fn=test_set.collate_fn_with_tf)
        else:
            audio_processor = mod.AudioPreprocessor(n_mels=config["n_mels"], n_dct_filters=config["n_dct_filters"], hop_ms=10, tf = [], sigma_level=config["sigma_level"])
            test_set.audio_processor = audio_processor
            sampler = RandomSampler(test_set, replacement=True, num_samples=config["num_samples"])
            test_loader = data.DataLoader(
                test_set,
                sampler=sampler,
                batch_size= config["batch_size"], # len(test_set),
                num_workers=32,
                collate_fn=test_set.collate_fn)
    
    if not model:
        model = config["model_class"](config)
        print(model)
        model.load(config["input_file"])
    # print("loaded model...")
    if not config["no_cuda"]:
        model= nn.DataParallel(model)
        model.to(device)
    model.eval()
    results = 0
    total = 0
    print("length of test_loader ",len(test_loader),test_loader)
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.to(device)
            labels = labels.to(device)
            # print("input shape = ",model_in.shape)
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        results += print_eval("test", scores, labels)
        total += model_in.size(0)
    print(f"Accuracy: {results / total}")
    cert_acc = proportion_confint(results, total, alpha=2 * config["alpha"], method="beta")[0]
    print(f"Certified accuracy: {cert_acc}")
    print(total)


def main():
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model_4cmd_normalize_no_tf.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False, n_epochs=50, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_file=output_file, cache_size=32768, momentum=0.9, weight_decay=0.00001, 
        sigma_level = 0.1, num_samples=100000, alpha = 0.001)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()

    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)
    
    evaluate(config)

if __name__ == "__main__":
    main()
