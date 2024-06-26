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

import _my_model as mod
from _my_manage_audio import AudioPreprocessor
from _my_trans import Transformation

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

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()
    print("{} accuracy: {:>5}, loss: {}".format(name, accuracy, loss), end=end)
    return accuracy.item()

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def train(config):
    output_dir = os.path.dirname(os.path.abspath(config["output_file"]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    
    # print(f'time to load data: {e-s}, train data: {train_set[0][0].shape}')
    
    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    if not config["no_cuda"]:
        model = nn.DataParallel(model)
        model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    # option to use TF from mic
    tf = mod.my_get_tf_phyaug(my_t)

    # transformation function
    trans = Transformation(tf=tf)
    
    train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
        num_workers = 32,
        collate_fn=train_set.collate_fn_transform_with_tf)
    dev_loader = data.DataLoader(
        dev_set,
        batch_size=min(len(dev_set), 64),
        shuffle=False,
        num_workers = 32,
        collate_fn=dev_set.collate_fn)
    test_loader = data.DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers = 32,
        collate_fn=test_set.collate_fn)
    
    step_no = 0
    train_start = time.time()
    for epoch_idx in range(config["n_epochs"]):
        epoch_start = time.time()
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.to(device)
                labels = labels.to(device)

            model_in = Variable(model_in, requires_grad=False)
            
            # Transformation
            model_in = trans(model_in).to(device)

            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
            print_eval("epoch {} train step #{}".format(epoch_idx,step_no), scores, labels, loss)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.to(device)
                    labels = labels.to(device)
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            print("final dev accuracy: {}".format(avg_acc))
            # implement early stopping
            # if avg_acc > 0.90 and abs(avg_acc - max_acc) < 0.005:
            #     print("early stopping saving best model...")
            #     model.save(config["output_file"])
            #     best_model=copy.deepcopy(model)
            #     max_acc = avg_acc
            #     break
            if avg_acc > max_acc:
                print("saving best model...")
                model.module.save(config["output_file"])
                best_model=copy.deepcopy(model)
                max_acc = avg_acc
        epoch_end = time.time()
        print("epoch ",epoch_idx, "execution time ",epoch_end-epoch_start)
    # evaluate(config, best_model, test_loader)
    train_end = time.time()
    print("train ended at ",epoch_idx, "total training time ",(train_end-train_start)/3600,"hours")
def main(sigma):
    model_name = f"aa1_sigma={sigma}.pt"
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
    train(config)


if __name__ == "__main__":
    for sigma in [0.5]:
        main(sigma)


# def evaluate(config, model=None, test_loader=None):
#     # print("before test_loader")
#     if not test_loader:
#         _, _, test_set = mod.SpeechDataset.splits(config)
#         print("test data ", test_set[0])
#         test_loader = data.DataLoader(
#             test_set,
#             batch_size= config["batch_size"], # len(test_set),
#             num_workers=32,
#             collate_fn=test_set.collate_fn)

#     if not model:
#         model = config["model_class"](config)
#         print(model)
#         model.load(config["input_file"])
#     # print("loaded model...")
#     if not config["no_cuda"]:
#         # model= nn.DataParallel(model)
#         model.to(device)
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     results = []
#     total = 0
#     print("length of test_loader ",len(test_loader),test_loader)
#     for model_in, labels in test_loader:
#         model_in = Variable(model_in, requires_grad=False)
#         if not config["no_cuda"]:
#             model_in = model_in.to(device)
#             labels = labels.to(device)
#             print("labels shape = ",labels.shape)
#         scores = model(model_in)
#         labels = Variable(labels, requires_grad=False)
#         loss = criterion(scores, labels)
#         results.append(print_eval("test", scores, labels, loss) * model_in.size(0))
#         total += model_in.size(0)
#     print("final test accuracy: {}".format(sum(results) / total))