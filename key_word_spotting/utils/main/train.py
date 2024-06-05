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

from tqdm import tqdm
import model as mod
from manage_audio import AudioPreprocessor
from trans import Transformation
from dro import DRO
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

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
    # print("{} accuracy: {:>5}, loss: {}".format(name, accuracy, loss), end=end)
    return accuracy.item()

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def train(config):
    output_dir = os.path.dirname(os.path.abspath(config["output_file"]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    if not config["no_cuda"]:
        model.init_weights_glorot()
        model = nn.DataParallel(model)
        model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    # option to use TF from mic
    tf = mod.get_tf_phyaug(my_t)
    # tf = mod.my_get_tf_phyaug(my_t)

    # transformation function
    trans = Transformation(tf=tf, sigma=config['sigma'])
    trans.to(device)
    dro = DRO(writer=writer, sigma=config['sigma'])
    
    train_set, dev_set, _ = mod.SpeechDataset.splits(config)
    train_set.audio_files_to_wav()
    dev_set.audio_files_to_wav()

    # ap = AudioPreprocessor(tf=tf)
    # train_set.audio_processor = ap

    target_loader = None
    source_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
        num_workers = 8,
        collate_fn=train_set.collate_fn)
        # collate_fn=train_set.collate_fn_with_tf)
    max_train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
        num_workers = 8,
        collate_fn=None)
    dev_loader = data.DataLoader(
        dev_set,
        batch_size=min(len(dev_set), 64),
        shuffle=False,
        num_workers = 8,
        collate_fn=dev_set.collate_fn)
    
    # populate main_set
    main_set = mod.TensorDataset()

    # aug_x, aug_y = [], []
    # model.eval()
    # requires_grad_(model, False)
    # trans.train()
    # requires_grad_(trans, True)
    # max_bar = tqdm(max_train_loader, total=len(max_train_loader))
    # for _, (inputs, labels) in enumerate(max_bar):
    #     for input in dro.forward(inputs, labels, model, trans, _):
    #         if not torch.isnan(input): 
    #             aug_x.append(input.detach().clone().cpu())
    #             aug_y.append(labels.cpu())
    #         # if torch.isnan(torch.max(input)) or torch.isnan(torch.min(input)):
    #         #     print(labels)
    #         #     print(input)
    # exit()

    step_no = 0
    train_start = time.time()
    for epoch_idx in range(config["n_epochs"]):
        print("Start training .....")
        train_loader = target_loader if target_loader is not None else source_loader
        train_bar = tqdm(train_loader, total=len(train_loader))
        running_results = {
            "samples": 0,
            "loss": 0,
            "acc": 0,
            "nan_loss": 0
        }
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            if epoch_idx == 0:
                main_set.add(list(torch.unbind(inputs, dim=0)), list(torch.unbind(labels, dim=0)))
            
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = Variable(inputs, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

            model.train()
            requires_grad_(model, True)

            scores = model(inputs)
            loss = criterion(scores, labels)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            else:
                running_results["nan_loss"] += 1

            # writer.add_scalar("Loss/train", loss.item(), step_no)
            running_results["samples"] += 1
            running_results["loss"] += loss.item()
            batch_size = labels.size(0)
            accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
            running_results["acc"] += accuracy
            train_bar.set_description(
                desc="[%d/%d] Loss: %f Acc: %f Nan Loss: %d"
                % (
                    epoch_idx,
                    config["n_epochs"],
                    running_results["loss"] / running_results["samples"],
                    running_results["acc"] / running_results["samples"],
                    running_results["nan_loss"],
                )
            )

            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        
        # START DRO
        if (epoch_idx + 1) % 5 == 0 and (epoch_idx+1) != config["n_epochs"]:
        # if epoch_idx == 0:
            print("Start DRO .....")
            aug_x, aug_y = [], []
            model.eval()
            requires_grad_(model, False)
            trans.train()
            requires_grad_(trans, True)
            # max_bar = tqdm(max_train_loader, total=len(max_train_loader))
            max_bar = tqdm(source_loader, total=len(source_loader))
            for _, (inputs, labels) in enumerate(max_bar):
                # for input in dro.forward(inputs, labels, model, trans, _):
                #     if not torch.isnan(input).any(): 
                #         aug_x.append(input.detach().clone().cpu())
                #         aug_y.append(labels.cpu())
                aug_x.append(inputs.detach().clone().cpu())
                aug_y.append(labels.cpu())
            
            aug_x = list(torch.unbind(torch.cat(aug_x)))
            aug_y = list(torch.unbind(torch.cat(aug_y)))
            main_set.add(aug_x, aug_y)
            target_loader = data.DataLoader(
                main_set,
                batch_size=config["batch_size"],
                shuffle=True, 
                # drop_last=True,
                num_workers = 8)
        # END DRO
        
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            requires_grad_(model, False)
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
            print("saving model...")
            model.module.save(config["output_file"])

    train_end = time.time()
    print("train ended at ",epoch_idx, "total training time ",(train_end-train_start)/3600,"hours")


def main(sigma):
    model_name = f"ac05_sigma={sigma}.pt"
    print("model name ",model_name)
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", model_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False, n_epochs=50, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_file=output_file, 
        cache_size=32768, 
        momentum=0.9, weight_decay=0.00001,
        sigma = sigma)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    
    # config["input_file"] = "./model/ac1_sigma=0.5.pt"

    set_seed(config)
    train(config)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    for sigma in [0.5]:
        main(sigma)
