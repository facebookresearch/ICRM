# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import wandb
import torch.optim as optim
import numpy as np
import torch
import hashlib
import random
import inspect
from datetime import date
from functools import partial
from matplotlib import pyplot as plt
import torch.nn.functional as F
import collections
from collections import defaultdict
from pathlib import Path
import json
from typing import List, Dict, Union
from collections import OrderedDict
import operator
from numbers import Number
import torch.distributed as dist
import seaborn as sns
import tqdm

from query import Q

DEBUG = False  # Set to True when debugging

def d_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs) 
        

def set_seed(seed, use_cuda):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f'=> Seed of the run set to {seed}')
    
def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

class  Logger(object):
    def __init__(self, args, tags=None):
        super(Logger, self).__init__()
        print("=> Project is", args.project)
        self.args=args
        tags=[args.user, tags] if tags is not None else [args.user]
        if args.resume:
            self.run = wandb.init(project=args.project, id = args.run_id, entity=args.user, resume="must", tags=tags)
        elif not args.debug:
            self.run = wandb.init(project=args.project, name = self.args.run_name, entity=args.user, reinit=True, tags=tags)
        config = wandb.config 
        curr_date = date.today()
        curr_date = curr_date.strftime("%B %d, %Y")
        wandb.config.update({"curr_date": curr_date}, allow_val_change=True) 
        wandb.config.update(args, allow_val_change=True) 
           

    def log(self, info):
        if not self.args.debug:
            wandb.log(info)


    def finish(self):
        if not self.args.debug:
            self.run.finish()
   
class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
                 
def save_checkpoint(algorithm, optimizer,hparams, args, metric_results, output_dir, filename = 'checkpoint.pth.tar', save_best = False):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_hparams": hparams,
            "results": metric_results,
            "state_dict": algorithm.state_dict(),
            'optimizer' : optimizer.state_dict()
        }
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model based on loss
        if save_best:
            torch.save(save_dict, os.path.join(output_dir, 'checkpoint_best.pth.tar'))
        # Save checkpoint model
        else:
            torch.save(save_dict, os.path.join(output_dir, filename))         
        
def load_checkpoint(path, epoch=None, best_model=False):
    if os.path.exists(path):
        try :
            if best_model is True:
                file = os.path.join(path, 'checkpoint_best.pth.tar')
                checkpoint = torch.load(file)
                print("=> Loading pre-trained model '{}'".format(file))           
            elif epoch is not None:
                file = os.path.join(path, 'checkpoint_step{}.pth.tar'.format(epoch))
                checkpoint = torch.load(file)
                print("=> Loading pre-trained model '{}'".format(file))
            else:
                file = os.path.join(path, 'checkpoint.pth.tar')
                checkpoint = torch.load(file)
                print("=> Loading pre-trained model '{}'".format(file))
            return checkpoint
        except FileNotFoundError:
            raise AssertionError(f"Specified path to checkpoint {file} doesn't exist :(")
    else:
        raise AssertionError(f"Specified path to checkpoint {path} doesn't exist :(")
    return None


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch
                
class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError 
    
class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers, drop_last = False):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=drop_last
        )
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))
        self._bs = batch_size
        self._length = len(batch_sampler)
        self.dataset = dataset

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
    
def batch_transform_to_context(xs, ys, context_len, xstart_token = None, ystart_token = None):
    bs, xdim = xs.shape
    _, ydim = ys.shape
    zs = torch.cat((xs, ys), dim=1)                                      # zs has shape (bs, xdim + ydim)
    if (bs % context_len) == 0:
        eff_bs, seq_len = bs // context_len, context_len
        zs = zs.view(bs // context_len, context_len, -1)                 # zs has shape (effective bs, context, xdim + ydim)
    else:
        eff_bs, seq_len = 1, context_len
        zs = zs.unsqueeze(0)                                             # zs has shape (1, bs, xdim + ydim)  
    xs, ys = zs[:, :, :xdim], zs[:, :, xdim:]
    if xstart_token is not None:
        xstart_token = xstart_token.expand(eff_bs, xstart_token.shape[1], xstart_token.shape[2])
        xs = torch.cat((xstart_token, xs), dim=1)    
    if ystart_token is not None:
        ystart_token = ystart_token.expand(eff_bs, ystart_token.shape[1], ystart_token.shape[2])
        ys = torch.cat((ystart_token, ys), dim=1)
    # print(xs.shape, ys.shape)
    return xs, ys

def context_transform_to_batch(ps):
    bs, context_len, xdim = ps.shape
    ps =  ps.reshape(-1, xdim)
    return ps

def get_loss_function(loss_name):
    loss_dict = {
        "mse": torch.nn.MSELoss(),
        "cross_entropy": torch.nn.CrossEntropyLoss(),
        "binary_cross_entropy": partial(torch.nn.functional.binary_cross_entropy_with_logits),
    }
    if loss_name in loss_dict:
        return loss_dict[loss_name]
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

def extract_optimizer(optimizer_name, parameters, **kwargs):
    # print(f'=> Using {optimizer_name} optimizer...')
    optimizer_class = getattr(optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Filter any arguments pertaining to other optimizers
    optimizer_args = inspect.getfullargspec(optimizer_class.__init__).args[1:]
    valid_args = {k: v for k, v in kwargs.items() if k in optimizer_args}
    invalid_args = set(kwargs) - set(optimizer_args)

    # Just FYI
    for arg in invalid_args:
        print(f"=> Warning: Invalid argument for {optimizer_name} optimizer: {arg}")
    return optimizer_class(parameters, **valid_args)


def get_activation(activation_name):
    activation_name = activation_name.lower()

    activation_map = {
        'relu': F.relu,
        'tanh': F.tanh,
        'sigmoid': F.sigmoid,
        'softmax': F.softmax,
        'softplus': F.softplus,
        'softsign': F.softsign,
        'leakyrelu': F.leaky_relu,
        'prelu': F.prelu,
        'rrelu': F.rrelu,
        'elu': F.elu,
        'selu': F.selu,
        'celu': F.celu,
        'gelu': F.gelu,
        'logsigmoid': F.logsigmoid,
        'hardsigmoid': F.hardsigmoid,
        'hardswish': F.hardswish
    }

    if activation_name in activation_map:
        return activation_map[activation_name]
    else:
        raise ValueError("Unsupported activation function: {}".format(activation_name))



def mse(ys_pred, ys):
    return (ys - ys_pred).square().mean().item()

def accuracy(ys_pred, ys):
    if ys_pred.size(1) == 1:
        correct = (ys_pred.gt(0).eq(ys).float()).sum().item()
    else:
        correct = (ys_pred.argmax(1).eq(ys).float()).sum().item()
    return 1. * correct / len(ys)

def compute_metric(metrics, ys_pred, ys):
    metric_dict = {
        "mse": partial(mse),
        "accuracy": partial(accuracy),
        "acc": partial(accuracy)
    }
    result = {metric: 0 for metric in metrics}
    for metric in metrics:
        if metric in metric_dict:
            result[metric] = metric_dict[metric](ys_pred, ys)
        else:
            raise ValueError(f"Unsupported metric function: {metric}")
    return result
    
def compute_additional_metrics(metrics: List[str], values: List[Dict[str, Union[int, float]]]) -> Dict[str, Union[int, float]]:
    """
    Computes additional metrics based on the given metrics and values.
    Args:
        metrics (List[str]): A list of metrics to compute. Supported metrics are 'worst_task' and 'average'.
        values (List[Dict[str, Union[int, float]]]): A list of dictionaries, where each dictionary represents a set of values for each metric.
    Returns:
        Dict[str, Union[int, float]]: A dictionary containing the computed additional metrics.
    """
    if metrics is None:
        return {}
    info = {}
    for metric in metrics:
        if metric == 'worst_group':
            out = {'wo_' + key: min(d[key] for d in values) for key in values[0].keys()}
        elif metric == 'average':
            out = {'avg_' + key: sum(d[key] for d in values) / len(values) for key in values[0].keys()}
        info.update(out)
    return info

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)
    
    
def extract_last_layer(model):
    if isinstance(model, torch.nn.Module):
        last_linear_layer = None
        
        # Find the last linear layer in the model
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear_layer = module
        
        if last_linear_layer is not None:
            return last_linear_layer
        else:
            raise ValueError("No linear layer found in the model")
    else:
        raise ValueError("Invalid model type. Expected torch.nn.Module.")



def find_device(model):
    if isinstance(model, torch.nn.Module):
        first_param = next(model.parameters(), None)        
        if first_param is not None:
            return first_param.device
        else:
            raise ValueError("Model has no parameters")
    else:
        raise ValueError("Invalid model type. Expected torch.nn.Module.")
    

def data_parallel(model):
    if torch.cuda.device_count() > 1:
        print("=> Let's use", torch.cuda.device_count(), "GPUs!")
        return torch.nn.DataParallel(model)
    else:
        return model
    
def setup(rank, world_size):
    # Use SLURM environment variables to set up distributed training.
    os.environ['MASTER_ADDR'] = os.environ['SLURM_JOB_NODELIST'].split(',')[0]  # Assumes a comma-separated list of nodes
    os.environ['MASTER_PORT'] = '12355'
    
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# taken from LEAF code
def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def get_samples_for_test(num_test_samples, loader):
    samples_x, samples_y = [], []
    remaining_elements = num_test_samples
    # Iterate over the data loader
    for x, y in loader: 
        if remaining_elements <= len(x):                                                   # If the remaining elements can be fulfilled by the current batch
            samples_x.append(x[:remaining_elements])
            samples_y.append(y[:remaining_elements])
            break
        samples_x.append(x)
        samples_y.append(y)
        remaining_elements -= len(x)
        if remaining_elements == 0:
            break
    test_samples_x = torch.cat(samples_x, dim=0)
    test_samples_y = torch.cat(samples_y, dim=0)
    return test_samples_x, test_samples_y



class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []
    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def model_difference(model1, model2):
    """Check if parameters of two models are different."""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            return True
    return False


# Reporting final numbers for sweeps
def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        results_path = os.path.join(path, subdir, "results.jsonl")
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        for test_env in r["args"]["test_envs"]:
            group = (r["args"]["trial_seed"],
                r["args"]["dataset"],
                r["args"]["algorithm"],
                r["args"]["shuffle_ctxt"],
                test_env)
            result[group].append(r)
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "shuffle_ctxt": s, "test_env": e, 
        "records": Q(r)} for (t,d,a,s,e),r in result.items()])


def calculate_mean_err(data):
    mean = 100 * np.mean(data)
    err = 100 * np.std(data) / np.sqrt(len(data))
    return "{:.1f} +/- {:.1f}".format(mean, err)

