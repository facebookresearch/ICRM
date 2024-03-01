# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import utils


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)

def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    
    SMALL_IMAGES = ['RareGroupRotatedMNIST', 'FEMNIST']
    
    hparams = {}

    def _hparam(name, default_val, random_val_fn, override=False):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(override or name not in hparams)
        random_state = np.random.RandomState(
            utils.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state)) 
  
    # Unconditional hparam definitions.
    _hparam('data_augmentation', True, lambda r: True)
    _hparam('freeze_bn', 0, lambda r: 0) 
    _hparam('resnet18', False, lambda r: False)
    _hparam('densenet121', False, lambda r: False)
    _hparam('resnet_dropout', 0, lambda r: 0)
    _hparam('is_transformer', 0, lambda r: 0)
    _hparam('loss','cross_entropy', lambda r: 'cross_entropy')
    _hparam('metrics',['acc'], lambda r: ['acc'])
    _hparam('optimizer_name', 'Adam', lambda r: 'Adam')
    _hparam('beta1', 0.5, lambda r: 0.5)
    _hparam('additonal_metrics', ['worst_group', 'average'], lambda r: ['worst_group', 'average']) 
    _hparam('n_sampled_tasks', 0, lambda r: 0)   
    _hparam('nonlinear_classifier', False, lambda r: False)
    _hparam('is_iid_tr', 0, lambda r: 0)
    _hparam('print_last', 1, lambda r: 1)
    _hparam('is_supervised', 0, lambda r: 0)

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-4, lambda r: 10**r.uniform(-4.5, -2.5))
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('num_features', 1, lambda r: 1)
        _hparam('batch_size', 100, lambda r: 100)
        _hparam('test_batch_size', 100, lambda r: 100)
    else:
        _hparam('lr', 1e-4, lambda r: 10**r.uniform(-5, -3.5))
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('num_features', 3, lambda r: 3)

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.
    if algorithm == 'ICRM':   
        _hparam('context_length', 100, lambda r: 100)                                   # because of start token
        _hparam('n_embd', 128, lambda r: 128)
        _hparam('n_layer', 12, lambda r: 12)
        _hparam('n_head', 4, lambda r: 4) 
        _hparam('freeze_bn', 1, lambda r: 1, override=True) 
    
    elif algorithm == 'ARM_CML': 
        _hparam('support_size', 100, lambda r: 100) 
        _hparam('adapt_bn', 0, lambda r: 0)
        _hparam('context_net', 'ContextNet', lambda r: 'ContextNet')            
        _hparam('test_support', None, lambda r: None) 

    elif algorithm == "TENT":
        _hparam('n_steps', 10, lambda r: 2)
        _hparam('episodic', 1, lambda r: 0)
       
    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.     
    
    if dataset == 'RareGroupRotatedMNIST':   
        _hparam('tr_fraction', 0.9, lambda r: 0.9) 
        _hparam('batch_size', 100, lambda r: 100, override = True)
        _hparam('test_batch_size', 1024, lambda r: 1024, override = True)  

    elif dataset == 'FEMNIST':   
        _hparam('n_sampled_tasks', 10, lambda r: 10, override = True)   
        
    elif dataset == 'TinyImagenet':    
        _hparam('n_sampled_tasks', 10, lambda r: 10, override = True)
        _hparam('batch_size',500, lambda r:500)
        _hparam('test_batch_size', 1000, lambda r: 1000) 
   
    elif dataset == 'WILDSCamelyon' : 
        _hparam('data_augmentation', False, lambda r: False, override = True)
        _hparam('batch_size', 400, lambda r: 400)
        _hparam('test_batch_size', 500, lambda r: 500)  
           
    return hparams
         
def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}



