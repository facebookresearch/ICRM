# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.import os

import os
import sys
import time
import argparse
import json
import random
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import utils
import dataset as dataset_file
import hparams_registry
import algorithms
import warnings
import string
import datetime
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('=> Home device: {}'.format(device))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default ='~/data/')
    parser.add_argument('--dataset', type=str, default="FEMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding shuffle dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=2001,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=30000,
        help='Checkpoint every N steps. Default is dataset-dependent.'
        )
    parser.add_argument('--print_freq', type=int, default=50,
        help='Printing after how many steps. Default is dataset-dependent.'
        )
    parser.add_argument('--eval_freq', type=int, default=50,
        help='Testing after how many steps. Default is dataset-dependent.'
        )
    parser.add_argument('--test_eval_freq', type=int, default=50,
        help='Testing after how many steps. Default is dataset-dependent.'
        )
    parser.add_argument('--test-args', default={},
        help='arguments for testing'
        )
    parser.add_argument('--custom-name', default='', type=str, help='custom log names to add')
    parser.add_argument('--colwidth', type=int, default=15)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[None])
    parser.add_argument('--output_dir', type=str, default="~/ICRM/results/")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--wandb', action='store_true', help = 'log into wandb')
    parser.add_argument('--run-name', type=str, default='', help='Choose name of wandb run')
    parser.add_argument('--project', default='', type=str, help='wandb project dataset_name')
    parser.add_argument('--user', default='', type=str, help='wandb username')
    parser.add_argument('--resume', action='store_true', help='resume wandb previous run')
    parser.add_argument('--run_id', default='', type=str, help='Run ID for Wandb model to resume')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--is_parallel', action='store_false')
    parser.add_argument('--show_env_results', action='store_true')
    parser.add_argument('--sweep', default=0, type=int, help='sweep mode or not')
    parser.add_argument('--mode', default='train', type=str, help='Training or inference', choices=['train', 'test'])


    args = parser.parse_args()
    start_step = 0
    
    if not args.sweep:
        if(args.run_name == ''):    
            args.run_name = f'{args.dataset}-{args.algorithm}-'
            args.run_name += ''.join(random.choice(string.ascii_letters) for i in range(10)) + '-' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        args.output_dir = os.path.join(args.output_dir, args.dataset)
        args.output_dir = os.path.join(args.output_dir, args.run_name, f'seed-{args.trial_seed}')   
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == 'train':
        sys.stdout = utils.Tee(os.path.join(args.output_dir, 'out.txt'))
        sys.stderr = utils.Tee(os.path.join(args.output_dir, 'err.txt'))
 
    print("=> Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))
    
    ## Path to save run artifacts
    os.environ['WANDB_DIR'] = args.output_dir
    if args.wandb:
        logger=utils.Logger(args)      
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, utils.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    
    hparams['device'] = device
    hparams['output_dir'] = args.output_dir
    hparams['overall_seed'] = args.seed
    hparams['is_parallel'] = args.is_parallel
    hparams['trial_seed'] = args.trial_seed
    hparams['terminal_command'] = " ".join(sys.argv)
    print('=> HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
        
    utils.set_seed(args.seed, device=="cuda")
    if hasattr(dataset_file, args.dataset):
        dataset = getattr(dataset_file, args.dataset)(args.data_dir, args.test_envs, hparams)               
    else:
        raise NotImplementedError 

    train_loaders = [utils.InfiniteDataLoader(
                    dataset=env,
                    weights=None,
                    batch_size=hparams['batch_size'],
                    num_workers=dataset.N_WORKERS)
                    for i, env in enumerate(dataset)]
    
    val_loaders = [utils.FastDataLoader(
                        dataset=env,
                        batch_size=hparams['test_batch_size'] if hparams.get('test_batch_size') is not None else hparams['batch_size'],
                        num_workers=dataset.N_WORKERS,
                        drop_last=True)
                        for i, env in enumerate(dataset.validation)]

    if dataset.holdout_test:
        test_loaders = [utils.FastDataLoader(
                        dataset=env,
                        batch_size=hparams['test_batch_size'] if hparams.get('test_batch_size') is not None else hparams['batch_size'],
                        num_workers=dataset.N_WORKERS,
                        drop_last=True)
                        for i, env in enumerate(dataset.holdout_test)]
    

    if args.algorithm == 'ICRM':
        validation_cache = [(x.to(device), y.to(device)) for x, y in zip(dataset.valid_cache_x,dataset.valid_cache_y) ]
        holdout_test_cache = [(x.to(device), y.to(device)) for x, y in zip(dataset.test_cache_x,dataset.test_cache_y) ]
    else:
        validation_cache, holdout_test_cache =  [(None, None) for i in dataset.valid_cache_x], [(None, None) for i in dataset.test_cache_x]
            
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, hparams)
              
    # Resume run
    if os.path.exists(os.path.join(args.output_dir, 'models', 'checkpoint.pth.tar')):
        ckpt = utils.load_checkpoint(os.path.join(args.output_dir, 'models'), epoch = None)   
        algorithm.load_state_dict(ckpt['state_dict'])
        start_step = ckpt['results']['step']
        algorithm.to(device)
        algorithm.optimizer.load_state_dict(ckpt['optimizer'])
        hparams['best_va'] = ckpt['model_hparams']['best_va']
        hparams['best_te'] = ckpt['model_hparams']['best_te']
        ckpt_metric = ckpt['model_hparams']['best_va']
        print(f'=> Checkpoint loaded and resuming at step {start_step}!')
    else:
        algorithm.to(device)
        hparams['best_va'], hparams['best_te'] = 0, 0

    train_minibatches_iterator = zip(*train_loaders)
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    ckpt_metric_name = algorithm._get_ckpt_metric()
    args.test_eval_freq = args.eval_freq if args.test_eval_freq is None else args.test_eval_freq
    print(f'=> Checkpointing based on {ckpt_metric_name}')
    
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        steps_per_epoch = min([len(env)/hparams['batch_size'] for env in dataset if env not in args.test_envs])
        info = {'step': step, 'step_time': time.time() - step_start_time}
        
        minibatches_device = [(x.to(device), y.to(device)) for x,y in next(train_minibatches_iterator)] # contains (x(e),y(e)) for all e where (x,y) are batches
        step_metrics = algorithm.update(minibatches_device)  
        info.update(step_metrics)


        ## Model evaluation (validation)
        if step % args.eval_freq == 0:
            consolidated_val_results = []
            for index, loader in enumerate(val_loaders):
                val_metric_results = algorithm.evaluate(loader, cache = validation_cache[index])                
                consolidated_val_results.append({f'va_{metric_name}': val for metric_name, val in val_metric_results.items()})
            consolidated_val_results = utils.compute_additional_metrics(hparams.get('additonal_metrics', ['acc']), consolidated_val_results)
            info.update(consolidated_val_results)
            ckpt_metric = consolidated_val_results[f'avg_va_{ckpt_metric_name}']
        
        ## Model evaluation (testing)
        if dataset.holdout_test and step % args.test_eval_freq == 0:
            consolidated_te_results = []
            env_te_results = {}
            for te_index, te_loader in enumerate(test_loaders):
                te_metric_results = algorithm.evaluate(te_loader, cache = holdout_test_cache[te_index])  
                consolidated_te_results.append({f'te_{metric_name}': val for metric_name, val in te_metric_results.items()})  
                if args.show_env_results:
                    env_te_results.update({f'te{te_index}_{metric_name}': val for metric_name, val in te_metric_results.items()})
            consolidated_te_results = utils.compute_additional_metrics(hparams.get('additonal_metrics', ['acc']), consolidated_te_results)
            info.update(consolidated_te_results)
            te_ckpt_metric = consolidated_te_results[f'avg_te_{ckpt_metric_name}']

        # Saving checkpoint and logging metrics (Don't save random training checkpoints)
        if args.save_model_every_checkpoint or step % checkpoint_freq == 0 and False:
            utils.save_checkpoint(algorithm, algorithm.optimizer, hparams, args, info, os.path.join(args.output_dir, 'models'), f'checkpoint_step{step}.pth.tar')
        
        if ckpt_metric >= hparams['best_va'] and step % args.test_eval_freq == 0:
            hparams['best_va'] = ckpt_metric
            hparams['best_te'] = te_ckpt_metric
            utils.save_checkpoint(algorithm, algorithm.optimizer, hparams, args, info, os.path.join(args.output_dir, 'models'), filename = None, save_best = True)
        utils.save_checkpoint(algorithm, algorithm.optimizer, hparams, args, info, os.path.join(args.output_dir, 'models'), filename = 'checkpoint.pth.tar', save_best = False)
   
        info['best_va'], info['best_te'] = hparams['best_va'], hparams['best_te']
        
        # Saving logs for sweeps and collecting results
        if step % args.test_eval_freq == 0:
            save_data = info.copy()
            save_data.update({'hparams': hparams, 'args': vars(args)})
            save_data['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
            save_data['hparams']['device'] = str(save_data['hparams']['device'])
            with open(os.path.join(args.output_dir, 'results.jsonl'), 'a') as f:
                f.write(json.dumps(save_data, sort_keys=True) + "\n")

        # Model training output      
        if step % args.print_freq == 0 or (step == n_steps - 1):
            if step == 0:
                utils.print_row([i for i in info.keys()], colwidth=args.colwidth)
            utils.print_row([info[key] for key in info.keys()], colwidth=args.colwidth)    
        
        if args.wandb:
            logger.log(info)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
        f.close()
                
        

    
    