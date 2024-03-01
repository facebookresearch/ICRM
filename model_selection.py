# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import numpy as np


def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        sorted_reversed_list =  (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc_100'])[::-1]
        )
        
        return sorted_reversed_list


    @classmethod
    def sweep_acc(self, records, mode = 'avg'):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            if mode == 'wo':
                return _hparams_accs[0][0]['wo_test_acc_0'], _hparams_accs[0][0]['wo_test_acc_25'], _hparams_accs[0][0]['wo_test_acc_50'], _hparams_accs[0][0]['wo_test_acc_75'], _hparams_accs[0][0]['wo_test_acc_100']
            elif mode == 'avg':
                return _hparams_accs[0][0]['test_acc_0'], _hparams_accs[0][0]['test_acc_25'], _hparams_accs[0][0]['test_acc_50'], _hparams_accs[0][0]['test_acc_75'], _hparams_accs[0][0]['test_acc_100']
            else:
                raise NotImplementedError()
        else:
            return None

class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]                                    # Here add multiple envs
        return {
            # Worst group accuracy
            'wo_val_acc_0': record[f'wo_va_acc(e-100)'],
            'wo_val_acc_25': record[f'wo_va_acc(e-75)'],
            'wo_val_acc_50': record[f'wo_va_acc(e-50)'],
            'wo_val_acc_75': record[f'wo_va_acc(e-25)'],
            'wo_val_acc_100': record[f'wo_va_acc(e-0)'],

            'wo_test_acc_0': record[f'wo_te_acc(e-100)'],
            'wo_test_acc_25': record[f'wo_te_acc(e-75)'],
            'wo_test_acc_50': record[f'wo_te_acc(e-50)'],
            'wo_test_acc_75': record[f'wo_te_acc(e-25)'],
            'wo_test_acc_100': record[f'wo_te_acc(e-0)'],

            # Average accuracy
            'step': record[f'step'],
            'val_acc_0': record[f'avg_va_acc(e-100)'],
            'val_acc_25': record[f'avg_va_acc(e-75)'],
            'val_acc_50': record[f'avg_va_acc(e-50)'],
            'val_acc_75': record[f'avg_va_acc(e-25)'],
            'val_acc_100': record[f'avg_va_acc(e-0)'],

            'test_acc_0': record[f'avg_te_acc(e-100)'],
            'test_acc_25': record[f'avg_te_acc(e-75)'],
            'test_acc_50': record[f'avg_te_acc(e-50)'],
            'test_acc_75': record[f'avg_te_acc(e-25)'],
            'test_acc_100': record[f'avg_te_acc(e-0)']
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc_100')
