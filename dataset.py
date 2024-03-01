# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
import utils
from pathlib import Path
from torchvision import transforms
from collections import Counter
from sklearn.svm import LinearSVC
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from PIL import Image
import os
import re
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.datasets.globalwheat_dataset import GlobalWheatDataset
from collections import Counter
import logging
import random
logging.basicConfig(level=logging.INFO)


DATASETS = ['FEMNIST', 'RareGroupRotatedMNIST', 'TinyImagenet', 'WILDSCamelyon', 'WILDSFMoW', 'PACS', 'TerraIncognita', 'DomainNet', 'ColoredMNIST', 'WILDSWaterbirds']

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in __datasets__:
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return __datasets__[dataset_name]

def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

class MultipleDomainDataset:
    N_STEPS = 10001          # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    EVAL_FREQ = 1            # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets) 
    
    def _get_info_(self):
        return {}
    
class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        generator = torch.Generator().manual_seed(hparams.get('trial_seed', 0))
        shuffle = torch.randperm(len(original_images), generator=generator)

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class FEMNIST(MultipleDomainDataset):
    N_WORKERS = 0
    def __init__(self, root, test_envs, hparams):
        self.ctxt = hparams.get('context_length', 1)
        self.seed = hparams.get('trial_seed',0)
        self.datasets = []
        
        if root is None:    raise ValueError('Data directory not specified!')
        self.root_dir = Path(root) / 'femnist-data' / 'train'
        clients, _ , self.data = utils.read_dir(self.root_dir)
        assert len(clients) == len(set(clients)), 'duplicate users'
        self.environments = list(range(len(clients)))
        
        # training data
        self.domain_counts = {}
        self.train_cache_x, self.train_cache_y = [], []
        for i, client in enumerate(clients):
            x, y, domain_count, cache_x, cache_y = self.make_dataset(self.data, client)
            self.domain_counts[client] = domain_count
            self.datasets.append(TensorDataset(x, y))
            self.train_cache_x.append(cache_x)
            self.train_cache_y.append(cache_y)

        val_clients, _, v_data = utils.read_dir(Path(root) / 'femnist-data' / 'val')
        datasets = [self.make_dataset(v_data, v_client, mode='val') for v_client in val_clients]
        self.validation = [TensorDataset(x, y) for x, y, *_ in datasets]
        validation_counts = [count for *_, count, _, _ in datasets]
        self.valid_cache_x = [cache_x for *_, _, cache_x, _ in datasets]
        self.valid_cache_y = [cache_y for *_, _, _, cache_y in datasets]
        
        test_clients, _ , te_data = utils.read_dir(Path(root) / 'femnist-data' / 'test') 
        test_datasets = [self.make_dataset(te_data, t_client, mode='test') for t_client in test_clients]
        test_counts = [count for *_, count, _, _ in test_datasets]

        self.holdout_test = [TensorDataset(x, y) for x, y, *_ in test_datasets]
        self.test_cache_x = [cache_x for *_, _, cache_x, _ in test_datasets]
        self.test_cache_y = [cache_y for *_, _, _, cache_y in test_datasets]
            
        print("=> n training domains: ", len(self.datasets))
        print("=> Smallest domain: ", min(list(self.domain_counts.values())) if not hparams.get('is_iid_tr',0)  else len(self.tr_iid_x))
        print("=> Largest domain: ", max(list(self.domain_counts.values())) if not hparams.get('is_iid_tr',0)  else len(self.tr_iid_x), "\n")
        print("=> n validation domains: ", len(self.validation))
        print("=> Smallest domain: ", min(validation_counts))
        print("=> Largest domain: ", max(validation_counts), "\n")
        print("=> n testing domains: ", len(self.holdout_test))
        print("=> Smallest domain: ", min(test_counts))
        print("=> Largest domain: ", max(test_counts), "\n")
        
        self.input_shape = (1, 28, 28)
        self.num_classes = 62
        
    def make_dataset(self, data, client, mode = 'train'):
        client_X, client_y = data[client]['x'], data[client]['y']
        assert len(client_X) == len(client_y), 'malformed user data'
        X_processed = np.array(client_X).reshape((len(client_X), 28, 28, 1))
        X_processed = (1.0 - X_processed)
        x = torch.transpose(torch.from_numpy(X_processed), 1, 3).float()
        y = torch.tensor(client_y, dtype=torch.long)
        rand_x, rand_y = None, None
        if self.ctxt > 1:
            generator = torch.Generator().manual_seed(self.seed)
            indx = torch.randint(0, x.size(0), (self.ctxt-1,), generator=generator)
            rand_x = x[indx].unsqueeze(0)
            rand_y = y[indx].unsqueeze(0)
            x = x if mode == 'train' else x.unsqueeze(1)
            y = y if mode == 'train' else y.unsqueeze(1)
        return x, y, len(client_X), rand_x, rand_y
      
    
class RareGroupRotatedMNIST(MultipleEnvironmentMNIST):
    class RotationConfig:
        n_groups = 14
        domain_values = torch.arange(n_groups, dtype=torch.long) * 10
        domain_probs = torch.zeros(n_groups, dtype=torch.float32)
        domain_probs[:3] = 0.7  # 0 - 20
        domain_probs[3:6] = 0.2  # 30 - 50
        domain_probs[6:9] = 0.06  # 60 - 80
        domain_probs[9:12] = 0.03  # 90 - 110
        domain_probs[12:] = 0.01
        
    def __init__(self, root, test_envs, hparams):
        if root is None:
            raise ValueError('Data directory not specified!')
        self.seed = hparams.get('trial_seed',0)
        self.ctxt = hparams.get('context_length', 1)
        
        tr = MNIST(root, train=True, download=True)
        te = MNIST(root, train=False, download=True)
        
        self._make_splits(tr, te, hparams['tr_fraction'], self.seed)
        self.datasets, self.validation, self.holdout_test, self.train_cache_x, self.valid_cache_x, self.test_cache_x, self.train_cache_y, self.valid_cache_y, self.test_cache_y = [], [], [], [], [], [], [], [], []
        validation_counts, test_counts = [], []
        self.domain_counts = {}
        
        for g_index in range(self.RotationConfig.n_groups):
            x, y,  domain_count, tr_cache_x, tr_cache_y = self.get_domain_data(self.x_tr, self.y_tr, g_index, self.RotationConfig.domain_probs[g_index], mode = 'train')
            self.domain_counts[g_index] = domain_count
            self.datasets.append(TensorDataset(x, y))
            self.train_cache_x.append(tr_cache_x)
            self.train_cache_y.append(tr_cache_y)

            # 
            x_va, y_va, va_counts, va_cache_x, va_cache_y = self.get_domain_data(self.x_va, self.y_va, g_index,self.RotationConfig.domain_probs[g_index], mode='val')
            self.valid_cache_x.append(va_cache_x)
            self.valid_cache_y.append(va_cache_y)
            self.validation.append(TensorDataset(x_va, y_va))
            validation_counts.append(va_counts)
        
            # 
            x_te, y_te, te_counts, te_cache_x, te_cache_y = self.get_domain_data(self.x_te, self.y_te, g_index, self.RotationConfig.domain_probs[g_index], mode='test')
            self.test_cache_x.append(te_cache_x)
            self.test_cache_y.append(te_cache_y)
            self.holdout_test.append(TensorDataset(x_te, y_te))
            test_counts.append(te_counts)
                 
        print("=> n training domains: ", len(self.datasets))
        print("=> Smallest training domain: ", min(list(self.domain_counts.values())) if not hparams.get('is_iid_tr',0)  else len(self.tr_iid_x))
        print("=> Largest training domain: ", max(list(self.domain_counts.values())) if not hparams.get('is_iid_tr',0)  else len(self.tr_iid_x))
        print("=> n validation domains: ", len(self.validation))
        print("=> Smallest validation domain: ", min(validation_counts))
        print("=> Largest validation domain: ", max(validation_counts))
        print("=> n testing domains: ", len(self.holdout_test))
        print("=> Smallest testing domain: ", min(test_counts))
        print("=> Largest testing domain: ", max(test_counts))
        
        self.input_shape = (1, 28, 28)
        self.num_classes = 10  
            
    def _make_splits(self, tr, te, tr_fraction = 0.9, seed = 0):
        generator = torch.Generator().manual_seed(seed)
        s_tr = torch.randperm(len(tr), generator=generator)
        x_tr, y_tr = tr.data[s_tr], tr.targets[s_tr]
        n_tr = int(len(x_tr) * tr_fraction)
        self.x_va, self.y_va = x_tr[n_tr:], y_tr[n_tr:]
        self.x_tr, self.y_tr = x_tr[:n_tr], y_tr[:n_tr]
        
        s_te = torch.randperm(len(te))
        self.x_te, self.y_te = te.data[s_te], te.targets[s_te]
        print(f'=> Dataset sizes are: [TRAIN] {len(self.y_tr)} [VAL] {len(self.y_va)} [TEST] {len(self.y_te)}')
                 
            
    def get_domain_data(self, images, labels, g_index, p, mode = 'train'):
        if mode == 'train':
            if p == 0:  return
            n = int(p * len(labels) / 5)
        else:
            n = len(labels)
        g_indices = np.random.choice(len(labels), size = n)
        x, y = self.rotate_dataset(images[g_indices], labels[g_indices], int(self.RotationConfig.domain_values[g_index]))

        rand_x, rand_y = None, None
        if self.ctxt > 1:
            generator = torch.Generator().manual_seed(self.seed)
            indx = torch.randint(0, x.size(0), (self.ctxt-1,), generator=generator)
            rand_x = x[indx].unsqueeze(0)
            rand_y = y[indx].unsqueeze(0)
            x = x if mode == 'train' else x.unsqueeze(1)
            y = y if mode == 'train' else y.unsqueeze(1)
        return x, y, len(g_indices), rand_x, rand_y                 

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])
        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])
        y = labels.view(-1)
        return x, y  
         
                    
class OnFlyEnvironment:
    def __init__(
            self,
            data_info,
            ctxt = 1, 
            seed = 0, 
            transform=None,
            mode = 'train'):
        self.seed= seed
        self.mode = mode
        self.data_info = data_info                                                   # list of dictionaries
        self.transform = transform
        self.ctxt = ctxt
        self.cache_x = None
        self.cache_y = None
        if self.ctxt > 1:
            self._make_context()

    def __getitem__(self, index):
        data =  self.data_info[index]
        if isinstance(data, dict):   
            x, y = self._process_dict(data) 
            if self.ctxt > 1 and self.mode != 'train':
                x, y = x.unsqueeze(0), y.unsqueeze(0)                                                 
        if isinstance(data, list):                                                  # list of samples for context
            x, y = zip(*[self._process_dict(sample) for sample in data])
            x,y =  torch.stack(x, dim=0), torch.stack(y, dim=0)
        return x, y

    def __len__(self):
        return len(self.data_info)

    def _make_context(self):
        random.seed(self.seed)
        random_indices = [random.randint(0, len(self.data_info)) for _ in range(self.ctxt-1)]
        samples = [self.data_info[ind] for ind in random_indices]
        x, y = zip(*[self._process_dict(sample) for sample in samples])            
        x = torch.stack(x, dim=0).unsqueeze(0)
        y = torch.stack(y, dim=0).unsqueeze(0)
        self.cache_x = x
        self.cache_y = y

    def _process_dict(self, data):
        if data.get('path'):
            x = self.transform(Image.open(data['path']))
        elif data.get('x'):
            x = self.transform(data['x'])
        y = torch.tensor(data['class'], dtype=torch.long)
        return x, y
   
  
class TinyImagenet(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        if root is None:
            raise ValueError('Data directory not specified!')
        self.seed = hparams['trial_seed']
        self.ctxt = hparams.get('context_length', 1)
       
        self.root_dir = Path(root) / 'Tiny-ImageNet-C-new' / 'train'
        corruptions = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'snow', 'brightness', 'contrast', 'pixelate']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        root_va = Path(root) / 'Tiny-ImageNet-C-new' / 'val'
        root_te = Path(root) / 'Tiny-ImageNet-C-new' / 'test'
        self.datasets, self.train_cache_x, self.train_cache_y = self.make_dataset(self.root_dir, frost_ids = [1, 2, 3], jpeg_ids = [1, 2, 3], corruptions = corruptions)
        self.validation, self.valid_cache_x, self.valid_cache_y  = self.make_dataset(root_va, frost_ids = [4], jpeg_ids = [5], corruptions = ['speckle_noise', 'gaussian_blur', 'saturate'], mode = 'val') 
        self.holdout_test, self.test_cache_x, self.test_cache_y  = self.make_dataset(root_te, frost_ids = [5], jpeg_ids = [4], corruptions = ['impulse_noise', 'motion_blur', 'fog', 'elastic_transform'], mode = 'test')        
       
        print("=> n training domains: ", len(self.datasets))
        print("=> Smallest domain: ", min([len(env) for env in self.datasets]))
        print("=> Largest domain: ", max([len(env) for env in self.datasets]), "\n")
        
        print("=> n validation domains: ", len(self.validation))
        print("=> Smallest domain: ", min([len(env) for env in self.validation]))
        print("=> Largest domain: ", max([len(env) for env in self.validation]), "\n")
        
        print("=> n testing domains: ", len(self.holdout_test))
        print("=> Smallest domain: ", min([len(env) for env in self.holdout_test]))
        print("=> Largest domain: ", max([len(env) for env in self.holdout_test]), "\n")
      
        self.input_shape = (3, 64, 64)
        self.num_classes = 200
        
    def make_dataset(self, root_dir, frost_ids, jpeg_ids, corruptions, mode = 'train'):
        data, all_data = [], []
        data_cache_x = []
        data_cache_y = []
        domain_count = []
        
        for level in frost_ids:
            out = self.construct_imdb(root_dir, 'frost', level)
            all_data.extend(out)
            env = OnFlyEnvironment(out, self.ctxt, self.seed, self.transform, mode)
            data.append(env)
            data_cache_x.append(env.cache_x)
            data_cache_y.append(env.cache_y)
            domain_count.append(len(out))
        for level in jpeg_ids:
            out = self.construct_imdb(root_dir, 'jpeg_compression', level)
            all_data.extend(out)
            env = OnFlyEnvironment(out, self.ctxt, self.seed, self.transform, mode)
            data.append(env)
            data_cache_x.append(env.cache_x)
            data_cache_y.append(env.cache_y)
            domain_count.append(len(out))
        for corruption in corruptions:
            for level in [1, 2, 3, 4, 5]:
                out = self.construct_imdb(root_dir, corruption, level)
                all_data.extend(out)
                env = OnFlyEnvironment(out, self.ctxt, self.seed, self.transform, mode)
                data.append(env)
                data_cache_x.append(env.cache_x)
                data_cache_y.append(env.cache_y)
                domain_count.append(len(out))   
        return data, data_cache_x, data_cache_y

    def construct_imdb(self, root_dir, corruption, level):
        """Constructs the imdb."""
        split_path = os.path.join(root_dir, corruption, str(level))                                            # Compile the split data path
        re_pattern = r"^n[0-9]+$"
        class_ids = sorted(f for f in os.listdir(split_path) if re.match(re_pattern, f))                            # Images are stored per class in subdirs (format: n<number>)
        class_id_cont_id = {v: i for i, v in enumerate(class_ids)}                                                  # Map ImageNet class ids to contiguous ids
        imdb = []                                                                                                   # Construct the image db
        for class_id in class_ids:
            cont_id = class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                imdb.append({"path": os.path.join(im_dir, im_name), "class": cont_id})
        return imdb

class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            ctxt = 1,
            seed = 0,
            transform=None,
            mode = 'train'):
        self.seed= seed
        self.mode = mode
        self.name = f"{metadata_name}_{metadata_value}"
        self.ctxt = ctxt
        self.dataset = wilds_dataset
        self.transform = transform
        subset_indices = self._get_subset_indices(wilds_dataset, metadata_name, metadata_value)
        self.cache_x,  self.cache_y = None, None
        if self.ctxt > 1:
            subset_indices = self._make_context(subset_indices)
        self.indices = subset_indices
    
    def _get_subset_indices(self, wilds_dataset, metadata_name, metadata_value):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)        
        if isinstance(metadata_value, list):
            print(f'=> Finding all indices for all metadata values in the list {metadata_value}')
            subset_indices = torch.cat([torch.where(wilds_dataset.metadata_array[:, metadata_index] == value)[0] for value in metadata_value])
        else:
            subset_indices = torch.where(wilds_dataset.metadata_array[:, metadata_index] == metadata_value)[0]
        
        generator = torch.Generator().manual_seed(self.seed)
        shuffle = torch.randperm(len(subset_indices), generator=generator)
        return subset_indices[shuffle].tolist()
    
    def _make_context(self, indices):
        random.seed(self.seed)
        random_indices = [random.randint(0, len(indices)) for _ in range(self.ctxt-1)]
        x, y = zip(*[self._process(index) for index in random_indices])
        x = torch.stack(x, dim=0).unsqueeze(0)
        y = torch.stack(y, dim=0).unsqueeze(0)
        self.cache_x = x
        self.cache_y = y
        return indices
            
    def _process(self, index):
        x = self.dataset.get_input(index)
        y = torch.Tensor(self.dataset.y_array[index])    
        if not isinstance(x, Image.Image) and type(x).__name__ != "Image":
            x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y 

    def __getitem__(self, i):
        val = self.indices[i]
        if isinstance(val, (int, torch.Tensor)):  
            x, y = self._process(val)
            if self.ctxt > 1 and self.mode != 'train':
                x, y = x.unsqueeze(0), y.unsqueeze(0)
        elif isinstance(val, list):
            x, y = zip(*[self._process(index) for index in val])
            x, y = torch.stack(x, dim=0), torch.stack(y, dim=0)
        else:
            raise NotImplementedError()
        return x, y

    def __len__(self):
        return len(self.indices)
    
class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, valid_envs, test_envs, augment, hparams):
        super().__init__()
        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes
        self.ctxt = hparams.get('context_length', 1)
        
        self.datasets, self.validation, self.holdout_test, self.valid_cache_x, self.valid_cache_y, self.test_cache_x, self.test_cache_y, self.train_cache_x, self.train_cache_y = self._create_environments(dataset, metadata_name, valid_envs, test_envs, augment, hparams)
        print("=> n training domains: ", len(self.datasets))
        print("=> Smallest domain: ", min([len(env) for env in self.datasets]))
        print("=> Largest domain: ", max([len(env) for env in self.datasets]), "\n")
        
        print("=> n validation domains: ", len(self.validation))
        print("=> Smallest domain: ", min([len(env) for env in self.validation]))
        print("=> Largest domain: ", max([len(env) for env in self.validation]), "\n")
        
        if self.holdout_test:
            print("=> n testing domains: ", len(self.holdout_test))
            print("=> Smallest domain: ", min([len(env) for env in self.holdout_test]))
            print("=> Largest domain: ", max([len(env) for env in self.holdout_test]), "\n")
        
        print(f'=> Number of classes {self.num_classes}')
 

    def _create_environments(self, dataset, metadata_name, valid_envs, test_envs, augment, hparams):
        datasets, validation, holdout_test, valid_cache_x, valid_cache_y, test_cache_x, test_cache_y  = [], [], [], [], [], [], []
        train_cache_x, train_cache_y = [], []
        if not augment:
            print('=> Training model WITHOUT any augmentations...')
        
        for i, metadata_value in enumerate(self.metadata_values(dataset, metadata_name)):
            env_transform = self._get_transform(augment and (i not in test_envs and i not in valid_envs))
            if i not in valid_envs and i not in test_envs:
                env_dataset = WILDSEnvironment(dataset, metadata_name, metadata_value, self.ctxt, hparams.get('trial_seed',0), env_transform, mode = 'train')
                train_cache_x.append(env_dataset.cache_x)
                train_cache_y.append(env_dataset.cache_y)
                datasets.append(env_dataset)
            elif i in valid_envs:
                env_dataset = WILDSEnvironment(dataset, metadata_name, metadata_value, self.ctxt, hparams.get('trial_seed',0), env_transform, mode = 'val')
                validation.append(env_dataset)
                valid_cache_x.append(env_dataset.cache_x)
                valid_cache_y.append(env_dataset.cache_y)
            elif i in test_envs:
                env_dataset = WILDSEnvironment(dataset, metadata_name, metadata_value, self.ctxt, hparams.get('trial_seed',0), env_transform, mode = 'test')
                holdout_test.append(env_dataset)
                test_cache_x.append(env_dataset.cache_x)
                test_cache_y.append(env_dataset.cache_y)

        return datasets, validation, holdout_test, valid_cache_x, valid_cache_y, test_cache_x, test_cache_y, train_cache_x, train_cache_y

    def _get_transform(self, augment):
        transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]    
        if augment:
            print('=> Using training augmentation')
            transform_list.insert(1, transforms.RandomResizedCrop(224, scale=(0.7, 1.0)))
            transform_list.insert(2, transforms.RandomHorizontalFlip())
            transform_list.insert(3, transforms.RandomGrayscale())
            transform_list.insert(3, transforms.ColorJitter(0.3, 0.3, 0.3, 0.3))
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))          # Normalization
        return transforms.Compose(transform_list)

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))
    
    
class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root, download = True)
        valid_envs = [1]
        test_envs = [2]
        super().__init__(
            dataset, "hospital", valid_envs, test_envs, hparams['data_augmentation'], hparams)
