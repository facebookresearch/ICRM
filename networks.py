# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from transformers import GPT2Model, GPT2Config
import torch.nn.functional as F
import utils
import copy


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""     
    if len(input_shape) <= 2 :
        return MLP(input_shape[-1], hparams["mlp_width"], hparams)  
    elif input_shape[1:3] == (28, 28):
        if hparams['is_transformer']:
            model = ContextNet(input_shape[0], hparams)
            return model
        else:
            return CNN(input_shape[0], hparams)
    elif input_shape[1:3] == (32, 32):
        return ResNet(input_shape[0], hparams)
    elif input_shape[1:3] == (64, 64) or input_shape[1:3] == (224, 224):
        return ResNet(input_shape[0], hparams)
    else:
        raise NotImplementedError
   
def Classifier(in_features, out_features, hparams):
    if hparams['is_transformer']:
        return GPT2Transformer(in_features, out_features, hparams)
    if hparams['nonlinear_classifier']:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class CNN(nn.Module):
    def __init__(self, n_inputs, hparams):
        super(CNN, self).__init__()
        k_size = hparams.get('kernel_size',5)
        self.is_small = hparams.get('model_size', 'small') == 'small'
        hidden_dim = hparams.get('hidden_dim', 128)
        padding = (k_size - 1) // 2
        num_channels = hparams.get('num_features', 1)
               
        if self.is_small:
            self.conv1 = nn.Sequential(
                            nn.Conv2d(n_inputs, hidden_dim, k_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )
        else:
            self.conv0 = nn.Sequential(
                        nn.Conv2d(num_channels, hidden_dim, k_size, padding=padding),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                    )

            self.conv1 = nn.Sequential(
                            nn.Conv2d(hidden_dim, hidden_dim, k_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, k_size),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.last = nn.Sequential(
                    nn.Linear(hidden_dim, 200),
                    nn.ReLU(),
                    Identity()
                  )
        self.n_outputs = 200


    def forward(self, x):
        """Returns logit with shape (batch_size, n_outputs)"""
        if self.is_small:
            out = self.conv1(x)
        else:
            out = self.conv0(x)
            out = self.conv1(out)
        out = self.conv2(out)
        out = self.adaptive_pool(out)                                                           # shape: batch_size, hidden_dim, 1, 1
        out = out.squeeze(dim=-1).squeeze(dim=-1)                                               # make sure not to squeeze the first dimension when batch size is 0.
        out = self.last(out)

        return out
            
class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hparams = hparams

        mlp_width = hparams.get('mlp_width', 500)
        mlp_depth = hparams.get('mlp_depth', 5)
        activation = hparams.get('activation', 'relu')
        mlp_dropout = hparams.get('mlp_dropout', 0)
        
        
        self.is_bn = hparams.get('mlp_bn', 0)
        self.is_ln = hparams.get('mlp_ln', 0)
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.activation = utils.get_activation(activation)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth - 2)])
        self.output = nn.Linear(mlp_width, n_outputs)

        if self.is_bn:  self.bn = nn.BatchNorm1d(mlp_width)
        if self.is_ln:  self.ln = nn.LayerNorm(mlp_width)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.activation(self.input(x)))
        if self.is_bn:  x = self.bn(x)
        if self.is_ln:  x = self.ln(x)
        for hidden in self.hiddens:
            x = self.dropout(self.activation(hidden(x)))
            if self.is_bn:  x = self.bn(x)
            if self.is_ln:  x = self.ln(x)
        x = self.output(x)
        return x
          
           
class GPT2Transformer(nn.Module):
    def __init__(self, n_inputs,  n_outputs, hparams):
        super(GPT2Transformer, self).__init__()
        print('=> Initializing a GPT2 Transformer...')
        n_embd = hparams['n_embd']
        n_layer = hparams['n_layer']
        n_head = hparams['n_head']
        context_len = hparams['context_length']

        configuration = GPT2Config(
            n_positions= context_len,                 
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache= True,
        )   
        self.context_len =context_len
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self._read_in = nn.Linear(n_inputs, n_embd)
        
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_outputs)
        self.initialize()
    
    def initialize(self):
        nn.init.xavier_uniform_(self._read_in.weight)
        nn.init.zeros_(self._read_in.bias)
        nn.init.xavier_uniform_(self._read_out.weight)
        nn.init.zeros_(self._read_out.bias)
        
        for name, param in self._backbone.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, inputs):
        xs, ys, inds, past_key_values = inputs
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
            
        embeds = self._read_in(xs)
        outputs = self._backbone(inputs_embeds=embeds, past_key_values=past_key_values)
        output = outputs.last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, :, :][:, inds], outputs.past_key_values
        

class ContextNet(nn.Module):        
    def __init__(self, n_inputs, hparams):
        super(ContextNet, self).__init__()
        # Keep same dimensions
        n_features = hparams.get('num_features', 1)
        if hparams['is_transformer']:
            self.n_outputs = n_features*784
        else:
            self.n_outputs = n_features
        k_size = hparams.get('kernel_size', 5)
        padding = (k_size - 1) // 2
        hidden_dim = hparams.get('hidden_dim', 128)
        self.context_net = nn.Sequential(
            nn.Conv2d(n_inputs, hidden_dim, k_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, k_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_features, k_size, padding=padding)
        )
    def forward(self, x):
        return self.context_net(x)

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
   
class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, n_inputs, hparams):
        super(ResNet, self).__init__()
        print('=> Training ResNet architecture...')
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # adapt number of channels
        nc = n_inputs
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()
            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        self.n_features = self.network.fc.in_features
        del self.network.fc
        self.network.fc = Identity()
        if hparams['freeze_bn']:
            self.freeze_bn()
            print('=> Training ResNet with frozen batch norm...')
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))
    

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams['freeze_bn']:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                

class DenseNet(nn.Module):
    """DenseNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, n_inputs, hparams):
        print('=> Training with DenseNet121')
        super(DenseNet, self).__init__()
        if hparams['densenet121']:
            self.network = torchvision.models.densenet121(pretrained=True)
            self.n_outputs = 1024
        else:
            raise NotImplementedError()

        # adapt number of channels
        nc = n_inputs
        if nc != 3:
            tmp = self.network.features.conv0.weight.data.clone()

            self.network.features.conv0 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.features.conv0.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        self.n_features = self.network.classifier.in_features
        del self.network.classifier
        self.network.classifier = Identity()
        
        if hparams['freeze_bn']:
            self.freeze_bn()
            print('=> Training DenseNet with frozen batch norm...')
            
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.get('densenet_dropout', 0))

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams['freeze_bn']:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                
                                
class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams)
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
    
    
