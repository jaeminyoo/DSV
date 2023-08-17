import math
from os import makedirs
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torchvision.models import resnet18

import data
import losses
from augment import PostTransform, collate_fn
from density import GaussianDensitySklearn, GaussianDensityTorch


class ProjectionNet(nn.Module):
    def __init__(self, head_layers=None, num_classes=2, pretrained=False):
        super().__init__()

        if head_layers is None:
            head_layers = [512, 512, 512, 512, 512, 512, 512, 512, 128]

        # Create an MLP head as in the following:
        # - https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # - Not sure if this is the right architecture used in CutPaste.
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons
        self.head = nn.Sequential(*sequential_layers)

        self.resnet18 = resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Identity()
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x, emb_type='first', emb_norm=True):
        emb1 = self.resnet18(x)
        emb2 = self.head(emb1)
        logits = self.out(emb2)
        if emb_type == 'first':
            emb = emb1
        elif emb_type == 'second':
            emb = emb2
        else:
            raise ValueError(emb_type)
        if emb_norm:
            emb = normalize(emb, p=2, dim=1)
        return emb, logits

    def freeze_resnet(self):
        # Freeze the ResNet18 network.
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # Unfreeze the MLP head.
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # Unfreeze all.
        for param in self.parameters():
            param.requires_grad = True


class Evaluator:
    def __init__(self, dataset, object_type, device, density='torch', batch_size=32):
        self.dataset = dataset
        self.object_type = object_type
        self.device = device
        self.batch_size = batch_size
        if density == 'torch':
            self.density = GaussianDensityTorch()
        elif density == 'sklearn':
            self.density = GaussianDensitySklearn()
        else:
            raise ValueError()

        # Cached embeddings and scores
        self.trn_emb = None
        self.trn_labels = None
        self.trn_scores = None
        self.test_emb = None
        self.test_labels = None
        self.test_scores = None

    def _get_training_embeddings(self, model):
        transform = PostTransform()
        trn_data = data.load_data(self.dataset, self.object_type, transform,
                                  mode='train')
        trn_loader = DataLoader(trn_data, self.batch_size, shuffle=False)
        emb_list = []
        for x in trn_loader:
            embed, logit = model(x.to(self.device))
            emb_list.append(embed.cpu())
        return torch.cat(emb_list)

    def _get_augmented_embeddings(self, model, augment1, augment2,
                                  anomaly_ratio=None):
        trn_data = data.load_data(self.dataset, self.object_type, augment1,
                                  mode='train')
        trn_loader = DataLoader(trn_data, self.batch_size, shuffle=True,
                                collate_fn=collate_fn)

        if anomaly_ratio is not None:
            multiplier = anomaly_ratio / (1 - anomaly_ratio)
            num_runs = math.ceil(multiplier)
            out_size = math.ceil(multiplier * len(trn_data))
        else:
            num_runs = 1
            out_size = len(trn_data)

        emb_list = []
        for _ in range(num_runs):
            for ori_x, aug_x in trn_loader:
                aug_x = augment2(aug_x.to(self.device))
                embed, logit = model(aug_x)
                emb_list.append(embed.cpu())
        return torch.cat(emb_list)[:out_size]

    def _get_test_embeddings(self, model):
        transform = PostTransform()
        test_data = data.load_data(self.dataset, self.object_type, transform,
                                   mode='test')
        test_loader = DataLoader(test_data, self.batch_size, shuffle=False)
        emb_list, label_list = [], []
        for x, y in test_loader:
            embed, logit = model(x.to(self.device))
            emb_list.append(embed.cpu())
            label_list.append(y)
        test_emb = torch.cat(emb_list)
        test_labels = torch.cat(label_list).to(torch.long)
        return test_emb, test_labels

    @torch.no_grad()
    def run_model(self, model, augment1, augment2):
        model.eval()
        test_emb, test_labels = self._get_test_embeddings(model)
        anomaly_ratio = test_labels.float().mean()
        trn_emb_clean = self._get_training_embeddings(model)
        trn_emb_augment = self._get_augmented_embeddings(
            model, augment1, augment2, anomaly_ratio
        )
        model.train()

        trn_emb = torch.cat([trn_emb_clean, trn_emb_augment])
        trn_labels = torch.tensor(
            data=[0] * len(trn_emb_clean) + [1] * len(trn_emb_augment),
            dtype=torch.int64
        )

        self.density.fit(trn_emb_clean)
        trn_scores = self.density.predict(trn_emb)
        test_scores = self.density.predict(test_emb)

        self.trn_emb = trn_emb
        self.trn_labels = trn_labels
        self.trn_scores = trn_scores
        self.test_emb = test_emb
        self.test_labels = test_labels
        self.test_scores = test_scores

    @torch.no_grad()
    def save_embeddings(self, path):
        def save(tensor, name):
            np.save(join(path, name), tensor.cpu().numpy())

        makedirs(path, exist_ok=True)
        save(self.trn_emb, 'trn_embeddings')
        save(self.trn_labels, 'trn_labels')
        save(self.trn_scores, 'trn_scores')
        save(self.test_emb, 'test_embeddings')
        save(self.test_labels, 'test_labels')
        save(self.test_scores, 'test_scores')

    def load_embeddings(self, path):
        def load(file):
            return torch.from_numpy(np.load(join(path, file)))

        self.trn_emb = load('trn_embeddings.npy')
        self.trn_labels = load('trn_labels.npy')
        self.trn_scores = load('trn_scores.npy')
        self.test_emb = load('test_embeddings.npy')
        self.test_labels = load('test_labels.npy')
        self.test_scores = load('test_scores.npy')

    @torch.no_grad()
    def measure_losses(self):
        # Need to consider the scale of anomaly scores.
        # - Check the implementation of the density estimator?
        # - Log scores would be better than the raw scores.

        mmd_loss = losses.BaseLoss()
        emb_sets = [
            self.trn_emb[self.trn_labels == 0],
            self.test_emb[self.test_labels == 0],
            self.trn_emb[self.trn_labels == 1],
            self.test_emb[self.test_labels == 1]
        ]
        loss_matrix = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                loss_matrix[i, j] = mmd_loss(emb_sets[i], emb_sets[j])

        names = losses.get_loss_names()
        loss_values = losses.get_all_losses(self.trn_emb,
                                            self.trn_labels,
                                            self.test_emb)

        out = dict(emb_matrix=loss_matrix)
        for name, loss_value in zip(names, loss_values):
            out[name] = loss_value
        return out

    def measure_auc(self):
        return roc_auc_score(
            self.test_labels.cpu().numpy(),
            self.test_scores.cpu().numpy()
        )

    @torch.no_grad()
    def get_scores(self):
        trn_scores = self.trn_scores.cpu().numpy()
        test_scores = self.test_scores.cpu().numpy()
        return trn_scores, test_scores
