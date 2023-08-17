import argparse
import json
import os
from argparse import Namespace
from os.path import join
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import losses
import models
import train
import utils
from augment import to_aug_function


def save_embeddings(evaluator, root):
    device = evaluator.device
    obj_type = evaluator.object_type

    model_path = join(root, 'models')
    model_names = list(Path(model_path).glob(f'model-{obj_type}*'))
    assert len(model_names) == 1
    model_name = model_names[0]

    with open(join(root, 'args.json')) as f:
        args_dict = json.load(f)
        if args_dict['augment'] == 'smooth':  # For compatibility
            args_dict['augment'] = 'cutdiff'
        trn_args = train.process_args(Namespace(**args_dict))

    augment1, augment2 = to_aug_function(
        name=trn_args.augment,
        area_ratio=trn_args.patch_size,
        aspect_ratio=trn_args.patch_aspect,
        angle=trn_args.patch_angle,
    )

    head_units = [512] * trn_args.head_layers + [128]
    weights = torch.load(model_name, map_location='cpu')
    num_classes = weights['out.weight'].shape[0]
    model = models.ProjectionNet(head_units, num_classes)
    model = model.to(device)
    model.load_state_dict(weights)

    emb_path = join(root, 'embeddings', obj_type)
    evaluator.run_model(model, augment1, augment2)
    evaluator.save_embeddings(emb_path)


def initialize_evaluator(dataset, obj_type, root, device):
    evaluator = models.Evaluator(dataset, obj_type, device)
    obj_type = evaluator.object_type
    emb_path = join(root, 'embeddings', obj_type)
    try:
        evaluator.load_embeddings(emb_path)
    except FileNotFoundError:
        save_embeddings(evaluator, root)
        evaluator.load_embeddings(emb_path)
    return evaluator


def visualize_embeddings(evaluator, auc, loss, path_out):
    def sample(a, s):
        return a[np.random.permutation(len(a))[:s]]

    obj_type = evaluator.object_type
    trn_emb = evaluator.trn_emb.numpy()
    trn_labels = evaluator.trn_labels.numpy()
    test_emb = evaluator.test_emb.numpy()
    test_labels = evaluator.test_labels.numpy()

    trn_normal = sample(trn_emb[trn_labels == 0], s=(test_labels == 0).sum())
    trn_anomaly = sample(trn_emb[trn_labels == 1], s=(test_labels == 1).sum())
    trn_emb = np.concatenate([trn_normal, trn_anomaly])
    trn_labels = np.array([0] * len(trn_normal) + [1] * len(trn_anomaly))

    test_labels[test_labels == 0] = 2
    test_labels[test_labels == 1] = 3
    labels = np.concatenate([trn_labels, test_labels])

    tsne = TSNE()
    emb_out = tsne.fit_transform(np.concatenate([trn_emb, test_emb]))

    names = ['Training normal',
             'Training augmented',
             'Test normal',
             'Test anomalies']
    markers = ['o', 'x', 'o', 'x']
    sizes = [15, 20, 15, 20]

    fig, ax = plt.subplots(figsize=(3.3, 3))
    ax.set_title(f'AUC={auc:.3f} & Loss={loss:.3f}')

    plots = []
    for i in range(4):
        plots.append(ax.scatter(
            emb_out[labels == i, 0],
            emb_out[labels == i, 1],
            s=sizes[i], c=f'C{i}', label=names[i], marker=markers[i]
        ))

    os.makedirs(path_out, exist_ok=True)
    file = join(path_out, f'{obj_type}.png')
    plt.savefig(file, bbox_inches='tight', dpi=600)
    plt.close()

    file_legend = join(path_out, 'legend.png')
    fig_legend = plt.figure()
    fig_legend.legend(plots, names, ncol=len(names), edgecolor='black')
    fig_legend.savefig(file_legend, bbox_inches='tight', dpi=600)
    plt.close()


def visualize_scores(evaluator, roc_auc, loss, path_out, scale='linear'):
    obj_type = evaluator.object_type
    trn_scores, test_scores = evaluator.get_scores()
    trn_labels = evaluator.trn_labels.numpy()
    test_labels = evaluator.test_labels.numpy()

    if scale == 'log':
        trn_scores = np.log(trn_scores)
        test_scores = np.log(test_scores)

    score_list = [
        trn_scores[trn_labels == 0],
        trn_scores[trn_labels == 1],
        test_scores[test_labels == 0],
        test_scores[test_labels == 1],
    ]
    all_scores = np.concatenate([trn_scores, test_scores])

    labels = ['Training normal',
              'Training augmented',
              'Test normal',
              'Test anomalies']
    colors = ['C1', 'C0', 'C2', 'C3']

    fig, ax = plt.subplots(figsize=(3.3, 3))
    ax.set_title(f'AUC={roc_auc:.3f} & Loss={loss:.3f}')
    ax.set_xlabel('Anomaly score')

    _, bin_edges = np.histogram(all_scores, bins=15)
    plots = []
    for curr_data, color in zip(score_list, colors):
        y_values, x_values = np.histogram(curr_data, bins=bin_edges)
        x_values = (x_values[:-1] + x_values[1:]) / 2
        y_values = y_values * 100 / y_values.sum()
        plots.append(ax.plot(x_values, y_values, color=color)[0])
        ax.fill_between(x_values, 0, y_values, color=color, alpha=0.5)
    ax.legend(plots, labels, ncol=1)

    file = join(path_out, f'{obj_type}.pdf')
    os.makedirs(path_out, exist_ok=True)
    plt.savefig(file, bbox_inches='tight')
    plt.close()


def evaluate_losses(path_in, path_out, dataset, obj_types, device):
    loss_names = losses.get_loss_names()

    out = []
    for obj_type in obj_types:
        evaluator = initialize_evaluator(dataset, obj_type, path_in, device)
        auc = evaluator.measure_auc()
        loss_dict = evaluator.measure_losses()
        out.append([obj_type, auc, *(loss_dict[n] for n in loss_names)])

        os.makedirs(join(path_out, 'losses'), exist_ok=True)
        df = pd.DataFrame(loss_dict['emb_matrix'])
        columns = ['trn_normal', 'test_normal', 'trn_augment', 'test_anomaly']
        df.index = columns
        df.columns = columns
        df.to_csv(join(path_out, 'losses', f'{obj_type}.tsv'), sep='\t')

    columns = ['type', 'auc'] + loss_names
    df = pd.DataFrame(out, columns=columns)
    os.makedirs(path_out, exist_ok=True)
    df.to_csv(join(path_out, 'out.tsv'), sep='\t', index=False)


def evaluate_ensemble(root_in, path_out, dataset, obj_types, settings, device):
    loss_names = losses.get_loss_names()

    out = []
    for obj_type in obj_types:
        score_list, test_labels = [], None
        for setting in settings:
            set_path = join(root_in, setting)
            evaluator = initialize_evaluator(dataset, obj_type, set_path, device)
            _, test_scores = evaluator.get_scores()
            score_list.append(test_scores)
            test_labels = evaluator.test_labels
        ensemble_scores = np.stack(score_list).mean(axis=0)
        auc = roc_auc_score(test_labels, ensemble_scores)
        out.append([obj_type, auc] + [0] * len(loss_names))

    columns = ['type', 'auc'] + loss_names
    df = pd.DataFrame(out, columns=columns)
    os.makedirs(path_out, exist_ok=True)
    df.to_csv(join(path_out, 'out.tsv'), sep='\t', index=False)


def visualize_model(path_in, path_out, dataset, obj_types, device):
    for obj_type in obj_types:
        evaluator = initialize_evaluator(dataset, obj_type, path_in, device)
        loss = evaluator.measure_losses()['ours']
        auc = evaluator.measure_auc()
        visualize_scores(evaluator, auc, loss, join(path_out, 'scores'))
        visualize_scores(evaluator, auc, loss, join(path_out, 'scores-log'), scale='log')
        visualize_embeddings(evaluator, auc, loss, join(path_out, 'embeddings'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mvtec')
    parser.add_argument('--name', type=str, default='1-cutout')
    parser.add_argument('--path-in', type=str, default='out-summary')
    parser.add_argument('--path-out', type=str, default='out-eval')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gpu', type=int, default=None)
    return parser.parse_args()


def read_settings(path, random=True):
    out = []
    for dir_name in os.listdir(path):
        if utils.is_number(dir_name):
            out.append(dir_name)
        elif random and dir_name == 'random':
            out.append(dir_name)
    return out


def main():
    args = parse_args()
    args.device = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
    utils.set_environment(args.seed, num_threads=8)

    root_in = join(utils.ROOT, args.path_in, args.name)
    root_out = join(utils.ROOT, args.path_out, args.name)
    objects = utils.get_objects(args.data.split('-')[0])

    settings = read_settings(root_in, random=True)
    for setting in tqdm(settings):
        path_in = join(root_in, setting)
        path_out = join(root_out, setting)
        evaluate_losses(path_in, path_out, args.data, objects, args.device)
        visualize_model(path_in, path_out, args.data, objects, args.device)

    # Ensemble of all settings
    path_out = join(root_out, 'ensemble')
    settings = read_settings(root_in, random=False)
    evaluate_ensemble(
        root_in, path_out, args.data, objects, settings, args.device
    )


if __name__ == '__main__':
    main()
