import argparse
from collections import defaultdict
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

import losses
import utils
from eval import read_settings


def to_df(values, objects, columns, random=None, ensemble=None, sort=False,
          transpose=True, stats=True):
    df1 = pd.DataFrame(values)
    if transpose:
        df1 = df1.transpose()
    df1.index = objects
    df1.columns = columns
    if sort:
        df1 = df1[sorted(df1.columns)]

    df2 = df1.copy()
    if stats:
        df2['average'] = df2.mean(1)
        df2['min'] = df2.min(1)
        df2['max'] = df2.max(1)
        if random is not None:
            df2['random'] = random
        if ensemble is not None:
            df2['ensemble'] = ensemble
    df2.loc['average'] = df2.mean(0)
    return df1, df2


def draw_corr_plot(auc_df, objects, loss_df, out_path):
    def normalize(v):
        return (v - v.min()) / (v.max() - v.min())

    def draw(x, y, xl, yl, name):
        plt.figure(figsize=(3, 2.7))
        plt.title(name)
        plt.ylabel(xl)
        plt.xlabel(yl)

        f = np.poly1d(np.polyfit(x, y, deg=1))
        x_min = np.min(x)
        x_max = np.max(x)
        plt.plot([x_min, x_max], [f(x_min), f(x_max)],
                 linestyle='dashed',
                 color='red',
                 zorder=0)

        plt.scatter(x, y)
        plt.savefig(join(out_path, f'{name}.pdf'), bbox_inches='tight')
        plt.close()

    makedirs(out_path, exist_ok=True)

    auc_list, loss_list = [], []
    for obj in objects:
        auc_row = auc_df.loc[obj]
        auc_list.extend(normalize(auc_row).tolist())
        loss_row = loss_df.loc[obj]
        loss_list.extend(normalize(loss_row).tolist())
        draw(loss_row, auc_row,
             xl='AUC',
             yl='MMD Loss',
             name=obj)

    draw(loss_list, auc_list,
         xl='Relative AUC',
         yl='Relative MMD Loss',
         name='all')


def run(path_in, path_out, loss_types):
    def read_auc(file):
        return pd.read_csv(file, sep='\t')['auc'].values

    # Read data
    settings = read_settings(path_in, random=False)
    loss_dict = defaultdict(lambda: [])
    auc_list = []
    objects = None
    for setting in settings:
        df = pd.read_csv(join(path_in, setting, 'out.tsv'), sep='\t')
        auc_list.append(df['auc'])
        objects = df['type']
        for loss_type in loss_types:
            loss_dict[loss_type].append(df[loss_type])

    columns = [float(e) for e in settings]
    rand_auc = read_auc(join(path_in, 'random', 'out.tsv'))
    ens_auc = read_auc(join(path_in, 'ensemble', 'out.tsv'))
    auc_df, auc_df_out = to_df(auc_list, objects, columns, rand_auc, ens_auc, sort=True)

    # Save AUC
    makedirs(path_out, exist_ok=True)
    auc_df_out.to_csv(join(path_out, 'auc.tsv'), sep='\t')

    # Save losses
    loss_df_dict = {}
    for loss_type in loss_types:
        dir_path = join(path_out, 'losses', loss_type)
        makedirs(dir_path, exist_ok=True)
        values = loss_dict[loss_type]
        loss_df, loss_df_out = to_df(values, objects, columns, sort=True)
        loss_df_out.to_csv(join(dir_path, 'loss.tsv'), sep='\t')
        loss_df_dict[loss_type] = loss_df

    # Compute their correlations
    results = [[] for _ in loss_types]
    for i, loss_type in enumerate(loss_types):
        curr_results = []
        for obj in objects:
            auc_row = auc_df.loc[obj].values
            loss_row = loss_df_dict[loss_type].loc[obj].values
            if np.isfinite(loss_row).all():
                spearman = scipy.stats.spearmanr(auc_row, loss_row)
                pearson = scipy.stats.pearsonr(auc_row, loss_row)
            else:
                spearman = (np.nan, np.nan)
                pearson = (np.nan, np.nan)
            curr_results.append((
                spearman[0], spearman[1], pearson[0], pearson[1]
            ))
            results[i].append(spearman[0])
        columns = ['spearman', 'p-value', 'pearson', 'p-value']
        corr_df, corr_df_out = to_df(curr_results, objects, columns, sort=False,
                                     transpose=False, stats=False)
        corr_df_out.to_csv(join(path_out, 'losses', loss_type, 'corr.tsv'), sep='\t')

    _, corr_df_out = to_df(results, objects, loss_types, stats=False)
    corr_df_out.to_csv(join(path_out, 'corr.tsv'), sep='\t')

    # Perform manual search
    auc_list = []
    for loss_type in loss_types:
        best_indices = loss_df_dict[loss_type].values.argmin(1)
        auc_list.append(
            auc_df.to_numpy()[np.arange(len(auc_df)), best_indices]
        )

    results = [
        *auc_list,
        rand_auc,
        auc_df.mean(1),
        auc_df.min(1),
        auc_df.max(1),
        ens_auc,
    ]
    columns = [*loss_types, 'random', 'average', 'min', 'max', 'ensemble']
    search_df, search_df_out = to_df(results, objects, columns, stats=False)
    search_df_out.to_csv(join(path_out, 'search.tsv'), sep='\t')

    for loss_type in loss_types:
        loss_df = loss_df_dict[loss_type]
        dir_out = join(path_out, 'losses', loss_type, 'figures')
        if np.isfinite(loss_df.values).all():
            draw_corr_plot(auc_df, objects, loss_df, dir_out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='1-cutout')
    parser.add_argument('--path', type=str, default='out-eval')
    return parser.parse_args()


def main():
    args = parse_args()
    path_in = join(utils.ROOT, args.path, args.name)
    path_out = join(utils.ROOT, args.path, f'{args.name}-search')
    loss_names = sorted(losses.get_loss_names())
    run(path_in, path_out, loss_names)


if __name__ == '__main__':
    main()
