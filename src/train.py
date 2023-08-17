import argparse
import datetime
from os import makedirs
from os.path import join
from pathlib import Path

import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import data
import models
import utils
from augment import collate_fn, to_aug_function
from augment.cutpaste import CutPaste3Way


def process_tuple(args: float or list):
    if isinstance(args, float):
        return args, args
    elif len(args) == 1:
        return args[0], args[0]
    elif len(args) == 2:
        return tuple(args)
    elif len(args) > 2:
        raise ValueError()


def process_args(args):
    assert not (args.cuda and args.gpu is None)
    if args.type == 'all':
        args.objects = utils.get_objects(args.data)
    else:
        args.objects = args.type.split(',')
    args.device = f'cuda:{args.gpu}' if args.cuda else 'cpu'
    args.patch_size = process_tuple(args.patch_size)
    args.patch_aspect = process_tuple(args.patch_aspect)
    return args


def parse_args():
    parser = argparse.ArgumentParser()

    # Environments
    parser.add_argument('--data', default='mvtec')
    parser.add_argument('--type', default='all', help='object types seperated by ","')
    parser.add_argument('--test-epochs', default=-1, type=int,
                        help='interval to calculate test AUC during training; '
                             '-1 means not calculating the test scores')
    parser.add_argument('--pretrained', type=utils.str2bool, default=False)
    parser.add_argument('--freeze-resnet', default=20, type=int,
                        help='number of epochs to freeze ResNet18')
    parser.add_argument('--load', type=str, default=join(utils.ROOT, 'out'))
    parser.add_argument('--out', type=str, default=join(utils.ROOT, 'out'))
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--verbose', type=utils.str2bool, default=True)

    # Efficiency
    parser.add_argument('--cuda', default=True, type=utils.str2bool)
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument('--workers', default=8, type=int)

    # Model
    parser.add_argument('--head-layers', default=1, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--lr', default=0.03, type=float)

    # Augmentation
    parser.add_argument('--augment', default='cutdiff', type=str)
    parser.add_argument('--patch-size', type=float, nargs='+', default=[0.02, 0.15])
    parser.add_argument('--patch-aspect', type=float, nargs='+', default=[0.3, 1.0])
    parser.add_argument('--patch-angle', type=float, default=0)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=None)

    return process_args(parser.parse_args())


def train(augment1=None,
          augment2=None,
          dataset='mvtec',
          obj_type='screw',
          evaluator=None,
          model_path=join(utils.ROOT, 'out/models'),
          log_path=join(utils.ROOT, 'out/logs'),
          epochs=256,
          pretrained=True,
          test_epochs=10,
          freeze_resnet=20,
          learning_rate=0.03,
          optim_name='SGD',
          batch_size=64,
          head_layers=8,
          device='cuda',
          workers=8,
          verbose=True):
    # The temperature hyperparameter is not implemented.

    weight_decay = 0.00003
    momentum = 0.9
    model_name = f'model-{obj_type}-' \
                 f'{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}'

    train_transform = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1
        ),
        augment1,
    ])
    trn_data = data.load_data(dataset, obj_type, train_transform, mode='train')
    data_loader = DataLoader(
        trn_data, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=workers, collate_fn=collate_fn,
        persistent_workers=True, pin_memory=True, prefetch_factor=5,
    )
    data_iterator = iter(data_loader)
    augment2 = augment2.to(device)

    head_units = [512] * head_layers + [128]
    num_classes = 3 if isinstance(augment1, CutPaste3Way) else 2
    model = models.ProjectionNet(head_units, num_classes, pretrained)
    model.to(device)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), learning_rate, momentum,
                              weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
    elif optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), learning_rate,
                               weight_decay=weight_decay)
        scheduler = None
    else:
        raise ValueError(f'ERROR unknown optimizer: {optim_name}')

    writer = SummaryWriter(join(log_path, model_name))
    for step in tqdm(range(epochs), disable=not verbose):
        epoch = int(step / 1)
        if epoch == freeze_resnet:
            model.unfreeze()
        writer.add_scalar('epoch', epoch, step)

        try:
            x_list = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            x_list = next(data_iterator)

        y = torch.arange(len(x_list), device=device) \
            .repeat_interleave(x_list[0].size(0))
        x = torch.cat(x_list, dim=0).to(device)
        x = torch.where(y.view(-1, 1, 1, 1) == 1, augment2(x), x)

        embeds, logits = model(x)
        loss = loss_fn(logits, y)
        writer.add_scalar('trn_loss', loss.item(), step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

        predicted = torch.argmax(logits, dim=1)
        accuracy = torch.mean((predicted == y).float())
        writer.add_scalar('trn_acc', accuracy, step)

        if test_epochs > 0 and epoch % test_epochs == 0:
            model.eval()
            roc_auc = evaluator(model)
            model.train()
            writer.add_scalar('test_auc', roc_auc, step)
    writer.close()

    torch.save(model.state_dict(), join(model_path, f'{model_name}.tch'))
    return model


def main():
    args = parse_args()
    assert args.epochs > 0

    utils.set_environment(args.seed)
    makedirs(args.out, exist_ok=True)
    utils.save_json(vars(args), join(args.out, 'args.json'))

    augment1, augment2 = to_aug_function(
        name=args.augment,
        area_ratio=args.patch_size,
        aspect_ratio=args.patch_aspect,
        angle=args.patch_angle,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
    )

    model_path = join(args.out, 'models')
    log_path = join(args.out, 'logs')
    Path(model_path).mkdir(exist_ok=True, parents=True)

    out = dict()
    for obj_type in args.objects:
        evaluator = models.Evaluator(args.data, obj_type, args.device)
        model = train(
            augment1=augment1,
            augment2=augment2,
            dataset=args.data,
            obj_type=obj_type,
            evaluator=evaluator,
            model_path=model_path,
            log_path=log_path,
            epochs=args.epochs,
            pretrained=args.pretrained,
            test_epochs=args.test_epochs,
            freeze_resnet=args.freeze_resnet,
            learning_rate=args.lr,
            optim_name=args.optim,
            batch_size=args.batch_size,
            head_layers=args.head_layers,
            device=args.device,
            workers=args.workers,
            verbose=args.verbose,
        )
        evaluator.run_model(model, augment1, augment2)
        evaluator.save_embeddings(join(args.out, 'embeddings', obj_type))
        out[obj_type] = evaluator.measure_auc()

    df = pd.DataFrame.from_dict(out, orient='index', columns=['auc'])
    df.index.name = 'type'
    df.to_csv(join(args.out, 'out.tsv'), sep='\t')


if __name__ == '__main__':
    main()
