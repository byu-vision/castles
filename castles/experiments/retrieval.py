'''Run image retrieval experiments'''
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import time

import pytorch_lightning as pl
from pytorch_metric_learning import samplers
import torch, torchvision

import dataset
import network


def train(args):
    '''Train an image retrieval model.
    Note that the use of a non-standard sampler here causes issues with the ddp
    backend: use dp instead'''
    pl.trainer.seed_everything(args.seed)
    
    datasplits, train_data, val_data = dataset.get_retrieval_datasets(
        args.data, args.split, trainval=True)
    
    tsampler = samplers.MPerClassSampler(
        train_data.targets, args.npos, args.bs, 300000)
    vsampler = samplers.MPerClassSampler(
        val_data.targets, args.npos, args.bs*2, 100000)

    
    train_load = dataset.get_loader(train_data, args.bs, args.workers,
                                    sampler=tsampler)
    val_load = dataset.get_loader(val_data, args.bs, args.workers,
                                  train=False, sampler=vsampler)
    
    model = network.CastleRetrieval(128, args)

    log = pl.loggers.CSVLogger(args.root, f'retrieval')
    cp = pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=1, mode='max')
    trainer = pl.Trainer.from_argparse_args(
        args, logger=log, checkpoint_callback=cp, default_root_dir=args.root,
        num_sanity_val_steps=0)
    trainer.fit(model, train_load, val_load)


def extract(args):
    '''Extract an embedding for each image in the test set using a trained
    model'''
    if not args.load_checkpoint:
        raise ValueError('Must provide a checkpoint to test from with '
                         '--load_checkpoint')
    ckpt = torch.load(args.load_checkpoint, map_location='cpu')
    cargs = ckpt['hyper_parameters']
    transfer = ['split','seed','net']
    for key in transfer:
        setattr(args, key, cargs[key])
    
    pl.trainer.seed_everything(args.seed)
    
    datasplits, test_data = dataset.get_retrieval_datasets(
        args.data, args.split, trainval=False)
    test_data.set_return_ind(True)
    test_load = dataset.get_loader(
        test_data, args.bs, args.workers, train=False)
    
    model = network.CastleRetrieval(128, args)
    model.load_state_dict(ckpt['state_dict'])
    model.output_file = str(Path(args.load_checkpoint).parent.joinpath('preds.pth'))
    
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=args.root)
    trainer.test(model, test_load)
    

def extract_train(args):
    '''Extract an embedding for each image in the training set using a trained
    model'''
    if not args.load_checkpoint:
        raise ValueError('Must provide a checkpoint to test from with '
                         '--load_checkpoint')
    ckpt = torch.load(args.load_checkpoint, map_location='cpu')
    cargs = ckpt['hyper_parameters']
    transfer = ['split','seed','net']
    for key in transfer:
        setattr(args, key, cargs[key])
    
    pl.trainer.seed_everything(args.seed)
    
    datasplits, train_data, val_data = dataset.get_retrieval_datasets(
        args.data, args.split, trainval=True)
    train_data.set_return_ind(True)
    train_load = dataset.get_loader(
        train_data, args.bs, args.workers, train=False)
    
    model = network.CastleRetrieval(128, args)
    model.load_state_dict(ckpt['state_dict'])
    model.output_file = str(Path(args.load_checkpoint).parent.joinpath('train_preds.pth'))
    
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=args.root)
    trainer.test(model, train_load)
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('run', type=str, choices=['train','test','extract','extract_train'])
    parser.add_argument('--root', type=str, default='checkpoints/')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--check_val_every_n_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--load_checkpoint', type=str, default=None)
    
    parser = network.CastleRetrieval.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    print(f'Starting: {time.ctime()}')
    if args.run == 'train':
        train(args)
    elif args.run == 'test':
        test(args)
    elif args.run == 'extract':
        extract(args)
    elif args.run == 'extract_train':
        extract_train(args)
    print(f'Finished: {time.ctime()}')
