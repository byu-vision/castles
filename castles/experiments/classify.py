'''Run classification experiments'''

from argparse import ArgumentParser
from pathlib import Path
import time

from PIL import Image
import pytorch_lightning as pl
import torch, torchvision

import dataset
import network
    
    
def train(args):
    # Train a classification model
    pl.trainer.seed_everything(args.seed)
    
    datasplits, train_data, val_data = dataset.get_class_datasets(
        args.data, args.split, args.class_type, trainval=True)
    train_load = dataset.get_loader(train_data, args.bs, args.workers)
    val_load = dataset.get_loader(val_data, args.bs, args.workers, train=False)
    
    model = network.CastleClassifier(train_data.nclass, args)

    log = pl.loggers.CSVLogger(args.root, f'classification')
    cp = pl.callbacks.ModelCheckpoint()
    trainer = pl.Trainer.from_argparse_args(
        args, logger=log, checkpoint_callback=cp, default_root_dir=args.root,
        num_sanity_val_steps=0)
    trainer.fit(model, train_load, val_load)
    

def test(args):
    # Compute test performance statistics
    if not args.load_checkpoint:
        raise ValueError('Must provide a checkpoint to test from with '
                         '--load_checkpoint')
    ckpt = torch.load(args.load_checkpoint, map_location='cpu')
    cargs = ckpt['hyper_parameters']
    transfer = ['split','seed','net']
    for key in transfer:
        setattr(args, key, cargs[key])
    
    pl.trainer.seed_everything(args.seed)
    
    datasplits, test_data = dataset.get_class_datasets(
        args.data, args.split, args.class_type, trainval=False)
    test_load = dataset.get_loader(
        test_data, args.bs, args.workers, train=False)
    
    cas2ctry, ctry2lab = dataset.castle_to_country(datasplits.name2label.keys())
    label_map = [ctry2lab[cas2ctry[x]] for x in cas2ctry]
    
    model = network.CastleClassTester(test_data.nclass, args)
    model.load_state_dict(ckpt['state_dict'])
    model.set_label_map(label_map)
    
    log = pl.loggers.CSVLogger(args.root, f'classification/test')
    trainer = pl.Trainer.from_argparse_args(
        args, logger=log, default_root_dir=args.root)
    trainer.test(model, test_load)
    
def extract(args):
    # Extract and save per-image predictions
    if not args.load_checkpoint:
        raise ValueError('Must provide a checkpoint to test from with '
                         '--load_checkpoint')
    ckpt = torch.load(args.load_checkpoint, map_location='cpu')
    cargs = ckpt['hyper_parameters']
    transfer = ['split','seed','net']
    for key in transfer:
        setattr(args, key, cargs[key])
    
    pl.trainer.seed_everything(args.seed)
    
    datasplits, test_data = dataset.get_class_datasets(
        args.data, args.split, args.class_type, trainval=False)
    test_data.set_return_ind(True)
    test_load = dataset.get_loader(
        test_data, args.bs, args.workers, train=False)
    
    model = network.CastleClassifier(test_data.nclass, args)
    model.load_state_dict(ckpt['state_dict'])
    model.output_file = str(Path(args.load_checkpoint).parent.joinpath('preds.pth'))
    
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=args.root)
    trainer.test(model, test_load)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('run', type=str, choices=['train','test','extract'])
    parser.add_argument('--root', type=str, default='checkpoints/')
    parser.add_argument('--class_type', type=str, choices=['castle','country'], default='castle')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--check_val_every_n_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--load_checkpoint', type=str, default=None)
    
    parser = network.CastleClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    print(f'Starting: {time.ctime()}')
    if args.run == 'train':
        train(args)
    elif args.run == 'test':
        test(args)
    elif args.run == 'extract':
        extract(args)
    print(f'Finished: {time.ctime()}')
