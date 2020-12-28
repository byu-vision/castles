'''Run cross-validation experiments for per-image difficulty analysis'''
from argparse import ArgumentParser
from itertools import combinations
from PIL import Image
import os

import pytorch_lightning as pl
import torch, torchvision

import dataset
import network
    
    
def train(train_data, val_data, args):
    train_load = dataset.get_loader(train_data, args)
    val_load = dataset.get_loader(val_data, args, train=False)
    
    model = network.CastleCrossClassifier(train_data.nclass, args)
    
    cp = pl.callbacks.ModelCheckpoint()
    
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=cp,
                                            default_root_dir=args.root,
                                            num_sanity_val_steps=0)
    trainer.fit(model, train_load, val_load)

    
def train_all(args):
    '''Consecutively train models on each cross-val combination of data.
    Note that since this calls trainer.fit multiple times, it does not work
    with ddp backend: use dp instead.'''
    t_tform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224,(.8,1.2),(.5,1)),
        torchvision.transforms.ColorJitter(.1,.1,.1),
        torchvision.transforms.RandomGrayscale(0.05),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    ])
    v_tform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    ])
    
    data_splits = dataset.DataSplits(args.data, args.split[0])
    combs = list(combinations(range(args.split[0]), args.split[1]))
    
    i = 1
    for ct, cv in zip(combs, combs[::-1]):
        ct = data_splits.get_split(ct)
        cv = data_splits.get_split(cv)
        train_data = data_splits.get_data(ct, t_tform)
        val_data = data_splits.get_data(cv, v_tform)

        train_load = dataset.get_loader(train_data, args.bs, args.workers)
        val_load = dataset.get_loader(val_data, args.bs, args.workers, train=False)
        
        cp = pl.callbacks.ModelCheckpoint()
        log = pl.loggers.CSVLogger(args.root, f'resnet34')
        
        kw = dict(
            logger=log,
            checkpoint_callback=cp,
            default_root_dir=args.root,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=5,
        )
        trainer = pl.Trainer.from_argparse_args(args, **kw)
        model = network.CastleCrossClassifier(train_data.nclass, args)
        trainer.fit(model, train_load, val_load)
        i += 1
        
def extract_predictions(args):
    '''Extract image-level predictions from each of the different cross-val
    models and combine them to get a spread of predictions for each image.
    Note that since this calls trainer.test multiple times, it does not work
    with ddp backend: use dp instead.'''
    tform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    ])
    
    data_splits = dataset.DataSplits(args.data, args.split[0])
    combs = list(combinations(range(args.split[0]), args.split[1]))
    
    trainer = pl.Trainer.from_argparse_args(args)
    results = {}
    for i, cv in enumerate(combs[::-1]):
        cv = data_splits.get_split(cv)
        data = data_splits.get_data(cv, tform)
        data.set_return_ind(True)
        loader = dataset.get_loader(data, args.bs, args.workers, train=False)
        
        d = f'checkpoints/crossval/resnet34/version_{i}/checkpoints/'
        wp = list(os.scandir(d))[0].path
        weights = torch.load(wp)['state_dict']
        model = network.CastleCrossClassifier(data.nclass, args)
        model.load_state_dict(weights)

        r = trainer.test(model, loader)
        
        preds, labels, inds = model.preds, model.labels, model.inds
        for pred, lab, ind in zip(preds, labels, inds):
            path = data.imgs[ind]
            rr = results.get(path, [])
            if len(rr):
                plist = rr[1]
            else:
                rr.append(lab)
                plist = []
                rr.append(plist)
            plist.append(pred)
            results[path] = rr
    paths = list(results.keys())
    data = list(results.values())
    labels = torch.as_tensor([x[0] for x in data])
    preds = torch.as_tensor([x[1] if len(x[1])==10 
                             else x[1]+[-1]*(10-len(x[1]))
                             for x in data])
    torch.save({'paths':paths, 'labels':labels, 'preds':preds},
               'checkpoints/crossval/resnet34/preds.pth')
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('run', type=str, choices=['train','extract'])
    parser.add_argument('--root', type=str, default='checkpoints/crossval/')
    parser.add_argument('--split', type=int, nargs=2, default=[6,3])
    
    parser = network.CastleCrossClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    if args.run == 'train':
        train_all(args)
    elif args.run == 'extract':
        extract_predictions(args)
