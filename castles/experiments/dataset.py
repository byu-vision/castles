'''Classes and functions for handling the data, including creating train/val/test
splits and pytorch Datasets and DataLoaders'''

from collections import Counter
from itertools import chain, groupby
import os
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch, torchvision


#################################################
# GLOBALS
#################################################

DATA_ROOT = './castle-images'  # make sure to update the path to the images
FIVE_SPLIT = [
    [[0, 1, 2], [3, 4]],
    [[1, 2, 3], [4, 0]],
    [[2, 3, 4], [0, 1]],
    [[3, 4, 0], [1, 2]],
    [[4, 0, 1], [2, 3]]
]

# train and val/test transforms
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


#################################################
# DATASET
#################################################

class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, ind):
        label = self.targets[ind]
        path = self.imgs[ind]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.tform: img = self.tform(img)
        if self.target_tform: label = self.target_tform(label)
        if self.return_ind:
            return img, label, ind
        return img, label
    
    def set_return_ind(self, opt=True):
        self.return_ind = opt


def get_img_files(root):
    '''Returns a list of image file paths found under `root`'''
    imgs = []
    for d, ds, fs in os.walk(root):
        if fs:
            p = os.path.split(d)[1]
            imgs.extend([os.path.join(p,x) for x in fs])
    imgs.sort()
    return imgs


def get_img_files_with_counts(root):
    '''Returns a list of image file paths found under `root`, along with a
    count of the number of images in each folder'''
    imgs = []
    counts = {}
    for d, ds, fs in os.walk(root):
        if fs:
            p = os.path.split(d)[1]
            ims = [os.path.join(p,x) for x in fs]
            imgs.extend(ims)
            counts[p] = len(ims)
    imgs.sort()
    return imgs, counts


def targets_from_folders(img_files):
    '''Return an integer target along with the mapping between folder names
    and targets from a list of images, divided into folders by class'''
    names = [x.split('/')[-2] for x in img_files]
    name2label = {n:i for i,n in enumerate(sorted(set(names)))}
    targets = [name2label[n] for n in names]
    return targets, name2label


def castle_to_country(castle_names, cutoff=None, return_counts=False):
    '''Return a mapping from castle name to country name, and a mapping from
    country name to integer country label'''
    castle2country = {x: x.split('_')[1] for x in castle_names}
    country_counts = Counter(castle2country.values())
    if cutoff:
        country_counts = {k:v for k,v in country_counts.items() if v >= cutoff}
    country2label = {x:i for i,x in enumerate(sorted(country_counts.keys()))}
    if return_counts:
        return castle2country, country2label, country_counts
    return castle2country, country2label


################################################
# K-FOLD SPLITS
################################################

def get_splits(labels, k=5):
    '''Returns indices to split data into `k` non-overlapping folds'''
    splitter = StratifiedKFold(k)
    splits = [x[1] for x in splitter.split(np.zeros_like(labels), labels)]
    return splits


class ImgDataset(_Dataset):
    def __init__(self, root, img_list, targets, nclass, 
                 tform=None, target_tform=None):
        self.root = root
        self.imgs = img_list
        self.targets = targets
        self.nclass = nclass
        self.tform = tform
        self.target_tform = target_tform
        self.return_ind = False
    
    
class DataSplits(object):
    '''Splits the data into multiple balanced, non-overlapping subsets which
    can be combined into different train/val/test splits'''
    def __init__(self, root, n=5):
        self.root = root
        self.n = n
        self.imgs = get_img_files(root)
        self.targets, self.name2label = targets_from_folders(self.imgs)
        self._set_splits()
    
    def _set_splits(self):
        self.splits = get_splits(self.targets, self.n)
        
    def n_splits(self):
        '''Return the number of splits'''
        return len(self.splits)
    
    def data_size(self):
        '''Return the total size of the data across all splits'''
        return sum(len(x) for x in self.splits)
    
    def get_split(self, splits, with_val=None, seed=0):
        '''Returns a split of the data consisting of several of the atomic
        splits concatenated together. `splits` is a list of integers to 
        determine which (numbered) splits should be combined. If `with_val` is
        True, this split is further divided into training and validation portions.
        '''
        if isinstance(splits, (list, tuple, slice, range)):
            pass
        elif isinstance(splits, int):
            splits = [splits]
        else:
            raise ValueError(f'Type {type(splits)} is not valid for `splits`')
        
        split = np.concatenate([self.splits[i] for i in splits])
        if with_val:
            targets = [self.targets[i] for i in split]
            train, val = train_test_split(
                split, test_size=with_val, random_state=seed, stratify=targets)
            train.sort()
            val.sort()
            return train, val
        split.sort()
        return split
    
    def get_data(self, split, tform=None, target_tform=None):
        '''Return an ImgDataset object containing data from one or more splits.
        
        Args:
            splits (int or array-like of int): the split indices to use
            tform (callable): a transforms callable to pass on to the dataset
                for transforming images
            target_tforms (callable): a transforms callable to pass on to the
                dataset for transforming targets
        '''
        targets = [self.targets[i] for i in split]
        imgs = [self.imgs[i] for i in split]
        
        return ImgDataset(self.root, imgs, targets, len(self.name2label),
                          tform, target_tform)
    
    
class CountryDataSplits(DataSplits):
    '''Creates splits at the country level instead of the instance level'''
    def _set_splits(self):
        ca_targets, name2lab = self.targets, self.name2label
        lab2name = {v:k for k,v in name2lab.items()}
        
        # map castle labels to lists of img indices
        castle2imgs = {k:list(g) for k,g in groupby(range(len(ca_targets)),
                                                    lambda i:ca_targets[i])}

        # ca2co = map from castle-name to country-name
        # co2la = map from country-name to numeric country label
        # ccs = map from country-name to number of castles in the country
        ca2co, co2la, ccs = castle_to_country(name2lab.keys(), 5, True)
        names = [x for x in name2lab if ca2co[x] in ccs]
        ctry_labels = [co2la[ca2co[x]] for x in names] # get the numeric country label for each castle
        ctry_split = get_splits(ctry_labels, 5) # split the castles into subsets
        
        # lists of img/target indices corresponding to the subset of castles
        # in each split
        splits = [np.concatenate([castle2imgs[name2lab[names[i]]]
                                  for i in csp]) 
                  for csp in ctry_split]
        
        # convert castle-level labels to country-level labels (-1 for castles
        # that aren't in the country set)
        targets = [co2la.get(ca2co[lab2name[t]],-1) for t in ca_targets]

        self.targets = targets
        self.castle2country = ca2co
        self.name2label = co2la
        self.splits = splits
        
        
class DisjointCastleSplits(DataSplits):
    '''Evenly partition the set of instances, so that all images of an instance
    are in the same split'''
    def _set_splits(self):
        # Split set of all castles into disjoint parts
        labels = list(self.name2label.values())
        labels = random.Random(0).sample(labels, k=len(labels))
        castle_split = [[x for x in labels[i::5]] for i in range(5)]
        
        # map castle labels to lists of img indices
        castle2imgs = {
            k:list(g) for k,g in 
            groupby(range(len(self.targets)), lambda i:self.targets[i])
        }
        
        # lists of img/target indices corresponding to the subset of castles
        # in each split
        splits = [np.concatenate([castle2imgs[i] for i in s])
                  for s in castle_split]
        self.splits = splits
        

#################################################
# CONVENIENCE FUNCTIONS
#################################################

def get_class_datasets(root, split, class_type='castle', trainval=True):
    split_is = FIVE_SPLIT[split]
    
    if class_type == 'castle':
        dataclass = DataSplits
    elif class_type == 'country':
        dataclass = CountryDataSplits
    else:
        raise ValueError(f'{class_type} is not a valid class_type option')
    
    datasplits = dataclass(root, 5)
    
    if trainval:
        trn_sp, val_sp = datasplits.get_split(split_is[0], with_val=0.33)
        train_data = datasplits.get_data(trn_sp, t_tform)
        val_data = datasplits.get_data(val_sp, v_tform)
        return datasplits, train_data, val_data
    else:
        tst_sp = datasplits.get_split(split_is[1])
        test_data = datasplits.get_data(tst_sp, v_tform)
        return datasplits, test_data
    
    
def get_retrieval_datasets(root, split, trainval=True):
    split_is = FIVE_SPLIT[split]
    
    datasplits = DisjointCastleSplits(root, 5)
    
    if trainval:
        trn_sp, val_sp = datasplits.get_split(split_is[0], with_val=0.33)
        train_data = datasplits.get_data(trn_sp, t_tform)
        val_data = datasplits.get_data(val_sp, v_tform)
        return datasplits, train_data, val_data
    else:
        tst_sp = datasplits.get_split(split_is[1])
        test_data = datasplits.get_data(tst_sp, v_tform)
        return datasplits, test_data

        
def get_loader(data, bs=32, workers=4, train=True, sampler=None):
    '''Return a torch.utils.data.DataLoader for `data`'''
    a = dict(num_workers=workers, pin_memory=True, sampler=sampler)
    shuffle = train and (sampler is None)
    drop = shuffle
    if train: a.update(batch_size=bs,shuffle=shuffle,drop_last=drop)
    else: a.update(batch_size=2*bs,shuffle=shuffle,drop_last=drop)
    return torch.utils.data.DataLoader(data, **a)