import sys, os, os.path as osp
import torch
from torch.utils.data import DataLoader, Subset,WeightedRandomSampler
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as tr
import pandas as pd
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, df, transforms):
        self.df=df
        try: self.im_list = df.file_path.values
        except: self.im_list = df['image_id'].values
        self.transforms = transforms

    def __getitem__(self, index):
        # load image and targets
        img = Image.open(self.im_list[index])
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.im_list)
class ClassDataset(Dataset):
    def __init__(self, csv_path, transforms):
        # assumes in the csv first column is file name, second column is target
        self.csv_path=csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df['image_id'].values
        self.target_list = df['label'].values
        self.classes = list(df['label'].unique())
        self.transforms = transforms

    def __getitem__(self, index):
        # load image and targets
        img = Image.open(self.im_list[index])
        target = self.target_list[index]
        img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.im_list)

def get_tr_vl_transforms_class(im_size, tr_aug=False):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    h_flip = tr.RandomHorizontalFlip()
    v_flip = tr.RandomVerticalFlip()
    rotate = tr.RandomRotation(degrees=30, fill=0)
    scale = tr.RandomAffine(degrees=0, scale=(0.95, 1.25), fill=0)
    transl = tr.RandomAffine(degrees=0, translate=(0.05, 0), fill=0)
    # either translate, rotate, or scale
    scale_transl_rot = tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.15, 0.15, 0.15, 0.05
    jitter = tr.ColorJitter(brightness, contrast, saturation, hue)
    sharpness = tr.RandomAdjustSharpness(sharpness_factor=0.25, p=0.5)
    # preparation
    resize = tr.Resize(im_size)
    tensorize = tr.ToTensor()
    normalize = tr.Normalize(mean, std)
    if tr_aug:
        tr_transforms = tr.Compose([resize, tr.TrivialAugmentWide(), tensorize, normalize])
    else:
        tr_transforms = tr.Compose([resize,scale_transl_rot, jitter, sharpness, h_flip, v_flip, tensorize, normalize])
    vl_transforms = tr.Compose([resize, tensorize, normalize])
    return tr_transforms, vl_transforms
def get_train_val_loaders(csv_path_tr, batch_size, im_size, tr_aug=False, num_workers=0):
    tr_transforms, vl_transforms = get_tr_vl_transforms_class(im_size, tr_aug=tr_aug)
    
    tr_ds = ClassDataset(csv_path_tr, transforms=tr_transforms)
    csv_path_vl = csv_path_tr.replace('tr', 'vl')
    vl_ds = ClassDataset(csv_path_vl, transforms=vl_transforms,)
    ovft_ds = ClassDataset(csv_path_tr, transforms=vl_transforms,)

    subset_size = len(vl_ds)
    subset_idxs = torch.randperm(len(ovft_ds))[:subset_size]
    ovft_ds = Subset(ovft_ds, subset_idxs)

    tr_loader = DataLoader(dataset=tr_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    vl_loader = DataLoader(dataset=vl_ds, batch_size=2*batch_size, num_workers=num_workers)
    ovft_loader= DataLoader(dataset=ovft_ds, batch_size=2*batch_size, num_workers=num_workers)

    print(20 * '*')
    for c in range(len(np.unique(tr_ds.target_list))):
        exs_train = np.count_nonzero(tr_ds.target_list == c)
        exs_val = np.count_nonzero(vl_ds.target_list == c)
        print('Found {:d}/{:d} train/val examples of class {}'.format(exs_train, exs_val, c))

    return tr_loader, ovft_loader, vl_loader

def get_test_loader(df_test, batch_size, im_size, num_workers=0):
    _, test_transforms = get_tr_vl_transforms_class(im_size)
    test_ds = TestDataset(df_test, transforms=test_transforms)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=num_workers)

    return test_loader


def get_sampling_probabilities(class_count, mode='instance'):
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities

def modify_loader(loader, oversampling=False, mode='sqrt'):
    if not oversampling:
        return loader
    else:
        class_count = np.unique(loader.dataset.target_list, return_counts=True)[1]
        sampling_probs = get_sampling_probabilities(class_count, mode=mode)
        sample_weights = sampling_probs[loader.dataset.target_list]

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    return mod_loader