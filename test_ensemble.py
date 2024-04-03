import sys, time
import os, os.path as osp

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from utils.model_factory_mh import get_model
from utils.data_load import get_test_loader
import random


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

def get_args_parser():
    import argparse

    def str2bool(v):
        # as seen here: https://stackoverflow.com/a/43357954/3208255
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', 'yes'):
            return True
        elif v.lower() in ('false', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected.')

    parser = argparse.ArgumentParser(description='Training for Biomedical Image Classification')
    parser.add_argument('--csv_path_test', type=str, default='test.csv', help='csv path test data')
    parser.add_argument('--csv_path_out', type=str, default='submission4.csv', help='csv path training data')
    parser.add_argument('--model_path_1', type=str, default='bceLS_rx50_bs8_512_OS_5x3', help='five-fold models stored F*/here')
    parser.add_argument('--model_name_1', type=str, default='resnext50', help='architecture')
    parser.add_argument('--model_path_2', type=str, default='bceLS_swin_bs8_512_OS_5x3', help='five-fold models stored F*/here')
    parser.add_argument('--model_name_2', type=str, default='swin_t', help='architecture')   
    parser.add_argument('--batch_size', type=int, default=16, help=' batch size')
    parser.add_argument('--im_size', type=str, default='512/512', help='im size/spatial xy dimension')
    parser.add_argument('--tta', type=str2bool, nargs='?', const=True, default=False, help='TTA')
    parser.add_argument('--seed', type=int, default=None, help='fixes random seed (slower!)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers')
    args = parser.parse_args()

    return args


def get_probs(model, test_loader, tta=False):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    y_prob = []
    for inputs in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        if tta:
            outputs += model(inputs.flip(-1))
            outputs += model(inputs.flip(-2))
            outputs += model(inputs.flip(-1).flip(-2))
            outputs /= 4.0
        y_prob.append(outputs.sigmoid().cpu().numpy())
    y_prob = np.concatenate(y_prob, axis=0)
    return y_prob


if __name__ == '__main__':
    args = get_args_parser()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)
    n_classes = 1

    # prepare data loader
    bs, nw = args.batch_size, 8
    im_size = args.im_size.split('/')
    im_size = tuple(map(int, im_size))
    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, nw))
    df_test = pd.read_csv(args.csv_path_test)
    df_test = pd.read_csv(args.csv_path_test)
    
    test_loader = get_test_loader(df_test, bs, im_size, num_workers=nw)

    print('* Starting to test\n', '-' * 10)
    start = time.time()
    y_prob_ens = []

    # PREDICTIONS MODEL 1: #################################################################################
    # prepare models for ensembling
    load_path_1 = osp.join('experiments', args.model_path_1)
    weight_paths = [osp.join(load_path_1, l, 'best_model.pth') for l in os.listdir(load_path_1) if not l.startswith('.')]
    
    print('* Predicting with model {}, I found {} trained weights'.format(args.model_path_1, len(weight_paths)))
    models = [get_model(args.model_name_1, n_classes=1, ).to(device) for _ in range(len(weight_paths))]
    states = [torch.load(w, map_location=device) for w in weight_paths]
    for i in range(len(models)):
        models[i].load_state_dict(states[i])
    with torch.inference_mode():
        for model in models:
            y_probs = get_probs(model, test_loader, tta=args.tta)
            y_prob_ens.append(y_probs)
    print('* Collected {} predictions so far'.format(len(y_prob_ens)))

    if args.model_path_2 is not None:
        # PREDICTIONS MODEL 2: #################################################################################
        # prepare models for ensembling
        load_path_2 = osp.join('experiments', args.model_path_2)
        weight_paths = [osp.join(load_path_2, l, 'best_model.pth') for l in os.listdir(load_path_2)]
        
        print('* Predicting with model {}, I found {} trained weights'.format(args.model_path_2, len(weight_paths)))
        models = [get_model(args.model_name_2, n_classes=1, ).to(device) for _ in range(len(weight_paths))]
        states = [torch.load(w, map_location=device) for w in weight_paths]
        for i in range(len(models)):
            models[i].load_state_dict(states[i])
        with torch.inference_mode():
            for model in models:
                y_probs = get_probs(model, test_loader, tta=False)
                y_prob_ens.append(y_probs)
        print('* Collected {} predictions so far'.format(len(y_prob_ens)))


    # ENSEMBLE ALL PREDICTIONS: #################################################################################
    n_preds = len(y_prob_ens)
    print('* Ensembling all {} predictions'.format(n_preds))
    y_prob_ens = np.stack(y_prob_ens, axis=0).mean(axis=0).flatten()
    print('* Ensembling all {} predictions -- done'.format(n_preds))

    print('* Generating CSV for submission')
    prediction_df = df_test[['image_id']].copy()
    prediction_df['prob'] = y_prob_ens
    filepath = osp.join('~/project/', args.csv_path_out)
    prediction_df.to_csv(filepath, index=False, header=False)
    print('* Generating CSV for submission -- done')

    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Testing time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

