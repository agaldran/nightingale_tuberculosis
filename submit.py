import sys, time
import os, os.path as osp
import argparse
import pandas as pd
import ngsci

def get_args_parser():

    parser = argparse.ArgumentParser(description='Training for Biomedical Image Classification')
    parser.add_argument('--csv_path_submission', type=str, default='project/submission4.csv', help='csv path training data')
    parser.add_argument('--description', type=str, default='4th and last submission, 5-fold with two models', help='message')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args_parser()
    ngsci.submit_contest_entry(args.csv_path_submission, description=args.description)