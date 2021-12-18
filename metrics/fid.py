import os
import argparse

def build_parser():
    parser = argparse.ArgumentParser(description='Perform FID computation')
    parser.add_argument('--dataset1', type=str, default='', help='Path to dataset 1')
    parser.add_argument('--dataset2', type=str, default='', help='Path to dataset 2')

    return parser

def compute_fid(dataset1: str, dataset2: str) -> float:
    """
    Computes the FID given two paths to dataset directories.
    """

    #install the following: https://github.com/mseitzer/pytorch-fid

    os.system(f"python -m pytorch_fid {dataset1} {dataset2}")
    

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    dataset1 = args.dataset1
    dataset2 = args.dataset2

    print('Dataset 1:', dataset1)
    print('Dataset 2:', dataset2)

    print('FID:', compute_fid(dataset1, dataset2))