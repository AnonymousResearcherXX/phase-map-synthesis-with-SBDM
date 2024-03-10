

import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3




def frechet_value(real_paths, fake_paths, device, dims=2048):
    # load InceptionV3 model 
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    # calculate statistics for both batch 
    m1, s1 = calculate_activation_statistics(real_paths, model, device=device)
    m2, s2 = calculate_activation_statistics(fake_paths, model, device=device)
    # calculate frechet value 
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def calc_frechet(model, dataset, size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # sample real images randomly 
    perm = torch.randperm(len(dataset))
    idxs = perm[:size]
    # img path names
    real_paths = [dataset[i][1] for i in idxs]
    # create fake images 
    z = torch.randn(size, 100, 1, 1)
    fake_imgs = model.generator(z).mul(0.5).add(0.5)
    # create directory for fake images 
    folder_name = "fake_images"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    # save fake images
    fake_paths = []
    for idx in range(fake_imgs.shape[0]):
        f = os.path.join(folder_name, f"fake_img{idx}.jpg")
        save_image(fake_imgs[idx], fp=f)
        fake_paths.append(f)
    # calculate frechet value 
    fid = frechet_value(real_paths, fake_paths, device)
    return fid


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
        return

    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()