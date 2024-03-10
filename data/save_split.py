


from datasets import CardiacDataset
import torch 
import numpy as np
from pathlib import Path 
import random



if __name__ == "__main__":

    # for reproducible results (important for train and test split)
    seed = 101
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    torch.autograd.set_detect_anomaly(True)

    dataset = CardiacDataset(data_path="../cardiac_data")
    # prepare train-val-test datasets 
    train_data, others = dataset.split(70, 24, seed)
    val_data, test_data = others.split(4, 20, seed)
    train_data.augmentation=True

    split_dir = Path.cwd() / "split_dict"
    # save the dictionaries for each set type 
    train_data.save_split(split_dir / "train_dict.pkl")
    val_data.save_split(split_dir / "val_dict.pkl")
    test_data.save_split(split_dir / "test_dict.pkl")
