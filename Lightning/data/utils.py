import os 
import sys
import h5py 
import logging
import numpy as np 
import time
import fastmri
import fastmri.fftc as fftc
import torch 
from fastmri.data.transforms import to_tensor, tensor_to_complex_np

## Helper functions 
def generate_dataset(data_path, folder_name="singlecoil_kspace_train", dataset="fastmri"):
    # create new folder for dataset
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    start = time.time()
    if dataset == "fastmri":
        file_names = os.listdir(data_path)
        for count, f in enumerate(file_names, 1):
            print(f"Volume idx: {count}/{len(file_names)}")
            # old and new scan paths
            old_scan_path = os.path.join(data_path, f)  # file path
            new_scan_path = os.path.join(folder_name, f.split(".")[0])  # folder path
            if not os.path.exists(new_scan_path):
                os.mkdir(new_scan_path)

            hf = h5py.File(old_scan_path, 'r')
            kspace_volume = hf.get('kspace')
            for i in range(kspace_volume.shape[0]):
                kspace_slice = kspace_volume[i]
                slice_path = os.path.join(new_scan_path, f"slice{i + 1}.npy")
                np.save(slice_path, kspace_slice)
            hf.close()

    elif dataset == "cardiac":

        def arr_load_save(load_path, save_path):
            arr = np.load(load_path)
            if not os.path.exists(save_path): os.mkdir(save_path)
            for i in range(arr.shape[2]):
                img = arr[..., i]
                img_path = os.path.join(save_path, f"slice{i + 1}.npy")
                np.save(img_path, img)

        folder_names = os.listdir(data_path)
        for count, folder in enumerate(folder_names, 1):
            folder_path_r = os.path.join(data_path, folder)  # read path
            folder_path_w = os.path.join(folder_name, folder)  # write path
            if not os.path.exists(folder_path_w): os.mkdir(folder_path_w)
            print(f"Volume idx: {count}/{len(folder_names)}")
            patients = os.listdir(folder_path_r)
            for patient in patients:
                patient_path_r = os.path.join(folder_path_r, patient)  # read path
                patient_path_w = os.path.join(folder_path_w, patient)  # write path
                if not os.path.exists(patient_path_w): os.mkdir(patient_path_w)
                scans = os.listdir(patient_path_r)
                for scan in scans:
                    if "magnitude" in scan or "phase" in scan:
                        scan_path_r = os.path.join(patient_path_r, scan)
                        scan_path_w = os.path.join(patient_path_w, scan.split(".")[0])
                        arr_load_save(scan_path_r, scan_path_w)
                    else:
                        raise ValueError(f"Undefined scan. What is {scan}?")
    else:
        raise ValueError(f"Wrong dataset entry! There is no dataset whose name is {dataset}.")

    end = time.time()
    print("FastMRI dataset is constructed successfully!")
    print(f"Time elapsed: {end - start:.4f}")

# Creata a logger for train history 
def log(path, file_name):
    if not os.path.exists("logs"):
        os.mkdir("logs")
    # check if the file exist
    log_file = os.path.join(path, file_name)
    # create a logger 
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    # create a file handler for output file
    handler = logging.FileHandler(log_file + ".log")
    # set the logging level for log file
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger


## Transformations 

class ToTensor(object):
    def __call__(self, sample):
        assert torch.is_tensor(sample)
        sample -= sample.min()
        return sample / sample.max() 

class InverseFourier(object):
    """Applies inverse Fourier transform to a sample."""
    def __call__(self, sample):
        assert torch.is_tensor(sample)
        return fftc.ifft2c_new(sample)

class Magnitude(object):
    """Calculates the absolute value of complex torch tensor."""
    def __call__(self, sample):
        assert torch.is_tensor(sample)
        return torch.norm(sample, dim=2)

class Phase(object):
    """Calculates the phase of a complex torch tensor."""
    def __call__(self, sample):
        assert torch.is_tensor(sample)
        return torch.arctan(sample[...,1] / sample[...,0])


