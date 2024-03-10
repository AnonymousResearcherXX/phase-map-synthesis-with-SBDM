
import os 
import h5py 
import numpy as np 
import random
import pickle
import logging
import copy 
import xml.etree.ElementTree as etree
import json
import pickle
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    OrderedDict
)
# torch imports 
import torch 
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as tvt
import torchvision.transforms.functional as F
# my imports
from .utils import ToTensor
from .transforms import AffineTransform, CenterCrop
#from utils import ToTensor
#from transforms import AffineTransform, CenterCrop
# fastMRI imports 
import fastmri
import fastmri.fftc as fftc
from fastmri.data.transforms import to_tensor
from .subsample import create_mask_for_mask_type


## FUNCTIONS 

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

## CLASSES 

class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]

class KneeVarNetDataset(Dataset):
    def __init__(self, data_path: Path, msk_fnc: Any) -> None:
        super().__init__()
        self.scan_list = data_path.glob("*.h5")
        self.msk_fnc = msk_fnc
    
    def __getitem__(self, index) -> Any:
        h5_file = self.scan_list[index]
        h5_object = h5py.File(h5_file)
        kspace = torch.from_numpy(np.array(h5_object['kspace']))
        kspace = kspace.unsqueeze(dim=1)
        masked_kspace, mask, num_low_frequencies = self._apply_mask(kspace, self.msk_fnc)
        
        complex_img = fastmri.fftc.ifft2c(torch.stack(kspace[...,0], kspace[...,1], -1))
        mag_img = torch.norm(complex_img, dim=3)
        return masked_kspace.unsqueeze(0), mask.bool(), num_low_frequencies, mag_img
    

class FastMRIDataset(Dataset):
    def __init__(self, data_path=None, transforms=None, path_scan_list=None):
        self.transforms = transforms
        if path_scan_list is not None:
            self.path_list, self.scan_list = path_scan_list 
        else:
            self.path_list = []
            self.scan_list = []

            folder_list = os.listdir(data_path)
            for folder_name in folder_list:
                folder_path = os.path.join(data_path, folder_name)
                self.scan_list.append(folder_path)
                file_list = os.listdir(folder_path)
                for f in file_list:
                    full_path = os.path.join(folder_path, f)
                    self.path_list.append(full_path)
            print("Dataset is constructed!")

    def random_scan(self, random_seed=None):
        if random_seed != None: 
            random.seed(random_seed)
        random_idx = random.randint(0, len(self.path_list))
        file_path = self.path_list[random_idx]
        dirname = os.path.dirname(file_path)
        # get the indices 
        indices = []
        for idx, path in enumerate(self.path_list):
            if dirname in path:
                indices.append(idx)
        volume = DataLoader(Subset(self, indices), batch_size=len(indices), shuffle=False)
        return next(iter(volume))

    def train_test_split(self, train_size=900, test_size=73):
        """Divides the current dataset into train and test sets."""
        if not (train_size + test_size) == len(self.scan_list):
            raise ValueError(f"The values for train and test sizes are not sum up to the total number of scans, which is {len(self.scan_list)}.")
        
        test_scan_idxs = np.random.choice(len(self.scan_list), test_size, replace=False)
        # Get the scan names corresponding to the indices 
        test_scan_names = []
        train_scan_names = []
        for idx, name in enumerate(self.scan_list):
            if idx in test_scan_idxs:
                test_scan_names.append(name)
            else: 
                train_scan_names.append(name)

        # Find the train and test indices 
        train_paths = []
        test_paths = []
        # Fill the train and test idxs lists   
        for idx, file_path in enumerate(self.path_list):
            test_scan_found = False
            for test_scan in test_scan_names:
                if test_scan in file_path:
                    test_paths.append(file_path)
                    test_scan_found = True
                    break
            if not test_scan_found:
                train_paths.append(file_path)

        train_path_scan_list = (train_paths, train_scan_names)
        test_path_scan_list = (test_paths, test_scan_names)
        return {"train": FastMRIDataset(transforms=self.transforms, path_scan_list=train_path_scan_list),
                "test": FastMRIDataset(transforms=self.transforms, path_scan_list=test_path_scan_list)}

    def _get_kspace(self, mag_img: torch.tensor, phase_img: torch.tensor) -> torch.tensor:

        mag_img = mag_img.unsqueeze(dim=2)
        phase_img = phase_img.unsqueeze(dim=2)
        # scale images 
        mag_img -= mag_img.min()
        mag_img /= mag_img.max()
        phase_img -= phase_img.min()
        phase_img = phase_img / phase_img.max() * 2 * torch.pi
        # get the complex image 
        complex_img = mag_img * torch.exp(1j * phase_img)
        real = torch.real(complex_img)
        imag = torch.imag(complex_img)
        complex_img = torch.concat([real, imag], dim=2)
        kspace = fftc.fft2c_new(complex_img)
        return kspace

    def _apply_mask(
        self,
        data,
        mask_func,
        offset=None,
        seed=None,
        padding=None,
    ):
        shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
        mask, num_low_frequencies = mask_func(shape, offset, seed)
        if padding is not None:
            mask[..., : padding[0], :] = 0
            mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros
        
        masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

        return masked_data, mask, num_low_frequencies

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        file_path = self.path_list[idx]
        kspace = np.load(file_path)

        ## Generate magnitude and phase images from k-space image 
        kspace = to_tensor(kspace).unsqueeze(dim=0)
        ifft_tensor = fftc.ifft2c_new(kspace)
        # Magnitude 
        mag_tensor = torch.norm(ifft_tensor, dim=3)
        mag_tensor = F.center_crop(mag_tensor, 320)
        # Phase 
        phase_tensor = torch.arctan(ifft_tensor[...,1] / ifft_tensor[...,0])
        phase_tensor = F.center_crop(phase_tensor, 320)

        padder = tvt.Pad(32)
        
        return padder(mag_tensor), padder(phase_tensor)

class CardiacDataset(Dataset):
    """
    Base class for Cardiac Dataset
    """
    def __init__(self, data_path="cardiac_data", data_dict=None, phase_type="ground_truth"):
        self.augmentation = False  # augmentation is OFF for test and val
        self.phase_type = phase_type
        if data_dict is not None:
            self.path_list_m, self.scan_list_m = data_dict["magnitude"]["path"], data_dict["magnitude"]["scan"]
            self.path_list_p, self.scan_list_p = data_dict["phase"]["path"], data_dict["phase"]["scan"]
        else:
            self.path_list_m, self.scan_list_m = [], []
            self.path_list_p, self.scan_list_p = [], []

            folder_list = os.listdir(data_path)
            for folder_name in folder_list:
                folder_path = os.path.join(data_path, folder_name)
                patient_list = os.listdir(folder_path)
                for patient in patient_list:
                    patient_path = os.path.join(folder_path, patient)
                    scan_list = os.listdir(patient_path)
                    for scan in scan_list:
                        scan_path = os.path.join(patient_path, scan)
                        if "magnitude" in scan:
                            scan_path_list = self.scan_list_m
                            slice_path_list = self.path_list_m
                        elif "phase" in scan:
                            scan_path_list = self.scan_list_p
                            slice_path_list = self.path_list_p
                        else:
                            raise ValueError(f"Invalid folder name. What is {scan}?")
                        scan_path_list.append(scan_path)
                        slice_list = os.listdir(scan_path)
                        for slice in slice_list:
                            slice_path = os.path.join(scan_path, slice)
                            slice_path_list.append(slice_path)
        
        # must have equal number of slices
        assert len(self.path_list_m) == len(self.path_list_p)
        assert len(self.scan_list_m) == len(self.scan_list_p)

        self.patient_list = []
        # create patient list
        for scan_path in self.scan_list_m:
            folders = scan_path.split("/")
            patient_path = os.path.join(folders[0], folders[1], folders[2])
            if patient_path not in self.patient_list:
                self.patient_list.append(patient_path)
    
    def save_split(self, path: Path) -> None:
        """Save the train-val-test split file paths as a dictionary"""
        if not path.parent.exists(): path.parent.mkdir(parents=True)

        data_dict = {
            "magnitude":
                {
                    "scan": [],
                    "path": []
                },
            "phase":
                {
                    "scan": [],
                    "path": []
                },
        }

        data_dict["magnitude"]["path"] = self.path_list_m
        data_dict["magnitude"]["scan"] = self.scan_list_m
        data_dict["phase"]["path"] = self.path_list_p
        data_dict["phase"]["scan"] = self.scan_list_p

        pkl_file = open(path, 'wb')
        pickle.dump(data_dict, pkl_file)
        pkl_file.close()
    
    def random_scan(self, random_seed=None):
        if random_seed != None:
            random.seed(random_seed+7348)
        random_idx = random.randint(0, len(self.path_list_m) - 1)
        #"../cardiac_data/PREHFPEF_NOSIEGI_54999_12_29_2020/meas_MID00095_FID14455_perf_REST_dilute_tpat3_midSliceOnly_mocoADV_f40_40frames/magnitude1/slice1.npy"#self.path_list_m[random_idx]
        #
        #file_path = Path("./cardiac_data/PREHFPEF_NOSIEGI_54999_12_29_2020/meas_MID00096_FID14456_perf_REST_full_tpat3_mocoADV_7ov8parF_85frames/magnitude1/slice1.npy")
        file_path = Path("../cardiac_data/PREHFPEF_NOSIEGI_54999_12_29_2020/meas_MID00095_FID14455_perf_REST_dilute_tpat3_midSliceOnly_mocoADV_f40_40frames/magnitude1/slice1.npy")
        dirname = os.path.dirname(file_path)
        # get the indices
        indices = []
        for idx, path in enumerate(self.path_list_m):
            #if dirname in path:
            if "../cardiac_data/PREHFPEF_NOSIEGI_54999_12_29_2020/meas_MID00095_FID14455_perf_REST_dilute_tpat3_midSliceOnly_mocoADV_f40_40frames/magnitude1/" in path:
                indices.append(idx)
        volume = DataLoader(Subset(self, indices), batch_size=len(indices), shuffle=False)
        return next(iter(volume))

    def split(self, train_size: int, test_size: int, seed: int):

        if not (train_size + test_size) == len(self.patient_list):
            raise ValueError(
                f"The values for train and test sizes are not sum up to the total number of scans, which is {len(self.patient_list)}.")

        random.seed(seed)
        random.shuffle(self.patient_list)
        # Get the scan names corresponding to the indices
        test_dict = {
            "magnitude":
                {
                    "scan": [],
                    "path": []
                },
            "phase":
                {
                    "scan": [],
                    "path": []
                },
        }

        train_dict = copy.deepcopy(test_dict)
        for idx, patient_path in enumerate(self.patient_list):
            # save data to train or test set
            if idx < train_size:
                data_dict = train_dict
            else:
                data_dict = test_dict
            # start walking through the files
            scan_list = os.listdir(patient_path)
            for sname in scan_list:
                scan_path = os.path.join(patient_path, sname)
                slice_list = os.listdir(scan_path)
                for slname in slice_list:
                    # THERE IS A PROBLEM HERE SIR!
                    if "magnitude" in slname:
                        scan_path_list = data_dict["magnitude"]["scan"]
                        slice_path_list = data_dict["magnitude"]["path"]
                    elif "phase" in slname:
                        scan_path_list = data_dict["phase"]["scan"]
                        slice_path_list = data_dict["phase"]["path"]
                    else:    
                        raise ValueError(f"Invalid folder name. What is {scan}?")
                    scan_path_list.append(scan_path)
                    sl_path = os.path.join(scan_path, slname)
                    slice_list = os.listdir(sl_path)
                    for fname in slice_list:
                        slice_path = os.path.join(sl_path, fname)
                        slice_path_list.append(slice_path)
        return (CardiacDataset(data_dict=train_dict, phase_type=self.phase_type), CardiacDataset(data_dict=test_dict, phase_type=self.phase_type))

    def transform(self, mag_img, phase_img, p=.3):  # p=0.3

        # Standard scale
        scaler = ToTensor()
        mag_img = scaler(mag_img)
        phase_img = scaler(phase_img)

        if self.augmentation and random.random() < p:
            ## Random Transforms
            # Random Rotation
            angle = random.random() * 120.0 - 60.0
            mag_img = F.rotate(mag_img, angle)
            phase_img = F.rotate(phase_img, angle)

            # Random Horizontal Flip
            """
            if random.random() > p:
                mag_img = F.hflip(mag_img)
                phase_img = F.hflip(phase_img)

            if random.random() > p:
                mag_img = F.vflip(mag_img)
                phase_img = F.vflip(phase_img)
            """

        # Center Crop
        crop = tvt.CenterCrop((128, 128))
        mag_img = crop(mag_img)
        phase_img = crop(phase_img)

        return mag_img, phase_img

    def get_patients(self):
        patient_list = []
        for path in self.path_list_m:
            name_list = path.split("/")
            patient = name_list[1]
            if patient not in patient_list:
                patient_list.append(patient)
        return patient_list

    def __len__(self):
        return len(self.path_list_m)

    def __getitem__(self, idx):
        if self.phase_type == "diffusion":
            path_m = self.path_list_m[idx].replace("phase", "synthetic_phase")
            path_p = self.path_list_p[idx].replace("phase", "synthetic_phase")
        elif self.phase_type == "ground_truth":
            path_m = self.path_list_m[idx]
            path_p = self.path_list_p[idx]
        else:
            raise ValueError(f"Undefined phase type. What is {self.phase_type}?")
        mag_img = torch.from_numpy(np.load(path_m)).type(torch.FloatTensor).unsqueeze(dim=0)
        phase_img = torch.from_numpy(np.load(path_p)).type(torch.FloatTensor).unsqueeze(dim=0)

        mag_img, phase_img = self.transform(mag_img, phase_img)

        return mag_img, phase_img
    

class OCMRDataset(Dataset):
    def __init__(
            self,
            msk_fnc: Any,
            patient_dict: Union[OrderedDict, Path],
            source_path: Optional[Path]=Path("../OCMR/ocmr_recon_data"),
            phase_type: str="gt_phase", 
            transforms: Optional[Any]=None,
            seed: Optional[bool]=None):
        
        if phase_type not in ("gt_phase", "diff_phase", "rand_phase", "zero_phase", "gan_phase"):
            raise ValueError(f"Type can be one of the following: gt_phase, diff_phase, rand_phase, gan_phase. Given type: {phase_type}")
        if isinstance(patient_dict, Path):
            print("Dict path:", patient_dict)
            self.patient_dict = pickle.load(open(patient_dict, 'rb'))
        else:
            self.patient_dict = patient_dict
        self.patients = list(self.patient_dict.keys())
        self.slice_list = []
        # create slice list
        for patient_id in self.patients:
            scans = self.patient_dict[patient_id]
            for scan_name in scans:
                self.slice_list.append(source_path / patient_id / scan_name.split(".npy")[0]) #.glob("gt_mag/*.npy")
        
        self.source_path = source_path
        self.phase_type = phase_type
        self.transforms = transforms
        self.seed = seed
        self.msk_fnc = msk_fnc
    
    def split(self, train_idxs, test_idxs):
        train_pdict = OrderedDict((key, value) for i, (key, value) in enumerate(self.patient_dict.items()) if i in train_idxs)
        test_pdict = OrderedDict((key, value) for i, (key, value) in enumerate(self.patient_dict.items()) if i in test_idxs)
        # train set 
        train_set = OCMRDataset(
            msk_fnc=self.msk_fnc,
            patient_dict=train_pdict, 
            source_path=self.source_path,
            transforms=self.transforms, 
            phase_type=self.phase_type, 
            seed=self.seed
        )
        # test set 
        test_set = OCMRDataset(
            msk_fnc=self.msk_fnc,
            patient_dict=test_pdict, 
            source_path=self.source_path, 
            transforms=self.transforms, 
            phase_type="gt_phase", 
            seed=self.seed
        )
        print("Train size:", len(train_set))
        print("Test size:", len(test_set))
        return train_set, test_set

    def _apply_mask(
        self,
        data,
        mask_func,
        offset=None,
        seed=None,
        padding=None,
    ):
        shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
        mask, num_low_frequencies = mask_func(shape, offset, seed)
        if padding is not None:
            mask[..., : padding[0], :] = 0
            mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros
        masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

        return masked_data, mask, num_low_frequencies
    
    def __len__(self):
        return len(self.slice_list)
    
    def random_scan(self, slice_path=None, random_seed=None):
        if random_seed != None:
            random.seed(random_seed+99343) #99343
        patient_id = random.choice(self.patients)
        scan_name = random.choice(self.patient_dict[patient_id]).split(".npy")[0]
        #scan_name = Path("./OCMR/ocmr_recon_data/fs_0079_1_5T/slice1")
        #"OCMR/ocmr_recon_data/fs_0028_3T/slice1"
        if slice_path == None:
            slice_path = Path(patient_id) / scan_name
        slice_path = str(slice_path)
        # get the indices 
        indices = []
        print("Slice Path:", slice_path)
        for idx, path in enumerate(self.slice_list):
            if slice_path in str(path):
                indices.append(idx)
        volume = DataLoader(Subset(self, indices), batch_size=len(indices), shuffle=False)
        return next(iter(volume))
    
    def __getitem__(self, index):
        slice_path = sorted(list(self.slice_list[index].glob("gt_mag/*.npy")), key=lambda x: int(str(x).split("frame")[1].split(".npy")[0]))
        batch_size = len(slice_path)
        frame0 = np.load(slice_path[0])
        H, W = frame0.shape
        mag_img = torch.zeros((batch_size, 1, H, W))
        for i in range(batch_size):
            frame_path = slice_path[i]
            mag_img[i] = torch.from_numpy(np.load(frame_path))
            #mag_img = torch.from_numpy(np.load(file))
            if self.phase_type in ("gt_phase", "diff_phase", "gan_phase"):
                phase_img = torch.from_numpy(np.load(str(frame_path).replace("gt_mag", self.phase_type)))
            elif self.phase_type == "rand_phase":
                phase_img = torch.randn_like(mag_img)
            elif self.phase_type == "zero_phase":
                phase_img = torch.zeros_like(mag_img)
        # augmentation 
        #normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        #mag_img = normalize(mag_img)
        #phase_img = normalize(phase_img) * 2 * np.pi - np.pi 
        if self.transforms is not None:
            mag_img, phase_img = self.transforms((mag_img, phase_img))
        # convert images to complex tensors
        #mag_img /= torch.max(mag_img)
        real_part = mag_img * torch.cos(phase_img)
        imag_part = mag_img * torch.sin(phase_img)
        target = torch.stack([real_part, imag_part], dim=-1)
        kspace = fftc.fft2c_new(target)
        masked_kspace, mask, num_low_frequencies = self._apply_mask(kspace, self.msk_fnc, offset=0) # add seed in inference 
        return masked_kspace, mask.bool(), num_low_frequencies, mag_img 

class KneeDataVarNet(Dataset):
    def __init__(
        self,
        msk_fnc: Any,
        patient_dict: Union[OrderedDict, Path],
        phase_type: str="gt_phase",
        source_path: Optional[Path]=Path("../KneeMRI/gen_test_data"),
        transforms: Optional[Any]=None,
        normalize_input: Optional[bool]=False
    ) -> None:
        
        if phase_type not in ("gt_phase", "gan_phase", "diff_phase", "rand_phase", "zero_phase", "gaus_phase", "gan2_phase"):
            raise ValueError(f"Type can be one of the following: gt_phase, diff_phase, rand_phase, gaus_phase. Given type: {phase_type}")
        # load patient-scan list 
        if isinstance(patient_dict, Path):
            self.patient_dict = pickle.load(open(patient_dict, 'rb'))
        else:
            self.patient_dict = patient_dict
        self.patients = list(self.patient_dict.keys())
        self.slice_list = []
        # create slice list
        for patient_id in self.patients:
            scans = self.patient_dict[patient_id]
            for scan_name in scans:
                self.slice_list += (source_path / scan_name).glob("gt_mag/*.npy")
        # other parameters
        self.phase_type = phase_type
        self.transforms  = transforms 
        self.msk_fnc = msk_fnc
        self.normalize_input = normalize_input
        self.resize = tvt.Resize(256, antialias=False)

    def random_scan(self, random_seed=None):
        if random_seed != None:
            random.seed(random_seed+777)
        patient_id = random.choice(self.patients)
        scan_name = random.choice(self.patient_dict[patient_id])
        # to display the images in sorted order
        indices = []
        paths = []
        for idx, path in enumerate(self.slice_list):
            if scan_name in str(path):
                indices.append(idx)
                paths.append(path)
        def return_indices(input_list):
            return sorted(range(len(input_list)), key=lambda x: int(str(input_list[x]).split("slice")[-1].split(".npy")[0]))
        sorted_indices = return_indices(paths)
        new_indices = [indices[idx] for idx in sorted_indices]
        volume = DataLoader(Subset(self, new_indices), batch_size=len(indices), shuffle=False)
        return next(iter(volume)), scan_name
    
    def __getitem__(self, index) -> Any:
        # load files
        file = self.slice_list[index]
        mag_img = self.resize(torch.from_numpy(np.load(file))[None,...])
        if self.phase_type in ("gt_phase", "diff_phase", "gan_phase"): 
            phase_img = torch.from_numpy(np.load(str(file).replace("gt_mag", self.phase_type)))
            phase_img = self.resize(phase_img[None,...])
        elif self.phase_type == "gan2_phase":
            phase_img = torch.from_numpy(np.load(str(file).replace("gt_mag", "gan2_phase/gan2_phase"))).unsqueeze(dim=0)
        elif self.phase_type == "rand_phase":
            phase_img = torch.rand_like(mag_img) * 2 * torch.pi 
        elif self.phase_type == "zero_phase":
            phase_img = torch.zeros_like(mag_img)
        elif self.phase_type == "gaus_phase":
            phase_img = torch.randn_like(mag_img)
        # augmentation 
        if self.transforms is not None:
            mag_img, phase_img = self.transforms((mag_img, phase_img)) 

        normalizer = lambda x : (x - x.min()) / (x.max() - x.min())
        mag_img = normalizer(mag_img)
        # normalize phase if needed
        if self.phase_type not in ("rand_phase", "zero_phase", "gaus_phase") and self.normalize_input: 
            phase_img = normalizer(phase_img) * 2 * torch.pi
        # scale images 
        #mag_img = mag_img.unsqueeze(dim=0)
        #phase_img = phase_img.unsqueeze(dim=0)
        real_part = mag_img * torch.cos(phase_img)
        imag_part = mag_img * torch.sin(phase_img)
        cimg = torch.stack([real_part, imag_part], dim=-1)
        kspace = fftc.fft2c_new(cimg)
        masked_kspace, mask, num_low_frequencies = self._apply_mask(kspace, self.msk_fnc, offset=0)
        return masked_kspace, mask.bool(), num_low_frequencies, mag_img
    
    def _apply_mask(
        self,
        data,
        mask_func,
        offset=None,
        seed=None,
        padding=None,
    ):
        shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
        mask, num_low_frequencies = mask_func(shape, offset, seed)
        if padding is not None:
            mask[..., : padding[0], :] = 0
            mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros
        
        masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

        return masked_data, mask, num_low_frequencies

    def __len__(self):
        return len(self.slice_list)

class BrainDataVarNet(Dataset):
    def __init__(
        self,
        msk_fnc: Any,
        patient_dict: Union[OrderedDict, Path],
        phase_type: str="gt_phase",
        source_path: Optional[Path]=Path("../BrainMRI/gen_test_data"),
        transforms: Optional[Any]=None,
        normalize_input: Optional[bool]=False
    ) -> None:
        
        if phase_type not in ("gt_phase", "gan_phase", "diff_phase", "rand_phase", "zero_phase", "gaus_phase"):
            raise ValueError(f"Type can be one of the following: gt_phase, diff_phase, rand_phase. Given type: {phase_type}")
        # load patient-scan list 
        if isinstance(patient_dict, Path):
            self.patient_dict = pickle.load(open(patient_dict, 'rb'))
        else:
            self.patient_dict = patient_dict
        self.patients = list(self.patient_dict.keys())
        self.slice_list = []
        # create slice list
        for patient_id in self.patients:
            scan_name = self.patient_dict[patient_id]
            self.slice_list += (source_path / scan_name).glob("gt_mag/*.npy")
        # other parameters
        self.phase_type = phase_type
        self.transforms  = transforms 
        self.msk_fnc = msk_fnc
        self.normalize_input = normalize_input

    def random_scan(self, random_seed=None):
        if random_seed != None:
            random.seed(random_seed+777)
        patient_id = random.choice(self.patients)
        scan_name = self.patient_dict[patient_id]
        # to display the images in sorted order
        indices = []
        paths = []
        for idx, path in enumerate(self.slice_list):
            if scan_name in str(path):
                indices.append(idx)
                paths.append(path)
        def return_indices(input_list):
            return sorted(range(len(input_list)), key=lambda x: int(str(input_list[x]).split("slice")[-1].split(".npy")[0]))
        sorted_indices = return_indices(paths)
        new_indices = [indices[idx] for idx in sorted_indices]
        volume = DataLoader(Subset(self, new_indices), batch_size=len(indices), shuffle=False)
        return next(iter(volume)), scan_name
    
    def __getitem__(self, index) -> Any:
        # load files
        file = self.slice_list[index]
        mag_img = torch.from_numpy(np.load(file))
        if self.phase_type in ("gt_phase", "diff_phase", "gan_phase"): 
            phase_img = torch.from_numpy(np.load(str(file).replace("gt_mag", self.phase_type)))
        elif self.phase_type == "rand_phase":
            phase_img = torch.rand_like(mag_img) * 2 * torch.pi 
        elif self.phase_type == "gaus_phase":
            phase_img = torch.randn_like(mag_img)
        elif self.phase_type == "zero_phase":
            phase_img = torch.zeros_like(mag_img)
        # augmentation 
        if self.transforms is not None:
            mag_img, phase_img = self.transforms((mag_img, phase_img)) 
        normalizer = lambda x : (x - x.min()) / (x.max() - x.min())
        mag_img = normalizer(mag_img)
        # normalize phase if needed
        if self.phase_type not in ("rand_phase", "zero_phase", "gaus_phase") and self.normalize_input: 
            phase_img = normalizer(phase_img) * 2 * torch.pi

        # scale images 
        mag_img = mag_img.unsqueeze(dim=0)
        phase_img = phase_img.unsqueeze(dim=0)
        real_part = mag_img * torch.cos(phase_img)
        imag_part = mag_img * torch.sin(phase_img)
        cimg = torch.stack([real_part, imag_part], dim=-1)
        kspace = fftc.fft2c_new(cimg)
        masked_kspace, mask, num_low_frequencies = self._apply_mask(kspace, self.msk_fnc, offset=0)
        return masked_kspace, mask.bool(), num_low_frequencies, mag_img
    
    def _apply_mask(
        self,
        data,
        mask_func,
        offset=None,
        seed=None,
        padding=None,
    ):
        shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
        mask, num_low_frequencies = mask_func(shape, offset, seed)
        if padding is not None:
            mask[..., : padding[0], :] = 0
            mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros
        
        masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

        return masked_data, mask, num_low_frequencies

    def __len__(self):
        return len(self.slice_list)

class CardiacSliceDataset(CardiacDataset):
    """
    A Pytorch Dataset that provides access to Cardiac MR image slices 
    """

    def __init__(
        self,
        root: Optional[Union[str, Path, os.PathLike]] = None, 
        data_dict: Optional[Dict[str, str]] = None,
        msk_fnc: Optional[Callable] = None,
        phase_type: str = "ground_truth",
        transforms: tvt.Compose = None,
        use_dataset_cache: bool = False,
        dataset_cache_file = Union[str, Path, os.PathLike],
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
    ) -> None:
        
        super().__init__(data_path=root, data_dict=data_dict, phase_type=phase_type)
        self.root = root
        self.data_dict = data_dict
        self.msk_fnc = msk_fnc
        self.phase_type = phase_type
        self.transforms = transforms


    def split(self, train_idxs: torch.Tensor, test_idxs: torch.Tensor, seed: int = 42):

        assert len(train_idxs) + len(test_idxs) == len(self.patient_list), f"The values for set 1 and set 2 are not sum up to the total number of scans, which is {len(self.patient_list)}. Your input is scan 1: {len(train_idxs)}, scan 2: {len(test_idxs)}"

        #random.seed(seed)
        #random.shuffle(self.patient_list)
        # Get the scan names corresponding to the indices
        test_dict = {
                    "magnitude":
                        {
                        "scan" : [],
                        "path" : []
                        },
                    "phase":
                        {
                        "scan" : [],
                        "path" : []
                        },
                    }

        train_dict = copy.deepcopy(test_dict)
        for idx, patient_path in enumerate(self.patient_list):
            # save data to train or test set 
            if idx in train_idxs:
                data_dict = train_dict
            else:
                data_dict = test_dict
            # start walking through the files 
            scan_list = os.listdir(patient_path)
            for sname in scan_list:
                scan_path = os.path.join(patient_path, sname)
                slice_list = os.listdir(scan_path)
                for slname in slice_list:
                    if "magnitude" in slname:
                        scan_path_list = data_dict["magnitude"]["scan"]
                        slice_path_list = data_dict["magnitude"]["path"]
                    elif "phase" in slname and "synthetic" not in slname:
                        scan_path_list = data_dict["phase"]["scan"]
                        slice_path_list = data_dict["phase"]["path"]
                    else:
                        continue
                    scan_path_list.append(scan_path)
                    sl_path = os.path.join(scan_path, slname)
                    slice_list = os.listdir(sl_path)
                    for fname in slice_list:
                        slice_path = os.path.join(sl_path, fname)
                        slice_path_list.append(slice_path)
        return (CardiacSliceDataset(self.root, train_dict, self.msk_fnc, self.phase_type),
                CardiacSliceDataset(self.root, test_dict, self.msk_fnc, "ground_truth"))

    def _get_kspace(self, mag_img: torch.tensor, phase_img: torch.tensor) -> torch.tensor:
        # get the complex image 
        complex_img = mag_img * torch.exp(1j * phase_img)
        real = torch.real(complex_img)
        imag = torch.imag(complex_img)
        complex_img = torch.concat([real, imag], dim=2)
        kspace = fftc.fft2c_new(complex_img)
        return kspace

    def _scale(self, img: torch.Tensor(), scale_type: str="mag") -> torch.Tensor():
        img = img.unsqueeze(dim=2)
        if scale_type == "mag":
            img -= img.min()
            img /= img.max()
        elif scale_type == "phase":
            img -= img.min()
            img = img / img.max() * torch.pi - torch.pi/2 
        else:
            raise ValueError(f"Invalid type. What is {scale_type}?")
        return img

    def _apply_mask(
        self,
        data,
        mask_func,
        offset=None,
        seed=None,
        padding=None,
    ):
        shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
        mask, num_low_frequencies = mask_func(shape, offset, seed)
        if padding is not None:
            mask[..., : padding[0], :] = 0
            mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros
        
        masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

        return masked_data, mask, num_low_frequencies

    def __getitem__(self, idx : int) -> Tuple[torch.tensor]:
        #print("Scan path:", self.path_list_m[idx])
        mag_img = torch.from_numpy(np.load(self.path_list_m[idx]).astype(np.float32))
        phase_img = torch.from_numpy(np.load(self.path_list_p[idx]).astype(np.float32))

        # get the relevant phase
        if self.phase_type == "ground_truth":
            syn_phase_img = phase_img
        elif self.phase_type == "diffusion":
            synthetic_path = self.path_list_p[idx].replace("phase", "synthetic_phase")
            syn_phase_img = torch.from_numpy(np.load(synthetic_path))
        elif self.phase_type == "diffusion_3D":
            synthetic_path = self.path_list_p[idx].replace("phase", "synthetic_phase_3D")
            syn_phase_img = torch.from_numpy(np.load(synthetic_path))
        elif self.phase_type == "gan":
            synthetic_path = self.path_list_p[idx].replace("phase", "synthetic_gan_phase")
            syn_phase_img = torch.from_numpy(np.load(synthetic_path))
        elif self.phase_type == "gan_3D":
            synthetic_path = self.path_list_p[idx].replace("phase", "synthetic_gan3D_phase")
            syn_phase_img = torch.from_numpy(np.load(synthetic_path))
        elif self.phase_type == "zero":
            syn_phase_img = torch.zeros_like(mag_img) 
        elif self.phase_type == "random":
            #syn_phase_img = torch.randn_like(mag_img)
            syn_phase_img = torch.rand_like(mag_img)
        else:
            raise ValueError(f"Invalid phase type. What is {self.phase_type}?")
        # augmentations & transformations
        if self.transforms is not None:
            mag_img, syn_phase_img = self.transforms((mag_img[None,...], syn_phase_img[None,...]))
            mag_img, syn_phase_img = mag_img.squeeze(dim=0), syn_phase_img.squeeze(dim=0)
        # scale images 
        mag_img = self._scale(mag_img, scale_type="mag")
        #phase_img = self._scale(phase_img, scale_type="phase")
        syn_phase_img = syn_phase_img.unsqueeze(dim=2) if self.phase_type == "zero" else self._scale(syn_phase_img, scale_type="phase")
        # do kspace transformation 
        kspace = self._get_kspace(mag_img, syn_phase_img)
        masked_kspace, mask, num_low_frequencies = self._apply_mask(kspace, self.msk_fnc, seed=88) # comment seed out in train
        #np.save("mask", mask.detach().cpu().numpy())
        #print(masked_kspace.shape)
        #np.save("masked_kspace", masked_kspace.detach().cpu().numpy())
        #np.save("kspace", kspace.detach().cpu().numpy())

        #import sys 
        #sys.exit()
        mag_img = mag_img.permute((2, 0, 1))
        return masked_kspace.unsqueeze(0), mask.bool(), num_low_frequencies, mag_img

class SliceDataset(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        phase_type: str = "ground_truth"
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)
        self.phase_type = phase_type

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)
    
    def _modify_kspace(self, kspace: torch.Tensor, phase_img: torch.Tensor):
        kspace = torch.view_as_real(kspace)
        complex_img = fftc.ifft2c_new(kspace)
        real = fastmri.math.complex_abs(complex_img)
        new_img = real * torch.exp(1j * phase_img)
        new_img = torch.view_as_real(new_img)
        new_kspace = fftc.fft2c_new(new_img)
        return torch.view_as_complex(new_kspace)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            if self.phase_type == "zero":
                kspace = torch.from_numpy(kspace)
                zero_phase = torch.zeros_like(kspace, dtype=torch.float32)
                kspace = self._modify_kspace(kspace, zero_phase).numpy()

            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)
        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)

        return sample


# test the code 
if __name__ == "__main__":

    # I just want to see whether we can train VarNet with Knee data 
    mask_type = "equispaced_fraction"
    center_fractions = [0.1]
    accelerations = [4] 


    mask = create_mask_for_mask_type(
        mask_type, center_fractions, accelerations
    )

    import pickle 
    
    data_dict = pickle.load(open(Path("datadict.pkl"), 'rb'))

    train_transform = tvt.Compose([
        AffineTransform(
            angle=[-45, 45],
            scale=[0.8, 1.2],
            shear=[-20, 20]
        ),
        CenterCrop(128)
    ])
    print(data_dict)
    print("Good job!")


