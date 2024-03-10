
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


