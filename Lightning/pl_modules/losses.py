
 
from torchmetrics import Metric, MeanSquaredError
from torch import Tensor
import torch.nn.functional as F 
import torch

__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.

    Returns
    -------
    float
        Mean Squared Error.

    Examples
    --------
    >>> from mridc.collections.common.metrics.reconstruction_metrics import mse
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> mse(datax, datay)
    0.17035991151556373
    """
    return np.mean((x - y) ** 2)  # type: ignore

def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(mse(x, y))


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Normalized Mean Squared Error (NMSE).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.

    Returns
    -------
    float
        Normalized Mean Squared Error.

    Examples
    --------
    >>> from mridc.collections.common.metrics.reconstruction_metrics import nmse
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> nmse(datax, datay)
    0.5001060028222054
    """
    return np.linalg.norm(x - y) ** 2 / np.linalg.norm(x) ** 2

def nrmse(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(nmse(x, y))


def psnr(x: np.ndarray, y: np.ndarray, maxval: np.ndarray = None) -> float:
    """
    Compute Peak Signal to Noise Ratio (PSNR).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.
    maxval : np.ndarray
        Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
        should be 1.

    Returns
    -------
    float
        Peak Signal to Noise Ratio.

    Examples
    --------
    >>> from mridc.collections.reconstruction.metrics.reconstruction_metrics import psnr
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> psnr(datax, datay)
    7.6700572264458

    .. note::
        x and y must be normalized to the same range, e.g. [0, 1].

        The PSNR is computed using the scikit-image implementation of the PSNR metric.
        Source: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio
    """
    if maxval is None:
        maxval = np.max(x)
    return peak_signal_noise_ratio(x, y, data_range=maxval)


def ssim(x: np.ndarray, y: np.ndarray, data_range: np.ndarray = None) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.
    maxval : np.ndarray
        Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
        should be 1.

    Returns
    -------
    float
        Structural Similarity Index Measure.

    Examples
    --------
    >>> from mridc.collections.common.metrics.reconstruction_metrics import ssim
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = datax * 0.5
    >>> ssim(datax, datay)
    0.01833040155119426

    .. note::
        x and y must be normalized to the same range, e.g. [0, 1].

        The SSIM is computed using the scikit-image implementation of the SSIM metric.
        Source: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity
    """

    if x.ndim != 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if x.ndim != y.ndim:
        raise ValueError("Ground truth dimensions does not match prediction dimensions.")

    maxval = np.max(x) if maxval is None else maxval
    ssim_score = sum(
        structural_similarity(x[slice_num], y[slice_num], data_range=maxval) for slice_num in range(x.shape[0])
    )
    return ssim_score / x.shape[0]


METRIC_FUNCS = {"MSE": mse, "NMSE": nmse, "PSNR": psnr, "SSIM": ssim}

class NMSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.base_metric = MeanSquaredError()


    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape == target.shape, "shape of the inputs should be the same!"
        batch_size, num_channels, height, width = preds.shape
        # normalize images 
        preds = preds.view(batch_size, -1)
        target = target.view(batch_size, -1)
        preds = F.normalize(preds, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        preds = preds.view(batch_size, num_channels, height, width)
        target = target.view(batch_size, num_channels, height, width)
        # calculate normalized MSE 
        self.base_metric.update(preds, target)
    
    def compute(self):
        return self.base_metric.compute()