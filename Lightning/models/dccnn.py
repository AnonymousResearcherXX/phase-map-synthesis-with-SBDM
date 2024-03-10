
import torch 
from torch import nn 
from typing import Optional, Tuple
import fastmri 

# Part of this code was taken from https://github.com/wdika/mridc/tree/main


class Conv2d(nn.Module):
    """
    Implementation of a simple cascade of 2D convolutions.
    If batchnorm is set to True, batch normalization layer is applied after each convolution.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, n_convs=3, activation=nn.PReLU(), batchnorm=False):
        """
        Inits Conv2d.

        Parameters
        ----------
        in_channels: Number of input channels.
            int
        out_channels: Number of output channels.
            int
        hidden_channels: Number of hidden channels.
            int
        n_convs: Number of convolutional layers.
            int
        activation: Activation function.
            torch.nn.Module
        batchnorm: If True a batch normalization layer is applied after every convolution.
            bool
        """
        super().__init__()

        self.conv = []
        for idx in range(n_convs):
            self.conv.append(
                nn.Conv2d(
                    in_channels if idx == 0 else hidden_channels,
                    hidden_channels if idx != n_convs - 1 else out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if batchnorm:
                self.conv.append(nn.BatchNorm2d(hidden_channels if idx != n_convs - 1 else out_channels, eps=1e-4))
            if idx != n_convs - 1:
                self.conv.append(activation)
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        """
        Performs the forward pass of Conv2d.

        Parameters
        ----------
        x: Input tensor.

        Returns
        -------
        Convoluted output.
        """
        if x.dim() == 5:
            x = x.squeeze(1)
            if x.shape[-1] == 2:
                x = x.permute(0, 3, 1, 2)
        return self.conv(x)
    
class CascadeNetBlock(torch.nn.Module):
    """
    Model block for CascadeNet & Convolution Recurrent Neural Network.

    This model applies a combination of soft data consistency with the input model as a regularizer.
    A series of these blocks can be stacked to form the full variational network.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        no_dc: bool = False,
    ):
        """
        Initializes the model block.

        Parameters
        ----------
        model: Model to apply soft data consistency.
            torch.nn.Module
        no_dc: Flag to disable the soft data consistency.
            bool
        """
        super().__init__()

        self.model = model
        self.no_dc = no_dc
        self.dc_weight = torch.nn.Parameter(torch.ones(1))

    def forward(
        self,
        pred: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(pred)
        #soft_dc = torch.where(mask.bool(), pred - ref_kspace, zero) * self.dc_weight
        pred = fastmri.ifft2c(pred)
        eta = self.model(pred.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        eta = fastmri.fft2c(eta)
        if not self.no_dc:
            #eta = pred - soft_dc.squeeze(dim=0) - eta
            eta = (1 - mask.bool()) * eta + mask.bool() * pred

        return eta
    
class CascadeNet(nn.Module):
    """
    Implementation of the Deep Cascade of Convolutional Neural Networks, as presented in Schlemper, J., \
    Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D.

    References
    ----------

    ..

        Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D., A Deep Cascade of Convolutional \
        Neural Networks for MR Image Reconstruction. Information Processing in Medical Imaging (IPMI), 2017. \
        Available at: https://arxiv.org/pdf/1703.00555.pdf

    """

    def __init__(
            self,
            hidden_channels: int, 
            n_convs: int, 
            batchnorm: bool,
            no_dc: bool,
            num_cascades: int, 
        ):
        # init superclass
        super().__init__()
        # Cascades of CascadeCNN blocks
        self.cascades = torch.nn.ModuleList(
            [
                CascadeNetBlock(
                    Conv2d(
                        in_channels=2,
                        out_channels=2,
                        hidden_channels=hidden_channels,
                        n_convs=n_convs,
                        batchnorm=batchnorm,
                    ),
                    no_dc=no_dc,
                )
                for _ in range(num_cascades)
            ]
        )

        self.accumulate_estimates = False
        #self.dc_weight = torch.nn.Parameter(torch.ones(1))

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        y: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        target: Target data to compute the loss.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        pred: list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or  torch.Tensor, shape [batch_size, n_x, n_y, 2]
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        y = y.squeeze(dim=1)
        pred = y.clone()
        for cascade in self.cascades:
            pred = cascade(pred, y, mask)
        #_, pred = utils.center_crop_to_smallest(target, pred)
        #return fastmri.complex_abs(fastmri.ifft2c(pred)).unsqueeze(dim=1)
        return fastmri.ifft2c(pred)
    
# test code 
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CascadeNet(
        hidden_channels=64, 
        n_convs=5, 
        batchnorm=False, 
        no_dc=False, 
        num_cascades=10
    ).to(device)

    random_tensor = torch.rand((16, 320, 320, 2)).to(device)
    mask = torch.zeros((1, 1, 320, 320, 1)).to(device)
    print("Model input data shape:", random_tensor.shape)
    output = model(random_tensor, mask)
    print("Output shape:", output.shape)
    print("Good job!")


    