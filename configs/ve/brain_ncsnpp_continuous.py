

import ml_collections
from configs.brainmri_default_configs import get_default_configs



def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.init_epoch = 1

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data
  data = config.data
  #data.dataset = 'fastmri_knee'
  data.is_multi = False
  data.is_complex = False
  #data.magpha = True
  #data.root = '/media/harry/tomo/fastmri'
  data.image_size = 320

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  # checkpoints 
  config.checkpts = checkpoints = ml_collections.ConfigDict()
  checkpoints.name = "checkpoint_150.pth" 
  checkpoints.load = False 

  return config
