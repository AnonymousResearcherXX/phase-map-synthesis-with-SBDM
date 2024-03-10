import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 12 # 16 
  # training.n_iters = 2400001
  training.epochs = 100 # 100
  training.snapshot_freq = 50000
  training.log_freq = 100
  #training.log_freq = 25
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075

  # data
  config.data = data = ml_collections.ConfigDict()
  data.type = "fastmri_knee"
  data.image_size = 384 # this was 320
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 2

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 1000 # this was 2000 (make this 1000)
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-4 # this was 2e-4 
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  # position encoding 
  config.pos_emb = False # positional encoding (like time conditioning)
  config.pos_emb2 = False # positional encoding vol. 2 (concatenating position encoding to input)
  #config.demb = 128

  config.seed = 101
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  return config