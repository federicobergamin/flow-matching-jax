import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
#   config.training = training = ml_collections.ConfigDict()
#   config.training.batch_size = 256
#   training.n_epochs = 100

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.image_size = 28
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
#   model.ema_rate = 0.9999
  model.normalization = "GroupNorm"
  model.nonlinearity = "swish"
  model.nf = 32
  model.ch_mult = (1, 2)
  model.num_res_blocks = 3
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.dropout = False

  config.data = data = ml_collections.ConfigDict()
  data.centered = False

  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
#   optim.weight_decay = 0
#   optim.optimizer = 'Adam'
#   optim.lr = 2e-4
#   optim.beta1 = 0.9
#   optim.eps = 1e-8
#   optim.warmup = 5000
#   optim.grad_clip = 1.

#   config.seed = 42

  return config