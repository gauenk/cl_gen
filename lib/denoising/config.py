"""
Configurations for denoising experiments

"""

# python imports
import argparse,json,uuid,yaml,io
from easydict import EasyDict as edict
from pathlib import Path

# only THIS project import is allows; all others banned.
import settings


def get_args():
    parser = argparse.ArgumentParser(description="Run a Denoising Experiment")
    parser.add_argument("--yaml", type=str, default=None,
                        help="Set all arguments using yaml file.")
    parser.add_argument("--name", type=str, default=None,
                        help="experiment name")
    jloads = json.loads
    msg = (
        "The experiment name to use for loading previous results. ",
        " The variable 'name' is used if this value is None.")
    parser.add_argument("--load-name", type=int, default=-1,
                        help=msg)
    msg = ("when running from an old experiment, ",
           "do we create a new experiment file?")
    parser.add_argument("--new",  action='store_true', help=msg)
    parser.add_argument("--epochs", type=int, default=200,
                        help="how many epochs do we train for?")
    parser.add_argument("--epoch-num", type=int, default=-1,
                        help="resume training from epoch-num")
    parser.add_argument("--mode", type=str, default="train",
                        help="train or test models")
    parser.add_argument("--N", type=int, default=2,
                        help="""number of noisy images to generate 
                        from each original image""")
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="experiment's dataset")
    parser.add_argument("--batch-size", type=int, default=1536,
                        help="batch-size for each item in world-size")
    parser.add_argument("--world_size", type=int, default=2,
                        help="number of training gpus")
    parser.add_argument("--gpuid", type=int, default=0,
                        help="if using one gpu, which gpu?")
    parser.add_argument("--init-lr", type=float, default=1e-3,
                        help="The initial learning rate for experiments")
    msg = "How does the learning rate scale with batch-size? "
    msg += "linear, sqrt, noner"
    parser.add_argument("--lr_bs_scale", type=str, default='none',
                        help=msg)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="How many workers per dataset?")
    parser.add_argument("--enc-size", type=int, default=768,
                        help="Dimension of encoder output.")
    parser.add_argument("--proj-size", type=int, default=64,
                        help="Dimension of projection output.")
    msg = """\
    which type of loss function do we want to use for the reconstructed images?
        "l2": the L_2 loss
        "simclr": the contrastive learning loss function
    """
    parser.add_argument("--img-loss-type", type=str, default='l2',
                        help=msg)

    msg = """\
    determines the type of aggregation method we use in experiment
        'id': this is the same as no aggregation
        'mean': this uses the arithmetic mean to aggregate results
        'graphconv': use graph convolution to aggregate information"""
    parser.add_argument("--agg-enc-fxn", type=str,default="mean",
                        help=msg)

    msg = """\
    how much aggregation do we do?
        'h': aggregate only the final output encoding
        'skip': aggregate only the skip connections
        'full': aggregate all activations
    """
    parser.add_argument("--agg-enc-type", type=str,default="full",
                        help=msg)

    msg = "determine if we use projection opt. before nt_xent"
    parser.add_argument("--proj-enc", type=bool, default=True,
                        help=msg)

    msg = """parameter settings for hyperparameters
        'h' (float): hyperparameter weight for nt_xent loss over enc repr.
    """
    parser.add_argument("--hyper-params", type=jloads, default='{"h":0.0}',
                        help=msg)

    msg = """what type of noise do we run experiments with?
        'g': Gaussian Noise
        'll': Low-light Noise
        'msg': Mutli-Scale Gaussian Noise
    """
    parser.add_argument("--noise-type", type=str, default='g', help=msg)

    msg = """parameters for noise generation
        'g': mean (float), stddev (float)
        'll': alpha (float)
        'msg': each_image (bool), stddev_range (tuple)
    """
    defaults = '{"g":{"mean":0.0,"stddev":10},\
    "ll":{"alpha":0.5},\
    "msg":{"each_image":0,"stddev_rand":[0,50]}\
    }'
    parser.add_argument("--noise-params", type=jloads,
                        default=defaults, help=msg)

    msg = """what type of optimizer? adam, sgd,
        'adam': Adam
        'sgd': SGD
        'lars': LARS
        'sched': pick Adam or SGD using scheduler choice
    """
    parser.add_argument("--optim-type", type=str,
                        default='adam', help=msg)

    msg = """parameters for optimizer
        'adam': betas (tuple[float]), eps (float), 
                weight_decay (float), amsgrad (bool)
        'sgd': momentum (float),  dampening (float), 
               nesterov (bool), weight_decay (float)
        'lars': momentum (float), weight_decay (float), 
                eta (float)
    """
    defaults = '{\
    "adam":{"betas":[0.9,0.999],"eps":1e-08,"weight_decay":0.0,"amsgrad":0},\
    "sgd":{"momentum":0.9,"weight_decay":0.0,"dampening":0.0,"nesterov":0},\
    "lars":{"momentum":0.9,"weight_decay":0.0,"eta":1e-3}\
    }'
    parser.add_argument("--optim-params", type=jloads,
                        default=defaults, help=msg)

    msg = """what type of scheduler?
        'lwca': linear warmup with cosine annealing
        'ms': mutlistep learning rate
        'none': no scheduler at all
    """
    parser.add_argument("--sched-type", type=str,
                        default="none", help=msg)

    msg = """parameters for scheduler
        'lwca': burnin (int)
        'ms': milestones (list[int])
    """
    defaults = '{\
    "lwca":{"burnin":10},\
    "ms":{"milestones":[150,400]}\
    }'
    parser.add_argument("--sched-params", type=jloads,
                        default=defaults, help=msg)

    parser.add_argument("--freeze-enc", action='store_true',
                        help="Freeze the weights for the encoder.")
    parser.add_argument("--freeze-dec", action='store_true',
                        help="Freeze the weights for the decoder.")
    parser.add_argument("--freeze-proj", action='store_true',
                        help="Freeze the weights for the projector.")
    args = parser.parse_args()
    return args

def set_cfg(args):
    """
    Expected argument fields:

    name (str): name of experiment, usually a uuid
    epoch_num (int): if loading from previous results, which model do we load from?
    dataset (str): determine experiment's dataset
    noise_level (float): the amount of noise, to be replaced by "noise_params"
    N (int): the number of noise images used in the forward pass
    img_loss_type (str): which type of loss function do we want to use for the reconstructed images?
        "l2": the L_2 loss
        "simclr": the contrastive learning loss function

    share_enc (bool): to be renamed "agg-enc"
    hyper_h (float): to be renamed "hyper_params", type (json)
    init_lr (float): initial learning rate
    lr_policy (str): to be renamed "sched_type"
    lr_params (json.loads): to be renamed "sched_params"
    batch_size (int): global batch size, split amongst world_size

    TODO:

    load_name (str): the experiment name to use for loading previous results. The "name" is used if this value is None.
    
    agg-enc-fxn (str): determines the type of aggregation method we use in experiment
        'id': this is the same as no aggregation
        'mean': this uses the arithmetic mean to aggregate results
        'graphconv': use graph convolution to aggregate information

    agg-enc-type (str): how much aggregation do we do?
        'h': aggregate only the final output encoding
        'skip': aggregate only the skip connections
        'full': aggregate all activations

    proj-h (bool): determine if we use projection opt. before nt_xent

    hyper-params (json.loads): parameter settings for hyperparameters
        'h' (float): hyperparameter weight for nt_xent loss over enc repr.

    noise-type (str): what type of noise do we run experiments with?
        'g': Gaussian Noise
        'll': Low-light Noise
        'msg': Mutli-Scale Gaussian Noise

    noise-params (json.loads): parameters for noise generation
        'g': mean (float), stddev (float)
        'll': alpha (float)
        'msg': each_image (bool), stddev_range (tuple)

    world-size (int): number of gpus

    optim-type (str): what type of optimizer? adam, sgd,
        'adam': Adam
        'sgd': SGD
        'lars': LARS
    
    optim-params (json.loads): parameters for optimizer
        'adam': gamma (float), beta1 (float), beta2 (float)
        'sgd': momentum (float), weight_decay (float), dampening (float),
               nesterov (bool), 
        'lars': momentum (float), weight_decay (float), eta (float)

    sched-type (str): what type of scheduler? 
        'lwca': linear warmup with cosine annealing
        'ms': mutlistep learning rate
        'none': no scheduler at all
           
    sched-params (json.loads): parameters for scheduler
        'lwca': burnin (int)
        'ms': milestones (list[int])

    freeze_enc (bool):
    freeze_dec (bool):
    freeze_proj (bool):

    """

    cfg = edict()

    cfg.device = 'cuda:%d' % args.gpuid
    cfg.use_ddp = True

    cfg.exp_name = args.name
    if cfg.exp_name is None:
        cfg.exp_name = str(uuid.uuid4())
    cfg.epochs = args.epochs
    cfg.load_name = args.load_name
    cfg.load = args.epoch_num > 0
    cfg.epoch_num = args.epoch_num
    cfg.mode = args.mode
    cfg.N = args.N

    cfg.dataset = edict()
    cfg.dataset.root = f"{settings.ROOT_PATH}/data/"
    cfg.dataset.n_classes = 10
    cfg.dataset.name = args.dataset

    cfg.batch_size = args.batch_size
    cfg.world_size = args.world_size
    cfg.init_lr = args.init_lr
    cfg.lr_bs_scale = args.lr_bs_scale
    cfg.img_loss_type = args.img_loss_type
    cfg.agg_enc_fxn = args.agg_enc_fxn
    cfg.agg_enc_type = args.agg_enc_type
    cfg.proj_enc = args.proj_enc
    cfg.hyper_params = args.hyper_params
    cfg.noise_type = args.noise_type
    cfg.noise_params = args.noise_params
    cfg.optim_type = args.optim_type
    cfg.optim_params = args.optim_params
    cfg.sched_type = args.sched_type
    cfg.sched_params = args.sched_params
    cfg.num_workers = args.num_workers
    cfg.enc_size = args.enc_size
    cfg.proj_size = args.proj_size

    cfg.sync_batchnorm = True # args.sync_batchnorm
    cfg.use_apex = False # args.use_apex
    cfg.test_with_psnr = True
    cfg.use_bn = True

    # todo: find and replace with
    # share_enc -> agg_enc_type + agg_enc_type
    # noise_level -> noisy_type + noise_params
    # hyper_h -> hyper_params
    # lr.policy -> sched_type 
    # lr.params -> sched_params
    # 

    dsname = cfg.dataset.name.lower()
    model_path = Path(f"{settings.ROOT_PATH}/output/denoise/{dsname}/{cfg.exp_name}/model/")
    optim_path = Path(f"{settings.ROOT_PATH}/output/denoise/{dsname}/{cfg.exp_name}/optim/")
    if not model_path.exists(): model_path.mkdir(parents=True)
    cfg.model_path = model_path
    cfg.optim_path = optim_path
    

    # logging and recording
    cfg.global_step = 0
    cfg.current_epoch = 0
    cfg.checkpoint_interval = 1
    cfg.test_interval = 5
    cfg.log_interval = 5
    
    # saving
    cfg.freeze_models = edict()
    cfg.freeze_models.encoder = args.freeze_enc
    cfg.freeze_models.decoder = args.freeze_dec
    cfg.freeze_models.projector = args.freeze_proj

    if cfg.dataset.name.lower() == "mnist":
        cfg.n_img_channels = 1
    else:
        cfg.n_img_channels = 3

    # include only for backward compatibility with datasets
    cfg.use_collate = True
    cfg.rank = -1 # only used in datasets!

    return cfg
    
#
# setting cfg from arguments or yaml file
#

def get_cfg(args):
    if args.yaml:
        return get_cfg_from_yaml(args.yaml)
    else:
        return get_cfg_from_args(args)

def get_cfg_from_yaml(yml_fn):
    fields = load_yaml_fields(yml_fn)
    return set_cfg(fields)

def get_cfg_from_args(args):
    return set_cfg(args)

#
# file io
#

def get_cfg_yaml_filename(exp_name):
    root_path = Path(settings.ROOT_PATH)
    cfg_path = root_path / Path("cfgs")
    exp_cfg_path = cfg_path / Path(exp_name + ".yml")
    return exp_cfg_path

def save_cfg(cfg,fn):
    exp_cfg_path = get_cfg_yaml_filename(cfg.exp_name)
    with io.open(exp_cfg_path,'w',encoding='utf8') as f:
        yaml.dump(cfg,f,default_flow_style=False,allow_unicode=True)
        
def load_cfg(exp_name):
    exp_cfg_path = get_cfg_yaml_filename(exp_name)
    if not exp_cfg_path.exists(): return None
    with open(exp_cfg_path,'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
