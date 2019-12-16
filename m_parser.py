'''
1.backbone
2.optim
3.loss
4.epoch
5.batchsize
6.learning rate
7.decay epoch
'''
import argparse
import torchvision.models as models
from build_net import model_names
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', required=True,
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')


parser.add_argument('--m_backbone',default='efficientnet-b0',type=str,help="choose backbone")
parser.add_argument('--optim',default='SGD',type=str,help="choose backbone")
parser.add_argument('--loss',default='EC',type=str,help="")
parser.add_argument('--decay_epoch',default='30',type=int,help="")
# parser.add_argument('--backbone',default='efficientnet-b0',type=str,help="choose backbone")



parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
parser.add_argument('--eval_pth', default='', type=str,
                    help='the *.pth model path need to be evaluated on validation set')
parser.add_argument('--pretrained', default=False, type=bool,
                    help='use pre-trained model or not')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# These arguments are added for adapting ModelArts
parser.add_argument('--num_classes', required=True, type=int, help='the num of classes which your task should classify')
parser.add_argument('--local_data_root', default='cache/', type=str,
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
parser.add_argument('--test_data_url', default='', type=str, help='the test data path')
parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
parser.add_argument('--test_data_local', default='', type=str, help='the test data path on local')
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
parser.add_argument('--train_local', default='', type=str, help='the training output results on local')
parser.add_argument('--tmp', default='', type=str, help='a temporary path on local')
parser.add_argument('--deploy_script_path', default='', type=str,
                    help='a path which contain config.json and customize_service.py, '
                         'if it is set, these two scripts will be copied to {train_url}/model directory')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')

best_acc1 = 0
