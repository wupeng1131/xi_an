
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import re

from prepare_data import prepare_data_on_modelarts
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数



def save_checkpoint(state, is_best, filename, args):
    if not is_best:
        torch.save(state, filename)
        if args.train_url.startswith('s3'):
            shutil.copy(filename,
                          args.train_url + '/' + os.path.basename(filename))
            os.remove(filename)


def save_best_checkpoint(best_acc1, args):
    best_acc1_suffix = '%s.pth' % str(round(best_acc1, 3))
    pth_files = os.listdir(args.train_url)
    for pth_name in pth_files:
        if pth_name.endswith(best_acc1_suffix):
            break

    # mox.file可兼容处理本地路径和OBS路径
    if not os.path.exists(os.path.join(args.train_url, 'model')):
        os.mkdir(os.path.join(args.train_url, 'model'))

    shutil.copy(os.path.join(args.train_url, pth_name), os.path.join(args.train_url, 'model/model_best.pth'))
    shutil.copy(os.path.join(args.deploy_script_path, 'config.json'),
                  os.path.join(args.train_url, 'model/config.json'))
    shutil.copy(os.path.join(args.deploy_script_path, 'customize_service.py'),
                  os.path.join(args.train_url, 'model/customize_service.py'))


    shutil.copy(os.path.join('./efficientnet', '__init__.py'),
                  os.path.join(args.train_url, 'model/__init__.py'))
    shutil.copy(os.path.join('./efficientnet', 'model.py'),
                os.path.join(args.train_url, 'model/model.py'))
    shutil.copy(os.path.join('./efficientnet', 'utils.py'),
                os.path.join(args.train_url, 'model/utils.py'))
    shutil.copy('./build_net.py',
                os.path.join(args.train_url, 'model/build_net.py'))
    if os.path.exists(os.path.join(args.train_url,'model/models')):
        shutil.rmtree(os.path.join(args.train_url,'model/models'),True)
    shutil.copytree('./models', os.path.join(args.train_url,'model/models'))
    if os.path.exists(os.path.join(args.train_url, 'model/config.json')) and \
            os.path.exists(os.path.join(args.train_url, 'model/customize_service.py')):
        print('copy config.json and customize_service.py success')
    else:
        print('copy config.json and customize_service.py failed')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.94 ** (epoch // args.decay_epoch))
    print("learning rate is:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output_list,target_list,output, target, topk=(1,), ):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        media = target.view(1, -1)
        media = target.view(1, -1).expand_as(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for i in range(correct.size(1)):
            output_list.append(pred[0][i].item())
            target_list.append(media[0][i].item())

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def accuracy1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output_list = [], target_list= [], output, target, topk=(1, 5))
        prec1, prec5 = accuracy1(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg, top5.avg)



class LabelSmoothSoftmaxCEV1(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', lb_ignore=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img



class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img

class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img

def validateTitle(title):
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", title)  # 替换为下划线
    return new_title

def cv_imwrite(write_path, img):
    cv2.imencode('.jpg', img,)[1].tofile(write_path)
def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv_imwrite(filename, input_tensor)


def default_loader(path):
     return Image.open(path).convert('RGB')  # operation object is the PIL image object
import torch.utils.data as data
class myImageFloder(data.Dataset):  # Class inheritance，继承Ｄａｔａｓｅｔ类
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        # fh = open(label)
        c = 0
        imgs = []
        class_names = ['regression']
        for line in label:  # label is a list
            cls = line.split()  # cls is a list
            fn = cls.pop(0)
            if os.path.isfile(os.path.join(root, fn)):
                imgs.append((fn, tuple([float(v) for v in cls[:len(cls)-1]])))
                # access the last label
                # images is the list,and the content is the tuple, every image corresponds to a label
                # despite the label's dimension
                # we can use the append way to append the element for list
            c = c + 1
        print('the total image is',c)
        print(class_names)
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]  # even though the imgs is just a list, it can return the elements of it
        # in a proper way
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label), fn    #　在这里返回图像数据以及对应的ｌａｂｅｌ以及对应的名称

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes