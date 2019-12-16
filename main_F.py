# -*- coding: utf-8 -*-
"""
基于 PyTorch resnet50 实现的图片分类代码
原代码地址：https://github.com/pytorch/examples/blob/master/imagenet/main.py
可以与原代码进行比较，查看需修改哪些代码才可以将其改造成可以在 ModelArts 上运行的代码
在ModelArts Notebook中的代码运行方法：
（0）准备数据
大赛发布的公开数据集是所有图片和标签txt都在一个目录中的格式
如果需要使用 torch.utils.data.DataLoader 来加载数据，则需要将数据的存储格式做如下改变：
1）划分训练集和验证集，分别存放为 train 和 val 目录；
2）train 和 val 目录下有按类别存放的子目录，子目录中都是同一个类的图片
prepare_data.py中的 split_train_val 函数就是实现如上功能，建议先在自己的机器上运行该函数，然后将处理好的数据上传到OBS
执行该函数的方法如下：
cd {prepare_data.py所在目录}
python prepare_data.py --input_dir '../datasets/train_data' --output_train_dir '../datasets/train_val/train' --output_val_dir '../datasets/train_val/val'

（1）从零训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --pretrained True --seed 0

（2）加载已有模型继续训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --seed 0 --resume '../model_snapshots/epoch_0_2.4.pth'

（3）评价单个pth文件
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --arch 'resnet50' --num_classes 54 --seed 0 --eval_pth '../model_snapshots/epoch_5_8.4.pth'
"""

# from Focal_Loss import focal_loss

from m_parser import *
from util import *
from config import *
from focalloss import *
from build_net import *
from utils.logger import *
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys


def main():

    # sys.stdout = open('output.log', mode='w', encoding='utf-8')

    args, unknown = parser.parse_known_args()
    args = prepare_data_on_modelarts(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = 1
    main_worker(args.gpu, ngpus_per_node, args) #here


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # model =  EfficientNet.from_pretrained('efficientnet-b0',num_classes=54)
    model = make_model(args)
    model = models.__dict__['resnext101_32x8d'](num_classes=54)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, args.num_classes)

    model = torch.nn.DataParallel(model).cuda()
    ##### loss
    print("loss is:",args.loss)
    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.loss == 'FL':
        criterion = FocalLoss(gamma=2)
    elif args.loss =='SM':
        criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1)
    elif args.loss =='CUT':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.loss == 'SM_CE':
        criterion =  nn.CrossEntropyLoss().cuda(args.gpu)

    #####  optim
    print("optim is:", args.optim)
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)


    # optionally resume from a checkpoint
    if args.resume:
        # if os.path.isfile(args.resume):
        if os.path.exists(args.resume) and (not os.path.isdir(args.resume)):
            if args.resume.startswith('s3://'):
                restore_model_name = args.resume.rsplit('/', 1)[1]
                shutil.copy(args.resume, '/cache/tmp/' + restore_model_name)
                args.resume = '/cache/tmp/' + restore_model_name
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if args.resume.startswith('/cache/tmp/'):
                os.remove(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        title = 'ImageNet-' + args.arch
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        logger = Logger(os.path.join(args.train_url, now + 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data_local, 'train')
    valdir = os.path.join(args.data_local, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.4)
    lighting = Lighting(alphastd=0.1,
                              eigval=[0.2175, 0.0188, 0.0045],
                              eigvec=[[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]])
    size = 224
    if args.loss =='CUT':
        size = 224
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                # Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
                # transforms.CenterCrop(size),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # RandomRotate(15, 0.3),
                # RandomGaussianBlur(),
                # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))
    else:
        print("use rotate")
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
                transforms.CenterCrop(size),
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                RandomRotate(15, 0.3),
                transforms.ToTensor(),
                normalize,
            ]))
    # ImageFolder类会将traindir目录下的每个子目录名映射为一个label id，然后将该id作为模型训练时的标签
    # 比如，traindir目录下的子目录名分别是0~53，ImageFolder类将这些目录名当做class_name，再做一次class_to_idx的映射
    # 最终得到这样的class_to_idx：{"0": 0, "1":1, "10":2, "11":3, ..., "19": 11, "2": 12, ...}
    # 其中key是class_name，value是idx，idx就是模型训练时的标签
    # 因此我们在保存训练模型时，需要保存这种idx与class_name的映射关系，以便在做模型推理时，能根据推理结果idx得到正确的class_name
    idx_to_class = OrderedDict()
    for key, value in train_dataset.class_to_idx.items():
        idx_to_class[value] = key

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    size = 224
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([

            # Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
            # transforms.CenterCrop(size),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc, train_5 = train(train_loader, model, criterion, optimizer, epoch, args)
        test_loss, test_acc, test_5 = test(val_loader, model, criterion, epoch, True)
        scheduler.step(test_loss)
        for param_group in optimizer.param_groups:
            print("#############",param_group['lr'])
        logger.append([param_group['lr'], train_loss, test_loss, train_acc, test_acc])

        # evaluate on validation set
        if (epoch + 1) % 1 == 0:
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = False
            best_acc1 = max(acc1.item(), best_acc1)
            pth_file_name = os.path.join(args.train_local, 'epoch_%s_%s.pth'
                                         % (str(epoch + 1), str(round(acc1.item(), 3))))
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'idx_to_class': idx_to_class
                }, is_best, pth_file_name, args)
    logger.close()
    # logger.plot()
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    savefig(os.path.join(args.train_url,now+ 'log.eps'))
    if args.epochs >= args.print_freq:
        save_best_checkpoint(best_acc1, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    # learn_rate = AverageMeter('learning_rate',' :.4f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    output_list = []
    target_list = []
    for i, (images, target) in enumerate(train_loader):
        # print()
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        r = np.random.rand(1)
        if args.loss == 'CUT':
            if args.beta > 0 and r < args.cutmix_prob:
                input = images.cuda()
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                # compute output
                input_var = torch.autograd.Variable(input, requires_grad=True)
                target_a_var = torch.autograd.Variable(target_a)
                target_b_var = torch.autograd.Variable(target_b)
                output = model(input_var)
                loss = criterion(output, target_a_var) * lam + criterion(output, target_b_var) * (1. - lam)
        else:
            # compute output
            output = model(images)
            # criterion = focal_loss(num_classes=54,gamma=1).cuda(args.gpu)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_list, target_list,output, target, topk=(1, 5))
        # learn_rate.update(float(optimizer.param_groups['lr']),images.size(0))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            progress.display(i)
        # progress.display(i)
    return top1.avg, top1.avg, top5.avg



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    output_list = []
    target_list = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output_list,target_list,output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    if print_matrix:
    # confusion_matrix
        cm = confusion_matrix(target_list,output_list  )
        for i in range(54):
            for j in range(54):
                print(str(int(cm[i][j])).ljust(4),end= '')
            print()

    return top1.avg




if __name__ == '__main__':
    # print("!!!!!!")
    main()