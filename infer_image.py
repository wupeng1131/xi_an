from util import *
from efficientnet.model import *
from build_net import *

count = 0
label_id_name_dict = \
            {
                "0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
                "2": "工艺品/布贴绣",
                "3": "工艺品/景泰蓝",
                "4": "工艺品/木马勺脸谱",
                "5": "工艺品/柳编",
                "6": "工艺品/葡萄花鸟纹银香囊",
                "7": "工艺品/西安剪纸",
                "8": "工艺品/陕历博唐妞系列",
                "9": "景点/关中书院",
                "10": "景点/兵马俑",
                "11": "景点/南五台",
                "12": "景点/大兴善寺",
                "13": "景点/大观楼",
                "14": "景点/大雁塔",
                "15": "景点/小雁塔",
                "16": "景点/未央宫城墙遗址",
                "17": "景点/水陆庵壁塑",
                "18": "景点/汉长安城遗址",
                "19": "景点/西安城墙",
                "20": "景点/钟楼",
                "21": "景点/长安华严寺",
                "22": "景点/阿房宫遗址",
                "23": "民俗/唢呐",
                "24": "民俗/皮影",
                "25": "特产/临潼火晶柿子",
                "26": "特产/山茱萸",
                "27": "特产/玉器",
                "28": "特产/阎良甜瓜",
                "29": "特产/陕北红小豆",
                "30": "特产/高陵冬枣",
                "31": "美食/八宝玫瑰镜糕",
                "32": "美食/凉皮",
                "33": "美食/凉鱼",
                "34": "美食/德懋恭水晶饼",
                "35": "美食/搅团",
                "36": "美食/枸杞炖银耳",
                "37": "美食/柿子饼",
                "38": "美食/浆水面",
                "39": "美食/灌汤包",
                "40": "美食/烧肘子",
                "41": "美食/石子饼",
                "42": "美食/神仙粉",
                "43": "美食/粉汤羊血",
                "44": "美食/羊肉泡馍",
                "45": "美食/肉夹馍",
                "46": "美食/荞面饸饹",
                "47": "美食/菠菜面",
                "48": "美食/蜂蜜凉粽子",
                "49": "美食/蜜饯张口酥饺",
                "50": "美食/西安油茶",
                "51": "美食/贵妃鸡翅",
                "52": "美食/醪糟",
                "53": "美食/金线油塔"
            }
batch_size = 1
workers = 0
# valdir = '/home/gu/Young/AI/datasets/add_data'
valdir = '/home/gu/Young/AI/train_data/new_train_val_1000/val'
# valdir = '/home/gu/WP/python/huawei/c_xian/train_data/train_val/train'
model_path =  r'/home/gu/Young/AI/wp_01/model/model_best.pth'
# err_imgs = '/home/gu/WP/python/huawei/c_xian/wp_01/err_imgs'
err_imgs = '/home/gu/Young/AI/wp_01/err_imgs'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
size = 224
val_dataset=datasets.ImageFolder(valdir, transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=0, pin_memory=False)

idx_to_class = OrderedDict()
for key, value in val_dataset.class_to_idx.items():
    idx_to_class[value] = key

def judge(output,target,images,i):
    global count
    with torch.no_grad():#images: 32,3,224,224
        # batch_size = target.size(0)
        _,pred = output.topk(1,1,True,True)
        pred = pred.t()#[1,32]
        # print(pred)
        target = target.view(1, -1).expand_as(pred)
        if pred[0][0] != target[0][0]:
            # image_tensor = images
            t =idx_to_class[(target[0][0].item())]
            t = str(t)
            p = idx_to_class[(pred[0][0].item())]
            p = str(p)
            gt_label = label_id_name_dict[t]
            pred_label = label_id_name_dict[p]
            count += 1
            print(val_dataset.imgs[i])
            print(count,"pred_id:",p,"pred_label:",pred_label,"gt_id:",t,"gt_label:",gt_label)
            # filename = pred_label+'*****'+gt_label +'.jpg'
            filename = str(count)+'_'+p+pred_label + 'true_is' + t+gt_label + '.jpg'
            filename = validateTitle(filename)
            print(filename)
            save_image_tensor2cv2(images, os.path.join(err_imgs, filename))
        # for i in range(batch_size):
        #     if pred[0][i] != target[0][i]:
        #         image_tensor = images[i]
        #         # image_tensor =
        #         k = 1


def validate_data(val_loader, model):
    model.eval()
    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
            # print(val_dataset.imgs[i])
            print(i)
            # file_name = 'a'+str(i)+'.jpg'
            # print(i,str(target[0][0].item()))
            # save_image_tensor2cv2(images, os.path.join(err_imgs, file_name))
            target = target#[32]
            output = model(images)#[32*54]
            judge(output, target,images,i)



if __name__ == '__main__':
    # checkpoint = torch.load(model_path)
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=54)
    model = make_model_by_name('resnext101_32x8d_wsl', 54)
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(checkpoint['state_dict'])
    # validate_data(val_loader,model)
    print('Using CPU for inference')
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = OrderedDict()
    # 训练脚本 main.py 中保存了'epoch', 'arch', 'state_dict', 'best_acc1', 'optimizer'五个key值，
    # 其中'state_dict'对应的value才是模型的参数。
    # 训练脚本 main.py 中创建模型时用了torch.nn.DataParallel，因此模型保存时的dict都会有‘module.’的前缀，
    # 下面 tmp = key[7:] 这行代码的作用就是去掉‘module.’前缀
    for key, value in checkpoint['state_dict'].items():
        tmp = key[7:]
        state_dict[tmp] = value
    model.load_state_dict(state_dict)
    model.eval()
    # idx_to_class = checkpoint['idx_to_class']
    validate_data(val_loader, model)

