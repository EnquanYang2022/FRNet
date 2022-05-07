import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale
from toolbox.utils import color_map
from torch import nn
from torch.autograd import Variable as V
import torch as t
class NYUv2(data.Dataset):

    def __init__(self, cfg, random_state=3, mode='train',):
        assert mode in ['train', 'test']

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']
        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        self.mode = mode
        self.class_weight = np.array([4.01302219, 5.17995767, 12.47921102, 13.79726557, 18.47574439, 19.97749822,
                                      21.10995738, 25.86733191, 27.50483598, 27.35425244, 25.12185149, 27.04617447,
                                      30.0332327, 29.30994935, 34.72009825, 33.66136128, 34.28715586, 32.69376342,
                                      33.71574286, 37.0865665, 39.70731054, 38.60681717, 36.37894266, 40.12142316,
                                      39.71753044, 39.27177794, 43.44761984, 42.96761184, 43.98874667, 43.43148409,
                                      43.29897719, 45.88895515, 44.31838311, 44.18898992, 42.93723439, 44.61617778,
                                      47.12778303, 46.21331253, 27.69259756, 25.89111664, 15.65148615, ])
        #train_test_split返回切分的数据集train/test
        self.train_ids, self.test_ids = train_test_split(np.arange(1449), train_size=795, random_state=random_state)


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        else:
            return len(self.test_ids)

    def __getitem__(self, index):
        # key=self.train_ids[index][0]

        if self.mode == 'train':
            image_index = self.train_ids[index]
            gate_gt = torch.zeros(1)
            # gate_gt[0] = key

        else:
            image_index = self.test_ids[index]
        
        image_path = f'all_data/image/{image_index}.jpg'
        depth_path = f'all_data/depth/{image_index}.png'
        label_path = f'all_data/label/{image_index}.png'
        # label_pathcxk = f'all_data/Label/{image_index}.png'
        # label_path = '/home/yangenquan/PycharmProjects/NYUv2/all_data/label/75.png'

        image = Image.open(os.path.join(self.root, image_path))  # RGB 0~255
        depth = Image.open(os.path.join(self.root, depth_path)).convert('RGB')  # 1 channel -> 3
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37
        # labelcxk = Image.open(os.path.join(self.root, label_pathcxk))

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
            # 'name' : image_index
            # 'labelcxk':labelcxk,
        }

        if self.mode == 'train':  # 只对训练集增强
            sample = self.aug(sample)


        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        # sample['labelcxk'] = torch.from_numpy(np.asarray(sample['labelcxk'], dtype=np.int64)).long()

        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        # sample['name'] = image_index
        return sample

    """ for train loader """

# def train_collate_fn(batch):
#     images, depths, labels, gate_gt = zip(*batch)
#     l = len(images[0])
#     images_t, depths_t, labels_t = {}, {}, {}
#     gates_t = {}
#     gate_gt = torch.stack(gate_gt)
#     for i in range(l):
#         images_t[i] = []
#         depths_t[i] = []
#         labels_t[i] = []
#         gates_t[i] = gate_gt
#
#     for i in range(len(images)):
#         for j in range(l):
#             images_t[j].append(images[i][j])
#             depths_t[j].append(depths[i][j])
#             labels_t[j].append(labels[i][j])
#
#     for i in range(l):
#         images_t[i] = torch.stack(images_t[i])
#         depths_t[i] = torch.stack(depths_t[i])
#         labels_t[i] = torch.stack(labels_t[i])
#
#     return images_t, depths_t, labels_t, gates_t
    @property
    def cmap(self):
        return [(0, 0, 0),
                (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0),
                (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128),
                (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0),
                (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128),
                (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64),
                (0, 128, 64), (128, 128, 64), (0, 0, 192), (128, 0, 192),
                (0, 128, 192), (128, 128, 192), (64, 0, 64)]  # 41
        # return color_map(N=41)


# if __name__ == '__main__':
#     import json
#
#     path = '../../configs/nyuv2.json'
#     with open(path, 'r') as fp:
#         cfg = json.load(fp)
#
#     dataset = NYUv2(cfg, mode='train')
#     from toolbox.utils import class_to_RGB
#     import matplotlib.pyplot as plt
#
#     for i in range(len(dataset)):
#         sample = dataset[i]
#
#         image = sample['image']
#         depth = sample['depth']
#         label = sample['label']
#
#         image = image.numpy()
#         image = image.transpose((1, 2, 0))
#         image *= np.asarray([0.229, 0.224, 0.225])
#         image += np.asarray([0.485, 0.456, 0.406])
#
#         depth = depth.numpy()
#         depth = depth.transpose((1, 2, 0))
#         depth *= np.asarray([0.226, 0.226, 0.226])
#         depth += np.asarray([0.449, 0.449, 0.449])
#
#         label = label.numpy()
#         label = class_to_RGB(label, N=41, cmap=dataset.cmap)
#
#         plt.subplot('131')  #行，列，那一幅图，如一共1*3图，该行的第一幅图
#         plt.imshow(image)
#         plt.subplot('132')
#         plt.imshow(depth)
#         plt.subplot('133')
#         plt.imshow(label)
#
#         plt.show()
if __name__ == '__main__':
    import json

    path = '/home/yangenquan/PycharmProjects/第一论文模型/(60.1)mymodel8/configs/nyuv2.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)

    dataset = NYUv2(cfg, mode='test')
    print(len(dataset))
    from toolbox.utils import class_to_RGB
    from PIL import Image
    import matplotlib.pyplot as plt

    # label = '/home/yangenquan/PycharmProjects/NYUv2/all_data/label/166.png'
    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        depth = sample['depth']
        label = sample['label']
        name = sample['name']

        image = image.numpy()
        image = image.transpose((1, 2, 0))
        image *= np.asarray([0.229, 0.224, 0.225])
        image += np.asarray([0.485, 0.456, 0.406])

        depth = depth.numpy()
        depth = depth.transpose((1, 2, 0))
        depth *= np.asarray([0.226, 0.226, 0.226])
        depth += np.asarray([0.449, 0.449, 0.449])
        # print(set(list(label)))
        label = label.numpy()
        # print(image)

        label = class_to_RGB(label, N=41, cmap=dataset.cmap)



        # print(dataset.cmap)
        # plt.subplot('131')  #行，列，那一幅图，如一共1*3图，该行的第一幅图
        # plt.imshow(image)
        # plt.subplot('132')
        # plt.imshow(depth)
        # plt.subplot('133')
        # plt.imshow(label)

        # plt.show()
        label = Image.fromarray(label)

        label.save(f'/home/yangenquan/PycharmProjects/NYUv2/all_data/change/label_color/{name}.png')
        # break
