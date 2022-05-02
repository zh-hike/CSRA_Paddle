from paddle.io import Dataset
from paddle.vision.transforms import transforms
from PIL import Image
import json
import numpy as np


class Voc(Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        file = './data/voc07/trainval_voc07.json'
        if train == False:
            file = './data/voc07/test_voc07.json'

        f = open(file, 'r')
        d = f.read()
        f.close()
        self.data = json.loads(d)

    def aug_train(self, x):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.args.img_size, scale=(0.7, 1.0)),
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
        ])

        return transform(x)

    def aug_test(self, x):
        transform = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
        ])
        return transform(x)

    def __getitem__(self, item):
        targets, img_path = self.data[item]['target'], self.data[item]['img_path']
        img = Image.open(img_path)
        targets = np.array(targets).astype('int64')
        if self.train:
            img = self.aug_train(img)
        else:
            img = self.aug_test(img)

        return img, targets

    def __len__(self):
        return len(self.data)
