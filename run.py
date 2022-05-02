import argparse
from Trainer import Trainer


parse = argparse.ArgumentParser('CSRA')

parse.add_argument('--img_size', type=int, help="图像大小", default=448)
parse.add_argument('--batch_size', type=int, help="批次大小", default=200)
parse.add_argument('--n_classes', type=int, help="类别数", default=20)
parse.add_argument('--epochs', type=int, help="训练轮次",default=30)
parse.add_argument('--lr', type=float, help="学习率", default=0.01)
parse.add_argument('--momentum', type=float,         default=0.9)
parse.add_argument("--w_d", type=float, help="weight_decay", default=0.0001)
parse.add_argument('--train', action='store_true')
# parse.add_argument('--data_path', type=str, help="数据集所在位置", default='./dataset/')
args = parse.parse_args()


if __name__ == '__main__':
    trainer = Trainer(args)

    if args.train:
        trainer.train()

