from paddle.io import DataLoader
from dataset.voc import Voc


class DL:
    def __init__(self, args):
        train_data = Voc(args, train=True)
        test_data = Voc(args, train=False)

        self.traindl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        self.testdl = DataLoader(test_data, batch_size=args.batch_size)
