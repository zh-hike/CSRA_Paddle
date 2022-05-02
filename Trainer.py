import paddle.nn.functional as F
import paddle
import paddle.nn as nn
from dataset.dataLoader import DL
from network import CSRA_resnet


class Trainer:
    def __init__(self, args):
        self.args = args
        self._init_data()
        self._init_model()

    def _init_data(self):
        data = DL(self.args)
        self.traindl = data.traindl
        self.testdl = data.testdl

    def _init_model(self):
        self.net = CSRA_resnet(self.args)
        self.cri = F.binary_cross_entropy_with_logits
        backbone, classifier = [], []
        for name, param in self.net.named_parameters():
            if 'classifier' in name:
                classifier.append(param)
            else:
                backbone.append(param)

        self.backbone_opt_lr = paddle.optimizer.lr.StepDecay(learning_rate=self.args.lr, step_size=4, gamma=0.1)
        self.classifier_opt_lr = paddle.optimizer.lr.StepDecay(learning_rate=self.args.lr*10, step_size=4, gamma=0.1)
        self.backbone_opt = paddle.optimizer.SGD(parameters=backbone, learning_rate=self.backbone_opt_lr,
                                                 weight_decay=self.args.w_d)
        self.classifier_opt = paddle.optimizer.SGD(parameters=classifier,
                                                   learning_rate=self.classifier_opt_lr,
                                                   weight_decay=self.args.w_d)





    def train(self):

        for epoch in range(1, self.args.epochs+1):

            for batch, (inputs, targets) in enumerate(self.traindl):

                output = self.net(inputs)

                print(output.shape)




            return










