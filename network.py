import paddle.nn as nn
from paddle.vision.models import resnet101
import paddle


class CSRA(nn.Layer):
    def __init__(self, in_dim, n_classes):
        super(CSRA, self).__init__()
        self.T = 1
        self.lambd = 0.1
        self.conv = nn.Conv2D(in_dim, n_classes, kernel_size=[1, 1])  # 缺一个偏置
        self.softmax = nn.Softmax(axis=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        score = self.head(x) / paddle.norm(self.head.weight, axis=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)

        base_logit = paddle.mean(score, axis=2)

        if self.T == 99:  # max-pooling
            att_logit = paddle.max(score, axis=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = paddle.sum(score * score_soft, axis=2)

        return base_logit + self.lam * att_logit


class MHA(nn.Layer):  # multi-head attention

    def __init__(self, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = [1]
        self.csra = CSRA(input_dim, num_classes)

    def forward(self, x):
        logit = self.csra(x)
        return logit


class CSRA_resnet(nn.Layer):
    def __init__(self, args):
        super(CSRA_resnet, self).__init__()
        self.args = args
        resnet = resnet101(pretrained=True)
        self.feature_extract = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,

        )

        self.classifier = MHA(2048, 20)

    def forward(self, img):
        x = self.feature_extract(img)
        x = self.classifier(x)
        return x
