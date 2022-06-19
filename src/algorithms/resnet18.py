import torch.nn as nn
import torch
import torch.nn.functional as F
from termcolor import colored
import time


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


# do classficiation
class RESNET(nn.Module):
    def __init__(self, img_size, num_class):
        super(RESNET, self).__init__()
        self.model = resnet50()
        in_channel = self.model.fc.in_features
        self.model.fc = nn.Linear(in_channel, num_class)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def train(self, dataloader_train, dataloader_val, optimizer, num_epochs, args):
        for epoch in range(num_epochs):
            current_time = time.time()
            for i, (img, label) in enumerate(dataloader_train):
                # img = img.view(img.size(0), 1, 28, 28)
                img = img.view(img.size(0), 1, 28, 28)
                img = img.expand(img.size(0), 3, 28, 28)
                img = img.to(torch.float32)
                label = label.to(torch.long)

                # use gpu
                if args.use_gpu:
                    img = img.cuda()
                    label = label.cuda()

                output = self(img)
                loss = F.cross_entropy(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with open('./loss_50.txt', 'a') as f:
                    f.write(str(epoch)+':'+str(i)+'\t'+str(loss.item())+'\n')
                if i % 1000 == 0:
                    new_time = time.time()
                    duration = new_time - current_time
                    current_time = new_time
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, num_epochs, i + 1,
                                                                                           len(dataloader_train),
                                                                                           loss.item(), duration))

            # validation
            correct = 0
            total = 0
            with torch.no_grad():
                for img, label in dataloader_val:
                    #img = img.view(img.size(0), 1, 28, 28)
                    img = img.view(img.size(0), 1, 28, 28)
                    img = img.expand(img.size(0), 3, 28, 28)
                    img = img.to(torch.float32)
                    label = label.to(torch.long)

                    # use gpu
                    if args.use_gpu:
                        img = img.cuda()
                        label = label.cuda()

                    output = self(img)
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            with open('./accuracy_50.txt','a') as f:
                f.write(str(epoch) + ':' + str(100 * correct / total) + '\n')
            print(
                colored('Accuracy of the network on the validation images: {} %'.format(100 * correct / total), 'red'))

    def test(self, dataloader_test, args):
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in dataloader_test:
                img = img.view(img.size(0), 1, 28, 28)
                img = img.expand(img.size(0), 3, 28, 28)
                img = img.to(torch.float32)
                label = label.to(torch.long)

                # use gpu
                if args.use_gpu:
                    img = img.cuda()
                    label = label.cuda()

                output = self(img)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        with open('./accuracy_test_50.txt', 'a') as f:
            f.write(str(100 * correct / total) + '\n')
        print(colored('Accuracy of the network on the test images: {} %'.format(100 * correct / total), 'cyan'))