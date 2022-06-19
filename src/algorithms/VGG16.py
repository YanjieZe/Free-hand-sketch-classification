import torch.nn as nn
import torch
import torch.nn.functional as F
from termcolor import colored
import time
import torch.utils.model_zoo as model_zoo


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x) #[512,1,1]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


# do classficiation
class VGG16_(nn.Module):
    def __init__(self, img_size, num_class):
        super(VGG16_, self).__init__()
        self.model = vgg16(pretrained=False)
        self.model.features[0] = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.model.features[2] = nn.ConvTranspose2d(16, 64, kernel_size=(3,3),
                                                    stride=(2,2), padding=(0,0),
                                                    bias=False)
        #in_channel = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(4096, num_class)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def train(self, dataloader_train, dataloader_val, optimizer, num_epochs, args):
        for epoch in range(num_epochs):
            current_time = time.time()
            for i, (img, label) in enumerate(dataloader_train):
                img = img.view(img.size(0), 1, 28, 28)
                #img = img.expand(img.size(0), 3, 28, 28)
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
                with open('./loss_vgg.txt', 'a') as f:
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
                    img = img.view(img.size(0), 1, 28, 28)
                    #img = img.expand(img.size(0), 3, 28, 28)
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
            with open('./accuracy_vgg.txt','a') as f:
                f.write(str(epoch) + ':' + str(100 * correct / total) + '\n')
            print(
                colored('Accuracy of the network on the validation images: {} %'.format(100 * correct / total), 'red'))

    def test(self, dataloader_test, args):
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in dataloader_test:
                img = img.view(img.size(0), 1, 28, 28)
                #img = img.expand(img.size(0), 3, 28, 28)
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
        with open('./accuracy_test_vgg.txt', 'a') as f:
            f.write(str(100 * correct / total) + '\n')
        print(colored('Accuracy of the network on the test images: {} %'.format(100 * correct / total), 'cyan'))