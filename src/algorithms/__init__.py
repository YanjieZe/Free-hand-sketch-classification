from .cnn import CNN
from .resnet18 import RESNET
from .VGG16 import VGG16_


def create_model(args):
    if args.alg == 'cnn':
        model = CNN(args.img_size, args.num_class)
    elif args.alg == 'resnet':
        model = RESNET(args.img_size, args.num_class)
    elif args.alg == 'vgg':
        model = VGG16_(args.img_size, args.num_class)
    else:
        raise ValueError('Unknown algorithm: {}'.format(args.alg))

    return model