from .cnn import CNN
from .sketch_rcnn import SketchR2CNN
from .modelzoo import CNN_MODELS
import torchvision

def create_model(args):
    if args.alg == 'cnn':
        model = CNN(args.img_size, args.num_class)
    elif args.alg == 'sketch_rcnn':
        cnn_fn = CNN_MODELS['resnet18']
        model = SketchR2CNN(cnn_fn=cnn_fn,
                 rnn_input_size=3,
                 rnn_dropout=1,
                 img_size=args.img_size,
                 thickness=1,
                 num_categories=args.num_class)
    else:
        raise ValueError('Unknown algorithm: {}'.format(args.alg))

    return model