from .cnn import CNN
from .sketch_rcnn import SketchR2CNN
from .vit import ViT
from .modelzoo import CNN_MODELS
import torchvision

def create_model(args):
    if args.alg == 'cnn':
        model = CNN(args.img_size, args.num_class)
    elif args.alg == 'sketch_r2cnn':
        cnn_fn = CNN_MODELS['resnet18']
        model = SketchR2CNN(cnn_fn=cnn_fn,
                 rnn_input_size=3,
                 rnn_dropout=1,
                 img_size=args.img_size,
                 thickness=1,
                 num_categories=args.num_class)
    elif args.alg == 'vit':
        model = ViT(image_size=args.img_size,
                    patch_size=args.patch_size, 
                    num_classes=args.num_class, 
                    dim=512,
                    depth=6, 
                    heads=8, 
                    mlp_dim=1024,
                    channels=1)
        
    else:
        raise ValueError('Unknown algorithm: {}'.format(args.alg))

    return model