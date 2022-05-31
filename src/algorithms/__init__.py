from .cnn import CNN


def create_model(args):
    if args.alg == 'cnn':
        model = CNN(args.img_size, args.num_class)
    else:
        raise ValueError('Unknown algorithm: {}'.format(args.alg))

    return model