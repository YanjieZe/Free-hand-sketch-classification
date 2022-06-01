import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Free-Hand Sketch Classifier')
    

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--use_gpu', action='store_true', help='use gpu')

    parser.add_argument('--alg', type=str, default='cnn', choices=['cnn'])



    # data
    parser.add_argument('--data_dir', type=str, default='dataset/quickdraw_png')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--num_class', type=int, default=25)



    # hyper param for training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)


    

    args = parser.parse_args()

    return args
