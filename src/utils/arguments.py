import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Free-Hand Sketch Classifier')
    parser.add_argument('--alg', type=str, default='svm', choices=['svm'])

    # data
    parser.add_argument('--data_dir', type=str, default='dataset/quickdraw')



    # hyper param for svm
    parser.add_argument('--C', type=float, default=1.0)
    

    args = parser.parse_args()

    return args
