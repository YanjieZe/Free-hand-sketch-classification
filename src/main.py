from utils.arguments import parse_args
from utils.dataloader import DatasetQuickdrawPNG, DatasetQuickdrawSVG
import utils.misc as misc
from algorithms import create_model
import torch.optim as optim
import torch
import os
import numpy as np

def train_data_collate_SVG(batch):

    length_list = [len(item['points3']) for item in batch] 
    max_length = max(length_list) 

    points3_padded_list = list()
    points3_offset_list = list()
    intensities_list = list()
    category_list = list()
    for item in batch:
        points3 = item['points3']
        points3_length = len(points3)
        points3_padded = np.zeros((max_length, 3), np.float32)
        points3_padded[:, 2] = np.ones((max_length,), np.float32)
        points3_padded[0:points3_length, :] = points3
        points3_padded_list.append(points3_padded)

        points3_offset = np.copy(points3_padded)
        points3_offset[1:points3_length, 0:2] = points3[1:, 0:2] - points3[:points3_length - 1, 0:2]
        points3_offset_list.append(points3_offset)

        intensities = np.zeros((max_length,), np.float32)
        intensities[:points3_length] = 1.0 - np.arange(points3_length, dtype=np.float32) / float(points3_length - 1)
        intensities_list.append(intensities)

        category_list.append(item['category'])

    batch_padded = {
        'points3': points3_padded_list,
        'points3_offset': points3_offset_list,
        'points3_length': length_list,
        'intensities': intensities_list,
        'category': category_list
    }

    sort_indices = np.argsort(-np.array(length_list))
    batch_collate = dict()
    for k, v in batch_padded.items():
        sorted_arr = np.array([v[idx] for idx in sort_indices])
        batch_collate[k] = torch.from_numpy(sorted_arr)
    return batch_collate



def main():
    args = parse_args()

    misc.seed_all(args.seed)

    # create dataset
    transform = None
    dataset_train = DatasetQuickdrawPNG(args.data_dir, transform, mode="train") if args.img_form == "png" else DatasetQuickdrawSVG(args.data_dir, transform, mode="train")
    dataset_val = DatasetQuickdrawPNG(args.data_dir, transform, mode="val") if args.img_form == "png" else DatasetQuickdrawSVG(args.data_dir, transform, mode="val")
    
    # create dataloader
    train_data_collate = train_data_collate_SVG if args.img_form == "svg" else None
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_data_collate)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_data_collate)



    # create model
    model = create_model(args)
    if args.use_gpu:
        model = model.cuda()

    # create sgd optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # train
    model.train(dataloader_train, dataloader_val, optimizer, args.num_epochs, args)

    # test
    dataset_test = DatasetQuickdrawPNG(args.data_dir, transform, mode="test") if args.img_form == "png" else DatasetQuickdrawSVG(args.data_dir, transform, mode="test")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model.test(dataloader_test, args)

    # save
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(model.state_dict(), os.path.join(args.save_dir, args.alg))





if __name__ == '__main__':
    main()