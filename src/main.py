from utils.arguments import parse_args
from utils.dataloader import DatasetQuickdraw
import utils.misc as misc
from algorithms import create_model
import torch.optim as optim
import torch
import os

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