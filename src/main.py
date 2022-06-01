import enum
from utils.arguments import parse_args
from utils.dataloader import DatasetQuickdraw
import utils.misc as misc
from algorithms.byol import BYOL 
import torch.optim as optim
from torchvision import models
import torch
import numpy as np

def main():
    args = parse_args()

    misc.seed_all(args.seed)

    # create dataset
    transform = None
    dataset_train = DatasetQuickdraw(args.data_dir, transform, mode="train")
    dataset_val = DatasetQuickdraw(args.data_dir, transform, mode="val")
    
    # create dataloader
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)



    # create model
    resnet = models.resnet50(pretrained=True)
    model = BYOL(resnet, image_size=28, hidden_layer='avgpool')
    if args.use_gpu:
        model = model.cuda()

    # create sgd optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train
    # model.train(dataloader_train, dataloader_val, optimizer, args.num_epochs, args)
    
    # for byol
    total_step = 0
    for i in range(args.num_epochs):
        model.train()
        for index, data in enumerate(dataloader_train):
            total_step += 1
            im = data[0].float()
            image = torch.zeros([im.shape[0], 3, im.shape[1], im.shape[2]])
            image[:,0,:,:] = im
            image[:,1,:,:] = im
            image[:,2,:,:] = im
            loss = model(image.cuda() if args.use_gpu else image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()
    # save model
    torch.save(model.state_dict(), 'model.pth')





if __name__ == '__main__':
    main()