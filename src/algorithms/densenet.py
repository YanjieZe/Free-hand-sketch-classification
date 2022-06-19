import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
import time
import torchvision
from .modelzoo import DenseNet121Backbone
import torchvision.transforms as transforms
# do classficiation
class DenseNet(nn.Module):
    def __init__(self, img_size, num_class):
        super(DenseNet, self).__init__()
        self.dense_layer =  DenseNet121Backbone(in_channels=1)
        self.mlp = nn.Linear(self.dense_layer.num_out_features, num_class)
        self.transforms = transforms.Resize(224)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.transforms(x)
        x = self.dense_layer(x)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)
    
    def train(self, dataloader_train, dataloader_val, optimizer, num_epochs, args):
        for epoch in range(num_epochs):
            current_time = time.time()
            for i, (img, label) in enumerate(dataloader_train):
                img = img.view(img.size(0), 1, 28, 28)
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
                if i % 1000 == 0:
                    new_time = time.time()
                    duration = new_time - current_time
                    current_time = new_time
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, num_epochs, i + 1, len(dataloader_train), loss.item(), duration))

            # validation
            correct = 0
            total = 0
            with torch.no_grad():
                for img, label in dataloader_val:
                    img = img.view(img.size(0), 1, 28, 28)
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
            print(colored('Accuracy of the network on the validation images: {} %'.format(100 * correct / total), 'red'))
    
    def test(self, dataloader_test, args):
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in dataloader_test:
                img = img.view(img.size(0), 1, 28, 28)
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
        print(colored('Accuracy of the network on the test images: {} %'.format(100 * correct / total), 'cyan'))

    
   
        