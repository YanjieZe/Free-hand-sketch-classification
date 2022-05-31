import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
import time

# do classficiation
class CNN(nn.Module):
    def __init__(self, img_size, num_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_class)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
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
                    output = self(img)
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            print(colored('Accuracy of the network on the validation images: {} %'.format(100 * correct / total), 'red'))

    
   
        