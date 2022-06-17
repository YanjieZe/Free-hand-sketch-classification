from __future__ import division
from __future__ import print_function
from .basemodel import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import os.path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from termcolor import colored

_project_folder_ = os.path.abspath('../')
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from neuralline.rasterize import RasterIntensityFunc


class SeqEncoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size=512,
                 num_layers=2,
                 out_channels=1,
                 batch_first=True,
                 bidirect=True,
                 dropout=0,
                 requires_grad=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.batch_first = batch_first
        self.bidirect = bidirect
        self.proj_last_hidden = False

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           bidirectional=bidirect,
                           dropout=dropout)

        num_directs = 2 if bidirect else 1
        self.attend_fc = nn.Linear(hidden_size * num_directs, out_channels)

        if self.proj_last_hidden:
            self.last_hidden_size = hidden_size
            self.last_hidden_fc = nn.Linear(num_directs * num_layers * hidden_size, self.last_hidden_size)
        else:
            self.last_hidden_size = num_directs * num_layers * hidden_size
            self.last_hidden_fc = None

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points, lengths):
        batch_size = points.shape[0]
        num_points = points.shape[1]
        point_dim = points.shape[2]

        if point_dim != self.input_size:
            points = points[:, :, :self.input_size]

        points_packed = pack_padded_sequence(points, lengths, batch_first=self.batch_first)
        hiddens_packed, (last_hidden, _) = self.rnn(points_packed) 

        intensities_act = torch.sigmoid(self.attend_fc(hiddens_packed.data))

        intensities_packed = PackedSequence(intensities_act, hiddens_packed.batch_sizes)
        intensities, _ = pad_packed_sequence(intensities_packed, batch_first=self.batch_first, total_length=num_points)

        last_hidden = last_hidden.view(batch_size, -1)

        if self.proj_last_hidden:
            last_hidden = F.relu(self.last_hidden_fc(last_hidden))

        return intensities, last_hidden


class SketchR2CNN(BaseModel):

    def __init__(self,
                 cnn_fn,
                 rnn_input_size,
                 rnn_dropout,
                 img_size,
                 thickness,
                 num_categories,
                 intensity_channels=8,
                 train_cnn=True,
                 device="cuda"):
        super().__init__()

        self.img_size = img_size
        self.thickness = thickness
        self.intensity_channels = intensity_channels
        self.eps = 1e-4
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.rnn = SeqEncoder(rnn_input_size, out_channels=intensity_channels, dropout=rnn_dropout)
        self.cnn = cnn_fn(pretrained=False, requires_grad=train_cnn, in_channels=intensity_channels)

        num_fc_in_features = self.cnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)

        nets.extend([self.rnn, self.cnn, self.fc])
        names.extend(['rnn', 'conv', 'fc'])
        train_flags.extend([True, train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, points, points_offset, lengths):
        intensities, _ = self.rnn(points_offset, lengths)

        images = RasterIntensityFunc.apply(points, intensities, self.img_size, self.thickness, self.eps, self.device)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        cnnfeat = self.cnn(images)
        logits = self.fc(cnnfeat)

        return logits, intensities, images

    def cuda(self):
        self.to('cuda')
        return self

    def parameters(self):
        return self.params_to_optimize()

    def train(self, dataloader_train, dataloader_val, optimizer, num_epochs, args):
        for epoch in range(num_epochs):
            current_time = time.time()
            for i, data_batch in enumerate(dataloader_train):


                label = data_batch['category'].long().to(self.device)

                points = data_batch['points3'].to(self.device)
                points_offset = data_batch['points3_offset'].to(self.device)
                points_length = data_batch['points3_length']
                        


                logits, attention, images = self(points, points_offset, points_length)
                loss = F.cross_entropy(logits, label)
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
                for i,  data_batch in enumerate(dataloader_val):
                    label = data_batch['category'].long().to(self.device)

                    points = data_batch['points3'].to(self.device)
                    points_offset = data_batch['points3_offset'].to(self.device)
                    points_length = data_batch['points3_length']
                            
                    logits, attention, images = self(points, points_offset, points_length)

                    _, predicted = torch.max(logits.data, 1)
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