import numpy as np
import os
import torch.utils.data as data
import tqdm 
from termcolor import colored

class DatasetQuickdrawPNG(data.Dataset):
    def __init__(self, data_dir, transform=None, mode="train"):
        self.data_dir = data_dir
        self.transform = transform
        self.data = None
        self.labels = None
        self.mode = mode

        self.load_data()

    def load_data(self):
        self.train_data = []
        self.train_label = []
        self.val_data = []
        self.val_label = []
        self.test_data = []
        self.test_label = []
        label_id = 0
        for filename in tqdm.tqdm(os.listdir(self.data_dir), desc='Data loading'):
            if filename.endswith('.npz'):
                file_loaded = np.load(os.path.join(self.data_dir, filename))
                keys = file_loaded.files
                if self.mode == "train":
                    train_data = file_loaded["train"]
                    train_label = np.ones(train_data.shape[0]) * label_id
                    self.train_data.append(train_data)
                    self.train_label.append(train_label)
                elif self.mode == "val":
                    val_data = file_loaded["valid"]
                    val_label = np.ones(val_data.shape[0]) * label_id
                    self.val_data.append(val_data)
                    self.val_label.append(val_label)
                elif self.mode == "test":
                    test_data = file_loaded["test"]
                    test_label = np.ones(test_data.shape[0]) * label_id
                    self.test_data.append(test_data)
                    self.test_label.append(test_label)

                label_id += 1
        # aggeragate
        if self.mode == "train":
            self.data = np.concatenate(self.train_data, axis=0)
            self.labels = np.concatenate(self.train_label, axis=0)
        elif self.mode == "val":
            self.data = np.concatenate(self.val_data, axis=0)
            self.labels = np.concatenate(self.val_label, axis=0)

        elif self.mode == "test":
            self.data = np.concatenate(self.test_data, axis=0)
            self.labels = np.concatenate(self.test_label, axis=0)

        perm = np.random.permutation(self.data.shape[0])
        self.data = self.data[perm]
        self.labels = self.labels[perm]
        print(colored('Data: Loaded %d samples' % len(self.data), 'red'))
        
        
    
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

class DatasetQuickdrawSVG(data.Dataset):
    def __init__(self, data_dir, transform=None, mode="train"):
        self.data_dir = data_dir
        self.transform = transform
        self.data = None
        self.labels = None
        self.mode = mode

        self.load_data()

    def load_data(self):
        self.train_data = []
        self.train_label = []
        self.val_data = []
        self.val_label = []
        self.test_data = []
        self.test_label = []
        label_id = 0
        for filename in tqdm.tqdm(os.listdir(self.data_dir), desc='Data loading'):
            if filename.endswith('.npz'):
                file_loaded = np.load(os.path.join(self.data_dir, filename), encoding='latin1', allow_pickle=True)
                keys = file_loaded.files
                if self.mode == "train":
                    train_data = file_loaded["train"]
                    train_label = np.ones(train_data.shape[0]) * label_id
                    self.train_data.append(train_data)
                    self.train_label.append(train_label)
                elif self.mode == "val":
                    val_data = file_loaded["valid"]
                    val_label = np.ones(val_data.shape[0]) * label_id
                    self.val_data.append(val_data)
                    self.val_label.append(val_label)
                elif self.mode == "test":
                    test_data = file_loaded["test"]
                    test_label = np.ones(test_data.shape[0]) * label_id
                    self.test_data.append(test_data)
                    self.test_label.append(test_label)

                label_id += 1
        # aggeragate
        if self.mode == "train":
            self.data = np.concatenate(self.train_data, axis=0)
            self.labels = np.concatenate(self.train_label, axis=0)
        elif self.mode == "val":
            self.data = np.concatenate(self.val_data, axis=0)
            self.labels = np.concatenate(self.val_label, axis=0)

        elif self.mode == "test":
            self.data = np.concatenate(self.test_data, axis=0)
            self.labels = np.concatenate(self.test_label, axis=0)

        perm = np.random.permutation(self.data.shape[0])
        self.data = self.data[perm]
        self.labels = self.labels[perm]
        print(colored('Data: Loaded %d samples' % len(self.data), 'red'))
        
        
    
    def __getitem__(self, index):
        points = self.data[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        items = dict()
        items["points3"] = points
        items["category"] = label
        return items

    def __len__(self):
        return len(self.data)


if __name__=="__main__":
    data_dir = "/data/yanjieze/projects/draft/hand_sketch_clf/dataset/quickdraw"
    dataset = DatasetQuickdrawSVG(data_dir)
    print(dataset.data[0].shape)
    print(dataset.labels[0].shape)