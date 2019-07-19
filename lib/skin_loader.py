from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import numpy as np
import torchvision.transforms as transforms

img_size = 32


class skin_dataset_test(Dataset):

    def __init__(self):
        super(skin_dataset_test, self)

        train_txt_file = "./txt_for_folds/fold0_train.txt"
        train_txt_file = "./txt_for_folds/fold0_val.txt"
        self.root = "/data1/kerrzwu/data/isic/Color_Constancy"

        self.abn_cls_idx = 7

        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
             transforms.ToTensor()
             ]
        )

        train_img_list = []
        train_label_list = []

        test_img_list = []
        test_label_list = []

        with open(train_txt_file, "r") as f:
            lines = list(f.readlines())

            for line in lines:
                img, label = tuple(line.split(","))
                train_img_list.append(img)
                train_label_list.append(int(label))

        with open(train_txt_file, "r") as f:
            lines = list(f.readlines())

            for line in lines:
                img, label = tuple(line.split(","))
                test_img_list.append(img)
                test_label_list.append(int(label))

        train_img_list = np.array(train_img_list)
        train_label_list = np.array(train_label_list)
        test_img_list = np.array(test_img_list)
        test_label_list = np.array(test_label_list)

        # Find idx, img, lbl for abnormal and normal on org dataset.
        abn_trn_idx = np.where(train_label_list == self.abn_cls_idx)[0]
        abn_trn_img = train_img_list[abn_trn_idx]  # Abnormal training images
        abn_trn_lbl = train_label_list[abn_trn_idx]  # Abnormal training labels.

        nrm_tst_idx = np.where(test_label_list != self.abn_cls_idx)[0]
        abn_tst_idx = np.where(test_label_list == self.abn_cls_idx)[0]
        nrm_tst_img = test_img_list[nrm_tst_idx]  # Normal training images
        abn_tst_img = test_img_list[abn_tst_idx]  # Abnormal training images.
        nrm_tst_lbl = test_label_list[nrm_tst_idx]  # Normal training labels
        abn_tst_lbl = test_label_list[abn_tst_idx]  # Abnormal training labels.

        # --
        # Assign labels to normal (0) and abnormals (1)
        nrm_tst_lbl[:] = 0
        abn_trn_lbl[:] = 1
        abn_tst_lbl[:] = 1

        # Create new anomaly dataset based on the following data structure:
        # - anomaly dataset
        #   . -> train
        #        . -> normal
        #   . -> test
        #        . -> normal
        #        . -> abnormal
        self.new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
        self.new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    def __getitem__(self, index):

        test_img = self.new_tst_img[index]
        test_img = os.path.join(self.root, test_img + ".jpg")
        test_img = Image.open(test_img)
        # test_img = np.asarray(test_img)
        if self.transform is not None:
            test_img = self.transform(test_img)

        label = self.new_tst_lbl[index]

        return test_img, label

        # nrm_tst_img = os.path.join(self.root, test_img[0] + ".jpg")
        # abn_trn_img = os.path.join(self.root, test_img[1] + ".jpg")
        # abn_tst_img = os.path.join(self.root, test_img[2] + ".jpg")
        #
        # nrm_tst_img = Image.open(nrm_tst_img).resize((img_size, img_size))
        # abn_trn_img = Image.open(abn_trn_img).resize((img_size, img_size))
        # abn_tst_img = Image.open(abn_tst_img).resize((img_size, img_size))
        #
        # nrm_tst_img = np.asarray(nrm_tst_img)
        # abn_trn_img = np.asarray(abn_trn_img)
        # abn_tst_img = np.asarray(abn_tst_img)

        # if self.transform is not None:
        #     nrm_tst_img = self.transform(nrm_tst_img)
        #     abn_trn_img = self.transform(abn_trn_img)
        #     abn_tst_img = self.transform(abn_tst_img)
        #
        # label = self.new_tst_lbl[index]
        #
        # return np.array([nrm_tst_img, abn_trn_img, abn_tst_img]), label

    def __len__(self):
        return len(self.new_tst_lbl)


class skin_dataset_train(Dataset):

    def __init__(self):
        super(skin_dataset_train, self)

        train_txt_file = "./txt_for_folds/fold0_train.txt"
        self.root = "/data1/kerrzwu/data/isic/Color_Constancy"

        self.abn_cls_idx = 7

        self.transform = transforms.Compose(
            [transforms.RandomRotation(180),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.Resize((img_size, img_size)),
             transforms.ToTensor()
             ]
        )

        img_list = []
        label_list = []

        with open(train_txt_file, "r") as f:
            lines = list(f.readlines())

            np.random.seed(1)
            np.random.shuffle(lines)

            for line in lines[:15000]:
                img, label = tuple(line.split(","))
                img_list.append(img)
                label_list.append(int(label))

        img_list = np.array(img_list)
        label_list = np.array(label_list)

        nrm_trn_idx = torch.from_numpy(np.where(label_list != self.abn_cls_idx)[0])

        self.nrm_trn_img = img_list[nrm_trn_idx]  # Normal training images
        self.nrm_trn_lbl = label_list[nrm_trn_idx]  # Normal training labels

        # Assign labels to normal (0) and abnormals (1)
        self.nrm_trn_lbl[:] = 0

    def __getitem__(self, index):
        img_path = self.nrm_trn_img[index]
        label = self.nrm_trn_lbl[index]
        img_path = os.path.join(self.root, img_path + ".jpg")
        img = Image.open(img_path)
        # img = np.asarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.nrm_trn_lbl)
