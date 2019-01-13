import os
import random
from PIL import Image
from torch.utils import data
import torch
import numpy as np
from torchvision import transforms as T
from opt import opt

random.seed(666)
np.random.seed(666)

class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)


class FlipLeftRight(object):
    def __init__(self, do):
        self.do = do
    def __call__(self, img):
        if self.do>0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img


def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])

class PoseTripletTrain(data.Dataset):
    def __init__(self):
        """
        主要目标： 获取训练三元组
        """
        data=[]
        f=open (opt.train_triplet_txt,'r')
        for line in f.readlines():
            images=line.split(',')
            data.append([images[:10],images[10:20],images[20:30]])

        self.data=data
        self.transform=T.Compose([
        	T.Resize((opt.image_size,opt.image_size)),
        	FixedRotation([0, 10, -10,20,-20]),
        	T.ToTensor(),
        	T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        """
        一次返回一个三元组的数据
        """
        a_p_n=self.data[index]
        r=random.random()

        # anchor=torch.cat([self.transform(Image.open(x).convert('RGB')).view(-1,3,300,300) for x in a_p_n[0]]).permute(1, 0, 2, 3)
        anchor=torch.stack([self.transform(FlipLeftRight(r)(Image.open(x).convert('RGB'))) for x in a_p_n[0]],0).permute(1, 0, 2, 3) #0维扩充，permute维度变换位置
        positive=torch.stack([self.transform(FlipLeftRight(r)(Image.open(x).convert('RGB'))) for x in a_p_n[1]],0).permute(1, 0, 2, 3)
        negative=torch.stack([self.transform(FlipLeftRight(r)(Image.open(x).convert('RGB'))) for x in a_p_n[2]],0).permute(1, 0, 2, 3)



        return anchor,positive,negative

    def __len__(self):
    	return len(self.data)

class PoseTripletValidation(data.Dataset):
    def __init__(self):
        """
        主要目标： 获取训练三元组
        """
        data=[]
        f=open (opt.validation_triplet_txt,'r')
        for line in f.readlines():
            images=line.split(',')
            data.append([images[:10],images[10:20],images[20:30]])

        self.data=data
        self.transform=T.Compose([
        	T.Resize((opt.image_size,opt.image_size)),
            FixedRotation([0, 10, -10,20,-20]),
        	T.ToTensor(),
        	T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        """
        一次返回一个三元组的数据
        """
        a_p_n=self.data[index]
        r=random.random()

        # anchor=torch.cat([self.transform(Image.open(x).convert('RGB')).view(-1,3,300,300) for x in a_p_n[0]]).permute(1, 0, 2, 3)
        anchor=torch.stack([self.transform(FlipLeftRight(r)(Image.open(x).convert('RGB'))) for x in a_p_n[0]],0).permute(1, 0, 2, 3) #0维扩充，permute维度变换位置
        positive=torch.stack([self.transform(FlipLeftRight(r)(Image.open(x).convert('RGB'))) for x in a_p_n[1]],0).permute(1, 0, 2, 3)
        negative=torch.stack([self.transform(FlipLeftRight(r)(Image.open(x).convert('RGB'))) for x in a_p_n[2]],0).permute(1, 0, 2, 3)


        return anchor,positive,negative

    def __len__(self):
    	return len(self.data)

class PoseTripletTest(data.Dataset):
    def __init__(self):
        """
        主要目标： 获取训练三元组
        """
        data=[]
        f=open (opt.test_triplet_txt,'r')
        for line in f.readlines():
            images=line.split(',')
            data.append([images[:10],images[10:20],images[20:30]])

        self.data=data
        self.transform=T.Compose([
        	T.Resize((opt.image_size,opt.image_size)),
        	T.ToTensor(),
        	T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        """
        一次返回一个三元组的数据
        """
        a_p_n=self.data[index]
        

        # anchor=torch.cat([self.transform(Image.open(x).convert('RGB')).view(-1,3,300,300) for x in a_p_n[0]]).permute(1, 0, 2, 3)
        anchor=torch.stack([self.transform(Image.open(x).convert('RGB')) for x in a_p_n[0]],0).permute(1, 0, 2, 3) #0维扩充，permute维度变换位置
        positive=torch.stack([self.transform(Image.open(x).convert('RGB')) for x in a_p_n[1]],0).permute(1, 0, 2, 3)
        negative=torch.stack([self.transform(Image.open(x).convert('RGB')) for x in a_p_n[2]],0).permute(1, 0, 2, 3)

        return anchor,positive,negative

    def __len__(self):
    	return len(self.data)


class PoseTripletPredict(data.Dataset):
    def __init__(self):
        """
        主要目标： 获取训练三元组
        """
        data=[]
        f=open ("data/txt/pre.txt",'r')
        for line in f.readlines():
            images=line.split(',')
            data.append([images[:10],images[10:20],images[20:30]])

        self.data=data
        self.transform=T.Compose([
            T.Resize((opt.image_size,opt.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        """
        一次返回一个三元组的数据
        """
        a_p_n=self.data[index]


        # anchor=torch.cat([self.transform(Image.open(x).convert('RGB')).view(-1,3,300,300) for x in a_p_n[0]]).permute(1, 0, 2, 3)
        anchor=torch.stack([self.transform(Image.open(x).convert('RGB')) for x in a_p_n[0]],0).permute(1, 0, 2, 3) #0维扩充，permute维度变换位置
        positive=torch.stack([self.transform(Image.open(x).convert('RGB')) for x in a_p_n[1]],0).permute(1, 0, 2, 3)
        negative=torch.stack([self.transform(Image.open(x).convert('RGB')) for x in a_p_n[2]],0).permute(1, 0, 2, 3)

        return anchor,positive,negative

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    pass