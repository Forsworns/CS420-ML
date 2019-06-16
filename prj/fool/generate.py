import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os

net = models.vgg16(pretrained=True)

# Switch to evaluation mode
net.eval()

path = "../cafffe_lenet/fer2013/dataset/train/"

if not os.path.exists("foolimage"):
    os.makedirs("foolimage")

for _,dirs,_ in os.walk(path):
    for d in dirs:
        for _,_,files in os.walk(path+d):
            for f in files:
                im_orig = Image.open(path+d+"/"+f)
                im_orig = Image.merge('RGB',[im_orig,im_orig,im_orig])

                mean = [ 0.485, 0.456, 0.406 ]
                std = [ 0.229, 0.224, 0.225 ]


                # Remove the mean
                im = transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean,
                                        std = std)])(im_orig)

                r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

                def clip_tensor(A, minv, maxv):
                    A = torch.max(A, minv*torch.ones(A.shape))
                    A = torch.min(A, maxv*torch.ones(A.shape))
                    return A

                clip = lambda x: clip_tensor(x, 0, 255)

                tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                        transforms.Lambda(clip),
                                        transforms.ToPILImage(),
                                        transforms.CenterCrop(224)])

                processed = tf(pert_image.cpu()[0])
                processed.save("foolimage/"+f)


'''
plt.figure()
plt.subplot(1,2,1)
plt.imshow(im_orig)
plt.title("orig")
plt.subplot(1,2,2)
plt.imshow(tf(pert_image.cpu()[0]))
plt.title("DeepFool")
plt.show()
'''
