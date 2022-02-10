import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = '/content/drive/MyDrive/colab/GenderClassification/New/output/model.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_OF_CLASSES = 2
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std)
])


def imshow(inp, title):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.figure (figsize = (12, 6))

    plt.imshow(inp)
    plt.title(title)
