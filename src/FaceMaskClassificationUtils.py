import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFont, ImageDraw

MODEL_PATH = '/content/drive/MyDrive/colab/GenderClassification/New/output/model.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")

NUM_OF_FACEMASK_CLASSES = 3
NUM_OF_OBJECTS_CLASSES = 10

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


def imshow(inp, title, output_path=None):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.figure(figsize=(12, 6))

    plt.imshow(inp)
    plt.title(title)
    if output_path is not None:
        plt.savefig(output_path + "-" + title.replace(" ", ""))


def draw_bounding_boxes(img, box, label):
    """
    Draw red bounding box around image
    :param img: image in PIL format
    :param box: box of face
    :param label: the label about it
    :return: the image in PIL format
    """
    w, _ = img.size

    font = ImageFont.truetype("/usr/share/fonts/truetype/ezra/SILEOT.ttf", round(w/50))
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)

    draw.text((box[0], box[1]-round(w/35)), label, (255, 0, 0), font=font)

    return img_draw
