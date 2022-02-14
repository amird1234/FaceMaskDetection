import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

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


def cv2_image_to_pil_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil_image_to_cv2_image(img):
    """
    :param img: PIL image in RGB format
    :return: cv2 image in BGE format
    """
    pil_image = img.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    return open_cv_image[:, :, ::-1].copy()


def draw_bounding_boxes_helper(image_data, inference_results, color):
    """
    Draw bounding boxes on an image.
    :param image_data:  image data in numpy array format
    :param inference_results: inference results array off object (l,t,w,h)
    :param color: Bounding box color candidates, list of RGB tuples.
    :return: image in cv2 format
    """
    for res in inference_results:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['left']) + int(res['width'])
        bottom = int(res['top']) + int(res['height'])
        label = res['label']
        img_height, img_width, _ = image_data.shape
        thick = int((img_height + img_width) // 900)
        print("{} {} {} {}".format(left, top, right, bottom))
        cv2.rectangle(image_data,(left, top), (right, bottom), color, thick)
        cv2.putText(image_data, label, (left, top - 12), 0, 1e-3 * img_height, color, thick//3)
    return image_data


def draw_bounding_boxes(img, left, top, width, height, label):
    """
    Draw red bounding box around image
    :param img: image in PIL format
    :param left: left coordinate
    :param top: upper coordinate
    :param width: object width
    :param height: object height
    :param label: the label about it
    :return: the imagr in PIL format
    """
    imgcv = pil_image_to_cv2_image(img)
    color = (0, 255, 0)
    results = [
        {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "label": label
        }
    ]
    out_imgcv = draw_bounding_boxes_helper(imgcv, results, color)
    return cv2_image_to_pil_image(out_imgcv)
