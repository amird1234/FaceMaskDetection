import os
import argparse
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import models
from FaceMaskClassificationUtils import imshow, DEVICE, CPU_DEVICE, split_prepare_dataset, CNN, train_model, evaluation, test_transform


natural_image_class_names_list = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
NUM_OF_NATURAL_IMAGE_CLASSES = len(natural_image_class_names_list)

batch_size = 8
num_workers = 4

_num_epochs = 10
_train_size, _validation_size, _test_size = 0.7, 0.15, 0.15


def print_labeled_samples(dataloaders, class_names):
    inputs, classes = next(iter(dataloaders['test']))

    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


def classify_natural_image(image_path, model):
    """
    Determines if and how a human wears a mask
    :param img: the human image to classify
    :param model: the model to be used
    :return: Mask enum (MASK_WORN_INCORRECT, WITH_MASK, WITHOUT_MASK)
    """
    img = test_transform(Image.open(image_path))
    with torch.no_grad():
        img = img.to(CPU_DEVICE)
        model.to(CPU_DEVICE)
        class_prediction = torch.argmax(model(img.unsqueeze(0))).item()
        mask = natural_image_class_names_list[class_prediction]
        print("classify_mask_usage: this is a {}.".format(mask))
        return mask


def train_natural_image_detection(model_path, data_dir):
    """
    train the model
    :return: path to the model
    """
    dataloaders, total_batch_sizes, class_names = split_prepare_dataset(data_dir, num_workers, batch_size)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_OF_NATURAL_IMAGE_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, _num_epochs, dataloaders, total_batch_sizes, batch_size)
    model.eval()
    evaluation(dataloaders, model, class_names)

    # save model
    checkpoint = {'model': models.resnet18(),
                  'state_dict': model.state_dict()}
    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(checkpoint, model_path)

    return model


def load_natural_image_model(model_path, class_num):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = checkpoint['model']
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_num)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-M', '--model_path', help='Model Path', required=True)
    parser.add_argument('-D', '--data_path', help='Data Path', required=False)
    parser.add_argument('-F', '--image_path', help='image to classify', required=True)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()
    print(args)

    # train if necessary
    if args.train:
        train_natural_image_detection(args.model_path, args.data_path, NUM_OF_NATURAL_IMAGE_CLASSES)

    # load the model
    model = load_natural_image_model(args.model_path, NUM_OF_NATURAL_IMAGE_CLASSES)

    # classify the requested image
    classify_natural_image(args.image_path, model)
