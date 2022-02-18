import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from FaceMaskClassificationUtils import DEVICE, CPU_DEVICE, train_model, split_prepare_dataset, evaluation

mask_class_names_list = ['mask_weared_incorrect', 'with_mask', 'without_mask']
NUM_OF_FACEMASK_CLASSES = len(mask_class_names_list)

batch_size = 8
num_workers = 4

_num_epochs = 15
_train_size, _validation_size, _test_size = 0.7, 0.15, 0.15


def classify_mask_usage(img, model):
    """
    Determines if and how a human wears a mask
    :param img: the human image to classify
    :param model: the model to be used
    :return: Mask enum (MASK_WORN_INCORRECT, WITH_MASK, WITHOUT_MASK)
    """
    with torch.no_grad():
        img = img.to(CPU_DEVICE)
        model.to(CPU_DEVICE)
        class_prediction = torch.argmax(model(img.unsqueeze(0))).item()
        mask = mask_class_names_list[class_prediction]
        print("classify_mask_usage: this is a {}.".format(mask))
        return mask


def train_face_mask_detection(model_path, data_dir):
    """
    train the model
    :return: path to the model
    """
    dataloaders, total_batch_sizes, class_names = split_prepare_dataset(data_dir, num_workers, batch_size)
    # print_labeled_samples(dataloaders, class_names)

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_OF_FACEMASK_CLASSES)

    #print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, _num_epochs, dataloaders, total_batch_sizes, batch_size, "FaceMask")
    model.eval()
    evaluation(dataloaders, model, class_names)

    # save model
    checkpoint = {'model': models.resnet18(),
                  'state_dict': model.state_dict()}
    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(checkpoint, model_path)

def load_face_mask_model(mask_model_path):
    checkpoint = torch.load(mask_model_path, map_location=DEVICE)
    mask_model = checkpoint['model']
    num_ftrs = mask_model.fc.in_features
    mask_model.fc = nn.Linear(num_ftrs, NUM_OF_FACEMASK_CLASSES)
    mask_model.load_state_dict(checkpoint['state_dict'])
    mask_model.eval()
    print("mask model successfully loaded")
    return mask_model


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
        train_face_mask_detection(args.model_path, args.data_path)

    # load the model
    model = load_face_mask_model(args.model_path)

    # classify the requested image
    classify_mask_usage(args.image_path, model)
