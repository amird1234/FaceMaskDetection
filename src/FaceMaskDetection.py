import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from FaceMaskClassificationUtils import DEVICE, NUM_OF_FACEMASK_CLASSES, CPU_DEVICE, train_model, split_prepare_dataset, evaluation


mask_class_names_list = ['mask_weared_incorrect', 'with_mask', 'without_mask']

batch_size = 8
num_workers = 4

_num_epochs = 10
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

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_OF_FACEMASK_CLASSES)

    #print(model)

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


if __name__ == "__main__":
    train_face_mask_detection()
