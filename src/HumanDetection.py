import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import models

from FaceMaskClassificationUtils import DEVICE, CPU_DEVICE, evaluation, split_prepare_dataset, train_model

combined_class_names_list = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'mask_weared_incorrect', 'motorbike',
                             'with_mask', 'without_mask']
NUM_OF_OBJECTS_CLASSES = len(combined_class_names_list)

batch_size = 8
num_workers = 4

_num_epochs = 10
_train_size, _validation_size, _test_size = 0.7, 0.15, 0.15


def classify_is_there_human_in_image(image_path, human_model, oc_model):
    """
    return True if there's a human in the image
    :param oc_model: object crop model
    :param image_path: the path of the image
    :param human_model: the model that determines if an object is human
    :return:
    """
    imgs = oc_model.objects_crop(image_path)
    print("classify_is_there_human_in_image")
    for i, img in enumerate(imgs):
        if classify_is_human(img, human_model):
            return True
    return False


def classify_is_human(img, model):
    """
    Determines if a specific picture has human image in it
    :param img: the image to classify
    :param model: the model to be used
    :return: Boolean (True: human, False: not human)
    """
    print("classify if it is human")
    with torch.no_grad():
        img = img.to(CPU_DEVICE)
        model.to(CPU_DEVICE)
        class_prediction = torch.argmax(model(img.unsqueeze(0))).item()
        species = combined_class_names_list[class_prediction]
        print("classify_is_human: this is a {}.".format(species))
        if species == 'mask_weared_incorrect' or \
                species == 'with_mask' or \
                species == 'without_mask':
            print("classify_is_human: it is human")
            return True
        else:
            print("classify_is_human: it is NOT human")
            return False


def train_human_detection(model_path, data_dir, start_with_model=None):
    """
    train the model
    :return: path to the model
    """
    dataloaders, total_batch_sizes, class_names = split_prepare_dataset(data_dir, num_workers, batch_size)
    # print_labeled_samples(dataloaders, class_names)

    if start_with_model is None:
        print("Not Retraining Natural Model")
        model = models.resnet18(pretrained=False)
    else:
        print("Retraining Natural model --> Human Model")
        model = start_with_model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_OF_OBJECTS_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, _num_epochs, dataloaders, total_batch_sizes, batch_size, "Human")
    model.eval()
    evaluation(dataloaders, model, class_names)

    # save model
    checkpoint = {'model': models.resnet18(),
                  'state_dict': model.state_dict()}
    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(checkpoint, model_path)

    return model_path


def load_human_model(human_model_path):
        checkpoint = torch.load(human_model_path, map_location=DEVICE)
        human_model = checkpoint['model']
        num_ftrs = human_model.fc.in_features
        human_model.fc = nn.Linear(num_ftrs, NUM_OF_OBJECTS_CLASSES)
        human_model.load_state_dict(checkpoint['state_dict'])
        human_model.eval()
        print("human model successfully loaded")
        return human_model


def retrain_human_model(model_path, data_dir, natural_model):
    train_human_detection(model_path, data_dir, start_with_model=natural_model)


if __name__ == "__main__":
    path = train_human_detection()
    print("model saved to " + path)
