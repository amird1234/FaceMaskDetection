import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchvision import models
from ObjectCrop import objects_crop

from FaceMaskClassificationUtils import imshow, std, mean, DEVICE, NUM_OF_OBJECTS_CLASSES, CPU_DEVICE

combined_class_names_list = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'mask_weared_incorrect', 'motorbike',
                             'with_mask', 'without_mask']

MODEL_PATH = '/content/drive/MyDrive/colab/Combined/out/CombinedModel.pth'

batch_size = 8
num_workers = 4
_data_dir = '/content/drive/MyDrive/colab/Combined/Dataset'

_num_epochs = 10
_train_size, _validation_size, _test_size = 0.7, 0.15, 0.15


def split(dataset, _train_size=0.7, _validation_size=0.15, _test_size=0.15):
    train_ds = int(_train_size * len(dataset))
    validation_ds = int(_validation_size * len(dataset))
    test_ds = len(dataset) - train_ds - validation_ds

    print("len: train valid test:" + str(train_ds) + " " + str(validation_ds) + " " + str(test_ds))
    train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [train_ds, validation_ds, test_ds])
    print("dataset is now " + str(len(train_data)))
    return train_data, validation_data, test_data


def split_prepare_dataset(data_dir):
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

    all_data = datasets.ImageFolder(root=data_dir, transform=train_transform)
    train_data, validation_data, test_data = split(all_data)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    validation_loader = torch.utils.data.DataLoader(validation_data,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'validation': validation_loader,
        'test': test_loader,
    }

    total_batch_sizes = {'train': len(train_loader), 'validation': len(validation_loader), 'test': len(test_loader)}

    print(train_data)
    print(test_data)

    class_names = all_data.classes

    print("Class Names: " + str(class_names))

    return dataloaders, total_batch_sizes, class_names


def print_labeled_samples(dataloaders, class_names):
    inputs, classes = next(iter(dataloaders['test']))

    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


class Amir(nn.Module):
    def __init__(self):
        super().__init__()
        self.nclasses = NUM_OF_OBJECTS_CLASSES
        self.in_channels = 3
        self.loss_func = nn.CrossEntropyLoss()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.nclasses))

    def forward(self, x):
        features = self.feature_extractor(x)
        class_scores = self.classifier(features)
        return class_scores


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, total_batch_sizes):
    model = model.to(DEVICE)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'validation']:

            if phase == 'train':

                scheduler.step()
                model.train()

            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            i = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / total_batch_sizes[phase]
            epoch_acc = running_corrects.double() / (total_batch_sizes[phase] * batch_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model


def evaluation(dataloaders, model, class_names):
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in dataloaders['test']:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {}%' \
              .format(100 * correct / total))

    with torch.no_grad():
        inputs, labels = iter(dataloaders['test']).next()
        inputs = inputs.to(DEVICE)

        inp = torchvision.utils.make_grid(inputs)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(len(inputs)):
            print("Acutal label", class_names[np.array(labels)[j]])
            inp = inputs.data[j]
            imshow(inp, 'predicted:' + class_names[preds[j]])


def classify_is_there_human_in_image(image_path, human_model):
    """
    return True if there's a human in the image
    :param image_path: the path of the image
    :param human_model: the model that determines if an object is human
    :return:
    """
    imgs = objects_crop(image_path)
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
        print(CPU_DEVICE)
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


def train_human_detection(model_path=MODEL_PATH, data_dir=_data_dir):
    """
    train the model
    :return: path to the model
    """
    dataloaders, total_batch_sizes, class_names = split_prepare_dataset(data_dir)
    print_labeled_samples(dataloaders, class_names)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_OF_OBJECTS_CLASSES)

    #print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, _num_epochs, dataloaders, total_batch_sizes)
    model.eval()
    evaluation(dataloaders, model, class_names)

    # save model
    checkpoint = {'model': models.resnet18(),
                  'state_dict': model.state_dict()}
    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(checkpoint, model_path)

    return model_path


if __name__ == "__main__":
    path = train_human_detection()
    print("model saved to " + path)
