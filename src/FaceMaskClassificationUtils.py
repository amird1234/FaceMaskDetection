import copy
import os

import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw

import torch
import numpy as np
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn

MODEL_PATH = '/content/drive/MyDrive/colab/GenderClassification/New/output/model.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")

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


def split(dataset, _train_size=0.7, _validation_size=0.15, _test_size=0.15):
    train_ds = int(_train_size * len(dataset))
    validation_ds = int(_validation_size * len(dataset))
    test_ds = len(dataset) - train_ds - validation_ds

    print("len: train valid test:" + str(train_ds) + " " + str(validation_ds) + " " + str(test_ds))
    train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [train_ds, validation_ds, test_ds])
    print("dataset is now " + str(len(train_data)))
    return train_data, validation_data, test_data


def split_prepare_dataset(data_dir, num_workers, batch_size):
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


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, total_batch_sizes, batch_size, title):
    model = model.to(DEVICE)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    measurements = {'Loss': {'train': [], 'validation': []}, 'Accuracy': {'train': [], 'validation': []}}

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

            # save measurements for later
            measurements['Accuracy'][phase].append(epoch_acc)
            measurements['Loss'][phase].append(epoch_loss)


            if phase == 'validation' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))
    reports = os.path.join(os.path.abspath(''), "Reports")

    print('saving report to ' + reports)
    for m in measurements:
        for phase in ['train', 'validation']:
            plt.clf()
            plt.figure(figsize=(6, 6))
            plt.plot([x for x in range(len(measurements[m][phase]))], measurements[m][phase], '-')
            if m == 'Accuracy':
                plt.plot(best_epoch, measurements[m][phase][best_epoch].item(), 'g*')
            else:
                plt.plot(best_epoch, measurements[m][phase][best_epoch], 'g*')
            plt.title(phase + " " + m)
            #plt.show()
            plt.savefig(os.path.join(reports, title + '-' + phase + '-' + m + '.png'))
            plt.clf()

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

        # inp = torchvision.utils.make_grid(inputs)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(len(inputs)):
            print("Acutal label", class_names[np.array(labels)[j]])
            inp = inputs.data[j]
            imshow(inp, 'predicted:' + class_names[preds[j]])


def print_labeled_samples(dataloaders, class_names):
    inputs, classes = next(iter(dataloaders['test']))

    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


class CNN(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.nclasses = class_num
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

