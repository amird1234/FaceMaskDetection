import argparse
import torch
from PIL import Image
from FaceMaskClassificationUtils import DEVICE, test_transform, NUM_OF_FACEMASK_CLASSES, NUM_OF_OBJECTS_CLASSES, imshow
from HumanDetection import classify_is_human, train_human_detection
from FaceMaskDetection import classify_mask_usage, train_face_mask_detection
from ObjectCrop import objects_crop
import torch.nn as nn
import time
import datetime


class FinalProject:
    def __init__(self, human_model_path, mask_model_path):
        checkpoint = torch.load(human_model_path, map_location=DEVICE)
        human_model = checkpoint['model']
        num_ftrs = human_model.fc.in_features
        human_model.fc = nn.Linear(num_ftrs, NUM_OF_OBJECTS_CLASSES)
        human_model.load_state_dict(checkpoint['state_dict'])
        human_model.eval()
        print("human model successfully loaded")

        checkpoint = torch.load(mask_model_path, map_location=DEVICE)
        mask_model = checkpoint['model']
        num_ftrs = mask_model.fc.in_features
        mask_model.fc = nn.Linear(num_ftrs, NUM_OF_FACEMASK_CLASSES)
        mask_model.load_state_dict(checkpoint['state_dict'])
        mask_model.eval()
        print("mask model successfully loaded")

        self.human_model = human_model
        self.mask_model = mask_model

    def single_image_classify(self, image_path):
        print("trying to classify image " + str(image_path))
        img = test_transform(Image.open(image_path))
        is_human = classify_is_human(img, self.human_model)
        if is_human is not True:
            return False
        print("cropping image")
        imgs = objects_crop(image_path)
        print("single_image_classify got " + str(len([img for img in imgs])) + " faces to classify")
        for i, img in enumerate(imgs):
            mask_usage = classify_mask_usage(img, self.mask_model)
            imshow(img, str(i) + ": mask usage is " + str(mask_usage))
        return mask_usage


def train_models(human_model_path, mask_model_path, human_data_path, mask_data_path):
    print("train_models: Training both models")
    train_start = time.time()
    train_human_detection(human_model_path, human_data_path)
    train_end = time.time()
    print("FinalProject: training human model took " + str(datetime.timedelta(seconds=(train_end - train_start))))

    train_start = time.time()
    train_face_mask_detection(mask_model_path, mask_data_path)
    train_end = time.time()
    print("FinalProject: training mask model took " + str(datetime.timedelta(seconds=(train_end - train_start))))


def load_models(human_model_path, mask_model_path):
    return FinalProject(human_model_path, mask_model_path)


if __name__ == '__main__':
    '''
    >>> python FinalProject.py -H '/content/drive/MyDrive/colab/Combined/out/CombinedModel.pth' -M '/content/drive/MyDrive/colab/FaceMaskDetection/out/MaskModel.pth' -F '/content/sample_data/twoWomen.jpeg'
    '''
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-H', '--human_model_path', help='Human Model Path', required=True)
    parser.add_argument('-a', '--human_data_path', help='Human Data Path', required=False)
    parser.add_argument('-M', '--mask_model_path', help='Face Mask Model path', required=True)
    parser.add_argument('-b', '--mask_data_path', help='Face Mask Data path', required=False)
    parser.add_argument('-F', '--image_path', help='image to classify', required=True)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()
    print(args)

    if args.train:
        print("FinalProject: should train")
        train_models(args.human_model_path, args.mask_model_path, args.human_data_path, args.mask_data_path)
    else:
        print("FinalProject: shouldn't train")

    start = time.time()
    fp = load_models(args.human_model_path, args.mask_model_path)
    end = time.time()
    print("FinalProject: loading models took " + str(datetime.timedelta(seconds=(end - start))))

    start = time.time()
    fp.single_image_classify(args.image_path)
    end = time.time()
    print("FinalProject: testing image took " + str(datetime.timedelta(seconds=(end - start))))
