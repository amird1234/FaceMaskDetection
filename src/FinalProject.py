import argparse
import torch
from PIL import Image
from FaceMaskClassificationUtils import DEVICE, test_transform, NUM_OF_FACEMASK_CLASSES, NUM_OF_OBJECTS_CLASSES, imshow
from HumanDetection import classify_is_human
from FaceMaskDetection import classify_mask_usage
from FaceCrop import faces_crop
import torch.nn as nn


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
        imgs = faces_crop(image_path)
        print("single_image_classify got " + str(len([img for img in imgs])) + " faces to classify")
        for i, img in enumerate(imgs):
            mask_usage = classify_mask_usage(img, self.mask_model)
            imshow(img, str(i) + ": mask usage is " + str(mask_usage))
        # return mask_usage


def load_models(human_model_path, mask_model_path):
    return FinalProject(human_model_path, mask_model_path)


if __name__ == '__main__':
    '''
    >>>python FinalProject.py -H '/content/drive/MyDrive/colab/Combined/out/CombinedModel.pth' -M '/content/drive/MyDrive/colab/FaceMaskDetection/out/MaskModel.pth' -F '/content/sample_data/twoWomen.jpeg'
    '''
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-H', '--human_model_path', help='Human Model Path', required=True)
    parser.add_argument('-M', '--mask_model_path', help='Face Mask Model path', required=True)
    parser.add_argument('-F', '--image_path', help='image to classify', required=False)
    args = parser.parse_args()
    print(args)
    fp = load_models(args.human_model_path, args.mask_model_path)
    fp.single_image_classify(args.image_path)
