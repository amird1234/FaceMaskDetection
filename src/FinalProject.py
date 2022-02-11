import argparse
import torch
from PIL import Image
from FaceMaskClassificationUtils import DEVICE, test_transform
from HumanDetection import classify_is_human
from FaceMaskDetection import classify_mask_usage
from FaceCrop import faces_crop


class FinalProject:
    def __init__(self, human_model_path, mask_model_path):
        checkpoint = torch.load(human_model_path, map_location=DEVICE)
        human_model = checkpoint['model']
        human_model.load_state_dict(checkpoint['state_dict'])
        human_model.eval()

        checkpoint = torch.load(mask_model_path, map_location=DEVICE)
        mask_model = checkpoint['model']
        mask_model.load_state_dict(checkpoint['state_dict'])
        mask_model.eval()

        self.human_model = human_model
        self.mask_model = mask_model

    def single_image_classify(self, image_path):
        img = test_transform(Image.open(image_path))
        is_human = classify_is_human(img, self.human_model)
        if is_human is True:
            return False
        mask_usage = classify_mask_usage(img, self.mask_model)
        print("mask usage is " + str(mask_usage))
        return mask_usage


def load_models(human_model_path, mask_model_path):
    return FinalProject(human_model_path, mask_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-H', '--human_model_path', help='Human Model Path', required=True)
    parser.add_argument('-M', '--mask_model_path', help='Face Mask Model path', required=True)
    parser.add_argument('-F', '--image_path', help='image to classify', required=False)
    args = parser.parse_args()
    print(args)
    fp = load_models(args.human_model_path, args.mask_model_path)
    fp.single_image_classify()

