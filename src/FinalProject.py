import argparse
from PIL import Image
import time
import datetime

from FaceMaskClassificationUtils import test_transform, imshow, draw_bounding_boxes
from HumanDetection import classify_is_human, train_human_detection, load_human_model, retrain_human_model
from FaceMaskDetection import classify_mask_usage, train_face_mask_detection, load_face_mask_model
from NaturalImages import train_natural_image_detection
from ObjectCrop import ObjectCrop


class FinalProject:
    def __init__(self, human_model_path, mask_model_path):
        self.human_model = load_human_model(human_model_path)
        self.mask_model = load_face_mask_model(mask_model_path)
        self.oc = ObjectCrop()

    def single_image_classify(self, image_path):
        print("trying to classify image " + str(image_path))
        orig_img = Image.open(image_path)
        img = test_transform(Image.open(image_path))
        is_human = classify_is_human(img, self.human_model)
        if is_human is not True:
            return None, None
        # TODO: is there a human in picture?
        print("cropping image")
        imgs = self.oc.objects_crop(image_path)
        if imgs is None:
            return None, None
        mask_usages = []
        print("single_image_classify got " + str(len([img for img in imgs])) + " faces to classify")
        np_bbox, _ = self.oc.bounding_box(orig_img)
        for i, img in enumerate(imgs):
            mask_usage = classify_mask_usage(img, self.mask_model)
            imshow(img, str(i) + ": mask usage is " + str(mask_usage))
            mask_usages.append((img, mask_usage))
            box = np_bbox[i]
            orig_img = draw_bounding_boxes(orig_img, box, mask_usage)
        return mask_usages, orig_img


def train_models(human_model_path, mask_model_path,
                 human_data_path, mask_data_path,
                 natural_model_path, natural_data_path):
    print("train_models: Training all models")

    train_start = time.time()
    natural_model = train_natural_image_detection(natural_model_path, natural_data_path)
    train_end = time.time()
    print("FinalProject: training human model took " + str(datetime.timedelta(seconds=(train_end - train_start))))

    train_start = time.time()
    # train_human_detection(human_model_path, human_data_path)
    retrain_human_model(human_model_path, human_data_path, natural_model)
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
    parser.add_argument('-N', '--natural_model_path', help='Natural Images Model path', required=True)
    parser.add_argument('-c', '--natural_data_path', help='Natural Images Data path', required=False)
    parser.add_argument('-F', '--image_path', help='image to classify', required=True)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()
    print(args)

    if args.train:
        print("FinalProject: should train")
        train_models(args.human_model_path, args.mask_model_path, args.human_data_path, args.mask_data_path, args.natural_model_path, args.natural_data_path)
    else:
        print("FinalProject: shouldn't train")

    start = time.time()
    fp = FinalProject(args.human_model_path, args.mask_model_path)
    end = time.time()
    print("FinalProject: loading models took " + str(datetime.timedelta(seconds=(end - start))))

    start = time.time()
    fp.single_image_classify(args.image_path)
    end = time.time()
    print("FinalProject: testing image took " + str(datetime.timedelta(seconds=(end - start))))
