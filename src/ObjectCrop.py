from FaceMaskClassificationUtils import DEVICE
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


class ObjectCrop:
    def __init__(self):
        # If required, create a face detection pipeline using MTCNN:
        self.mtcnn = MTCNN(keep_all=True, image_size=224, device=DEVICE)

    def objects_crop(self, in_path):
        """
        crop an image found in in_path and return the output
        :param in_path:
        :return: cropped images tensor array
        """
        img = Image.open(in_path)
        # Get cropped and pre whitened image tensor
        img_cropped = self.mtcnn(img)

        return img_cropped

    def bounding_box(self, img):
        """
        bounding box coordinates
        :param img: image in PIL format
        :return: np array coordinates
        """
        return self.mtcnn.detect(img)
