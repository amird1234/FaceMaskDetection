from FaceMaskClassificationUtils import DEVICE
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def face_crop(in_path):
    """
    crop an image found in in_path and return the output
    :param in_path:
    :return: cropped images tensor array
    """
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(keep_all=True, image_size=256, device=DEVICE)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    img = Image.open(in_path)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)

    return img_cropped
