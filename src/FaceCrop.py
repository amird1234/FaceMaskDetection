from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def face_crop(in_path, out_path):
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    img = Image.open(in_path)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=out_path)

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))

    # Or, if using for VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))


def main():
    face_crop('/tmp/Amir-Photo.jpg', '/tmp/Amir-Photo-Cropped.jpg')


if __name__ == "__main__":
    main()
