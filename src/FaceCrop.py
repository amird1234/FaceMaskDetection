from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def face_crop(in_path, out_path):
    from facenet_pytorch import MTCNN, InceptionResnetV1

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(keep_all=True,image_size=256, device=device)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    img = Image.open('/content/sample_data/twoWomen.jpeg')
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path='/content/sample_data/twoImage-cropped.jpg')

    # Calculate embedding (unsqueeze to add batch dimension)
    #img_embedding = resnet(img_cropped.unsqueeze(0))  # comes with keep_all=False
    img_embedding = resnet(img_cropped) # comes with keep_all=True
