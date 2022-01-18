from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision
import sys
from os.path import dirname

print(torchvision.__path__)
sys.path.append(dirname(__file__))
print(torchvision.__path__)

for line in sys.path : print(line)

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=256)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image

img = Image.open('/Users/amirdahan/Desktop/Amir-Photo.jpg')
# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path='/Users/amirdahan/Desktop/Amir-Photo-Cropped.jpg')

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))
