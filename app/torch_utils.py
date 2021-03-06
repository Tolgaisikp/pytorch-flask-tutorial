import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

#1 load model
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes)
device = torch.device('cpu')

PATH = 'app/mnist_ffn.pth'
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

#2 image to tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1 ),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, ),(0.3081, ))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

#3 predict
def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    _, predicted = torch.max(outputs, dim = 1)
    return predicted