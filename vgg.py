from torchvision import models
import torch.nn as nn
import torch
from torchsummary import summary

def vgg_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.vgg16().to(device)
    model.classifier[-1] = nn.Linear(4096, 10)
    model.load_state_dict(torch.load('animal_classify/animals_classify.pth'))

    # print(summary(model=model, input_size=(3, 224, 224)))

    weigths = models.VGG16_Weights.DEFAULT
    transform = weigths.transforms()

    return model, transform

if __name__ == '__main__':
    vgg_model()