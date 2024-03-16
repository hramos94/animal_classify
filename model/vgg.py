from torchvision import models
import torch.nn as nn
import torch
from torchsummary import summary

class VggModel:
    def __init__(self) -> None:
        self.model = models.vgg16()

    def vgg_model(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.classifier[-1] = nn.Linear(4096, 10)
        self.model.load_state_dict(torch.load('animal_classify/animals_classify.pth'))

        # print(summary(model=model, input_size=(3, 224, 224)))

        # weigths = models.VGG16_Weights.DEFAULT
        # transform = weigths.transforms()

        return self.model
        #transform

if __name__ == '__main__':
    model = VggModel()
    model = model.vgg_model()