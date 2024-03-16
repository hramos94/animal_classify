from torchvision import models
import torch.nn as nn
import torch
from torchsummary import summary

class VggModel:
    """Classe para instaciar o modelo vgg16
    """
    def __init__(self) -> None:
        self.model = models.vgg16()

    def vgg_model(self):
        """Carregar o modelo junto com o load
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.classifier[-1] = nn.Linear(4096, 10)
        #  carrega o modelo com o modelo salvo pelo treinamento no kaggle
        self.model.load_state_dict(torch.load('animal_classify/animals_classify.pth'))

        return self.model

if __name__ == '__main__':
    model = VggModel()
    model = model.vgg_model()