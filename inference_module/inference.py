import os
from torchvision import transforms
from model.vgg import VggModel
from torchvision.io import read_image
import torch

class Inference:
    """Classe relacionada a inferencia das imagens
    """
    def __init__(self, class_names, folder, images, model) -> None:
        self.class_names = class_names
        self.folder = folder
        self.images = images
        self.model = model
    
    def do_inference(self):
        """realiza a inferencia com base na lista de imagens criadas pelo diretorio escolhido

        Args:
            imagens (list): Uma lista de imagens
        """
        
        # passa cada imagem para realizar a inferencia
        inferencia = []
        for imagem in self.images:
            custom_image = read_image(str(self.folder+'/'+imagem)).type(torch.float32)
            custom_image = custom_image / 255

            custom_image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            ])

            # transforma a imagem de entrada para se alinhar com a rede (224,224)
            custom_image_transformed = custom_image_transform(custom_image)
            # self.model
            # self.model, _ = vgg_model()
            self.model.eval()
            with torch.inference_mode():
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
            
                # Make a prediction on image with an extra dimension
                custom_image_pred = self.model(custom_image_transformed.unsqueeze(dim=0).to(device))
                # print(torch.max(custom_image_pred))

                # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
                custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)

                # Convert prediction probabilities -> prediction labels
                custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
                custom_image_pred_class = self.class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
                
                if torch.max(custom_image_pred) < 5:
                    custom_image_pred_class = "Nenhum"
                inferencia.append(custom_image_pred_class)
                print(imagem, custom_image_pred_class)
        return (self.images, inferencia)