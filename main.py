from tkinter import Tk, Button
from tkinter.filedialog import askdirectory
import pandas as pd
import os
from torchvision import transforms
from vgg import vgg_model
from torchvision.io import read_image
import torch

class_names=['aranha', 'borboleta', 'cachorro', 'cavalo', 'elefante', 'esquilo', 'galinha', 'gato', 'ovelha', 'vaca']

def main():
    """Selecione um diretorio contendo imagens para realizar a inferencia
    """

    def select_folder():
        folder = askdirectory()
        if folder:
            listar_images(folder)

    root = Tk()
    root.title("Selecione a pasta de imagens")
    btn_select_folder = Button(root, text='Selecionar a pasta', command=select_folder)
    btn_select_folder.pack(pady=10)
    root.mainloop()

def listar_images(folder):
    """Salva as imagens do diretorio escolhido que tenham terminacao jpg,jpeg, png

    Args:
        folder: Pasta selecionado que contenham as imagens
    """
    imagens = []
    for arquivo in os.listdir(folder):
        if arquivo.endswith(('.jpg', 'jpeg', 'png')):
            imagens.append(arquivo)

    if imagens:
        do_inference(imagens,folder)

def do_inference(imagens: list, folder):
    """realiza a inferencia com base na lista de imagens criadas pelo diretorio escolhido

    Args:
        imagens (list): Uma lista de imagens
    """
    # model, transform = vgg_model()
    # resize = Resize((224,224))
    # image_lida = read_image(folder+"/"+imagens[1])
    
    # passa cada imagem para realizar a inferencia
    inferencia = []
    for imagem in imagens:
        custom_image = read_image(str(folder+'/'+imagem)).type(torch.float32)
        custom_image = custom_image / 255

        custom_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        ])

        # Transform target image
        custom_image_transformed = custom_image_transform(custom_image)

        model_1, _ = vgg_model()
        model_1.eval()
        with torch.inference_mode():
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
        
            # Make a prediction on image with an extra dimension
            custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))
            # print(torch.max(custom_image_pred))

            # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
            custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)

            # Convert prediction probabilities -> prediction labels
            custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
            custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
            
            if torch.max(custom_image_pred) < 5:
                custom_image_pred_class = "Nenhum"
            inferencia.append(custom_image_pred_class)
            print(imagem, custom_image_pred_class)

    create_csv_file(imagens, inferencia)

def create_csv_file(imagens, inferencia):
    """Cria o arquivo csv de acordo com a imagem de entrada e a saida da inferencia

    Args:
        imagem : o arquivo imagem
        inferencia : o resultado da inferencia da rede
    """
    csv_file = {'Inferencia':[]}

    for i in range(len(imagens)):
        csv_file['Inferencia'].append(f'{imagens[i]};{inferencia[i]}')

    data_frame = pd.DataFrame(csv_file)
    data_frame.to_csv('Inferencias_das_Imagens',index=False,lineterminator=',')
    print("csv criado.")


if __name__ == '__main__':
    main()