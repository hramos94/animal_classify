from tkinter import Tk, Button
from tkinter.filedialog import askdirectory
from csv_module.csv_file_creator import CsvCreator
from inference_module.inference import Inference
from images_module.images_catcher import ImageCatcher
from model.vgg import VggModel


def main():
    """Selecione um diretorio contendo imagens para realizar a inferencia
    """
    class_names=['aranha', 'borboleta', 'cachorro', 'cavalo', 'elefante', 'esquilo', 'galinha', 'gato', 'ovelha', 'vaca']
    model = VggModel()
    model = model.vgg_model()
    def select_folder():
        folder = askdirectory()
        if folder:
            image_manipulation = ImageCatcher(folder=folder)
            images, folder = image_manipulation.listar_images()
            inferencia = Inference(class_names=class_names, folder=folder, model=model, images=images)
            image, inferencia = inferencia.do_inference()
            csv = CsvCreator(inferencia=inferencia,images=image)
            csv.create_csv_file()

    root = Tk()
    root.title("Selecione a pasta de imagens")
    btn_select_folder = Button(root, text='Selecionar a pasta', command=select_folder)
    btn_select_folder.pack(pady=10)
    root.mainloop()

    


if __name__ == '__main__':
    main()