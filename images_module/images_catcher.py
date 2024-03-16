import os

class ImageCatcher:
    """classe realacionada com a manipulacao das imagens do diretorio (folder)
    """
    def __init__(self, folder) -> None:
        self.folder = folder
        self.images = []
    
    def listar_images(self):
        """Salva as imagens do diretorio escolhido que tenham terminacao jpg,jpeg, png

        Args:
            folder: Pasta selecionada que contenham as imagens
        """
        self.images = []
        #  lista os arquivos presentes no diretorio e adiciona a lista imagens
        #  somente os arq que terminem em jpg,jpeg e png
        for image_file in os.listdir(self.folder):
            if image_file.endswith(('.jpg', 'jpeg', 'png')):
                self.images.append(image_file)

        if self.images:
            return (self.images, self.folder)
