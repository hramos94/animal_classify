import pandas as pd

class CsvCreator:
    def __init__(self, images, inferencia) -> None:
        self.images = images
        self.inferencia = inferencia

    def create_csv_file(self):
        """Cria o arquivo csv de acordo com a imagem de entrada e a saida da inferencia

        Args:
            imagem : o arquivo imagem
            inferencia : o resultado da inferencia da rede
        """
        csv_file = {'Inferencia':[]}

        for i in range(len(self.images)):
            csv_file['Inferencia'].append(f'{self.images[i]};{self.inferencia[i]}')

        data_frame = pd.DataFrame(csv_file['Inferencia'])
        data_frame.to_csv('Inferencias_das_Imagens',
                          index=False,
                          lineterminator=',',header=None,
                          sep=',')
        print("csv criado.")