# animal_classify
Repositório criado para resolver um problema de classifacação de diferentes animais.

## Dataset
O dataset utilizado foi o Animals-10 (Kaggle: [https://www.kaggle.com/datasets/alessiocorrado99/animals10/code?datasetId=59760&sortBy=voteCount](https://www.kaggle.com/datasets/alessiocorrado99/animals10/code?datasetId=59760&sortBy=voteCount)).

Explorando um pouco do dataset podemos encontrar 10 classes :

| Classe    | Quantidade |
|-----------|------------|
| Cachorro  | 4863       |
| Elefante  | 1446       |
| Borboleta | 2112       |
| Galinha   | 3098       |
| Cavalo    | 2623       |
| Gato      | 1668       |
| Vaca      | 1866       |
| Ovelha    | 1820       |
| Esquilo   | 1862       |
| Aranha    | 4821       |

Podem ser melhor visualizadas com:

![Alt text](https://github.com/hramos94/animal_classify/blob/main/assets/image.png)


Como há um desbalanço entre as classes, se utilizou de uma estratégia de diminuir a quantidade das imagens das classes que continham mais imagens, ou seja, limitou-se o nº de imagens em 2000/classe.

# Rede Utilizada
## VGG16 pre treinada
A rede utilizada foi uma vgg16 pre treinada com os pesos da IMAGINENET1K_V1. Mas por que escolher uma rede pre treinada?
1) Devido ao curto intervalo de tempo, não seria tão interessante desenvolver uma rede do zero, já que existiria o risco de algo acontecer e o tempo que restaria para o resto do desenvolvimento poderia ser compremetido.
2) Uma rede pre treinada pode ser utilizada como um ponto de partida para executar tarefas customizadas, por meio do transfer learning, congela-se as camadas inferiores (convoluções) e somente as camadas mais superiores (classificação) são retreinadas com um dataset personalizado.
3) Permite uma melhor generalização das inferências, entre outras vantagens.

![Alt text](https://github.com/hramos94/animal_classify/blob/main/assets/image-1.png)

## Treinamento
O treinamento da rede com transfer learning foi feito no notebook do Kaggle, já que ele possui uma GPU estável, permitando melhores aproveitamentos dos valores de treino (batch, workers etc). 
O notebook se encontra no link [https://www.kaggle.com/code/heitorleiteramos/animal-classification-with-pretrained-vgg16](https://www.kaggle.com/code/heitorleiteramos/animal-classification-with-pretrained-vgg16)


# Como utilizar
1) baixe o arquivo zip "animals_classify.zip"
2) descompacte o zip na mesma pasta do animals_classify
3) execute o main
4) Clique no botão "Selecionar Diretório"
5) Selecione um diretório e "selecionar pasta"
6) Espere a rede fazer a inferencia das imagens dentro da pasta
7) Dentro da pasta "animals_classify" será criado um arquivo "Inferencias_das_Imagens.csv"
8) O arquivo csv contém as inferências feitas pela rede.
9) Para realizar uma nova inferência, selecione outra pasta e repita os passos 4-> 8
