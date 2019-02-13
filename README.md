# desafioSpi2019
Desafio da seleção de Bolsista SPI 2019

Crie uma aplicação Web (usando uma tecnologia a sua escolha) para que dadas como entrada 14 características, compute e exiba o preço de uma casa com base em um modelo preditivo criado a partir de um algoritmo de regressão. 
O modelo deve ser treinado usando os dados obtidos no seguinte link. 
Refere-se a valores habitacionais nos bairros de Boston.
Deve-se usar 80% do dataset para treino e 20% para teste. 
Você deve comparar algoritmos de regressão e usar a métrica RMSE para justificar a escolha do algoritmo de regressão usado na aplicação Web. 
Essa comparação deve ser exibida através de um gráfico na aplicação Web.

O modelo preditivo é implementado utilizando a biblioteca scikit-learn, para aprendizagem de maquina, inicialmente no ambiente Jupyter.
Para aplicação web é usado o framework Flask para facilitar integração do modelo implementado em Python. 
Foi utilizado a biblioteca Chart js para plotagem do gráfico comparando alguns classficadores de regressão.
