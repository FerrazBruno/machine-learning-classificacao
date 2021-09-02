import pandas as pd
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

# teste inicial: home, buscas, logado => comprou
# home, buscas
# home, logado
# busca, logado
# busca: 85,71% (7 testes)

#  Importando o arquivo e separando as caracteristicas das marcacoes
df = pd.read_csv('busca2.csv')
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

#  Dummie classifier
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

# Treinos e testes
porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1
tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = int(porcentagem_de_teste * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

fim_de_treino = tamanho_de_treino + tamanho_de_teste
treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

teste_dados = X[tamanho_de_treino:fim_de_treino]
teste_marcacoes = Y[tamanho_de_treino:fim_de_treino]

validacao_dados = X[fim_de_treino:]
validacao_marcacoes = Y[fim_de_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    #  Criando o modelo
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    acertos = resultado == teste_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = total_de_acertos / total_de_elementos * 100
    msg = f'Taxa de acerto do algoritmo {nome}: {taxa_de_acerto}'

    print(msg)
    return taxa_de_acerto


def teste_real(modelo, validacao_dados, validacao_marcacoes):

    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = total_de_acertos / total_de_elementos * 100

    msg = f'Taxa de acerto do vencedor entre esses dois algoritmos no mundo real: {taxa_de_acerto}'
    print(msg)

modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict('MultinomialNB', modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict('AdaBoostClassifier', modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if (resultadoMultinomial > resultadoAdaBoost):
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoost

teste_real(vencedor, validacao_dados, validacao_marcacoes)

# A eficacia do algoritimo que chuta tudo um unico valor
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_acerto_base = 100 * acerto_base / len(validacao_marcacoes)
print(f'Taxa de acerto base: {taxa_acerto_base}')

total_de_elementos = len(validacao_dados)
print(total_de_elementos)

