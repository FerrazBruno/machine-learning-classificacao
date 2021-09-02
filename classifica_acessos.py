from sklearn.naive_bayes import MultinomialNB
from dados import carregar_acessos

# separando as caracteristicas das marcacoes
X, Y = carregar_acessos()

# separando os dados de treino
treino_dados = X[:90]
treino_marcacoes = Y[:90]

# separando os dados de teste
teste_dados = X[90:]
teste_marcacoes = Y[90:]

# criando o modelo
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

# treinando com os testes
resultado = modelo.predict(teste_dados)

# diferenca entre resultado e marcacoes
diferenca = resultado - teste_marcacoes

# obtendo o total de acertos e o total de elementos
acertos = [d for d in diferenca if d == 0]
total_acertos = len(acertos)
total_elementos = len(teste_dados)

# obtendo a taxa de acertos
taxa_de_acerto = (total_acertos / total_elementos) * 100

# printando os resultdos
print(taxa_de_acerto)
print(total_elementos)
