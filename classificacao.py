from sklearn.naive_bayes import MultinomialNB

# Caracteristicas: eh gordo?, tem perna curta?, late?
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]

cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
marcacoes = [1, 1, 1, -1, -1, -1]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

marcacao_teste = [-1, 1, -1]

testes = [misterioso1, misterioso2, misterioso3]
resultado = modelo.predict(testes)

diferencas = resultado - marcacao_teste

acertos = [d for d in diferencas if d == 0]

total_acertos = len(acertos)

total_de_elementos = len(testes)

taxa_de_acertos = 100.0 * total_acertos / total_de_elementos

print(resultado)
print(diferencas)
print(taxa_de_acertos)
