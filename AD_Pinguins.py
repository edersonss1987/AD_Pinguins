# **Análise de Dados: Aprendizado de Máquina, Regressão**

## Usaremos um dataset já disponivel no pacote seaborn, para analise de dados, tratamento dos vares missing e treinamento do nosso modelo preditivo
## 1\. Pinguins
#Vamos utilizar uma base de dados com informações sobre penguins. A idéia é prever o peso do penguin (**body_mass_g**) baseado em suas características físicas e geográficas (variáveis preditivas).
### Importação dos pacotes que serão úteis para as próximas etapas

# Importação das Bibliotecas
import os
from getpass import getpass
import numpy as np
import pandas as pd
# Bibliotecas de geradoras de graficos
import seaborn as sns
import matplotlib.pyplot as plt
# Biblioteca de hotEncoding
from sklearn.preprocessing import OneHotEncoder
# Biblioteca para fazer a PADRONIZAÇÃO DOS DADOS
from sklearn.preprocessing import StandardScaler
# Biblioteca para fazer a NORMALIZAÇÃO DOS DADOS
from sklearn.preprocessing import MinMaxScaler
# Biblioteca para separação dos dados em treino e teste
from sklearn.model_selection import train_test_split
# Biblioteca de calculo da média
from sklearn.metrics import mean_squared_error
# Biblioteca de regressão linear
from sklearn.linear_model import LinearRegression
# Biblioteca de calculo de acuracia
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


# importando o dataset
penguim = sns.load_dataset('penguins')

# Renomeando os nomes de colunas
penguim = penguim.rename(columns={'species': 'especie', 'island':'ilha','bill_length_mm':'comprimento_do_bico',\
                            'bill_depth_mm':'profundidade_do_bico','flipper_length_mm':'comprimento_da_nadadeira',\
                             'body_mass_g':'massa_corporal','sex':'sexo'})


# Verificando os valores após alteração dos nomes atribuidos as colunas, a fim de facilitar o entendimento
penguim.head()

# somando os valores vazios de cada coluna
penguim.isna().sum()

### **1.1. Analise exploratoria baseada em Gráficos**

 #- Atributos por sexo:
#*   1º - comprimento_do_bico x profundidade_do_bico: Pode se dizer que, tanto machos e fêmeas possuem fraca relação entre as duas variaveis, sendo que os.
#*   2º - comprimento_do_bico x comprimento_da_nadadeira: Podemos identificar que o comprimento de bicos e comprimento de nadadeiras tem correlação, sendo que na grande maioria os machos tem maior comprimento de bico.
#*   3º - comprimento_do_bico x massa_corporal: conforme dados, as variáveis tem correlação, porem as fêmeas são mais correlacionadas quando comparadas com as de machos
#*   4º - profundidade_do_bico x comprimento_da_nadadeira: Possuem correlação negativa, sendo que a mair correlacionada é de fêmeas.
#*   5º - profundidade_do_bico x massa_corporal: Fraca correlação negativa de ambos os sexos.
#*   6º - massa_corporal x comprimento_da_nadadeira: Forte correlação positiva de ambos os sexos
with sns.axes_style('whitegrid'):

  grafico = sns.pairplot(data=penguim, hue="sexo",kind='reg', palette="pastel", height=4.5)


 #- Atributos por espécie:
 #*   1º - comprimento_do_bico x profundidade_do_bico:
 #Adelie e Chinstrap possuem pouca correlação, sendo que Adele possuí caracteristicas me métricas menores, e Chinstrap caracteristicas de métricas intermediarias e Gentoo com uma certa correlação e maiores métricas.
#*   2º - comprimento_do_bico x comprimento_da_nadadeira:
#todas as espécies possuem pouca correlação, sendo que a espécie de Adele possuem menores métricas de comprimento de bicos e nadadeiras, já a de Gentoo possuí maiores nadadeiras e métricas intermediaria de comprimento de bico, os de espécie Chinstrap, possuí maiores métricas de nadadeiras e tamanho comprimento de bico menor,  quando comparado aos de Gento.
#*   3º - comprimento_do_bico x massa_corporal:
#Adelie tem menor métricas de peso e comprimento de bico, quando olhamos para as outras duas espécies, Chinstrap tem sua espécie de massa corporal intermediária e comprimento de bico relativamente parecido com as da espécie de  Gentoo.. Já a espécie Gentoo possuí a maior massa corporal e tamnho de bico relativamente parecido com as de Chinstrap
#*   4º - profundidade_do_bico x comprimento_da_nadadeira: A espécie de Gentoo tem forte correlção entre as duas variaveis, quanto maior a profundidade do bico, maior é o comprimento da nadadeira, já as outras duas espécies Adelie
# e Chinstrap tem o métricas parecidas e não possuem correlação..
#*   5º - profundidade_do_bico x massa_corporal:  Chinstrap Adele Possuem correlação de profundidade de bicos e massa parecidos, sendo as profundidades de bicos elavados quando comparado com as de Gento e também sua correlação parece ser fraca, já a correlação entre as duas variaves de Gento parece ser forte, quanto maior a profundidade do bico, maior sua massa corporal.
#*   6º - massa_corporal x comprimento_da_nadadeira:Todas as espécies possuem correlação, sendo que as de Adele possouem menor massa e menor comprimento de nadadeira, a de Gentoo com maiores massas e maiores comprimentos de nadadeiras e as de Chinstrap está com métricas intermediarias.
with sns.axes_style('whitegrid'):

  grafico = sns.pairplot(data=penguim, hue="especie", palette="pastel", kind='reg',height=5)


 #- Atributos por ilha:
 #*   1º - comprimento_do_bico x profundidade_do_bico: Todas as ilhas não possuem correlação entre as duas variaveis, porem podemos afirmar que os da ilha Torgersen tem menores comprimentos de bico, já a ilha de Biscoe tem menor profundidade de bicos e a ilha de Dream está com métricas intermediarias.
#*   2º - comprimento_do_bico x comprimento_da_nadadeira: As variaveis possuem uma certa correlação, mais olhando para o grafico podemos perceber que os penguind da ilha de Dream tem comprimento de nadadeiras menores, quando comparadas com as de Biscoe, já o comprimento de bicos entre as duas são relativamente parecidas, já as de Torgersen tem as menores métricas
#*   3º - comprimento_do_bico x massa_corporal: Analisando o gráfico, os penguins da ilha Dream possuem as moires massas corporais e o comprimento de bico é bem variado, ou seja, não possuí relação entre as duas variáveis, contrário aos penguins da ilha de Biscoe, que possuí correlação, conforme o comprimento de bico aumenta, aumenta tambem a massa corporal, já a ilha de Torgersen	não possuí correlação significativa e os seus penguins são de menores comprimento de bicos e menores massa corporal.
#*   4º - profundidade_do_bico x comprimento_da_nadadeira: Biscoe tem maiores comprimentos de nadadeiras já as menores medidas de profundidade de bico, já os penguins da ilha de Dream tem grande profundidade de bico, porem sua nadadeira está dentro de uma faixa pequena de medida, os da ilha de Torgersen tem suas métricas bastante variadas, sem correlação.
#*   5º - profundidade_do_bico x massa_corporal: Os penguins das ilhas de Torgersen e Dream possuem correlação entre as duas variáveis e suas métricas são bem parecidas, diferentemente dos penguins da ilha de Biscoe, onde seus valores estão bastante distribuídos no gráfico.
#*   6º - massa_corporal x comprimento_da_nadadeira: As três ilhas possuem forte correlação entre as variáveis, sendo que os da ilha Dream e de Torgersen tem menores valores de massa e compromento de nadadeiras, difente dos da ilha de Biscoe que tem moires massas e maiores comprimentos de nadadeiras.
with sns.axes_style('whitegrid'):

  grafico = sns.pairplot(data=penguim, hue="ilha", palette="pastel", kind='reg',height=5)


#verificando as variaveis numéricas, onde possamos identificar possiveis outlayers

variaveis_nun = [] # Lista vazia
for i in penguim.columns[2:-1].tolist(): # percorrendo as variaveis
  if penguim.dtypes[i] == 'int64' or penguim.dtypes[i] == 'float64': # somente quando float e int
    variaveis_nun.append(i) # atribuindo a lista vazia


plt.rcParams['figure.figsize'] = [8.00, 6.00] # atribuindo o tamanho da imagem
plt.rcParams['figure.autolayout'] = True
f, axes = plt.subplots(2,2, dpi=150) # informando que o plot são duas linhas e 3 colunas de imagens

linha = 0
coluna = 0

for i in variaveis_nun:
  sns.boxplot(data=penguim, y=i, ax=axes[linha][coluna])# percorrendo as variáveis do dataframe e atribuindo os valores de "i" no eixo y e os valores
  coluna += 1                                           # de linhas e colunas já definidos quando executado o for dentro das nossas vairaveis
  if coluna == 2:                                       # de
    linha += 1
    coluna = 0

plt.show()



## 2\. Dados
### **2.1. Valores nulos**
#A base de dados possui valores faltantes, por isso vamos aplicar o conceito de atribuição de valores pela média, 
#poderíamos excluir os dados faltante, pois não fariam diferença no nosso treinamento de modelo, 
#mas para enriquecimento, decidi usar o método "bfill"

# incluindo os valores nulos com o valor anterios a sua linha, isso iguala a quantidade de generos
penguim['sexo'].fillna(method='bfill', inplace=True)


# contando valore da coluna que será tratada
penguim.sexo.value_counts()

# resposta da questão 2.1
# conforme analise, resolvi preencher com a média os valores faltantes, já que ele se aproxima da mediana.

penguim.loc[penguim.comprimento_do_bico.isna(),'comprimento_do_bico'] = penguim.comprimento_do_bico.mean()
penguim.loc[penguim.profundidade_do_bico.isna(),'profundidade_do_bico'] = penguim.profundidade_do_bico .mean()
penguim.loc[penguim.comprimento_da_nadadeira.isna(),'comprimento_da_nadadeira'] = penguim.comprimento_da_nadadeira.mean()
penguim.loc[penguim.massa_corporal.isna(),'massa_corporal'] = penguim.massa_corporal.mean()


# somando os valores vazios de cada coluna, validando que os dados vazios foram eliminados
print('Dados tratados, não sobrando valores nulos \n\n',penguim.isna().sum())


### **2.2. Variáveis numéricas**
#Identificando as variáveis numéricas e criando uma nova coluna. 
#A nova coluna deve tem  o mesmo nome da coluna original acrescidade de "*_std*" 
#apenas para entender que essas já foram tratadas.
# resposta da questão 2.2
penguim['comprimento_do_bico_std'] = penguim['comprimento_do_bico']
penguim['profundidade_do_bico_std'] = penguim['profundidade_do_bico']
penguim['comprimento_da_nadadeira_std'] = penguim['comprimento_da_nadadeira']
penguim.head()


### **2.3. Variáveis categóricas**
#Identificando as variáveis numéricas e criando uma nova coluna. 
#A nova coluna deve tem  o mesmo nome da coluna original acrescidade de "*_nom*" ou "*_ord*" 
#apenas para entender que essas já foram tratadas..
# resposta da questão 2.3
# indicando as variáveis categóricas nominais
penguim['especie_nom'] = penguim['especie']
penguim['ilha_nom'] = penguim['ilha']
penguim['sexo_nom'] = penguim['sexo']


### **2.4. Limpeza**
#Descartaremos as colunas originais e vamos manter apenas a variável resposta e as variáveis preditivas com 
#o sufixo *_std*", *_nom*" e "*_ord*", com isso iniciaremos o OneHotEncoding e fit dos dados,
# e transformaremos em um outro dataframe novamente.
# resposta da questão 2.4

# usando a OneHotEncoder do pacote sklearn
ohe = OneHotEncoder(handle_unknown='ignore')


# agora vamos realizar o FIT dos penguim e em seguida vamos transformar os valores em colunas em forma de array, depois trasnformamos em dadosframe novamente
especie_nom = ohe.fit(penguim[['especie_nom']])
ohe.transform(penguim[['especie_nom']]).toarray()
especie_nom = pd.DataFrame(ohe.transform(penguim[['especie_nom']]).toarray(),columns=ohe.get_feature_names_out())


# agora vamos realizar o FIT dos dados e em seguida vamos transformar os valores em colunas em forma de array, depois trasnformamos em dadosframe novamente
ilha_nom = ohe.fit(penguim[['ilha_nom']])
ohe.transform(penguim[['ilha_nom']]).toarray()
ilha_nom= pd.DataFrame(ohe.transform(penguim[['ilha_nom']]).toarray(),columns=ohe.get_feature_names_out())


# agora vamos realizar o FIT dos penguim e em seguida vamos transformar os valores em colunas em forma de array, depois trasnformamos em DataFrame novamente
sexo_nom= ohe.fit(penguim[['sexo_nom']])
ohe.transform(penguim[['sexo_nom']]).toarray()
sexo_nom= pd.DataFrame(ohe.transform(penguim[['sexo_nom']]).toarray(),columns=ohe.get_feature_names_out())


# DataFrameenando os DataFrames tratados anteriormente
dados_nom = pd.concat([especie_nom,ilha_nom,sexo_nom], axis=1)
df = pd.concat([penguim, dados_nom], axis=1)

# descartando as colunas originais e mantendo apenas as de interesse, nas quais foram tratadas

df = df.drop(['especie','ilha','comprimento_do_bico','comprimento_da_nadadeira',\
              'sexo','especie_nom','ilha_nom','sexo_nom','profundidade_do_bico'],axis=1)


# reorganizando a disposição das colunas
df = df[['especie_nom_Adelie','especie_nom_Chinstrap','especie_nom_Gentoo','ilha_nom_Biscoe','ilha_nom_Dream',\
         'ilha_nom_Torgersen','comprimento_do_bico_std','profundidade_do_bico_std','comprimento_da_nadadeira_std',\
         'massa_corporal','sexo_nom_Female','sexo_nom_Male']]


df.head()


### **2.5. Treino/Teste**
### Modelo Padronizado
#Separação das variaveis preditoras e variável alvo do modelo padronizado, 
#em seguida vamos separar as variáveis para treinamento e teste.
# Definindo as variáveis preditoras e Variavel alvo dos dados PADRONIZADOS
df_padronizado_preditoras = df.drop('massa_corporal',axis=1)
df_padronizado_alvo = df['massa_corporal']

# Separando os dados de treino e teste
X_train_pad, X_test_pad,  Y_train_pad, Y_test_pad = train_test_split(df_padronizado_preditoras, df_padronizado_alvo, test_size=0.3, random_state=123)


# Fazendo a PADRONIZAÇÃO DOS DADOS
obj_padronizacao = StandardScaler().fit(X_train_pad)
obj_padronizacao = StandardScaler().fit(X_test_pad)
df_padronizado = obj_padronizacao.transform(X_train_pad)
df_padronizado = obj_padronizacao.transform(X_test_pad)
df_padronizado = pd.DataFrame(df_padronizado)
df_padronizado.columns = ['especie_nom_Adelie','especie_nom_Chinstrap','especie_nom_Gentoo','ilha_nom_Biscoe','ilha_nom_Dream',\
         'ilha_nom_Torgersen','comprimento_do_bico_std','profundidade_do_bico_std','comprimento_da_nadadeira_std',\
         'sexo_nom_Female','sexo_nom_Male']

df_padronizado.head()


## Modelo Normalizado
#Separação das variaveis preditoras e variável alvo do modelo normalizado, 
#em seguida vamos separar as variáveis para treinamento e teste.
# Definindo as variáveis preditoras e Variavel alvo
df_normalizado_preditoras = df.drop('massa_corporal',axis=1)
df_normalizado_alvo = df['massa_corporal']

# Separando os dados de treino e teste dos dados NORMALIZADOS
X_train_nor, X_test_nor,  Y_train_nor, Y_test_nor = train_test_split(df_normalizado_preditoras, df_normalizado_alvo, test_size=0.3, random_state=123)


# Fazendo a NORMALIZAÇÃO DOS DADOS
obj_normalizacao = MinMaxScaler().fit(X_train_nor)
obj_normalizacao = MinMaxScaler().fit(X_test_nor)
df_normalizado = obj_normalizacao.transform(X_train_pad)
df_normalizado = obj_normalizacao.transform(X_test_pad)

df_normalizado = pd.DataFrame(df_normalizado)
df_normalizado.columns = ['especie_nom_Adelie','especie_nom_Chinstrap','especie_nom_Gentoo','ilha_nom_Biscoe','ilha_nom_Dream',\
         'ilha_nom_Torgersen','comprimento_do_bico_std','profundidade_do_bico_std','comprimento_da_nadadeira_std',\
         'sexo_nom_Female','sexo_nom_Male']

df_normalizado.head()


## Modelo Sem Normalização ou Padronização
#Separação das variaveis preditoras e variável alvo do modelo com os dados sem a padronização ou a normalização, 
#em seguida vamos separar as variáveis para treinamento e teste.
# Definindo as variáveis preditoras e Variavel alvo
df_ori_preditoras = df.drop('massa_corporal',axis=1)
df_ori_alvo = df['massa_corporal']

# Separando os dados de treino e teste dos dados NORMALIZADOS
X_train_ori, X_test_ori,  Y_train_ori, Y_test_ori = train_test_split(df_ori_preditoras, df_ori_alvo, test_size=0.3, random_state=123)


## 3\. Modelagem
### **3.1. Treino**
#Treino dos modelos de **regressão linear** com os **dados de treino**. 
## Treinamento do Modelo Padronizado
modelo_pad = LinearRegression()
modelo_pad = modelo_pad.fit(X_train_pad,Y_train_pad)
modelo_pad.__dict__


## Treinamento do Modelo Normalizado
modelo_nor = LinearRegression()
modelo_nor = modelo_nor.fit(X_train_nor,Y_train_nor)
modelo_nor.__dict__


## Treinamento do Modelo Sem Normalização nem Padronização
modelo_ori = LinearRegression()
modelo_ori = modelo_ori.fit(X_train_ori,Y_train_ori)
modelo_ori.__dict__


### **3.2. Avaliação**
#Calculo do **RMSE** para o modelo de **regressão linear** treinado com os **dados de teste** 
#> **Nota:** Como referência, eu consegui um RMSE de approx. 293.7g, 7% da média do peso, 
#uma performance razoável.

# Avaliação do Modelo Padronizado
# Passando o objeto do modelo, para avaliar
preditor_teste_pad = modelo_pad.predict(X_test_pad)
alvo_teste_pad = Y_test_pad

# Calculando o RMSE(Root Mean Squared Error) do modelo, passando  Y_teste do modelo treinado e a variavel preditora
rmse_pad = np.sqrt(mean_squared_error( Y_test_pad,preditor_teste_pad))
print('O valor de RMSE do modelo PADRONIZADO foi:',rmse_pad)

# Calculando a acuracia do modelo, passando  Y_teste do modelo treinado e a variavel preditora
score_pad = r2_score(Y_test_pad,preditor_teste_pad)
print('A  acuracia do modelo é de:',score_pad )


## Avaliação do Modelo Normalizado
# Passando o objeto do modelo, para avaliar
preditor_teste_nor = modelo_nor.predict(X_test_nor)
alvo_teste_nor = Y_test_nor

# Calculando o RMSE(Root Mean Squared Error) do modelo, passando  Y_teste do modelo treinado e a variavel preditora
rmse_nor = np.sqrt(mean_squared_error( Y_test_nor,preditor_teste_nor))
print('O valor de RMSE do modelo NORMALIZADO foi:',rmse_nor)

# Calculando a acuracia do modelo, passando  Y_teste do modelo treinado e a variavel preditora
score_nor = r2_score(alvo_teste_nor,preditor_teste_nor)
print('A  acuracia do modelo é de:',score_nor)


## Avaliação do Modelo Sem Normalização ou Padronização
# Passando o objeto do modelo, para avaliar
preditor_teste_ori = modelo_ori.predict(X_test_ori)
alvo_teste_ori = Y_test_ori

# Calculando o RMSE(Root Mean Squared Error) do modelo, passando  Y_teste do modelo treinado e a variavel preditora
rmse_ori = np.sqrt(mean_squared_error( Y_test_ori,preditor_teste_ori))
print('O valor de RMSE do modelo Sem normalização ou padronização foi:', rmse_ori)

# Calculando a acuracia do modelo, passando  Y_teste do modelo treinado e a variavel preditora
score_ori = r2_score(alvo_teste_ori,preditor_teste_ori)
print('A  acuracia do modelo é de:',score_ori)


## 4\. Predição
### **4.1. Novo penguim**
#Qual o peso de um penguim com as seguintes características?

#- Vamos silumar a entrada de um novo valor, e conforme as predições, iremos aplicar os nossos modelos, para predizer o peso do nosso novo penguim

#| species	| island | bill_length_mm | bill_depth_mm | flipper_length_mm | sex |
#| --- | --- | --- | --- | --- | --- |
#| Adelie | Biscoe | 38.2 | 18.1 | 185.0 | Male |
#Criando um novo valor,no DataFrame, para que possamos aplicar a predição dos valores
novo_penguim = {'especie_nom_Adelie':[1],'especie_nom_Chinstrap':[0],'especie_nom_Gentoo':[0],'ilha_nom_Biscoe':[1],'ilha_nom_Dream':[0],\
         'ilha_nom_Torgersen':[0],'comprimento_do_bico_std':[38.2],'profundidade_do_bico_std':[18.1],'comprimento_da_nadadeira_std':[185.0],\
         'sexo_nom_Female':[0],'sexo_nom_Male':[1]}


# Reshape
x = np.array(novo_penguim).reshape(1, -1)
novo_penguim = pd.DataFrame(novo_penguim)
novo_penguim.head()


## Predição do modelo Padronizado
predicao_modelo_padronizado = modelo_pad.predict(novo_penguim)

## Predição do modelo Normalizado
predicao_modelo_normalizado = modelo_nor.predict(novo_penguim)

## Predição do modelo sobre os dados Originais, sem normalização ou padronização
predicao_modelo_original = modelo_ori.predict(novo_penguim)

print(f'VALOR PREDITO DOS DADOS PADRONIZADOS {predicao_modelo_padronizado}\
 \nVALOR PREDITO DOS DADOS NORMALIZADOS {predicao_modelo_normalizado}\
\nVALOR PREDITO DOS DADOS ORIGINAIS {predicao_modelo_original}')


# Fizemos um filtro apenas para validar se faria sentido os dados atribuidos a espécie, quando falamos da ilha de Biscoe
# Filtro dos dados atribuidos dos dados de importação inicial, apenas para conferência dos valores
ilha_de_Biscoe = penguim.loc[penguim['especie'] == 'Adelie']
ilha_de_Biscoe = penguim.loc[penguim['ilha'] == 'Biscoe']
ilha_de_Biscoe = ilha_de_Biscoe.drop(['comprimento_do_bico_std','profundidade_do_bico_std','comprimento_da_nadadeira_std','especie_nom','ilha_nom','sexo_nom'], axis=1)
ilha_de_Biscoe.head()








#__________________________________________________________________________________________________________________
# Analise de dados, voltada a anlise exploratória e tratamento das variaveis e OneHotEncoding
# importando as bibliotecas

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


# importando o dataset para tratamento dos dados

data = sns.load_dataset('penguins')

# renomeando os cabeçalhos do dataset para melhor compreensão do usuario

data = data.rename(columns={'species': 'espécie', 'island':'ilha','bill_length_mm':'comprimento_do_bico_mm',\
                            'bill_depth_mm':'profundidade_do_bico_mm','flipper_length_mm':'comprimento_da_nadadeira',\
                             'body_mass_g':'massa_corporal','sex':'sexo'})

# identificando as informações de numeros não nulos (nan) e tipo das colunas

data.info()

# resposta da questão 1.1
# somando os valores vazios de cada coluna

data.isna().sum()


# printando os valores nulos de cada coluna para identificar 

display(data[data.comprimento_do_bico_mm.isnull()])
print('_______________________________________________________________________________________________________________________')
print('_______________________________________________________________________________________________________________________')

display(data[data.profundidade_do_bico_mm.isnull()])
print('_______________________________________________________________________________________________________________________')
print('_______________________________________________________________________________________________________________________')

display(data[data.comprimento_da_nadadeira.isnull()])
print('_______________________________________________________________________________________________________________________')
print('_______________________________________________________________________________________________________________________')

display(data[data.massa_corporal.isnull()])
print('_______________________________________________________________________________________________________________________')
print('_______________________________________________________________________________________________________________________')

display(data[data.sexo.isnull()])

# resumo estatístico

data.describe()


# verificando o tamanho do nosso dataframe

data.shape

# identificando os valores que se repetem por cada coluna

data.nunique()

# olhando o tipo de dados de cada coluna

data.dtypes


# contando valore da coluna que será tratada

data.sexo.value_counts()

# somando os valores vazios de cada coluna

data.isna().sum()

# incluindo os valores nulos com o valor anterios a sua linha, isso iguala a quantidade de generos 

data['sexo'].fillna(method='bfill', inplace=True)


# contando valore da coluna que será tratada

data.sexo.value_counts()

# conforme analise, resolvi preencher com a média os valores faltantes, já que ele se aproxima da mediana.

data.loc[data.comprimento_do_bico_mm.isna(),'comprimento_do_bico_mm'] = data.comprimento_do_bico_mm.mean()
data.loc[data.profundidade_do_bico_mm.isna(),'profundidade_do_bico_mm'] = data.profundidade_do_bico_mm.mean()
data.loc[data.comprimento_da_nadadeira.isna(),'comprimento_da_nadadeira'] = data.comprimento_da_nadadeira.mean()
data.loc[data.massa_corporal.isna(),'massa_corporal'] = data.massa_corporal.mean()


# somando os valores vazios de cada coluna, validando que os dados vazios foram eliminados

print('Dados tratados, não sobrando valores nulos \n\n',data.isna().sum())

# resposta da questão 1.2
# identificando os valores numéricos

data.columns[data.dtypes == 'float64']


# criando e atribuindo nomes nas colunas que indicam variável numérica

data['comprimento_do_bico_mm_std'] = data['comprimento_do_bico_mm']
data['profundidade_do_bico_mm_std'] = data['profundidade_do_bico_mm']
data['comprimento_da_nadadeira_std'] = data['comprimento_da_nadadeira']
data['massa_corporal_std'] = data['massa_corporal']

data.head()


# resposta da questão 1.3
# identificando as variaveis categoricas nominais e ordinais

data.columns[data.dtypes == 'object']

# indicando as variáveis categóricas nominais

data['espécie_nom'] = data['espécie']
data['ilha_nom'] = data['ilha']
data['sexo_nom'] = data['sexo']



# usando a OneHotEncoder do pacote sklearn
ohe = OneHotEncoder(handle_unknown='ignore')


# agora vamos realizar o FIT dos data e em seguida vamos transformar os valores em colunas em forma de array, depois trasnformamos em dadosframe novamente
especie_nom = ohe.fit(data[['espécie_nom']])
ohe.transform(data[['espécie_nom']]).toarray()
especie_nom = pd.DataFrame(ohe.transform(data[['espécie_nom']]).toarray(),columns=ohe.get_feature_names_out())


# agora vamos realizar o FIT dos dados e em seguida vamos transformar os valores em colunas em forma de array, depois trasnformamos em dadosframe novamente
ilha_nom = ohe.fit(data[['ilha_nom']])
ohe.transform(data[['ilha_nom']]).toarray()
ilha_nom= pd.DataFrame(ohe.transform(data[['ilha_nom']]).toarray(),columns=ohe.get_feature_names_out())


# agora vamos realizar o FIT dos data e em seguida vamos transformar os valores em colunas em forma de array, depois trasnformamos em dataframe novamente
sexo_nom= ohe.fit(data[['sexo_nom']])
ohe.transform(data[['sexo_nom']]).toarray()
sexo_nom= pd.DataFrame(ohe.transform(data[['sexo_nom']]).toarray(),columns=ohe.get_feature_names_out())



# concatenando os dataframes tratados anteriormente
dados_nom = pd.concat([especie_nom,ilha_nom,sexo_nom], axis=1)
df = pd.concat([data, dados_nom], axis=1)
df.head()


# descartando as colunas originais e mantendo apenas as de interesse, nas quais foram tratadas

df = df.drop(['espécie','ilha','comprimento_do_bico_mm','comprimento_da_nadadeira','massa_corporal',\
              'sexo','espécie_nom','ilha_nom','sexo_nom','profundidade_do_bico_mm'],axis=1)


# reorganizando a disposição das colunas
df = df[['espécie_nom_Adelie','espécie_nom_Chinstrap','espécie_nom_Gentoo','ilha_nom_Biscoe','ilha_nom_Dream',\
         'ilha_nom_Torgersen','comprimento_do_bico_mm_std','profundidade_do_bico_mm_std','comprimento_da_nadadeira_std',\
         'massa_corporal_std','sexo_nom_Female','sexo_nom_Male']]


df.head()

##_______________________________________________________________________________________________________________________

