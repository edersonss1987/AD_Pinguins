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
