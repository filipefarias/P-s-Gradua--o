from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib, os

filePath = pathlib.Path(__file__).parent.resolve()
os.chdir(filePath)

dfDadosClinicos = pd.read_csv('.\dados_clinicos.csv', sep= ';')
dfDadosPacientes = pd.read_csv('.\dados_pacientes.csv', sep= ';')
dfEstadosRegiao = pd.read_csv('.\estado_regiao.csv', sep= ';', encoding= 'latin-1')

dfDadosClinicos.drop_duplicates(inplace= True)
dfDadosClinicos.dropna(inplace= True)

# print(dfDadosClinicos.info())
# print(dfDadosPacientes.info())
# print(dfEstadosRegiao.info())

modaTrabalho = dfDadosPacientes['classe_trabalho'].mode()[0]
mediaFilhos = dfDadosPacientes['qtde_filhos'].mean().round()
dfDadosPacientes.fillna({'classe_trabalho': modaTrabalho, 'qtde_filhos': mediaFilhos}, inplace= True)
# print(dfDadosPacientes.info())

dfDadosCompletos = pd.merge(dfDadosPacientes, dfDadosClinicos, how= 'inner', on='id_cliente').sort_values(by='id_cliente')
dfDadosCompletos = pd.merge(dfDadosCompletos, dfEstadosRegiao, how='left', on='id_estado').sort_values(by='id_cliente')
dfDadosCompletos = dfDadosCompletos[['id_cliente', 'idade', 'genero', 'peso', 'colesterol', 'classe_trabalho', 'salario', 'escolaridade', 'estado_civil', 'raca', 'qtde_filhos', 'id_estado', 'sigla', 'estado', 'regiao', 'pais']].reset_index(drop=True)
# print(dfDadosCompletos)

# dfDadosCompletos.query('regiao == "Sudeste"').hist(column='idade', figsize=(15,7), bins= 20)
# plt.show()

def calcular_wcss(dados_cliente):
    wcss = []
    for k in range(1,11):
        kmeans = KMeans(n_clusters = k, random_state=42, init='k-means++', n_init= 10)
        kmeans.fit(X=dados_cliente)        
        wcss.append(kmeans.inertia_)
    return wcss

print(calcular_wcss(dfDadosCompletos[['peso', 'colesterol']]))

kmeansPacientes = KMeans(n_clusters=3, random_state=42, init='k-means++' ,n_init=10)
dfDadosCompletos['cluster'] = kmeansPacientes.fit_predict(dfDadosCompletos[['peso', 'colesterol']])
# print(dfDadosCompletos)
centroides_clusters = kmeansPacientes.cluster_centers_
# print(centroides_clusters)
# graficoKmeans = px.scatter(x = dfDadosCompletos['peso'], 
#                             y = dfDadosCompletos['colesterol'], 
#                             color= dfDadosCompletos['cluster'])
# fig = go.Figure(graficoKmeans)
# fig.show()

dfDadosCompletos.loc[dfDadosCompletos['cluster']==1,'nome_cluster'] = 'Alto Risco'
dfDadosCompletos.loc[dfDadosCompletos['cluster']==0,'nome_cluster'] = 'Baixo Risco'
dfDadosCompletos.loc[dfDadosCompletos['cluster']==2,'nome_cluster'] = 'Risco Moderado'
# print(dfDadosCompletos)
# print(dfDadosCompletos.query('escolaridade == "Mestrado"').groupby('estado')['escolaridade'].describe())

# sns.set_theme(style='darkgrid')
# plt.figure(figsize=(10, 6))
# scatter_plot = sns.scatterplot(data=dfDadosCompletos, 
#                                x='salario', 
#                                y='idade', 
#                                hue='genero', 
#                                palette={'Masculino': 'blue', 
#                                         'Feminino': 'orange'})
# scatter_plot.legend(loc='upper right', facecolor='white')
# plt.title('Relação entre salário e idade dos pacientes')
# plt.show()

# print(dfDadosCompletos.query('regiao == "Nordeste"')['salario'].mean())
# print(dfDadosCompletos.query('regiao"')['idade'])