from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib, os

filePath = pathlib.Path(__file__).parent.resolve()
os.chdir(filePath)

dfDadosClinicos = pd.read_csv('.\dados_clinicos.csv', sep= ';')
# print(dfDadosClinicos.info())
mediaPeso = dfDadosClinicos['peso'].mean().round(2)
dfDadosClinicos.fillna({'peso': mediaPeso}, inplace=True)
# print(dfDadosClinicos.duplicated())
dfDadosClinicos.drop_duplicates(inplace=True)

xColesterolPeso = dfDadosClinicos.iloc[:, 1].values.reshape(-1,1)
yColesterolPeso = dfDadosClinicos.iloc[:, 2].values
# print(np.corrcoef(xDadosClinicos, yDadosClinicos))

previsaoColesterolPeso = LinearRegression()
previsaoColesterolPeso.fit(xColesterolPeso, yColesterolPeso)
previsaoRegressao = previsaoColesterolPeso.predict(xColesterolPeso)
# print(previsaoColesterolPeso.intercept_.round(2))
print(previsaoColesterolPeso.coef_)

sns.set_theme(style='darkgrid')
plt.figure(figsize=(10, 6))

grafico = sns.scatterplot(data=dfDadosClinicos, 
                          x='peso',
                          y='colesterol',
                          hue='genero',
                          palette={'Masculino' : 'blue',
                                   'Feminino' : 'orange'})

grafico.legend(loc='upper left', facecolor='white')
plt.title('Relação entre peso e colesterol')
sns.regplot(data=dfDadosClinicos, x='peso', y='colesterol', scatter=False, color='black' )
#plt.show()

# print(previsaoColesterolPeso.predict([[95]]))
# print(previsaoColesterolPeso.predict([[220]]))

# print(previsaoColesterolPeso.score(xColesterolPeso, yColesterolPeso))
print(mean_absolute_error(yColesterolPeso, previsaoRegressao))