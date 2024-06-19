# Importações necessárias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error

# Lendo a base de dados

df = pd.read_csv('Cardetails.csv')

# Visualizando os 5 primeiros registros

df.head()

# Verificando os 5 últimos registros

df.tail()

# Informação sobre a tabela

df.info()

# Breve descrição da tabela

df.describe()

# Fazendo a soma dos valores nulos

df.isnull().sum()

# Distribuição de preço

plt.figure(figsize= (12, 8))
sns.histplot(df['selling_price'], kde= True, bins= 25, color='indigo') #bins= número de barras
plt.title('Quantidade de carros por preço', fontsize= 16, fontweight= 'bold')
plt.xlabel('Preço', fontsize= 14, fontweight= 'bold')
plt.ylabel('Contagem', fontsize= 14, fontweight= 'bold')
plt.show();

# Distribuição por anos

plt.figure(figsize=(10, 8)) # criando e definindo o tamanho da figura
df['year'].plot(kind='hist', bins=25, color= 'navy') # frequência da coluna 'year' e definindo cor
plt.title('Distribuição por anos (2010 a 2022)', fontsize= 16, fontweight='bold') # título
plt.grid(True, linestyle=':') # linha de divisão do gráfico
plt.ylabel('Frequência', fontweight= 'bold') # legenda y
plt.xlabel('Anos', fontweight= 'bold', fontsize= 12) # legenda x
plt.show(); # plotando o gráfico

