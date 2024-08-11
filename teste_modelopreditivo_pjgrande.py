# %%
# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%
# Configurar estilo do seaborn
sns.set(style="whitegrid")

# %%
delimiters = [',', ';', '\t', '|']
for delim in delimiters:
    try:
        data = pd.read_csv(r"C:\Users\bruno\Documents\Python\base_iguatu_ativos_pjgrande.CSV", delimiter=delim)
        print(f"Dados carregados com delimitador '{delim}':")
        print(data.head())
        if len(data.columns) > 1:
            break
    except Exception as e:
        print(f"Erro ao carregar dados com delimitador '{delim}': {e}")

# %%
# Exibir os nomes das colunas para verificação
print("Colunas do DataFrame original:")
print(data.columns)

# %%
# Verificar o número de colunas
if len(data.columns) != 4:
    raise ValueError(f"O número de colunas no DataFrame não corresponde ao esperado. Colunas encontradas: {len(data.columns)}")

# %%
# Renomear colunas para padronização
data.columns = ['data', 'segmento', 'subsegmento', 'quantidade']

# %%
# Verificar e transformar a coluna de data
data['data'] = pd.to_datetime(data['data'], format='%d/%m/%Y')
data['mês'] = data['data'].dt.month
data['ano'] = data['data'].dt.year

# %%
#Transformar variáveis categóricas em variáveis dummy (one-hot encoding)
data_encoded = pd.get_dummies(data, columns=['segmento', 'subsegmento'], drop_first=True)

# %%
# Exibir as primeiras linhas do DataFrame após a codificação
print("Dados codificados:")
print(data_encoded.head())

# %%
# Selecionar variáveis independentes e dependentes
X = data_encoded.drop(columns=['quantidade', 'data'])
y = data_encoded['quantidade']

# %%
# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# %%
# Avaliar o modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# %%
print(f'MSE: {mse}')
print(f'R²: {r2}')

# %%
# Criar DataFrame para os próximos meses (julho a dezembro de 2024)
future_data = pd.DataFrame({
    'mês': [7, 8, 9, 10, 11, 12],
    'ano': [2024]*6,
    'segmento': ['Pessoa Jurídica']*6,  # Ajuste conforme necessário
    'subsegmento': ['Grande']*6  # Ajuste conforme necessário
})

# %%
# Transformar variáveis categóricas em variáveis dummy
future_data_encoded = pd.get_dummies(future_data, columns=['segmento', 'subsegmento'], drop_first=True)

# %%
# Garantir que as colunas do conjunto futuro correspondam às colunas do conjunto de treinamento
future_data_encoded = future_data_encoded.reindex(columns=X.columns, fill_value=0)

# %%
# Fazer previsões
future_predictions = model.predict(future_data_encoded)

# %%
# Calcular a margem de erro com base no desvio padrão dos resíduos
residuals = y_test - y_pred
margin_of_error = np.std(residuals)

# %%
# Criar cenários
future_data['previsão_mediana'] = future_predictions
future_data['previsão_pessimista'] = future_predictions - margin_of_error
future_data['previsão_otimista'] = future_predictions + margin_of_error

# %%
# Exibir as previsões futuras
print("Previsões futuras:")
print(future_data)

# %%
# Concatenar dados históricos e previsões para plotagem
data['tipo'] = 'histórico'
future_data['tipo'] = 'previsão'

# %%
combined_data = pd.concat([
    data[['mês', 'ano', 'quantidade', 'tipo']],
    future_data[['mês', 'ano', 'previsão_mediana', 'previsão_pessimista', 'previsão_otimista', 'tipo']].melt(id_vars=['mês', 'ano', 'tipo'], value_vars=['previsão_mediana', 'previsão_pessimista', 'previsão_otimista'], var_name='cenário', value_name='quantidade')
])

# %%
# Plotar os resultados
plt.figure(figsize=(14, 7))

# %%
# Histórico por ano
sns.lineplot(data=combined_data[(combined_data['tipo'] == 'histórico')], x='mês', y='quantidade', hue='ano', marker='o', palette='tab10')

# %%
# Cenários futuros
sns.lineplot(data=combined_data[(combined_data['tipo'] == 'previsão') & (combined_data['cenário'] == 'previsão_mediana')], x='mês', y='quantidade', label='Previsão Mediana', marker='o')
sns.lineplot(data=combined_data[(combined_data['tipo'] == 'previsão') & (combined_data['cenário'] == 'previsão_pessimista')], x='mês', y='quantidade', label='Previsão Pessimista', marker='o', linestyle='--')
sns.lineplot(data=combined_data[(combined_data['tipo'] == 'previsão') & (combined_data['cenário'] == 'previsão_otimista')], x='mês', y='quantidade', label='Previsão Otimista', marker='o', linestyle='--')

# %%
# Adicionar anotações com os valores
for line in combined_data.itertuples():
    plt.annotate(f'{line.quantidade:.0f}', xy=(line.mês, line.quantidade), textcoords="offset points", xytext=(0,10), ha='center')

# %%
# Configurações adicionais do gráfico
plt.title('Quantidade de Clientes - Histórico e Previsões')
plt.xlabel('Mês')
plt.ylabel('Quantidade de Clientes - Iguatu')
plt.legend(title='Tipo', loc='upper left')  # Ajustar a posição da legenda
plt.xticks(rotation=45)
plt.grid(True)


# %%
# Mostrar o gráfico
plt.show()


