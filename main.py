
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
import json
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

DATA_INICIO = '2010-01-01'
DATA_FIM = date.today().strftime('%Y-%m-%d')

st.title('Análise de ações')


#Criando a sidebar
st.sidebar.header('Escolha a ação')

n_dias = st.slider('Quantidade de dias de previsão',7, 365)

def pegar_dados_acoes():
    #path = 'C:/Users/matheus.bertuci/DS/pythonProject1/acoes.csv'
    path = 'acoes.csv'
    return pd.read_csv(path, delimiter=';')

df = pegar_dados_acoes()

acao = df['snome']
nome_acao_escolhida = st.sidebar.selectbox('Escolha uma ação:', acao)

df_acao = df[df['snome'] == nome_acao_escolhida]
acao_escolhida = df_acao.iloc[0]['sigla_acao']
acao_escolhida = acao_escolhida + '.SA'

@st.cache
def pegar_Valores_online(sigla_acao):
    df = yf.download(sigla_acao, DATA_INICIO, DATA_FIM)
    df.reset_index(inplace=True)
    return df

df_valores = pegar_Valores_online(acao_escolhida)

st.subheader('Tabela de valores - ' + nome_acao_escolhida)
st.write(df_valores.tail(10))

#Criar um gráfico

st.subheader('Gráfico de preços')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_valores['Date'],
                         y = df_valores['Close'],
                         name='Preço Fechamento',
                         line_color = 'black'))
fig.add_trace(go.Scatter(x=df_valores['Date'],
                         y = df_valores['Open'],
                         name='Preço Abertura',
                         line_color='blue'))

st.plotly_chart(fig)

#previsão




df_treino = df_valores[['Date', 'Close']]
df_treino['Date'] = df_treino['Date'].dt.tz_localize(None)

#renomear colunas
df_treino = df_treino.rename(columns={'Date': 'ds', 'Close': 'y'})

modelo = Prophet()
modelo.fit(df_treino)

futuro = modelo.make_future_dataframe(periods=n_dias, freq='B')
previsao = modelo.predict(futuro)

st.subheader('Previsão')
st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias))


