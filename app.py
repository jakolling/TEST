# Football Analytics App - Versão Completa e Corrigida
# Sistema de Análise de Jogadores v3.0

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from io import BytesIO
import kaleido

# =============================================
# Configuração Inicial
# =============================================
st.set_page_config(
    page_title='Football Analytics',
    layout='wide',
    page_icon="⚽"
)

# Cabeçalho com logo
coluna_esquerda, coluna_central, coluna_direita = st.columns([1, 3, 1])
with coluna_central:
    st.image('vif_logo.png', width=400)

st.title('Departamento de Scouting Técnico')
st.subheader('Painel de Análise de Futebol')
st.caption("Desenvolvido por João Alberto Kolling | Sistema de Análise de Jogadores v3.0")

# Guia do Usuário
with st.expander("📘 Guia do Usuário & Instruções", expanded=False):
    st.markdown("""
    **⚠️ Pré-requisitos:**  
    1. Instale as dependências:  
    `pip install kaleido==0.2.1.post1 xlsxwriter`  
    2. Os dados devem conter colunas: Jogador, Idade, Posição, Métricas, Time  
    
    **Funcionalidades Principais:**  
    - Comparação entre jogadores com gráficos de radar e barras  
    - Análise de correlação entre métricas  
    - Sistema de filtros avançados  
    - Exportação profissional de gráficos (300 DPI)  
    - Índice composto por PCA  
    """)

# =============================================
# Funções Principais
# =============================================
if 'metadados_arquivo' not in st.session_state:
    st.session_state.metadados_arquivo = {}

def carregar_e_limpar_dados(arquivos_enviados):
    lista_dataframes = []
    for arquivo in arquivos_enviados[:15]:
        dataframe = pd.read_excel(arquivo)
        dataframe.dropna(how="all", inplace=True)
        dataframe = dataframe.loc[:, dataframe.columns.notnull()]
        dataframe.columns = [str(coluna).strip() for coluna in dataframe.columns]
        
        metadados = st.session_state.metadados_arquivo[arquivo.name]
        dataframe['Origem Dados'] = metadados['rotulo']
        dataframe['Temporada'] = metadados['temporada']
        
        lista_dataframes.append(dataframe)
    return pd.concat(lista_dataframes, ignore_index=True)

@st.cache_data
def calcular_percentil(serie_dados, valor):
    return (serie_dados <= valor).sum() / len(serie_dados)

def obter_informacoes_contexto(dataframe, intervalo_minutos, intervalo_minutos_por_jogo, intervalo_idade, posicoes_selecionadas):
    return {
        'ligas': ', '.join(dataframe['Origem Dados'].unique()),
        'temporadas': ', '.join(dataframe['Temporada'].unique()),
        'total_jogadores': len(dataframe),
        'min_minutos': intervalo_minutos[0],
        'max_minutos': intervalo_minutos[1],
        'min_minutos_por_jogo': intervalo_minutos_por_jogo[0],
        'max_minutos_por_jogo': intervalo_minutos_por_jogo[1],
        'min_idade': intervalo_idade[0],
        'max_idade': intervalo_idade[1],
        'posicoes': ', '.join(posicoes_selecionadas) if posicoes_selecionadas else 'Todas'
    }

# =============================================
# Filtros da Barra Lateral
# =============================================
st.sidebar.header('Filtros')
with st.sidebar.expander("⚙️ Filtros Avançados", expanded=True):
    arquivos_enviados = st.file_uploader(
        "Carregue até 15 arquivos Excel do Wyscout", 
        type=["xlsx"], 
        accept_multiple_files=True
    )

if arquivos_enviados:
    # Coleta de Metadados
    novos_arquivos = [arquivo for arquivo in arquivos_enviados if arquivo.name not in st.session_state.metadados_arquivo]
    
    for arquivo in novos_arquivos:
        with st.form(key=f'metadados_{arquivo.name}'):
            st.subheader(f"Metadados para: {arquivo.name}")
            rotulo = st.text_input("Rótulo de origem dos dados (ex: Bundesliga 2)", key=f"rotulo_{arquivo.name}")
            
            temporadas = [f"{ano}/{ano+1}" for ano in range(2020, 2026)] + [str(ano) for ano in range(2020, 2026)]
            temporada = st.selectbox("Temporada", temporadas, key=f"temporada_{arquivo.name}")
            
            if st.form_submit_button("Confirmar"):
                st.session_state.metadados_arquivo[arquivo.name] = {'rotulo': rotulo, 'temporada': temporada}
                st.rerun()

    if metadados_faltantes := [arquivo.name for arquivo in arquivos_enviados if arquivo.name not in st.session_state.metadados_arquivo]:
        st.warning("Por favor, forneça metadados para todos os arquivos carregados")
        st.stop()

    try:
        dataframe_completo = carregar_e_limpar_dados(arquivos_enviados)

        # Aplicação dos Filtros Principais
        minutos_minimos, minutos_maximos = int(dataframe_completo['Minutes played'].min()), int(dataframe_completo['Minutes played'].max())
        intervalo_minutos = st.sidebar.slider('Minutos Jogados', minutos_minimos, minutos_maximos, (minutos_minimos, minutos_maximos))
        dataframe_filtrado_minutos = dataframe_completo[dataframe_completo['Minutes played'].between(*intervalo_minutos)].copy()

        dataframe_filtrado_minutos['Minutos por Jogo'] = (
            dataframe_filtrado_minutos['Minutes played'] / 
            dataframe_filtrado_minutos['Matches played'].replace(0, np.nan)
        )
        dataframe_filtrado_minutos['Minutos por Jogo'] = dataframe_filtrado_minutos['Minutos por Jogo'].fillna(0).clip(0, 120)
        
        mpg_min, mpg_max = int(dataframe_filtrado_minutos['Minutos por Jogo'].min()), int(dataframe_filtrado_minutos['Minutos por Jogo'].max())
        intervalo_mpg = st.sidebar.slider('Minutos por Jogo', mpg_min, mpg_max, (mpg_min, mpg_max))
        dataframe_filtrado_minutos = dataframe_filtrado_minutos[dataframe_filtrado_minutos['Minutos por Jogo'].between(*intervalo_mpg)]

        idade_minima, idade_maxima = int(dataframe_filtrado_minutos['Age'].min()), int(dataframe_filtrado_minutos['Age'].max())
        intervalo_idade = st.sidebar.slider('Faixa Etária', idade_minima, idade_maxima, (idade_minima, idade_maxima))

        if 'Position' in dataframe_filtrado_minutos.columns:
            dataframe_filtrado_minutos['Posicoes_Separadas'] = (
                dataframe_filtrado_minutos['Position']
                .astype(str)
                .apply(lambda posicoes: [posicao.strip() for posicao in posicoes.split(',')])
            )
            todas_posicoes = sorted({posicao for lista_posicoes in dataframe_filtrado_minutos['Posicoes_Separadas'] for posicao in lista_posicoes})
            posicoes_selecionadas = st.sidebar.multiselect('Posições', todas_posicoes, default=todas_posicoes)
        else:
            posicoes_selecionadas = []

        contexto = obter_informacoes_contexto(dataframe_filtrado_minutos, intervalo_minutos, intervalo_mpg, intervalo_idade, posicoes_selecionadas)
        jogadores = sorted(dataframe_filtrado_minutos['Player'].unique())
        jogador_selecionado_1 = st.sidebar.selectbox('Selecionar Jogador 1', jogadores)
        jogador_selecionado_2 = st.sidebar.selectbox('Selecionar Jogador 2', [jogador for jogador in jogadores if jogador != jogador_selecionado_1])

        colunas_numericas = dataframe_filtrado_minutos.select_dtypes(include=[np.number]).columns.tolist()
        abas_analise = st.tabs(['Radar', 'Barras', 'Dispersão', 'Perfilador', 'Correlação', 'Índice Composto (PCA)'])

        # =============================================
        # Gráfico de Radar (Aba 1)
        # =============================================
        with abas_analise[0]:
            st.header('Comparação por Radar')
            metricas_selecionadas = st.multiselect('Selecionar Métricas para Radar (6–12)', colunas_numericas, default=colunas_numericas[:6])
            
            if 6 <= len(metricas_selecionadas) <= 12:
                dados_jogador1 = dataframe_filtrado_minutos[dataframe_filtrado_minutos['Player'] == jogador_selecionado_1].iloc[0]
                dados_jogador2 = dataframe_filtrado_minutos[dataframe_filtrado_minutos['Player'] == jogador_selecionado_2].iloc[0]
                
                percentis_jogador1 = {
                    metrica: calcular_percentil(dataframe_filtrado_minutos[metrica], dados_jogador1[metrica]) 
                    for metrica in metricas_selecionadas
                }
                percentis_jogador2 = {
                    metrica: calcular_percentil(dataframe_filtrado_minutos[metrica], dados_jogador2[metrica]) 
                    for metrica in metricas_selecionadas
                }

                if posicoes_selecionadas:
                    grupo_filtrado = dataframe_filtrado_minutos[
                        dataframe_filtrado_minutos['Posicoes_Separadas'].apply(
                            lambda posicoes: any(posicao in posicoes for posicao in posicoes_selecionadas)
                        )
                    ]
                else:
                    grupo_filtrado = dataframe_filtrado_minutos

                medias_grupo = {metrica: grupo_filtrado[metrica].mean() for metrica in metricas_selecionadas}
                percentis_media_grupo = {
                    metrica: calcular_percentil(dataframe_filtrado_minutos[metrica], medias_grupo[metrica]) 
                    for metrica in metricas_selecionadas
                }

                dataframe_comparacao = pd.DataFrame({
                    'Métrica': metricas_selecionadas,
                    jogador_selecionado_1: [dados_jogador1[metrica] for metrica in metricas_selecionadas],
                    jogador_selecionado_2: [dados_jogador2[metrica] for metrica in metricas_selecionadas],
                    'Média do Grupo': [medias_grupo[metrica] for metrica in metricas_selecionadas]
                }).set_index('Métrica').round(2)

                mostrar_media = st.checkbox('Mostrar Média do Grupo', True)
                figura_radar = go.Figure()
                
                figura_radar.add_trace(go.Scatterpolar(
                    r=[percentis_jogador1[metrica] * 100 for metrica in metricas_selecionadas],
                    theta=metricas_selecionadas,
                    fill='toself',
                    name=jogador_selecionado_1
                ))
                
                figura_radar.add_trace(go.Scatterpolar(
                    r=[percentis_jogador2[metrica] * 100 for metrica in metricas_selecionadas],
                    theta=metricas_selecionadas,
                    fill='toself',
                    name=jogador_selecionado_2
                ))
                
                if mostrar_media:
                    figura_radar.add_trace(go.Scatterpolar(
                        r=[percentis_media_grupo[metrica] * 100 for metrica in metricas_selecionadas],
                        theta=metricas_selecionadas,
                        fill='toself',
                        name='Média do Grupo'
                    ))
                
                titulo_radar = (
                    f"<b>{jogador_selecionado_1} vs {jogador_selecionado_2}</b><br>"
                    f"<sup>Ligas: {contexto['ligas']} | Temporadas: {contexto['temporadas']}<br>"
                    f"Filtros: {contexto['min_minutos']}-{contexto['max_minutos']} minutos | "
                    f"{contexto['min_minutos_por_jogo']}-{contexto['max_minutos_por_jogo']} min/jogo | "
                    f"Idade {contexto['min_idade']}-{contexto['max_idade']} | Posições: {contexto['posicoes']}</sup>"
                )
                
                figura_radar.update_layout(
                    title=dict(text=titulo_radar, x=0.03, xanchor='left', font=dict(size=18)),
                    polar=dict(radialaxis=dict(range=[0, 100])),
                    template='plotly_white',
                    margin=dict(t=200, b=100, l=100, r=100),
                    height=700
                )
                st.plotly_chart(figura_radar)
                
                st.subheader('Comparação de Valores Nominais')
                st.dataframe(
                    dataframe_comparacao.style
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]
                    }])
                    .format(precision=2)
                )
                
                if st.button('Exportar Gráfico de Radar (300 DPI)', key='exportar_radar'):
                    figura_radar.update_layout(margin=dict(t=250))
                    bytes_imagem = figura_radar.to_image(format="png", width=1600, height=900, scale=3)
                    st.download_button(
                        "⬇️ Baixar Gráfico de Radar", 
                        data=bytes_imagem, 
                        file_name=f"radar_{jogador_selecionado_1}_vs_{jogador_selecionado_2}.png", 
                        mime="image/png"
                    )

        # =============================================
        # Gráficos de Barras (Aba 2)
        # =============================================
        with abas_analise[1]:
            st.header('Comparação por Barras')
            metricas_selecionadas = st.multiselect('Selecionar Métricas (máx. 5)', colunas_numericas, default=colunas_numericas[:1])
            
            if len(metricas_selecionadas) > 5:
                st.error("Máximo de 5 métricas permitidas!")
                st.stop()
            
            if len(metricas_selecionadas) >= 1:
                figura_barras = make_subplots(
                    rows=len(metricas_selecionadas),
                    cols=1,
                    subplot_titles=metricas_selecionadas,
                    vertical_spacing=0.15
                )
                
                for indice, metrica in enumerate(metricas_selecionadas, 1):
                    valor_jogador1 = dataframe_filtrado_minutos[dataframe_filtrado_minutos['Player'] == jogador_selecionado_1][metrica].iloc[0]
                    valor_jogador2 = dataframe_filtrado_minutos[dataframe_filtrado_minutos['Player'] == jogador_selecionado_2][metrica].iloc[0]
                    
                    if posicoes_selecionadas:
                        grupo_filtrado = dataframe_filtrado_minutos[
                            dataframe_filtrado_minutos['Posicoes_Separadas'].apply(
                                lambda posicoes: any(posicao in posicoes for posicao in posicoes_selecionadas)
                            )
                        ]
                    else:
                        grupo_filtrado = dataframe_filtrado_minutos
                    
                    valor_medio = grupo_filtrado[metrica].mean()
                    
                    figura_barras.add_trace(go.Bar(
                        y=[jogador_selecionado_1], 
                        x=[valor_jogador1], 
                        orientation='h',
                        name=jogador_selecionado_1, 
                        marker_color='#1f77b4', 
                        showlegend=(indice == 1)
                    ), row=indice, col=1)
                    
                    figura_barras.add_trace(go.Bar(
                        y=[jogador_selecionado_2], 
                        x=[valor_jogador2], 
                        orientation='h',
                        name=jogador_selecionado_2, 
                        marker_color='#ff7f0e', 
                        showlegend=(indice == 1)
                    ), row=indice, col=1)
                    
                    figura_barras.add_vline(
                        x=valor_medio, 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text="Média do Grupo", 
                        row=indice, 
                        col=1
                    )
                
                titulo_barras = (
                    f"<b>Comparação de Métricas</b><br>"
                    f"<sup>Contexto: {contexto['ligas']} ({contexto['temporadas']}) | "
                    f"Jogadores: {contexto['total_jogadores']} | Filtros: {contexto['min_idade']}-{contexto['max_idade']} anos</sup>"
                )
                
                figura_barras.update_layout(
                    title=dict(text=titulo_barras, x=0.03, xanchor='left', font=dict(size=18)),
                    height=300*len(metricas_selecionadas),
                    width=800,
                    template='plotly_white',
                    barmode='group',
                    margin=dict(t=200, b=100, l=100, r=100)
                st.plotly_chart(figura_barras)
                
                if st.button('Exportar Gráficos de Barras (300 DPI)', key='exportar_barras'):
                    figura_barras.update_layout(margin=dict(t=250))
                    bytes_imagem = figura_barras.to_image(format="png", width=1600, height=300*len(metricas_selecionadas)+300, scale=3)
                    st.download_button(
                        "⬇️ Baixar Gráficos", 
                        data=bytes_imagem, 
                        file_name="graficos_barras.png", 
                        mime="image/png"
                    )

        # =============================================
        # Gráfico de Dispersão (Aba 3)
        # =============================================
        with abas_analise[2]:
            st.header('Gráfico de Dispersão')
            metrica_x = st.selectbox('Métrica X', colunas_numericas)
            metrica_y = st.selectbox('Métrica Y', colunas_numericas)
            jogadores_destaque = st.multiselect('Destacar até 5 jogadores', jogadores, default=[jogador_selecionado_1, jogador_selecionado_2])[:5]
            
            dados_filtrados = dataframe_filtrado_minutos[dataframe_filtrado_minutos['Age'].between(*intervalo_idade)]
            
            # Correção aplicada aqui
            if posicoes_selecionadas:
                dados_filtrados = dados_filtrados[dados_filtrados['Posicoes_Separadas'].apply(
                    lambda posicoes: any(posicao in posicoes for posicao in posicoes_selecionadas)
                )]
            
            figura_dispersao = go.Figure()
            figura_dispersao.add_trace(go.Scatter(
                x=dados_filtrados[metrica_x], 
                y=dados_filtrados[metrica_y], 
                mode='markers', 
                marker=dict(color='cornflowerblue', opacity=0.5, size=8), 
                text=dados_filtrados['Player'], 
                hoverinfo='text', 
                name='Todos'
            ))
            
            cores = ['red','blue','green','orange','purple']
            for indice, jogador in enumerate(jogadores_destaque):
                dados_jogador = dados_filtrados[dados_filtrados['Player'] == jogador]
                if not dados_jogador.empty:
                    figura_dispersao.add_trace(go.Scatter(
                        x=dados_jogador[metrica_x], 
                        y=dados_jogador[metrica_y], 
                        text=dados_jogador['Player'], 
                        mode='markers+text', 
                        marker=dict(size=12, color=cores[indice]), 
                        name=jogador
                    ))
            
            titulo_dispersao = (
                f"<b>{metrica_x} vs {metrica_y}</b><br>"
                f"<sup>Fonte: {contexto['ligas']} ({contexto['temporadas']})<br>"
                f"Filtros: {contexto['total_jogadores']} jogadores | "
                f"{contexto['min_minutos_por_jogo']}+ min/jogo | {contexto['posicoes']}</sup>"
            )
            
            figura_dispersao.update_layout(
                title=dict(text=titulo_dispersao, x=0.03, xanchor='left', font=dict(size=18)),
                width=1000, 
                height=700,
                template='plotly_dark',
                margin=dict(t=200, b=100, l=100, r=100)
            )
            st.plotly_chart(figura_dispersao)
            
            if st.button('Exportar Gráfico de Dispersão (300 DPI)', key='exportar_dispersao'):
                figura_dispersao.update_layout(margin=dict(t=250))
                bytes_imagem = figura_dispersao.to_image(format="png", width=1800, height=1200, scale=3)
                st.download_button(
                    "⬇️ Baixar Gráfico de Dispersão", 
                    data=bytes_imagem, 
                    file_name=f"dispersao_{metrica_x}_vs_{metrica_y}.png", 
                    mime="image/png"
                )

        # =============================================
        # Perfilador (Aba 4)
        # =============================================
        with abas_analise[3]:
            st.header('Perfilador de Jogadores')
            metricas_selecionadas = st.multiselect('Selecionar 4–12 métricas', colunas_numericas)
            
            if 4 <= len(metricas_selecionadas) <= 12:
                percentis = {metrica: dataframe_filtrado_minutos[metrica].rank(pct=True) for metrica in metricas_selecionadas}
                minimos_percentis = {metrica: st.slider(f'Percentil mínimo para {metrica}', 0, 100, 50) for metrica in metricas_selecionadas}
                mascara_filtro = np.logical_and.reduce([percentis[metrica]*100 >= minimos_percentis[metrica] for metrica in metricas_selecionadas])
                st.dataframe(dataframe_filtrado_minutos.loc[mascara_filtro, ['Player', 'Team'] + metricas_selecionadas].reset_index(drop=True))
            else:
                st.warning('Selecione entre 4 e 12 métricas.')

        # =============================================
        # Matriz de Correlação (Aba 5)
        # =============================================
        with abas_analise[4]:
            st.header('Matriz de Correlação')
            metricas_selecionadas = st.multiselect('Métricas para correlacionar', colunas_numericas, default=colunas_numericas)
            
            if len(metricas_selecionadas) >= 2:
                matriz_correlacao = dataframe_filtrado_minutos[metricas_selecionadas].corr()
                figura_correlacao = go.Figure(data=go.Heatmap(
                    z=matriz_correlacao.values, 
                    x=metricas_selecionadas, 
                    y=metricas_selecionadas, 
                    zmin=-1, 
                    zmax=1, 
                    colorscale='Viridis'
                ))
                
                titulo_correlacao = (
                    f"<b>Relações entre Métricas</b><br>"
                    f"<sup>Base: {contexto['ligas']} ({contexto['temporadas']})<br>"
                    f"Jogadores: {contexto['total_jogadores']} | Mín. Minutos: {contexto['min_minutos']}+</sup>"
                )
                
                figura_correlacao.update_layout(
                    title=dict(text=titulo_correlacao, x=0.03, xanchor='left', font=dict(size=18)),
                    template='plotly_dark',
                    margin=dict(t=200, b=100, l=100, r=100)
                )
                st.plotly_chart(figura_correlacao)
                
                if st.button('Exportar Matriz de Correlação (300 DPI)', key='exportar_correlacao'):
                    figura_correlacao.update_layout(margin=dict(t=250))
                    bytes_imagem = figura_correlacao.to_image(format="png", width=1400, height=1400, scale=3)
                    st.download_button(
                        "⬇️ Baixar Matriz de Correlação", 
                        data=bytes_imagem, 
                        file_name="matriz_correlacao.png", 
                        mime="image/png"
                    )

        # =============================================
        # Índice Composto por PCA (Aba 6)
        # =============================================
        with abas_analise[5]:
            st.header('Índice Composto por PCA + Exportação Excel')
            metricas_desempenho = [coluna for coluna in colunas_numericas if coluna not in ['Age','Height','Country','Minutes played','Position']]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                tipo_kernel = st.selectbox('Tipo de Kernel',['linear','rbf'], index=1)
            with col2:
                gamma = st.number_input('Gamma', value=0.1, min_value=0.0, step=0.1, disabled=(tipo_kernel=='linear'))
            with col3:
                limiar_correlacao = st.slider(
                    'Limiar de Correlação', 
                    0.0, 1.0, 0.5, 0.05,
                    help='Correlação média mínima para inclusão de métricas',
                    disabled=st.session_state.get('pesos_manuais', False)
            with col4:
                pesos_manuais = st.checkbox('Pesos Manuais', key='pesos_manuais')

            metricas_selecionadas = st.multiselect('Selecionar métricas de desempenho', metricas_desempenho)
            
            if len(metricas_selecionadas)<2:
                st.warning('Selecione pelo menos duas métricas de desempenho.')
                st.stop()

            if pesos_manuais:
                st.subheader('Ajuste Manual de Pesos')
                controles_peso = {}
                colunas = st.columns(3)
                for indice, metrica in enumerate(metricas_selecionadas):
                    with colunas[indice%3]:
                        controles_peso[metrica] = st.slider(f'Peso para {metrica}', 0.0, 1.0, 0.5, key=f'peso_{metrica}')
                pesos = pd.Series(controles_peso)
                metricas_excluidas = []
            else:
                @st.cache_data
                def calcular_pesos(dataframe, features, threshold):
                    matriz_correlacao = dataframe[features].corr().abs()
                    correlacoes_medias = matriz_correlacao.mean(axis=1)
                    return correlacoes_medias.where(correlacoes_medias>threshold, 0)
                pesos = calcular_pesos(dataframe_filtrado_minutos, metricas_selecionadas, limiar_correlacao)
                metricas_excluidas = pesos[pesos==0].index.tolist()

            if metricas_excluidas and not pesos_manuais:
                st.warning(f'Métricas excluídas (baixa correlação): {", ".join(metricas_excluidas)}')
                metricas_selecionadas = [metrica for metrica in metricas_selecionadas if metrica not in metricas_excluidas]

            class PCA_Ponderado:
                def __init__(self, kernel='rbf', gamma=None):
                    self.kernel = kernel
                    self.gamma = gamma
                    self.scaler = StandardScaler()
                
                def fit_transform(self, X, pesos):
                    X_escalonado = self.scaler.fit_transform(X)
                    X_ponderado = X_escalonado * pesos.values
                    self.kpca = KernelPCA(
                        n_components=1,
                        kernel=self.kernel,
                        gamma=self.gamma,
                        random_state=42
                    )
                    return self.kpca.fit_transform(X_ponderado).flatten()

            if len(metricas_selecionadas)>=2:
                try:
                    pca = PCA_Ponderado(kernel=tipo_kernel, gamma=(None if tipo_kernel=='linear' else gamma))
                    dados_selecionados = dataframe_filtrado_minutos[metricas_selecionadas].dropna()
                    scores_pca = pca.fit_transform(dados_selecionados, pesos)
                    indices = dados_selecionados.index
                    
                    dataframe_pca = pd.DataFrame({
                        'Player': dataframe_filtrado_minutos.loc[indices, 'Player'],
                        'Team': dataframe_filtrado_minutos.loc[indices, 'Team'],
                        'Score PCA': scores_pca,
                        'Age': dataframe_filtrado_minutos.loc[indices, 'Age'],
                        'Position': dataframe_filtrado_minutos.loc[indices, 'Position'],
                        'Data Origin': dataframe_filtrado_minutos.loc[indices, 'Origem Dados'],
                        'Season': dataframe_filtrado_minutos.loc[indices, 'Temporada']
                    })

                    st.write('**Pesos das Métricas**')
                    dataframe_pesos = pd.DataFrame({
                        'Métrica': pesos.index,
                        'Peso': pesos.values
                    }).sort_values('Peso', ascending=False)
                    st.dataframe(dataframe_pesos.style.format({'Peso':'{:.2f}'}))

                    filtro_idade = dataframe_pca['Age'].between(*intervalo_idade)
                    filtro_posicao = (
                        dataframe_pca['Position'].astype(str).apply(lambda x: any(pos in x for pos in posicoes_selecionadas)) 
                        if posicoes_selecionadas 
                        else pd.Series(True, index=dataframe_pca.index)
                    )
                    dataframe_filtrado_pca = dataframe_pca[filtro_idade & filtro_posicao]

                    if not dataframe_filtrado_pca.empty:
                        score_min, score_max = dataframe_filtrado_pca['Score PCA'].min(), dataframe_filtrado_pca['Score PCA'].max()
                        intervalo_score = st.slider(
                            'Filtrar por Score PCA',
                            min_value=float(score_min),
                            max_value=float(score_max),
                            value=(float(score_min), float(score_max))
                        )
                        
                        dataframe_final = dataframe_filtrado_pca[dataframe_filtrado_pca['Score PCA'].between(*intervalo_score)]
                        if dataframe_final.empty:
                            st.warning('Nenhum jogador no intervalo selecionado.')
                        else:
                            st.write(f'**Jogadores Correspondentes ({len(dataframe_final)})**')
                            st.dataframe(dataframe_final.sort_values('Score PCA', ascending=False).reset_index(drop=True))
                            
                            st.write('**Distribuição de Scores**')
                            figura_pca = go.Figure(data=[go.Bar(x=dataframe_final['Player'], y=dataframe_final['Score PCA'])])
                            
                            titulo_pca = (
                                f"<b>Scores PCA</b><br>"
                                f"<sup>Contexto: {contexto['ligas']} ({contexto['temporadas']})<br>"
                                f"Filtros: Idade {contexto['min_idade']}-{contexto['max_idade']} | "
                                f"{contexto['posicoes']} | Métricas: {len(metricas_selecionadas)} selecionadas</sup>"
                            )
                            
                            figura_pca.update_layout(
                                title=dict(text=titulo_pca, x=0.03, xanchor='left', font=dict(size=18)),
                                template='plotly_dark',
                                margin=dict(t=200, b=100, l=100, r=100)
                            )
                            st.plotly_chart(figura_pca)
                            
                            if st.button('Exportar Scores PCA (300 DPI)', key='exportar_pca'):
                                figura_pca.update_layout(margin=dict(t=250))
                                bytes_imagem = figura_pca.to_image(format="png", width=1600, height=900, scale=3)
                                st.download_button(
                                    "⬇️ Baixar Gráfico PCA", 
                                    data=bytes_imagem, 
                                    file_name="scores_pca.png", 
                                    mime="image/png"
                                )
                            
                            # Exportação Excel
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                dataframe_final.to_excel(writer, sheet_name='Resultados PCA', index=False)
                            buffer.seek(0)
                            st.download_button(
                                '📥 Baixar Resultados em Excel',
                                data=buffer,
                                file_name='resultados_pca.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                    else:
                        st.warning('Nenhum jogador corresponde aos filtros atuais.')
                except Exception as erro:
                    st.error(f'Erro no cálculo PCA: {str(erro)}')
            else:
                st.error('Métricas insuficientes após filtragem.')

    except Exception as erro:
        st.error(f'Erro: {erro}')
else:
    st.info('Por favor, carregue até 15 arquivos Excel do Wyscout para iniciar a análise')
    st.warning("⚠️ Necessário: `pip install kaleido==0.2.1.post1`")
