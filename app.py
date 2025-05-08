# Football Analytics App - Enhanced Version with mplsoccer
# Player Comparison and Similarity Analysis System - v4.0

import streamlit as st
# Deve ser a primeira chamada Streamlit
st.set_page_config(page_title='Football Analytics',
                   layout='wide',
                   page_icon="⚽")

# Inicializar session_state para nossas variáveis principais
if 'saved_similarity_metrics' not in st.session_state:
    st.session_state.saved_similarity_metrics = []

if 'saved_bar_metrics' not in st.session_state:
    st.session_state.saved_bar_metrics = []

if 'saved_scatter_x_metric' not in st.session_state:
    st.session_state.saved_scatter_x_metric = None

if 'saved_scatter_y_metric' not in st.session_state:
    st.session_state.saved_scatter_y_metric = None

if 'benchmark_loaded' not in st.session_state:
    st.session_state.benchmark_loaded = False
    
# Adicionar controle de abas para evitar que o app volte para a aba original
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0
    
# Função para atualizar a aba atual
def update_tab(tab_index):
    st.session_state.current_tab = tab_index
    st.session_state.benchmark_df = None
    st.session_state.benchmark_name = None
    
# Função auxiliar para processar listas de métricas coladas pelo usuário
def process_pasted_metrics(pasted_text, available_metrics):
    """
    Processa uma lista de métricas colada pelo usuário e retorna aquelas que existem na lista de métricas disponíveis.
    
    Args:
        pasted_text: Texto colado pelo usuário, com métricas separadas por linhas ou vírgulas
        available_metrics: Lista de métricas disponíveis no dataset
        
    Returns:
        Lista de métricas válidas encontradas no texto colado
    """
    if not pasted_text:
        return []
        
    # Substituir vírgulas por quebras de linha para padronizar
    text = pasted_text.replace(',', '\n')
    
    # Dividir o texto em linhas e limpar espaços extras
    metrics_list = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Filtrar apenas as métricas que existem no dataset
    valid_metrics = [m for m in metrics_list if m in available_metrics]
    
    return valid_metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import io
from io import BytesIO
import os
import random
from mplsoccer import Radar, FontManager, grid, PyPizza
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cdist
import base64

# =============================================
# Configuration
# =============================================

# Set up consistent fonts for mplsoccer
plt.rcParams['font.family'] = 'sans-serif'

# =============================================
# Helper Functions
# =============================================
# Função para detectar automaticamente todos os arquivos Excel com base na fonte de dados
@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_available_leagues(data_source='skillcorner'):
    """
    Detecta automaticamente todos os arquivos Excel (.xlsx) de uma fonte específica
    e os adiciona como ligas disponíveis.
    Limitado a 15 arquivos para melhorar performance.
    
    Args:
        data_source: 'skillcorner' para os dados integrados SkillCorner da pasta attached_assets, 
                    'wyscout' para os dados WyScout da pasta wyscout_data
    """
    leagues_dict = {}
    max_files = 15  # Limitar a 15 arquivos para melhorar performance

    if data_source == 'skillcorner':
        # Buscar arquivos locais na pasta attached_assets (SkillCorner Integrated)
        assets_dir = "attached_assets"

        # Verificar se a pasta existe
        file_count = 0
        if os.path.exists(assets_dir) and os.path.isdir(assets_dir):
            # Obter lista de arquivos Excel e ordenar por data de modificação (mais recentes primeiro)
            excel_files = []
            for filename in os.listdir(assets_dir):
                if filename.endswith(".xlsx"):
                    file_path = os.path.join(assets_dir, filename)
                    try:
                        mod_time = os.path.getmtime(file_path)
                        excel_files.append((filename, mod_time))
                    except:
                        excel_files.append((filename, 0))
            
            # Ordenar por data de modificação (mais recentes primeiro)
            excel_files.sort(key=lambda x: x[1], reverse=True)
            
            # Processar até o limite máximo de arquivos
            for filename, _ in excel_files[:max_files]:
                # Usar o nome do arquivo (sem extensão) como nome da liga
                league_name = os.path.splitext(filename)[0]
                # Remover sufixos comuns como "WySC" ou números de versão para deixar o nome mais limpo
                league_name = league_name.replace(" - WySC",
                                                  "").replace(" WySC", "")
                league_name = league_name.split(" (")[
                    0]  # Remover qualquer coisa após "(" como "(1)"

                # Adicionar ao dicionário
                leagues_dict[league_name] = os.path.join(
                    assets_dir, filename)
                file_count += 1
                
                if file_count >= max_files:
                    break

        # Caso não encontre nenhum arquivo, usar valores padrão
        if not leagues_dict:
            leagues_dict = {
                "Austria 2.Liga":
                "attached_assets/Austria 2.Liga - WySC (1).xlsx"
            }
            print(
                "Aviso: Nenhum arquivo Excel encontrado na pasta attached_assets. Usando valores padrão."
            )

    elif data_source == 'wyscout':
        # Buscar arquivos locais na pasta wyscout_data
        wyscout_dir = "wyscout_data"

        # Verificar se a pasta existe
        file_count = 0
        if os.path.exists(wyscout_dir) and os.path.isdir(wyscout_dir):
            # Obter lista de arquivos Excel e ordenar por data de modificação (mais recentes primeiro)
            excel_files = []
            for filename in os.listdir(wyscout_dir):
                if filename.endswith(".xlsx"):
                    file_path = os.path.join(wyscout_dir, filename)
                    try:
                        mod_time = os.path.getmtime(file_path)
                        excel_files.append((filename, mod_time))
                    except:
                        excel_files.append((filename, 0))
            
            # Ordenar por data de modificação (mais recentes primeiro)
            excel_files.sort(key=lambda x: x[1], reverse=True)
            
            # Processar até o limite máximo de arquivos
            for filename, _ in excel_files[:max_files]:
                # Usar o nome do arquivo como nome da liga para WyScout
                league_name = os.path.splitext(filename)[0]

                # Adicionar ao dicionário
                leagues_dict[league_name] = os.path.join(
                    wyscout_dir, filename)
                file_count += 1
                
                if file_count >= max_files:
                    break
        
        # Caso não encontre nenhum arquivo WyScout, informar ao usuário
        if not leagues_dict:
            print("Aviso: Nenhum arquivo WyScout encontrado na pasta wyscout_data.")
            print("Por favor, adicione arquivos .xlsx na pasta wyscout_data.")

    return leagues_dict


# Inicializar variáveis de sessão necessárias
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'skillcorner'
if 'selected_leagues_skillcorner' not in st.session_state:
    st.session_state.selected_leagues_skillcorner = []
if 'selected_leagues_wyscout' not in st.session_state:
    st.session_state.selected_leagues_wyscout = []

# Carregar ligas disponíveis automaticamente conforme a fonte de dados atual
AVAILABLE_LEAGUES = load_available_leagues(st.session_state.data_source)


@st.cache_data(ttl=600)  # Cache por 10 minutos para melhorar performance
def load_league_data(selected_leagues):
    """
    Carrega dados das ligas selecionadas.
    Suporta tanto carregar de arquivos locais quanto de arquivos enviados pelo usuário.
    Se st.session_state.combine_leagues for True, combina todas as ligas em um único dataframe.
    Se False, processa cada liga separadamente para cálculo de percentis.
    """
    dfs = []
    league_dfs = {}  # Para armazenar cada dataframe separadamente por liga
    total_players = 0
    
    # Usando spinner para melhorar a experiência de carregamento
    with st.spinner(f"Loading data from {len(selected_leagues)} leagues..."):
        # Verificar se estamos usando arquivos carregados pelo usuário
        if st.session_state.uploaded_files:
            # Carregar dados dos arquivos enviados pelo usuário
            for file_name, file_info in st.session_state.uploaded_files.items():
                league_name = file_info['league_name']
                
                # Verificar se esta liga foi selecionada
                if league_name in selected_leagues:
                    # Adicionar o DataFrame já processado
                    df = file_info['df']
                    dfs.append(df)
                    league_dfs[league_name] = df
                    total_players += len(df)
                    
                    st.success(f"Loaded from uploaded file: {league_name} ({len(df)} players)")
        else:
            # Método tradicional: carregar de arquivos locais
            for league_name in selected_leagues:
                try:
                    # Verificar se a liga está disponível
                    if league_name not in AVAILABLE_LEAGUES:
                        st.warning(f"League '{league_name}' not found in available data.")
                        continue
                    
                    # Obter caminho do arquivo    
                    file_path = AVAILABLE_LEAGUES[league_name]
                    
                    # Verificar se o arquivo existe
                    if not os.path.exists(file_path):
                        st.error(f"File not found: {file_path}")
                        continue
                    
                    # Tentar carregar o arquivo Excel
                    df = None
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    except Exception as excel_error:
                        st.error(f"Error reading Excel file {league_name}: {str(excel_error)}")
                        continue
                    
                    if df is None or df.empty:
                        st.warning(f"No data found in file for {league_name}")
                        continue
                        
                    # Processamento básico
                    df = df.dropna(how="all")
                    df = df.loc[:, df.columns.notnull()]
                    df.columns = [str(c).strip() for c in df.columns]
                    df['Data Origin'] = league_name
                    df['Season'] = "2023/2024"
                    
                    # Calcular métricas avançadas
                    df = calculate_offensive_metrics(df)
                    
                    # Otimização de tipos de dados
                    for col in df.columns:
                        if df[col].dtype == 'float64' and col not in ['xG', 'npxG', 'G-xG', 'xA']:
                            df[col] = pd.to_numeric(df[col], downcast='float')
                        elif df[col].dtype == 'int64':
                            df[col] = pd.to_numeric(df[col], downcast='integer')
                    
                    # Adicionar ao conjunto de dados
                    dfs.append(df)
                    league_dfs[league_name] = df
                    total_players += len(df)
                    
                    st.success(f"Loaded {league_name}: {len(df)} players")
                
                except Exception as e:
                    st.error(f"Error loading {league_name}: {str(e)}")
                    continue

    if not dfs:
        st.warning("No data was loaded. Please select at least one league and try again.")
        return pd.DataFrame()

    # Verificar o modo de processamento
    if 'combine_leagues' not in st.session_state:
        st.session_state.combine_leagues = True
    combine_leagues = st.session_state.combine_leagues

    if combine_leagues:
        # Modo padrão: combinar todas as ligas e calcular percentis globalmente
        with st.spinner(f"Combining data from {len(dfs)} leagues ({total_players} players)..."):
            combined_df = pd.concat(dfs, ignore_index=True)
            st.success(f"Successfully loaded combined data with {len(combined_df)} players.")
            return combined_df
    else:
        # Modo novo: calcular percentis separadamente para cada liga e depois combinar
        processed_dfs = []
        with st.spinner("Calculating league-specific percentiles..."):
            # Identificar todas as colunas métricas (excluindo colunas categóricas/identificadoras)
            # Assumindo que o primeiro dataframe tem todas as colunas que queremos processar
            first_df = dfs[0]
            metric_cols = [
                col for col in first_df.columns if col not in
                ['Player', 'Team', 'Position', 'Data Origin', 'Season']
            ]

            # Processar cada liga separadamente - com otimização para evitar cálculos desnecessários
            for league_name, league_df in league_dfs.items():
                # Para cada métrica no dataframe, calcular percentil dentro de cada liga
                # Processar apenas colunas numéricas para evitar erros e cálculos desnecessários
                numeric_cols = [col for col in metric_cols if col in league_df.columns and 
                               pd.api.types.is_numeric_dtype(league_df[col])]
                
                # Calcular percentis em lote para melhor performance
                for col in numeric_cols:
                    # Criar nova coluna com percentis
                    col_percentile = f"{col}_percentile_in_{league_name}"
                    league_df[col_percentile] = league_df[col].rank(pct=True) * 100

                processed_dfs.append(league_df)

            # Combinar todos os dataframes processados
            combined_df = pd.concat(processed_dfs, ignore_index=True)

            # Exibir informação sobre o modo de processamento
            st.info(
                "Percentiles calculated separately within each league. Check columns with '_percentile_in_' suffix for league-specific percentiles."
            )
            
            st.success(f"Successfully loaded data with {len(combined_df)} players and league-specific percentiles.")
            return combined_df


# Inicializar o session_state para manter os dados entre recargas
if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}
    
# Armazenar arquivos carregados pelo usuário
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

# Session state para persistência de filtros e seleções
if 'last_players' not in st.session_state:
    st.session_state.last_players = []
if 'last_metrics' not in st.session_state:
    st.session_state.last_metrics = []
if 'last_selected_p1' not in st.session_state:
    st.session_state.last_selected_p1 = None
if 'last_selected_p2' not in st.session_state:
    st.session_state.last_selected_p2 = None
if 'last_minutes_range' not in st.session_state:
    st.session_state.last_minutes_range = [0, 5000]
if 'last_mpg_range' not in st.session_state:
    st.session_state.last_mpg_range = [0, 100]
if 'last_age_range' not in st.session_state:
    st.session_state.last_age_range = [15, 40]
if 'last_positions' not in st.session_state:
    st.session_state.last_positions = []
    
# Inicializar variáveis temporárias para filtros com valores baseados nos filtros atuais
if 'temp_minutes_range' not in st.session_state:
    st.session_state.temp_minutes_range = st.session_state.last_minutes_range

if 'temp_mpg_range' not in st.session_state:
    st.session_state.temp_mpg_range = st.session_state.last_mpg_range
    
if 'temp_age_range' not in st.session_state:
    st.session_state.temp_age_range = st.session_state.last_age_range
    
if 'temp_positions' not in st.session_state:
    st.session_state.temp_positions = st.session_state.last_positions.copy() if st.session_state.last_positions else []

# Variáveis para Pizza Chart
if 'saved_pizza_p1' not in st.session_state:
    st.session_state.saved_pizza_p1 = None
if 'saved_pizza_p2' not in st.session_state:
    st.session_state.saved_pizza_p2 = None
if 'saved_pizza_metrics' not in st.session_state:
    st.session_state.saved_pizza_metrics = []
if 'saved_pizza_comparison_option' not in st.session_state:
    st.session_state.saved_pizza_comparison_option = 0
if 'pizza_selected_groups' not in st.session_state:
    st.session_state.pizza_selected_groups = []

# Variáveis para Bar Chart
if 'saved_bar_p1' not in st.session_state:
    st.session_state.saved_bar_p1 = None
if 'saved_bar_p2' not in st.session_state:
    st.session_state.saved_bar_p2 = None
if 'saved_bar_metrics' not in st.session_state:
    st.session_state.saved_bar_metrics = []

# Variáveis para Scatter Plot
if 'saved_scatter_x_metric' not in st.session_state:
    st.session_state.saved_scatter_x_metric = None
if 'saved_scatter_y_metric' not in st.session_state:
    st.session_state.saved_scatter_y_metric = None

# Variáveis para Similarity
if 'saved_similarity_player' not in st.session_state:
    st.session_state.saved_similarity_player = None
if 'saved_similarity_metrics' not in st.session_state:
    st.session_state.saved_similarity_metrics = []
if 'saved_similarity_method' not in st.session_state:
    st.session_state.saved_similarity_method = 'pca_kmeans'

# Variáveis de persistência na UI
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None 
if 'players' not in st.session_state:
    st.session_state.players = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = []
if 'selected_p1' not in st.session_state:
    st.session_state.selected_p1 = None
if 'selected_p2' not in st.session_state:
    st.session_state.selected_p2 = None

# Variáveis legadas (compatibilidade)
if 'bar_p1' not in st.session_state:
    st.session_state.bar_p1 = st.session_state.saved_bar_p1
if 'bar_p2' not in st.session_state:
    st.session_state.bar_p2 = st.session_state.saved_bar_p2
if 'bar_metrics' not in st.session_state:
    st.session_state.bar_metrics = st.session_state.saved_bar_metrics
if 'scatter_x_metric' not in st.session_state:
    st.session_state.scatter_x_metric = st.session_state.saved_scatter_x_metric
if 'scatter_y_metric' not in st.session_state:
    st.session_state.scatter_y_metric = st.session_state.saved_scatter_y_metric
if 'similarity_player' not in st.session_state:
    st.session_state.similarity_player = st.session_state.saved_similarity_player
if 'similarity_method' not in st.session_state:
    st.session_state.similarity_method = st.session_state.saved_similarity_method
    
# Variáveis de salto direto para menus
if 'jump_to_pizza' not in st.session_state:
    st.session_state.jump_to_pizza = False
if 'jump_to_sim' not in st.session_state:
    st.session_state.jump_to_sim = False


# Função para tratamento de erros/exceções de forma centralizada
def safe_operation(func, error_msg, fallback=None, *args, **kwargs):
    """Execute uma função e capture exceções com uma mensagem amigável"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"{error_msg}: {str(e)}")
        return fallback


@st.cache_data
def calculate_offensive_metrics(df):
    """
    Calculate advanced offensive metrics:
    1. npxG (non-penalty expected Goals): xG - (0.81 * Penalties taken)
    2. G-xG (Goals minus Expected Goals)
    3. npxG per Shot: (xG - (0.81 * Penalties taken)) / (Shots - Penalties taken)
    4. npxG per 90: npxG * 90 / Minutes played
    5. Box Efficiency: (npxG_per_90 + xA_per_90) / Touches_in_Box_per_90

    Args:
        df: DataFrame containing player data with the required metrics

    Returns:
        DataFrame with the new metrics added
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    # Check if required columns exist
    required_cols = [
        'xG', 'Penalties taken', 'Goals', 'Shots', 'Minutes played',
        'xA per 90', 'Touches in box per 90'
    ]
    missing_cols = [col for col in required_cols if col not in df_copy.columns]

    # Replace missing columns with zeros to allow calculation
    for col in missing_cols:
        df_copy[col] = 0

    # Calculate npxG (non-penalty expected Goals)
    df_copy['npxG'] = df_copy['xG'] - (0.81 * df_copy['Penalties taken'])

    # Calculate G-xG (Goals minus Expected Goals)
    df_copy['G-xG'] = df_copy['Goals'] - df_copy['xG']

    # Calculate npxG per Shot
    # Avoid division by zero by using numpy.where
    shots_minus_penalties = df_copy['Shots'] - df_copy['Penalties taken']
    df_copy['npxG per Shot'] = np.where(
        shots_minus_penalties > 0,
        df_copy['npxG'] / shots_minus_penalties,
        0  # Default value when denominator is zero
    )

    # Calculate npxG per 90 minutes
    df_copy['npxG per 90'] = np.where(
        df_copy['Minutes played'] > 0,
        df_copy['npxG'] * 90 / df_copy['Minutes played'],
        0  # Default value when minutes played is zero
    )

    # Calculate Box Efficiency
    # Use our calculated npxG per 90 and get other per90 metrics from the dataset
    xa_per_90_col = 'xA per 90' if 'xA per 90' in df_copy.columns else 'xA per 90'
    touches_box_col = 'Touches in box per 90' if 'Touches in box per 90' in df_copy.columns else 'Touches in box per 90'

    # Avoid division by zero
    df_copy['Box Efficiency'] = np.where(
        df_copy[touches_box_col] > 0,
        (df_copy['npxG per 90'] + df_copy[xa_per_90_col]) /
        df_copy[touches_box_col],
        0  # Default value when denominator is zero
    )

    return df_copy


def load_and_clean(files):
    """Load and preprocess Excel files"""
    dfs = []
    for file in files[:15]:  # Limit to 15 files
        df = pd.read_excel(file)
        df.dropna(how="all", inplace=True)
        df = df.loc[:, df.columns.notnull()]
        df.columns = [str(c).strip() for c in df.columns]

        metadata = st.session_state.file_metadata[file.name]
        df['Data Origin'] = metadata['label']
        df['Season'] = metadata['season']

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Calculate advanced offensive metrics
    combined_df = calculate_offensive_metrics(combined_df)

    return combined_df


@st.cache_data
def calc_percentile(series, value, benchmark_series=None):
    """
    Calculate percentile rank of a value within a series

    Args:
        series: Pandas Series containing values for percentile calculation
        value: Value to calculate percentile for
        benchmark_series: Optional benchmark series to use instead of main series

    Returns:
        Percentile rank on 0-1 scale
    """
    # IMPORTANTE: Usar o dataframe filtrado APENAS por minutos para cálculos de percentis
    # Se df_percentiles está disponível na sessão, usamos ele em vez do dataframe filtrado
    # (df_percentiles é filtrado apenas por minutos e mpg, NÃO por idade ou posição)
    if 'df_percentiles' in st.session_state and benchmark_series is None:
        metric_name = series.name
        # Verificar se a métrica existe no dataframe de percentis
        if metric_name in st.session_state.df_percentiles.columns:
            # Usar a série da métrica no dataframe de percentis (sem filtros de idade/posição)
            percentile_series = st.session_state.df_percentiles[metric_name]
            # Aplicar o resto da lógica com a série filtrada corretamente
            
            # Se o valor é zero, verificamos se deve ser percentil 0 ou 0.5
            if value == 0:
                # Verificar se todos os valores são zero
                check_series = percentile_series.dropna()
                if len(check_series) > 0 and check_series.max() == 0:
                    return 0.5  # Todos são zero
                elif len(check_series) > 0 and check_series.max() > 0:
                    return 0.0  # Existem valores maiores
                    
            # Calcular percentil de forma normal usando a série correta
            calc_series = percentile_series.dropna()
            if len(calc_series) == 0:
                return 0.5
            return (calc_series <= value).sum() / len(calc_series)
    
    # Fluxo original para casos onde não temos df_percentiles ou estamos usando benchmark
    # Se o valor é zero e assumimos que a métrica não pode ser negativa,
    # retornamos 0 como percentil, a menos que TODOS os valores sejam 0
    if value == 0:
        # Determine a série a ser usada para verificação
        check_series = benchmark_series if benchmark_series is not None else series
        check_series = check_series.dropna()

        # Se todos os valores são zero, retorna 0.5 (média)
        if len(check_series) > 0 and check_series.max() == 0:
            return 0.5

        # Se existem valores maiores que zero, e este valor é zero, retorna 0
        elif len(check_series) > 0 and check_series.max() > 0:
            return 0.0

    # Para todos os outros casos, siga o cálculo regular
    # Se foi fornecida uma série de benchmark, use-a ao invés da série principal
    calc_series = benchmark_series if benchmark_series is not None else series

    # Remover valores NaN para cálculo mais preciso
    calc_series = calc_series.dropna()

    # Se a série estiver vazia, retorne 0.5 (50º percentil)
    if len(calc_series) == 0:
        return 0.5

    # Calcular o percentil (escala 0-1)
    return (calc_series <= value).sum() / len(calc_series)


def apply_benchmark_filter(benchmark_df,
                           minutes_range,
                           mpg_range=None,
                           age_range=None,
                           positions=None):
    """
    Apply the same filters to benchmark database as applied to main database

    Args:
        benchmark_df: The benchmark DataFrame
        minutes_range: Tuple of (min, max) minutes played
        mpg_range: Optional tuple of (min, max) minutes per game
        age_range: Optional tuple of (min, max) age
        positions: Optional list of positions to include

    Returns:
        Filtered benchmark DataFrame
    """
    try:
        if benchmark_df is None or benchmark_df.empty:
            return None

        # Apply minutes played filter
        filtered_df = benchmark_df[benchmark_df['Minutes played'].between(
            *minutes_range)].copy()

        # Apply minutes per game filter if provided
        if mpg_range and 'Minutes per game' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Minutes per game'].between(
                *mpg_range)]

        # Apply age filter if provided
        if age_range and 'Age' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Age'].between(*age_range)]

        # Apply position filter if provided
        if positions and 'Position_split' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Position_split'].apply(
                lambda x: any(pos in x for pos in positions))]

        return filtered_df
    except Exception as e:
        print(f"Error applying benchmark filter: {str(e)}")
        return benchmark_df


def get_context_info(df, minutes_range, mpg_range, age_range, sel_pos):
    """Get contextual information for visualization titling"""
    return {
        'leagues': ', '.join(df['Data Origin'].unique()),
        'seasons': ', '.join(df['Season'].unique()),
        'total_players': len(df),
        'min_min': minutes_range[0],
        'max_min': minutes_range[1],
        'min_mpg': mpg_range[0],
        'max_mpg': mpg_range[1],
        'min_age': age_range[0],
        'max_age': age_range[1],
        'positions': ', '.join(sel_pos) if sel_pos else 'All'
    }


def abbreviate_metric_name(metric):
    """
    Abreviar nomes de métricas para o gráfico de pizza
    """
    # Substituir palavras comuns por abreviações
    shortened = metric
    shortened = shortened.replace("Progressive", "Prog")
    shortened = shortened.replace("Accurate", "Acc")
    shortened = shortened.replace("Defensive", "Def")
    shortened = shortened.replace("Attacking", "Att")
    shortened = shortened.replace("per 90", "/90")
    shortened = shortened.replace("Completion", "Comp")
    shortened = shortened.replace("Completions", "Comp")

    # Se ainda for muito longo, tentar abreviar mais
    if len(shortened) > 20:
        # Remover palavras comuns que não mudam o significado da métrica
        shortened = shortened.replace(" in ", " ")

    return shortened


def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for download"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf


@st.cache_data
def create_pizza_chart(params=None,
                       values_p1=None,
                       values_p2=None,
                       values_avg=None,
                       title=None,
                       subtitle=None,
                       p1_name="Player 1",
                       p2_name="Player 2"):
    """
    Cria um pizza chart profissional usando o mesmo estilo do gráfico comparativo.
    Usa o PyPizza da mesma maneira que o gráfico de comparação para garantir consistência visual.
    Inclui logo no centro do gráfico para manter consistência visual com outros gráficos.
    """
    try:
        # Métricas padrão se não forem fornecidas
        if params is None:
            params = [
                "xG per 90", "Shots on target, %", "Goal conversion, %",
                "Touches in box per 90", "Progressive runs per 90",
                "Explosive Acceleration to Sprint P90", "M/min P90",
                "Sprint Count P90", "Accurate passes, %",
                "Deep completions per 90", "Progressive passes per 90",
                "Key passes per 90"
            ]
            # Criar valores aleatórios para demonstração se não houver dados
            if values_p1 is None:
                values_p1 = [
                    random.randint(50, 95) for _ in range(len(params))
                ]

        # Verificações de segurança
        if values_p1 is not None and len(params) != len(values_p1):
            raise ValueError(
                f"Número de parâmetros ({len(params)}) não corresponde ao número de valores ({len(values_p1)})"
            )

        # Estamos sempre usando percentis para a visualização
        # Não precisamos mais do código de valores originais, pois simplificamos
        # para usar apenas percentis conforme solicitado pelo usuário

        # Para esta função, SEMPRE usamos os percentis para visualização
        # Ou seja, não fazemos normalização aqui - já que estamos sempre trabalhando com percentis

        # Arredondar valores para inteiros (tanto percentis quanto valores normalizados)
        if values_p1 is not None:
            values_p1 = [round(v) for v in values_p1]
        if values_p2 is not None:
            values_p2 = [round(v) for v in values_p2]
        if values_avg is not None:
            values_avg = [round(v) for v in values_avg]

        # Esquema uniforme de cores - azul real e azul claro
        player1_color = "#0052CC"  # Azul real
        player2_color = "#00A3FF"  # Azul claro
        avg_color = "#B3CFFF"  # Azul muito claro para média
        text_color = "#000000"  # Preto para texto
        background_color = "#F5F5F5"  # Cinza claro para fundo

        # Preparar mínimos e máximos para cada parâmetro
        min_values = [0] * len(params)
        max_values = [100] * len(params)

        # Instanciar o objeto PyPizza
        baker = PyPizza(
            params=params,  # lista de parâmetros
            background_color=background_color,  # cor de fundo
            straight_line_color="#FFFFFF",  # cor das linhas retas (branco)
            straight_line_lw=1.5,  # largura das linhas retas
            last_circle_lw=1.5,  # largura do último círculo
            other_circle_color="#FFFFFF",  # cor das linhas circulares (branco)
            other_circle_lw=1.5,  # largura das linhas circulares
            other_circle_ls="-",  # estilo de linha sólida
            inner_circle_size=20  # tamanho do círculo interior
        )

        # Criar figura e eixos com projeção polar (tamanho menor para visualização no app)
        fig, ax = plt.subplots(figsize=(8, 8),
                               facecolor=background_color,
                               subplot_kw={"projection": "polar"},
                               dpi=100)

        # Definir DPI maior para exportação, mas tamanho menor para visualização
        fig.set_dpi(300)  # Usado apenas na exportação

        # Centralizar e ajustar a figura com mais espaço para a legenda
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)

        # Limitar o tamanho do gráfico (reduzir raio)
        ax.set_ylim(0, 0.9)  # Reduzir o raio máximo para 0.9 (ao invés de 1.0)

        # Removemos a adição do logo ANTES de criar a pizza
        # O logo será adicionado DEPOIS do plot do gráfico para não ficar sobreposto

        # Criar pizza para jogador 1 (principal)
        values = values_p1

        # Cores iguais para todas as fatias
        slice_colors = [player1_color] * len(params)
        text_colors = ["#FF0000"] * len(
            params)  # Texto vermelho para os valores

        # Melhorar o grid
        # Criar círculos de referência mais definidos (25%, 50%, 75%, 100%)
        circles = [0.25, 0.5, 0.75]
        for circle in circles:
            ax.plot(np.linspace(0, 2 * np.pi, 100), [circle] * 100,
                    color='#AAAAAA',
                    linestyle='-',
                    linewidth=0.8,
                    zorder=1,
                    alpha=0.7)

        # Abreviar os rótulos dos parâmetros para melhor visualização
        abbreviated_params = [
            abbreviate_metric_name(param) for param in params
        ]

        # Fazer o plot principal com grid melhorado e círculo interno de tamanho adequado
        baker = PyPizza(
            params=abbreviated_params,  # parâmetros abreviados
            min_range=min_values,  # valores mínimos
            max_range=max_values,  # valores máximos
            background_color=background_color,
            straight_line_color="#FFFFFF",  # linhas radiais brancas
            straight_line_lw=1.5,  # linhas mais grossas
            last_circle_lw=1.5,  # círculo externo mais visível
            other_circle_color="#FFFFFF",  # linhas circulares brancas
            other_circle_lw=1.5,  # largura das linhas circulares
            other_circle_ls="-",  # linhas sólidas para círculos
            inner_circle_size=10  # círculo interno BEM MENOR (10% do raio total)
        )

        # Criar a pizza para o jogador 1
        baker.make_pizza(
            values,  # valores
            ax=ax,  # axis
            color_blank_space="same",  # espaço em branco com mesma cor
            slice_colors=slice_colors,  # cores das fatias
            value_colors=text_colors,  # cores dos valores (vermelho)
            value_bck_colors=["#FFFFFF"] *
            len(params),  # fundo branco para valores
            blank_alpha=0.4,  # transparência do espaço em branco
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000",
                               fontsize=8,
                               fontweight="normal",
                               fontfamily="DejaVu Sans",
                               va="center",
                               zorder=3),
            kwargs_values=dict(color="#FF0000",
                               fontsize=11,
                               fontweight="bold",
                               zorder=5,
                               bbox=dict(edgecolor="#000000",
                                         facecolor="#FFFFFF",
                                         boxstyle="round,pad=0.2",
                                         lw=1,
                                         alpha=0.9)))

        # Adicionar jogador 2 se fornecido
        if values_p2 is not None:
            if len(values_p2) != len(params):
                raise ValueError(
                    f"Número de valores do jogador 2 ({len(values_p2)}) não corresponde ao número de parâmetros ({len(params)})"
                )

            # Adicionar linhas para o jogador 2
            for i, value in enumerate(values_p2):
                angle = (i / len(params)) * 2 * np.pi
                ax.plot([angle, angle], [0, value / 100],
                        color=player2_color,
                        linewidth=2.5,
                        linestyle='-',
                        zorder=10)

                # Adicionar valor em caixa para o jogador 2 (fundo branco e texto vermelho)
                if value > 25:  # Mostrar apenas valores relevantes
                    radius = value / 100
                    ax.text(angle,
                            radius + 0.05,
                            f"{value}",
                            color='#FF0000',
                            fontsize=9,
                            ha='center',
                            va='center',
                            fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor='#FFFFFF',
                                      alpha=0.9,
                                      edgecolor='#000000',
                                      linewidth=1))

        # Adicionar média do grupo se fornecida
        if values_avg is not None:
            if len(values_avg) != len(params):
                raise ValueError(
                    f"Número de valores médios ({len(values_avg)}) não corresponde ao número de parâmetros ({len(params)})"
                )

            # Adicionar linhas para média
            for i, value in enumerate(values_avg):
                angle = (i / len(params)) * 2 * np.pi
                ax.plot([angle, angle], [0, value / 100],
                        color=avg_color,
                        linewidth=2,
                        linestyle='--',
                        zorder=5,
                        alpha=0.7)

        # Adicionar título centralizado
        if title:
            title_text = title
        else:
            title_text = f"{p1_name}" + (f" vs {p2_name}"
                                         if values_p2 is not None else "")

        fig.text(0.5,
                 0.97,
                 title_text,
                 size=16,
                 ha="center",
                 fontweight="bold",
                 color="#000000")

        # Adicionar subtítulo centralizado
        if subtitle:
            fig.text(0.5,
                     0.93,
                     subtitle,
                     size=12,
                     ha="center",
                     color="#666666")

        # Adicionar créditos no canto inferior direito em itálico
        fig.text(0.95,
                 0.02,
                 "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
                 size=8,
                 ha="right",
                 color="#666666",
                 style='italic')

        # Valores nominais foram removidos conforme solicitado

        # Remover grade e ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Adicionar logo do Vålerenga ao centro do gráfico APÓS criar a pizza
        try:
            print(
                f"Tentando adicionar logo do Vålerenga ao gráfico pizza (DEPOIS DO PLOT)"
            )

            # Carregar o arquivo de imagem do Vålerenga diretamente
            logo_file = "attached_assets/Vålerenga_Oslo_logo.svg.png"

            if os.path.exists(logo_file):
                # Desenhar um círculo branco no centro para servir de fundo para o logo
                center_circle = plt.Circle((0, 0),
                                           0.08,
                                           facecolor='white',
                                           edgecolor='none',
                                           zorder=20)
                ax.add_patch(center_circle)

                # Carregar logo
                logo_img = plt.imread(logo_file)

                # Criar um novo axes no centro com tamanho apropriado
                # Posição relativa à figura: [left, bottom, width, height]
                logo_ax = fig.add_axes([0.46, 0.46, 0.08, 0.08], zorder=30)

                # Mostrar logo e remover eixos
                logo_ax.imshow(logo_img)
                logo_ax.axis('off')

                print(f"Logo adicionado com sucesso DEPOIS da pizza")
            else:
                print(f"Arquivo de logo não encontrado: {logo_file}")
        except Exception as e:
            print(f"Erro ao adicionar logo: {str(e)}")
            import traceback
            print(traceback.format_exc())

        # Adicionar legenda se necessário
        if values_p2 is not None or values_avg is not None:
            legend_elements = []

            # Jogador 1
            legend_elements.append(
                plt.Rectangle((0, 0),
                              1,
                              1,
                              facecolor=player1_color,
                              edgecolor='white',
                              label=p1_name))

            # Jogador 2
            if values_p2 is not None:
                legend_elements.append(
                    plt.Rectangle((0, 0),
                                  1,
                                  1,
                                  facecolor=player2_color,
                                  edgecolor='white',
                                  label=p2_name))

            # Média
            if values_avg is not None:
                legend_elements.append(
                    plt.Line2D([0], [0],
                               color=avg_color,
                               linewidth=2,
                               linestyle='--',
                               label='Média do Grupo'))

            # Posicionar legenda centralizada abaixo do gráfico
            ax.legend(handles=legend_elements,
                      loc='lower center',
                      bbox_to_anchor=(0.5, -0.1),
                      ncol=len(legend_elements),
                      frameon=True,
                      facecolor='white',
                      edgecolor='#CCCCCC')

    except Exception as e:
        # Em caso de erro, criar um gráfico com mensagem e imprimir o erro
        st.error(f"Erro detalhado: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5,
                0.5,
                f"Erro ao criar pizza chart: {str(e)}",
                ha='center',
                va='center',
                fontsize=12,
                color='#333333')
        ax.axis('off')

    return fig


@st.cache_data
def create_comparison_pizza_chart(params,
                                  values_p1,
                                  values_p2=None,
                                  values_avg=None,
                                  title=None,
                                  subtitle=None,
                                  p1_name="Player 1",
                                  p2_name="Player 2"):
    """
    Cria um pizza chart para comparação entre dois jogadores ou jogador vs média,
    usando o estilo do gráfico padrão mas com sobreposição direta das fatias.
    Inclui logo no centro do gráfico.
    """
    try:
        # Verificações de segurança
        if values_p1 is not None and len(params) != len(values_p1):
            raise ValueError(
                f"Número de parâmetros ({len(params)}) não corresponde ao número de valores ({len(values_p1)})"
            )

        # Determinar quais valores usar para comparação (valores_p2 ou values_avg)
        compare_values = None
        compare_name = p2_name
        if values_p2 is not None and len(values_p2) == len(params):
            compare_values = values_p2
        elif values_avg is not None and len(values_avg) == len(params):
            compare_values = values_avg
            compare_name = "Group Average"

        if compare_values is None:
            raise ValueError(
                "Valores de comparação não fornecidos (Player 2 ou Group Average)"
            )

        # Estamos sempre usando percentis para a visualização
        # Não precisamos mais do código de valores originais, pois simplificamos
        # para usar apenas percentis conforme solicitado pelo usuário

        # Para esta função, SEMPRE usamos os percentis para visualização
        # Removemos o parâmetro is_percentile pois sempre usamos percentis para consistência

        # Arredondar valores para inteiros (tanto percentis quanto valores normalizados)
        values_p1 = [round(v) for v in values_p1]
        compare_values = [round(v) for v in compare_values]

        # Definir cores - azul e vermelho para player vs player
        player1_color = "#1A78CF"  # Azul real para jogador 1
        player2_color = "#E41A1C"  # Vermelho para jogador 2 (mesma cor que usamos para média)
        avg_color = "#E41A1C"  # Vermelho para média do grupo
        text_color = "#000000"  # Preto para texto
        background_color = "#F5F5F5"  # Cinza claro para fundo

        # Usar vermelho como cor padrão para comparação (tanto para jogador 2 quanto para média)
        compare_color = player2_color  # Sempre vermelho

        # Ajustar os limites mínimos e máximos para os valores
        # Usar lógica do script de exemplo para ajustar o range dos valores
        min_range = [0] * len(params)  # Começar de zero sempre
        max_range = [100] * len(params)  # Máximo é sempre 100 para percentis

        # Abreviar os rótulos dos parâmetros para melhor visualização
        abbreviated_params = [
            abbreviate_metric_name(param) for param in params
        ]

        # Instanciar o objeto PyPizza com círculo interno padrão, logo será menor
        baker = PyPizza(
            params=abbreviated_params,  # parâmetros abreviados
            min_range=min_range,
            max_range=max_range,
            background_color=background_color,
            straight_line_color="#FFFFFF",  # linhas radiais brancas
            straight_line_lw=1.5,  # linhas mais grossas
            last_circle_lw=1.5,  # círculo externo mais visível
            other_circle_color="#FFFFFF",  # linhas circulares brancas
            other_circle_lw=1.5,  # largura das linhas circulares
            other_circle_ls="-",  # linhas sólidas para círculos
            inner_circle_size=5  # tamanho padrão, menor, para não interferir
        )

        # Usar o método make_pizza do PyPizza, que aceita compare_values diretamente
        # Isso criará automaticamente um gráfico com os dois jogadores sobrepostos
        fig, ax = baker.make_pizza(
            values_p1,  # valores do jogador 1
            compare_values=compare_values,  # valores do jogador 2 ou média
            figsize=(10, 10),  # tamanho da figura
            color_blank_space="same",  # espaço em branco com mesma cor
            blank_alpha=0.4,  # transparência do espaço em branco
            param_location=
            110,  # localização dos parâmetros (um pouco afastados)
            kwargs_slices=dict(facecolor=player1_color,
                               edgecolor="#F2F2F2",
                               zorder=2,
                               linewidth=1),
            kwargs_compare=dict(facecolor=compare_color,
                                edgecolor="#000000",
                                zorder=3,
                                linewidth=1,
                                alpha=0.8),
            kwargs_params=dict(color="#000000",
                               fontsize=8,
                               fontweight="normal",
                               fontfamily="DejaVu Sans",
                               va="center",
                               zorder=3),
            kwargs_values=dict(color=player1_color,
                               fontsize=11,
                               fontweight="bold",
                               zorder=5,
                               bbox=dict(edgecolor="#000000",
                                         facecolor="#FFFFFF",
                                         boxstyle="round,pad=0.2",
                                         lw=1,
                                         alpha=0.9)),
            kwargs_compare_values=dict(color=compare_color,
                                       fontsize=11,
                                       fontweight="bold",
                                       zorder=6,
                                       bbox=dict(edgecolor="#000000",
                                                 facecolor="#FFFFFF",
                                                 boxstyle="round,pad=0.2",
                                                 lw=1,
                                                 alpha=0.9)))

        # Ajustar os textos para evitar sobreposição (como no script de exemplo)
        params_offset = [True] * len(params)
        baker.adjust_texts(params_offset, offset=-0.15)

        # Centralizar e ajustar a figura com mais espaço para a legenda
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)

        # Adicionar título centralizado
        if title:
            title_text = title
        else:
            title_text = f"{p1_name}" + (f" vs {compare_name}"
                                         if compare_values is not None else "")

        fig.text(0.5,
                 0.97,
                 title_text,
                 size=16,
                 ha="center",
                 fontweight="bold",
                 color="#000000")

        # Adicionar subtítulo centralizado
        if subtitle:
            fig.text(0.5,
                     0.93,
                     subtitle,
                     size=12,
                     ha="center",
                     color="#666666")

        # Remover grade e ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Adicionar legenda para identificar os jogadores/média
        legend_elements = []

        # Jogador 1
        legend_elements.append(
            plt.Rectangle((0, 0),
                          1,
                          1,
                          facecolor=player1_color,
                          edgecolor='white',
                          label=p1_name))

        # Jogador 2 ou média
        legend_elements.append(
            plt.Rectangle((0, 0),
                          1,
                          1,
                          facecolor=compare_color,
                          edgecolor='white',
                          alpha=0.8,
                          label=compare_name))

        # Posicionar legenda centralizada abaixo do gráfico
        ax.legend(handles=legend_elements,
                  loc='lower center',
                  bbox_to_anchor=(0.5, -0.1),
                  ncol=len(legend_elements),
                  frameon=True,
                  facecolor='white',
                  edgecolor='#CCCCCC')

        # No gráfico de comparação não adicionamos logo

        # Adicionar créditos no canto inferior direito em itálico
        fig.text(0.95,
                 0.02,
                 "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
                 size=8,
                 ha="right",
                 color="#666666",
                 style='italic')

        return fig

    except Exception as e:
        # Em caso de erro, criar um gráfico com mensagem
        st.error(f"Erro na criação do pizza chart comparativo: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5,
                0.5,
                f"Erro ao criar pizza chart: {str(e)}",
                ha='center',
                va='center',
                fontsize=12,
                color='#333333')
        ax.axis('off')
        return fig


@st.cache_data
def create_bar_chart(metrics,
                     p1_name,
                     p1_values,
                     p2_name,
                     p2_values,
                     avg_values,
                     title=None,
                     subtitle=None):
    """Create horizontal bar chart using matplotlib"""
    # Aumentar o espaço superior para evitar sobreposição do título com legendas
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))

    # Handle single metric case
    if len(metrics) == 1:
        axes = [axes]

    # Definir cores consistentes com o gráfico pizza
    player1_color = "#1A78CF"  # Azul real para jogador 1
    player2_color = "#E41A1C"  # Vermelho para jogador 2

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot bars (usando as cores azul real e vermelho)
        y_pos = [0, 1]
        ax.barh(y_pos[0], p1_values[i], color=player1_color, height=0.6)
        ax.barh(y_pos[1], p2_values[i], color=player2_color, height=0.6)

        # Plot average line
        ax.axvline(x=avg_values[i], color='#2ca02c', linestyle='--', alpha=0.7)
        ax.text(avg_values[i],
                0.5,
                f'Group Avg',
                va='center',
                ha='right',
                rotation=90,
                color='#2ca02c',
                backgroundcolor='white',
                alpha=0.8)

        # Add metric name as title para cada subplot (não principal)
        ax.set_title(metrics[i], fontsize=12, pad=10)

        # Add player labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([p1_name, p2_name])

        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        # Add value labels
        ax.text(p1_values[i],
                y_pos[0],
                f' {p1_values[i]:.2f}',
                va='center',
                color=player1_color,
                fontweight='bold')
        ax.text(p2_values[i],
                y_pos[1],
                f' {p2_values[i]:.2f}',
                va='center',
                color=player2_color,
                fontweight='bold')

    # Ajustar layout primeiro para dar espaço adequado ao título
    plt.tight_layout(pad=1.2, h_pad=0.8, w_pad=0.5, rect=[0, 0, 1, 0.92])

    # Adicionar título principal e subtítulo com maior espaço acima
    if title:
        # Título acima dos gráficos com espaço suficiente para não sobrepor
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    if subtitle:
        # Subtítulo abaixo do título principal
        plt.figtext(0.5, 0.95, subtitle, ha='center', fontsize=10, wrap=True)

    # Adicionar legenda única para o gráfico inteiro
    legend_elements = [
        plt.Rectangle((0, 0),
                      1,
                      1,
                      facecolor=player1_color,
                      edgecolor='white',
                      label=p1_name),
        plt.Rectangle((0, 0),
                      1,
                      1,
                      facecolor=player2_color,
                      edgecolor='white',
                      label=p2_name),
        plt.Line2D([0], [0],
                   color='#2ca02c',
                   lw=2,
                   linestyle='--',
                   label='Group Average')
    ]

    # Posicionar a legenda centralizada no topo da figura (acima do título)
    # Isso evita sobreposições com os subplots individuais
    fig.legend(handles=legend_elements,
               loc='lower center',
               bbox_to_anchor=(0.5, 1.0),
               ncol=3,
               frameon=True,
               facecolor='white',
               edgecolor='#CCCCCC')

    return fig


@st.cache_data
def create_scatter_plot(df, x_metric, y_metric, title=None):
    """Create scatter plot using matplotlib with player names and hover annotations"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Definir cor consistente
    player1_color = "#1A78CF"  # Azul real para jogador 1

    # Identificar a coluna que contém os nomes dos jogadores
    player_column = None
    for possible_column in ['Player', 'player', 'NAME', 'Name', 'name']:
        if possible_column in df.columns:
            player_column = possible_column
            break
    
    # Se não encontrar nenhuma coluna de jogador, criar índices numéricos em vez de nomes
    if player_column is None:
        st.warning("Player column not found. Using numeric indices instead.")
        df = df.copy()
        df['PlayerIndex'] = [f"Player {i+1}" for i in range(len(df))]
        player_column = 'PlayerIndex'

    # Criar um dict para armazenar nomes de todos os jogadores
    player_names = {}

    # Calcular o número máximo de jogadores para rotular (por questão de legibilidade)
    max_labels = min(50, len(df))

    # Plot all players (diminuir tamanho dos pontos para um visual mais limpo)
    all_players = df[player_column].values
    x_values = df[x_metric].values
    y_values = df[y_metric].values

    # Scatter plot com pontos menores e maior nitidez
    sc = ax.scatter(x_values, y_values, alpha=0.7, s=30, c='gray')

    # Adicionar nomes abaixo dos pontos para todos os jogadores (discretamente)
    # Usar um algoritmo para evitar sobreposição de textos

    # Coletar coordenadas dos pontos para todos os jogadores
    player_positions = {}
    for i, player in enumerate(all_players):
        if i < len(x_values) and i < len(y_values):
            player_names[i] = player
            player_positions[player] = (x_values[i], y_values[i])

    # Adicionar rótulos para todos os jogadores (limitado a max_labels)
    # Utilizamos uma abordagem mais discreta, com textos pequenos e transparentes
    for i, (player,
            (x, y)) in enumerate(list(player_positions.items())[:max_labels]):
        # Adicionar textos abaixo dos pontos usando um estilo mais nítido
        ax.annotate(
            player,
            (x, y),
            xytext=(0, -7),  # Posição abaixo do ponto
            textcoords='offset points',
            fontsize=7,  # Fonte pequena
            ha='center',  # Centralizado
            va='top',  # Alinhado ao topo do texto
            alpha=0.8,  # Maior nitidez
            color='#333333')  # Cor mais escura para melhor visibilidade

    # Usar mplcursors para adicionar interatividade (hover)
    try:
        import mpld3
        from mpld3 import plugins

        # Adicionar tooltip com os nomes dos jogadores ao passar o mouse
        tooltip = plugins.PointHTMLTooltip(
            sc,
            labels=[
                f"<b>{p}</b><br>{x_metric}: {x_values[i]:.2f}<br>{y_metric}: {y_values[i]:.2f}"
                for i, p in player_names.items()
            ],
            voffset=10,
            hoffset=10)
        mpld3.plugins.connect(fig, tooltip)
    except Exception as e:
        # Em caso de erro com mpld3, silenciosamente prosseguir sem tooltip
        pass

    # Add mean lines
    ax.axvline(df[x_metric].mean(), color='#333333', linestyle='--', alpha=0.5)
    ax.axhline(df[y_metric].mean(), color='#333333', linestyle='--', alpha=0.5)

    # Add labels and title
    ax.set_xlabel(x_metric, fontsize=12)
    ax.set_ylabel(y_metric, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, pad=20)

    # Não temos mais highlight players, então não precisamos mais da legenda

    ax.grid(True, alpha=0.3)

    return fig


@st.cache_data
def create_similarity_viz(selected_player,
                          similar_players,
                          metrics,
                          df,
                          method='pca_kmeans'):
    """
    Create player similarity visualization based on the model from the reference website.
    Includes PCA visualization and radar charts in a single visualization.

    Args:
        selected_player: Name of the reference player
        similar_players: List of tuples (player_name, similarity_score)
        metrics: List of metrics used for similarity calculation
        df: DataFrame containing player data
        method: Similarity method used ('pca_kmeans', 'cosine', or 'euclidean')

    Returns:
        matplotlib figure object
    """
    # Check if we have any similar players
    if not similar_players:
        # Create a simple figure with just a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"No similar players found for {selected_player} based on the selected metrics.",
            ha='center',
            va='center',
            fontsize=14)
        ax.axis('off')
        return fig

    try:
        # Definir cores consistentes com o resto da aplicação
        player1_color = "#1A78CF"  # Azul real para jogador principal
        player2_color = "#E41A1C"  # Vermelho para jogador similar
        cluster_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]  # Cores para clusters

        # Criar figura principal
        fig = plt.figure(figsize=(15, 9))

        # Título principal
        fig.suptitle(f"Player Similarity Analysis: {selected_player}",
                     fontsize=18,
                     y=0.98)
        plt.figtext(
            0.5,
            0.95,
            f"Based on {len(metrics)} metrics including: {', '.join(metrics[:5])}{'...' if len(metrics) > 5 else ''}",
            ha='center',
            fontsize=10,
            fontstyle='italic')

        # Create grid layout for subplots
        gs = fig.add_gridspec(3, 6, height_ratios=[2, 2, 1])

        # Se estivermos usando o método PCA+K-Means, mostrar a visualização PCA
        if method == 'pca_kmeans':
            # Criar o dataframe PCA
            pca_df = create_pca_kmeans_df(df, metrics)

            if pca_df is not None:
                # Grid para o plot PCA
                ax_pca = fig.add_subplot(gs[0:2, 0:3])

                # Plotar todos os jogadores em cores baseadas no cluster
                for cluster_id in pca_df['cluster'].unique():
                    cluster_data = pca_df[pca_df['cluster'] == cluster_id]
                    ax_pca.scatter(cluster_data['x'],
                                   cluster_data['y'],
                                   alpha=0.5,
                                   s=50,
                                   c=cluster_colors[int(cluster_id) %
                                                    len(cluster_colors)],
                                   label=f'Cluster {cluster_id+1}')

                # Destacar o jogador principal
                player_point = pca_df[pca_df['Player'] == selected_player]
                ax_pca.scatter(player_point['x'],
                               player_point['y'],
                               s=150,
                               c=player1_color,
                               marker='*',
                               edgecolor='black',
                               linewidth=1.5,
                               label=selected_player)

                # Destacar jogadores similares
                similar_players_list = [p[0] for p in similar_players]
                similar_points = pca_df[pca_df['Player'].isin(
                    similar_players_list)]
                ax_pca.scatter(similar_points['x'],
                               similar_points['y'],
                               s=100,
                               c=player2_color,
                               marker='o',
                               edgecolor='black',
                               linewidth=1,
                               label='Similar Players')

                # Adicionar textos para o jogador selecionado e similares
                for _, row in player_point.iterrows():
                    ax_pca.annotate(row['Player'], (row['x'], row['y']),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=12,
                                    fontweight='bold',
                                    color=player1_color,
                                    bbox=dict(facecolor='white',
                                              alpha=0.8,
                                              edgecolor=player1_color,
                                              boxstyle="round,pad=0.2"))

                for _, row in similar_points.iterrows():
                    ax_pca.annotate(row['Player'], (row['x'], row['y']),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=10,
                                    color=player2_color,
                                    bbox=dict(facecolor='white',
                                              alpha=0.8,
                                              edgecolor=player2_color,
                                              boxstyle="round,pad=0.2"))

                # Configurações do plot PCA
                ax_pca.set_title('Player Similarity - PCA Visualization',
                                 fontsize=14)
                ax_pca.set_xlabel('Principal Component 1', fontsize=10)
                ax_pca.set_ylabel('Principal Component 2', fontsize=10)
                ax_pca.legend(loc='upper right', fontsize=8)
                ax_pca.grid(alpha=0.3)

                # Adicionar uma explicação sobre PCA
                explanation_text = (
                    "Principal Component Analysis (PCA) reduces multiple metrics to two dimensions, "
                    "allowing visualization of player similarity. Players closer together have more similar styles."
                )
                ax_pca.text(0.5,
                            -0.08,
                            explanation_text,
                            transform=ax_pca.transAxes,
                            fontsize=8,
                            ha='center',
                            va='center',
                            bbox=dict(facecolor='white',
                                      alpha=0.8,
                                      boxstyle="round,pad=0.3"))

        # Top similar players table
        ax_table = fig.add_subplot(gs[0:2, 3:6])
        ax_table.axis('off')

        # Mostrar tabela de similaridade
        table_data = []
        for i, (player_name, similarity) in enumerate(similar_players[:10], 1):
            # Obter dados básicos do jogador
            player_info = df[df['Player'] == player_name]
            if not player_info.empty:
                team = player_info.iloc[0].get('Team', 'N/A')
                age = player_info.iloc[0].get('Age', 'N/A')
                position = player_info.iloc[0].get('Position', 'N/A')

                table_data.append([
                    f"{i}. {player_name}", team,
                    f"{age:.1f}" if isinstance(age, (int, float)) else age,
                    position, f"{similarity:.1f}%"
                ])

        # Criar a tabela
        if table_data:
            table = ax_table.table(
                cellText=table_data,
                colLabels=['Player', 'Team', 'Age', 'Position', 'Similarity'],
                cellLoc='center',
                loc='center',
                colWidths=[0.35, 0.25, 0.1, 0.15, 0.15])

            # Configurações de estilo da tabela
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Estilizar o cabeçalho
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Cabeçalho
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor(player1_color)
                else:
                    # Alternar cores das linhas
                    if i % 2 == 0:
                        cell.set_facecolor('#f0f0f0')
                    # Colorir célula de similaridade
                    if j == 4:  # Coluna de similaridade
                        similarity_value = float(table_data[i -
                                                            1][4].strip('%'))
                        # Gradiente de cor vermelho-amarelo-verde baseado na similaridade
                        if similarity_value >= 80:
                            cell.set_facecolor('#d4f7d4')  # Verde claro
                        elif similarity_value >= 60:
                            cell.set_facecolor('#fffacd')  # Amarelo claro
                        else:
                            cell.set_facecolor('#ffecec')  # Vermelho claro

            ax_table.set_title('Top Similar Players', fontsize=14)

        # Criar radar chart para o jogador selecionado e o jogador mais similar
        if similar_players:
            # Obter os dados do jogador selecionado e do mais similar
            selected_data = df[df['Player'] == selected_player]
            similar_data = df[df['Player'] == similar_players[0][0]]

            if not selected_data.empty and not similar_data.empty:
                # Criar um grid para o radar
                ax_radar = fig.add_subplot(gs[2, 1:5], polar=True)

                # Selecionar métricas para o radar (limitando a 8 para não ficar poluído)
                radar_metrics = metrics[:8] if len(metrics) > 8 else metrics

                # Calcular percentis
                values_selected = [
                    calc_percentile(df[m], selected_data.iloc[0][m]) * 100
                    for m in radar_metrics
                ]
                values_similar = [
                    calc_percentile(df[m], similar_data.iloc[0][m]) * 100
                    for m in radar_metrics
                ]

                # Inicializar o radar
                radar = Radar(radar_metrics,
                              min_range=[0] * len(radar_metrics),
                              max_range=[100] * len(radar_metrics),
                              round_int=[True] * len(radar_metrics),
                              num_rings=4,
                              ring_width=1,
                              center_circle_radius=1)

                # Preparar o eixo e desenhar os círculos
                ax_radar = radar.setup_axis(ax=ax_radar)
                rings_inner = radar.draw_circles(ax=ax_radar,
                                                 facecolor='#f9f9f9',
                                                 edgecolor='#c5c5c5')

                # Desenhar o radar para o jogador selecionado
                radar_poly1, rings_outer1, vertices1 = radar.draw_radar(
                    values_selected,
                    ax=ax_radar,
                    kwargs_radar={
                        'facecolor': player1_color,
                        'alpha': 0.6,
                        'edgecolor': player1_color,
                        'linewidth': 1.5
                    },
                    kwargs_rings={
                        'facecolor': player1_color,
                        'alpha': 0.1
                    })

                # Desenhar o radar para o jogador similar
                radar_poly2, rings_outer2, vertices2 = radar.draw_radar(
                    values_similar,
                    ax=ax_radar,
                    kwargs_radar={
                        'facecolor': player2_color,
                        'alpha': 0.6,
                        'edgecolor': player2_color,
                        'linewidth': 1.5
                    },
                    kwargs_rings={
                        'facecolor': player2_color,
                        'alpha': 0.1
                    })

                # Adicionar legenda
                legend_elements = [
                    plt.Line2D([0], [0],
                               marker='o',
                               color='w',
                               markerfacecolor=player1_color,
                               markersize=10,
                               label=selected_player),
                    plt.Line2D([0], [0],
                               marker='o',
                               color='w',
                               markerfacecolor=player2_color,
                               markersize=10,
                               label=similar_players[0][0])
                ]
                ax_radar.legend(handles=legend_elements,
                                loc='upper right',
                                fontsize=9)

                # Título do radar
                ax_radar.set_title(
                    f"Percentile Radar Comparison with Most Similar Player",
                    fontsize=12)

        # Adicionar créditos no canto inferior direito em itálico
        fig.text(0.95,
                 0.01,
                 "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
                 size=8,
                 ha="right",
                 color="#666666",
                 style='italic')

        # Adjust spacing and layout - use larger bottom margin to fix the "bottom" error
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig

    except Exception as e:
        # Create a simple figure with error description in English
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5,
                0.5,
                f"Could not generate similarity visualization:\n{str(e)}",
                ha='center',
                va='center',
                fontsize=14,
                wrap=True)
        ax.axis('off')
        return fig


@st.cache_data
def create_pca_kmeans_df(df, metrics, n_clusters=8):
    """
    Create a PCA and K-Means clustering model for player similarity.
    This follows the methodology from the reference website.

    Args:
        df: DataFrame containing player data
        metrics: List of metrics to use for similarity calculation
        n_clusters: Number of clusters for K-Means

    Returns:
        DataFrame with player information, PCA coordinates, and cluster assignments
    """
    try:
        # Safety check: ensure all metrics exist in dataframe
        valid_metrics = [m for m in metrics if m in df.columns]
        if len(valid_metrics) != len(metrics):
            missing = set(metrics) - set(valid_metrics)
            st.warning(f"Some metrics were not found: {missing}")
            if not valid_metrics:
                return None

        # Keep player information
        player_info = df[['Player', 'Team', 'Position', 'Age']].copy()

        # Extract feature columns for valid metrics only
        X = df[valid_metrics].copy()

        # Handle missing values in features
        if X.isna().any().any():
            st.info(
                "Missing values detected in metrics. They will be filled with column means."
            )
            X = X.fillna(X.mean())

        # Normalize/Scale the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Reduce dimensionality with PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)

        # Add PCA coordinates to player info
        player_info['x'] = pca_result[:, 0]
        player_info['y'] = pca_result[:, 1]

        # Compute K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Add cluster assignments to player info
        player_info['cluster'] = cluster_labels

        # Get variance explained by PCA
        variance_explained = pca.explained_variance_ratio_.sum()
        st.info(
            f"PCA with 2 components explains {variance_explained:.1%} of the variance in the data"
        )

        return player_info

    except Exception as e:
        st.error(f"Error in K-Means clustering: {str(e)}")
        return None


def compute_player_similarity(df, player, metrics, n=5, method='pca_kmeans'):
    """
        Compute player similarity using PCA and K-Means clustering.
        This follows the methodology from the reference website.

        Args:
            df: DataFrame containing player data
            player: Name of the reference player
            metrics: List of metrics to use for similarity calculation
            n: Number of similar players to return
            method: Similarity calculation method ('pca_kmeans', 'cosine', or 'euclidean')

        Returns:
            List of tuples (player_name, similarity_score)
        """
    # Safety check: make sure player exists in the dataframe
    if player not in df['Player'].values:
        st.error(f"Player '{player}' not found in the filtered dataset!")
        return []

    try:
        # If using the new PCA + K-Means method
        if method == 'pca_kmeans':
            # Create PCA + K-Means DataFrame
            pca_df = create_pca_kmeans_df(df, metrics)

            if pca_df is None:
                return []

            # Get the reference player
            ref_player = pca_df[pca_df['Player'] == player].iloc[0]

            # Calculate Euclidean distance in PCA space
            pca_df['distance'] = np.sqrt((pca_df['x'] - ref_player['x'])**2 +
                                         (pca_df['y'] - ref_player['y'])**2)

            # Convert distance to similarity percentage (higher is more similar)
            max_distance = pca_df['distance'].max()
            pca_df['similarity'] = (
                (max_distance - pca_df['distance']) / max_distance) * 100

            # Get the most similar players (excluding the reference player)
            similar = pca_df.sort_values('distance').reset_index(drop=True)
            similar = similar[similar['Player'] != player].head(n)

            # Check if we have any similar players
            if similar.empty:
                return []

            # Format and return the similar players
            similar_players = [(row['Player'], row['similarity'])
                               for _, row in similar.iterrows()]
            return similar_players

        # Using the original vector-based methods
        else:
            # Safety check: ensure all metrics exist in dataframe
            valid_metrics = [m for m in metrics if m in df.columns]
            if len(valid_metrics) != len(metrics):
                missing = set(metrics) - set(valid_metrics)
                st.warning(f"Some metrics were not found: {missing}")
                if not valid_metrics:
                    return []

            # Extract feature columns for valid metrics only
            X = df[valid_metrics].values

            # Handle missing values in features
            if np.isnan(X).any():
                st.warning(
                    "Missing values detected in metrics. They will be filled with column means."
                )
                # Replace NaNs with column means
                col_means = np.nanmean(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_means, inds[1])

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Get index of player - using DataFrame.index now for safety
            player_data = df[df['Player'] == player]
            if player_data.empty:
                st.error(f"Player '{player}' not found in dataset")
                return []

            player_idx = player_data.index[0]

            # Make sure player_idx is within the bounds of X_scaled
            if player_idx >= X_scaled.shape[0]:
                st.error(
                    f"Player index out of bounds: {player_idx} >= {X_scaled.shape[0]}"
                )
                return []

            # Compute similarities based on method
            if method == 'cosine':
                # Cosine similarity (higher is more similar)
                similarities = cosine_similarity([X_scaled[player_idx]],
                                                 X_scaled)[0]
                # Convert to similarity score (0 to 1, higher is better)
                similarities = (similarities +
                                1) / 2  # Convert from [-1,1] to [0,1]
            else:
                # Euclidean distance (lower is more similar)
                distances = cdist([X_scaled[player_idx]], X_scaled,
                                  'euclidean')[0]
                # Convert to similarity score (0 to 1, higher is better)
                max_dist = np.max(distances)
                similarities = 1 - (distances / max_dist)

            # Get player indices sorted by similarity (excluding the player itself)
            similar_indices = np.argsort(similarities)[::-1]
            similar_indices = [i for i in similar_indices
                               if i != player_idx][:n]

            # Return player names and similarity scores
            similar_players = [(df.iloc[i]['Player'], similarities[i] * 100)
                               for i in similar_indices]
            return similar_players

    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return []


# =============================================
# Main Application
# =============================================

# Inicializar variáveis adicionais de cache no session_state
if 'cached_df' not in st.session_state:
    st.session_state.cached_df = None
if 'cached_leagues' not in st.session_state:
    st.session_state.cached_leagues = None
if 'last_data_update' not in st.session_state:
    st.session_state.last_data_update = None
if 'data_needs_reload' not in st.session_state:
    st.session_state.data_needs_reload = True
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'file_upload_key' not in st.session_state:
    st.session_state.file_upload_key = 0

# Cabeçalho com logo
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    # Instead of using an image file, let's use a title and emoji
    st.title("⚽ Football Analytics")

st.header('Technical Scouting Department')
st.subheader('Football Analytics Dashboard')
st.caption(
    "Created by João Alberto Kolling | Enhanced Player Analysis System v4.0")

# Inicializar variável de sessão para fonte de dados
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'skillcorner'

# Guia do Usuário
with st.expander("📘 User Guide & Instructions", expanded=False):
    st.markdown("""
    **⚠️ Requirements:**  
    1. Data must contain columns: Player, Age, Position, Metrics, Team  

    **Key Features:**  
    - Player comparison with radar/barcharts  
    - Metric correlation analysis  
    - Advanced filtering system  
    - Player similarity modeling  
    - Professional 300 DPI exports  
    """)

# =============================================
# Filtros da Barra Lateral
# =============================================
st.sidebar.header('Filters')

# Inicializar a fonte de dados no session_state, se não existir
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'skillcorner'

# Opção para escolher a fonte de dados (SkillCorner ou WyScout)
data_source = st.sidebar.checkbox(
    "Use WyScout data",
    value=True if st.session_state.data_source == 'wyscout' else False,
    key="source_checkbox",
    help=
    "When checked, the application will load WyScout data from the 'wyscout_data' folder. When unchecked, it will use SkillCorner Integrated data from 'attached_assets'."
)

# Atualizar a fonte de dados no estado da sessão
previous_source = st.session_state.data_source
if data_source:
    # Checkbox marcado = WyScout
    if st.session_state.data_source != 'wyscout':
        st.session_state.data_source = 'wyscout'
        # Recarregar as ligas disponíveis
        AVAILABLE_LEAGUES = load_available_leagues('wyscout')
        # Marcar que os dados precisam ser recarregados
        st.session_state.data_needs_reload = True
        # Limpar os filtros avançados ao mudar de fonte para evitar inconsistências
        st.session_state.last_players = []
        st.session_state.saved_pizza_p1 = None
        st.session_state.saved_pizza_p2 = None
else:
    # Checkbox desmarcado = SkillCorner
    if st.session_state.data_source != 'skillcorner':
        st.session_state.data_source = 'skillcorner'
        # Recarregar as ligas disponíveis
        AVAILABLE_LEAGUES = load_available_leagues('skillcorner')
        # Marcar que os dados precisam ser recarregados
        st.session_state.data_needs_reload = True
        # Limpar os filtros avançados ao mudar de fonte para evitar inconsistências
        st.session_state.last_players = []
        st.session_state.saved_pizza_p1 = None
        st.session_state.saved_pizza_p2 = None

# Se mudou de fonte de dados, limpar o cache
if previous_source != st.session_state.data_source:
    st.session_state.cached_df = None
    st.session_state.cached_leagues = None

# Criar a pasta para dados WyScout se ela não existir
if st.session_state.data_source == 'wyscout':
    wyscout_dir = "wyscout_data"
    if not os.path.exists(wyscout_dir):
        os.makedirs(wyscout_dir)
        st.sidebar.info(
            f"Created directory '{wyscout_dir}' for WyScout data. Please add your Excel files there."
        )
    
    # Verificar se há arquivos na pasta
    if os.path.exists(wyscout_dir) and os.path.isdir(wyscout_dir):
        excel_files = [f for f in os.listdir(wyscout_dir) if f.endswith('.xlsx')]
        if not excel_files:
            st.sidebar.warning(f"No Excel files found in '{wyscout_dir}' directory. Please add some Excel files.")
        else:
            st.sidebar.success(f"Found {len(excel_files)} Excel files in '{wyscout_dir}' directory.")

# Função para processar arquivos carregados via upload
def process_uploaded_file(uploaded_file, league_name=None):
    """
    Processa um arquivo Excel carregado via upload.
    
    Args:
        uploaded_file: Arquivo Excel carregado via st.file_uploader
        league_name: Nome opcional da liga para identificação (usa nome do arquivo se não for fornecido)
    
    Returns:
        DataFrame processado, nome da liga
    """
    try:
        # Carregar o arquivo em um DataFrame
        df = pd.read_excel(uploaded_file)
        
        # Limpeza básica
        df.dropna(how="all", inplace=True)
        df = df.loc[:, df.columns.notnull()]
        df.columns = [str(c).strip() for c in df.columns]
        
        # Obter nome da liga do nome do arquivo se não for fornecido
        if league_name is None:
            # Pegar o nome do arquivo sem extensão
            league_name = os.path.splitext(uploaded_file.name)[0]
        
        # Adicionar informações de origem
        df['Data Origin'] = league_name
        df['Season'] = "2023/2024"
        
        # Calcular métricas ofensivas
        df = calculate_offensive_metrics(df)
        
        return df, league_name
    except Exception as e:
        st.error(f"Error processing uploaded file {uploaded_file.name}: {str(e)}")
        return None, None

# Interface para upload e seleção de arquivos/ligas
with st.sidebar.expander("📤 Upload Excel Files", expanded=True):
    st.markdown("### Upload your own Excel files")
    
    # Upload de arquivos Excel
    uploaded_files = st.file_uploader(
        "Upload Excel files with player data",
        type=["xlsx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_upload_key}"
    )
    
    # Processar arquivos carregados
    if uploaded_files:
        if st.button("Process Uploaded Files"):
            # Limpar cache anterior se necessário
            st.session_state.cached_df = None
            st.session_state.cached_leagues = None
            st.session_state.data_needs_reload = True
            
            # Inicializar dicionário de arquivos carregados se não existir
            if 'uploaded_files' not in st.session_state:
                st.session_state.uploaded_files = {}
            
            # Verificar se atingimos o limite de 15 ligas
            current_league_count = len(st.session_state.uploaded_files)
            max_leagues = 15
            
            with st.spinner("Processing uploaded files..."):
                for file in uploaded_files:
                    # Verificar se este arquivo já foi processado
                    if file.name not in st.session_state.uploaded_files:
                        # Verificar o limite de ligas
                        if current_league_count >= max_leagues:
                            st.warning(f"Maximum limit of {max_leagues} leagues reached. Remove some files to add more.")
                            break
                        
                        # Processar o arquivo
                        df, league_name = process_uploaded_file(file)
                        
                        if df is not None:
                            # Armazenar o DataFrame no session_state
                            st.session_state.uploaded_files[file.name] = {
                                'df': df,
                                'league_name': league_name
                            }
                            st.success(f"Processed: {file.name} ({len(df)} players)")
                            current_league_count += 1
            
            # Incrementar a chave para forçar o recarregamento do uploader (evita problemas com cache)
            st.session_state.file_upload_key += 1
            
            # Recarregar a página para aplicar as alterações
            st.rerun()
    
    # Mostrar arquivos já carregados
    if st.session_state.uploaded_files:
        st.markdown("### Uploaded Files")
        
        uploaded_leagues = [info['league_name'] for info in st.session_state.uploaded_files.values()]
        total_players = sum(len(info['df']) for info in st.session_state.uploaded_files.values())
        
        st.success(f"You have {len(st.session_state.uploaded_files)} files loaded ({total_players} players total)")
        
        # Mostrar tabela com arquivos carregados e opção de remover individualmente
        file_names = list(st.session_state.uploaded_files.keys())
        league_names = [st.session_state.uploaded_files[f]['league_name'] for f in file_names]
        player_counts = [len(st.session_state.uploaded_files[f]['df']) for f in file_names]
        
        # Criar DataFrame para exibição
        files_df = pd.DataFrame({
            'File': file_names,
            'League': league_names,
            'Players': player_counts
        })
        
        # Exibir tabela com os arquivos
        st.dataframe(files_df)
        
        # Opção para remover arquivos específicos
        remove_file = st.selectbox("Select file to remove", 
                                  ["None"] + file_names,
                                  index=0,
                                  help="Select a file to remove from the loaded files")
        
        if remove_file != "None" and st.button(f"Remove '{remove_file}'"):
            # Remover o arquivo do dicionário de arquivos
            if remove_file in st.session_state.uploaded_files:
                del st.session_state.uploaded_files[remove_file]
                st.session_state.cached_df = None
                st.session_state.cached_leagues = None
                st.session_state.data_needs_reload = True
                st.success(f"Removed file: {remove_file}")
                st.rerun()
        
        # Permitir selecionar quais arquivos carregados usar
        selected_leagues = st.multiselect(
            "Select leagues to analyze",
            options=uploaded_leagues,
            default=uploaded_leagues[:min(3, len(uploaded_leagues))]
        )
        
        # Opção para escolher como calcular os percentis
        if 'combine_leagues' not in st.session_state:
            st.session_state.combine_leagues = True
            
        combine_leagues = st.checkbox(
            "Combine all selected leagues into one dataset for percentile calculation",
            value=st.session_state.combine_leagues,
            help="When checked, all leagues are combined into one dataset and percentiles are calculated globally. When unchecked, percentiles are calculated separately within each league."
        )
        
        # Atualizar o estado da sessão se houver mudança
        if combine_leagues != st.session_state.combine_leagues:
            st.session_state.combine_leagues = combine_leagues
            st.session_state.cached_df = None  # Forçar recálculo
            st.session_state.data_needs_reload = True
        
        # Se todos os arquivos forem removidos, limpar o cache
        if not selected_leagues and st.session_state.uploaded_files:
            if st.button("Clear All Uploaded Files"):
                st.session_state.uploaded_files = {}
                st.session_state.cached_df = None
                st.session_state.cached_leagues = None
                st.session_state.data_needs_reload = True
                st.rerun()
    else:
        st.info("No files uploaded yet. Please upload Excel files with player data.")
        selected_leagues = []

# Se não houver arquivos carregados via upload, mostrar a seleção tradicional de ligas
if not st.session_state.uploaded_files:
    with st.sidebar.expander("⚙️ Select Leagues", expanded=True):
        current_source_text = "WyScout" if st.session_state.data_source == 'wyscout' else "SkillCorner Integrated"
        st.markdown(f"**Current data source: {current_source_text}**")
    
        if len(AVAILABLE_LEAGUES) == 0:
            st.warning(
                f"No {current_source_text} data files found. Please add data files to the appropriate folder."
            )
            # Mostrar mensagem específica para fonte de dados
            if st.session_state.data_source == 'wyscout':
                st.info(
                    "Please add your WyScout Excel files to the 'wyscout_data' folder."
                )
            else:
                st.info(
                    "Please add your SkillCorner Excel files to the 'attached_assets' folder."
                )
            
            # Atualizar a variável de seleção específica da fonte atual
            if st.session_state.data_source == 'wyscout':
                st.session_state.selected_leagues_wyscout = []
                selected_leagues = []
            else:
                st.session_state.selected_leagues_skillcorner = []
                selected_leagues = []
        else:
            # Verificar se há ligas disponíveis que não estão mais na lista de opções
            # para limpar seleções inválidas
            valid_leagues = list(AVAILABLE_LEAGUES.keys())
            
            # Obter a seleção anterior específica para a fonte de dados atual
            if st.session_state.data_source == 'wyscout':
                previous_selection = st.session_state.selected_leagues_wyscout
            else:
                previous_selection = st.session_state.selected_leagues_skillcorner
                
            # Filtrar para manter apenas ligas válidas
            filtered_selection = [
                league for league in previous_selection
                if league in valid_leagues
            ]
            
            # Se não houver seleção válida, definir o padrão para a primeira liga disponível
            default_selection = filtered_selection
            if not default_selection and AVAILABLE_LEAGUES:
                default_selection = [list(AVAILABLE_LEAGUES.keys())[0]]
            
            # Usar st.multiselect com o valor salvo específico para a fonte atual
            selected_leagues = st.multiselect(
                "Select leagues to analyze",
                options=valid_leagues,
                default=default_selection)
            
            # Atualizar a seleção específica da fonte atual no session_state
            if st.session_state.data_source == 'wyscout':
                st.session_state.selected_leagues_wyscout = selected_leagues
            else:
                st.session_state.selected_leagues_skillcorner = selected_leagues

    # Opção para escolher como calcular os percentis
    if 'combine_leagues' not in st.session_state:
        st.session_state.combine_leagues = True

    combine_leagues = st.checkbox(
        "Combine all selected leagues into one dataset for percentile calculation",
        value=st.session_state.combine_leagues,
        help=
        "When checked, all leagues are combined into one dataset and percentiles are calculated globally. When unchecked, percentiles are calculated separately within each league."
    )

    # Atualizar o estado da sessão se houver mudança
    if combine_leagues != st.session_state.combine_leagues:
        st.session_state.combine_leagues = combine_leagues

    # Separador visual
    st.sidebar.markdown("---")

    # Opção para carregar benchmark
    st.sidebar.subheader("Benchmark Database")

    # Inicializar variáveis de estado para o benchmark
    if 'benchmark_loaded' not in st.session_state:
        st.session_state.benchmark_loaded = False
        st.session_state.benchmark_df = None
        st.session_state.benchmark_name = ""

    # Abas para carregar benchmark de diferentes fontes
    benchmark_source = st.sidebar.radio(
        "Benchmark Source:", ["From Local Files", "Upload File"],
        help="Choose where to load the benchmark data from")

    if benchmark_source == "From Local Files":
        # Opção para usar dados locais não selecionados como dados principais
        # Evitar escolher ligas que já estão na seleção principal
        current_main_leagues = selected_leagues if selected_leagues else []

        # Filtrar ligas disponíveis para não duplicar com as já selecionadas
        available_benchmark_leagues = [
            league for league in AVAILABLE_LEAGUES.keys()
            if league not in current_main_leagues
        ]

        if available_benchmark_leagues:
            benchmark_leagues = st.sidebar.multiselect(
                "Select benchmark league(s)",
                options=available_benchmark_leagues,
                help="Select one or more leagues to use as benchmark")

            if st.sidebar.button("Load Local Benchmark"):
                try:
                    with st.spinner("Loading benchmark data..."):
                        benchmark_dfs = []

                        for league_name in benchmark_leagues:
                            try:
                                file_path = AVAILABLE_LEAGUES[league_name]
                                df = pd.read_excel(file_path)
                                df.dropna(how="all", inplace=True)
                                df = df.loc[:, df.columns.notnull()]
                                df.columns = [
                                    str(c).strip() for c in df.columns
                                ]
                                df['Data Origin'] = league_name
                                df['Season'] = "2023/2024"
                                benchmark_dfs.append(df)
                            except Exception as e:
                                st.sidebar.error(
                                    f"Error loading {league_name}: {str(e)}")

                        if benchmark_dfs:
                            # Combinar os dataframes se houver mais de um
                            benchmark_df = pd.concat(benchmark_dfs,
                                                     ignore_index=True)

                            # Verificar e processar as colunas necessárias
                            required_cols = [
                                'Player', 'Team', 'Age', 'Position',
                                'Minutes played'
                            ]
                            if all(col in benchmark_df.columns
                                   for col in required_cols):
                                # Processamento básico
                                if 'Minutes per game' not in benchmark_df.columns and 'Matches played' in benchmark_df.columns:
                                    benchmark_df[
                                        'Minutes per game'] = benchmark_df[
                                            'Minutes played'] / benchmark_df[
                                                'Matches played']

                                # Limpeza básica
                                benchmark_df = benchmark_df.fillna(0)

                                # Calcular métricas ofensivas avançadas para o benchmark
                                benchmark_df = calculate_offensive_metrics(
                                    benchmark_df)

                                # Guardar no session_state
                                st.session_state.benchmark_df = benchmark_df
                                st.session_state.benchmark_loaded = True
                                # Nome do benchmark combinado
                                benchmark_name = " & ".join(benchmark_leagues)
                                st.session_state.benchmark_name = benchmark_name

                                st.sidebar.success(
                                    f"Benchmark '{benchmark_name}' loaded successfully with {len(benchmark_df)} players!"
                                )
                            else:
                                missing = [
                                    col for col in required_cols
                                    if col not in benchmark_df.columns
                                ]
                                st.sidebar.error(
                                    f"Benchmark file is missing required columns: {', '.join(missing)}"
                                )
                        else:
                            st.sidebar.error(
                                "No benchmark data was loaded. Please select at least one league."
                            )
                except Exception as e:
                    st.sidebar.error(f"Error loading benchmark: {str(e)}")
        else:
            st.sidebar.warning(
                "All available leagues are already selected as main data. Please deselect some leagues from main data to use them as benchmark."
            )

    else:  # Upload File option
        benchmark_file = st.sidebar.file_uploader(
            "Upload benchmark Excel file",
            type=["xlsx"],
            help=
            "Upload a separate database (e.g., Premier League) to use as a benchmark for comparison"
        )

        # Se um arquivo de benchmark foi carregado
        if benchmark_file:
            benchmark_name = st.sidebar.text_input(
                "Benchmark name (e.g., 'Premier League')",
                key="benchmark_name_input",
                value="Benchmark League")

            if st.sidebar.button("Load File Benchmark"):
                try:
                    with st.spinner("Loading benchmark data..."):
                        # Carregar o arquivo de benchmark
                        benchmark_df = pd.read_excel(benchmark_file)

                        # Verificar e processar as colunas necessárias
                        required_cols = [
                            'Player', 'Team', 'Age', 'Position',
                            'Minutes played'
                        ]
                        if all(col in benchmark_df.columns
                               for col in required_cols):
                            # Processamento básico (o mesmo aplicado aos arquivos regulares)
                            if 'Minutes per game' not in benchmark_df.columns and 'Matches played' in benchmark_df.columns:
                                benchmark_df[
                                    'Minutes per game'] = benchmark_df[
                                        'Minutes played'] / benchmark_df[
                                            'Matches played']

                            # Limpeza básica
                            benchmark_df = benchmark_df.fillna(0)

                            # Calcular métricas ofensivas avançadas para o benchmark
                            benchmark_df = calculate_offensive_metrics(
                                benchmark_df)

                            # Guardar no session_state
                            st.session_state.benchmark_df = benchmark_df
                            st.session_state.benchmark_loaded = True
                            st.session_state.benchmark_name = benchmark_name

                            st.sidebar.success(
                                f"Benchmark '{benchmark_name}' loaded successfully with {len(benchmark_df)} players!"
                            )
                        else:
                            missing = [
                                col for col in required_cols
                                if col not in benchmark_df.columns
                            ]
                            st.sidebar.error(
                                f"Benchmark file is missing required columns: {', '.join(missing)}"
                            )
                except Exception as e:
                    st.sidebar.error(f"Error loading benchmark: {str(e)}")

    # Status do benchmark
    if st.session_state.benchmark_loaded:
        st.sidebar.info(
            f"Current benchmark: {st.session_state.benchmark_name} ({len(st.session_state.benchmark_df)} players)"
        )
        if st.sidebar.button("Reset Benchmark"):
            st.session_state.benchmark_loaded = False
            st.session_state.benchmark_df = None
            st.session_state.benchmark_name = ""
            st.sidebar.success("Benchmark has been reset")

st.sidebar.markdown("---")
st.sidebar.subheader("Dataframe Filters")

# CONTEÚDO PRINCIPAL - Começa aqui (fora do sidebar)
if selected_leagues:
    # Verificar se podemos usar dados em cache
    need_to_reload = False
    
    # Verificar se os dados já estão em cache e se são as mesmas ligas
    if (st.session_state.cached_df is None or 
        st.session_state.cached_leagues is None or 
        st.session_state.data_needs_reload or
        set(st.session_state.cached_leagues) != set(selected_leagues)):
        need_to_reload = True
    
    if need_to_reload:
        try:
            # Mostrar status de carregamento
            with st.spinner(f"Loading data from {len(selected_leagues)} leagues..."):
                df = load_league_data(selected_leagues)
                
                # Salvar no cache do session_state
                st.session_state.cached_df = df
                st.session_state.cached_leagues = selected_leagues.copy()
                st.session_state.last_data_update = pd.Timestamp.now()
                st.session_state.data_needs_reload = False
                
                # Indicar ao usuário que os dados foram atualizados
                st.success(f"Data refreshed at {st.session_state.last_data_update.strftime('%H:%M:%S')}")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.exception(e)
            st.stop()
    else:
        # Usar dados do cache
        df = st.session_state.cached_df
        if st.session_state.last_data_update:
            st.info(f"Using cached data from {st.session_state.last_data_update.strftime('%H:%M:%S')} - {len(selected_leagues)} leagues ({len(df)} players)")
        
        # Botão para forçar recarregar dados se necessário
        if st.button("Refresh Data", help="Force reload data from the selected leagues"):
            st.session_state.data_needs_reload = True
            st.rerun()

    if df.empty:
        st.error(
            "Failed to load data from selected leagues or no data available."
        )
        st.stop()

    # Verificar se colunas essenciais existem
    required_cols = ['Player', 'Minutes played', 'Matches played', 'Age']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(
            f"Required columns are missing: {', '.join(missing_cols)}")
        st.stop()

    # Adicionar controle para processamento de filtros
    st.sidebar.subheader("Player Filters")
    
    # Inicializar variáveis para armazenar as seleções temporárias
    if 'temp_minutes_range' not in st.session_state:
        st.session_state.temp_minutes_range = st.session_state.last_minutes_range
    if 'temp_mpg_range' not in st.session_state:
        st.session_state.temp_mpg_range = st.session_state.last_mpg_range
    if 'temp_age_range' not in st.session_state:
        st.session_state.temp_age_range = st.session_state.last_age_range
    if 'temp_positions' not in st.session_state:
        st.session_state.temp_positions = st.session_state.last_positions.copy() if st.session_state.last_positions else []
    
    # Inicializar o dataframe filtrado
    # IMPORTANTE: Criamos duas cópias - uma para filtros de minutos apenas (para percentis)
    # e outra para todos os filtros (para visualização)
    df_percentiles = df.copy()  # Este será filtrado APENAS por minutos para cálculo de percentis
    df_filtered = df.copy()     # Este será filtrado por todos os critérios para visualização

    # Carregar os valores atuais dos filtros
    try:
        # Filtro de minutos jogados
        min_min, max_min = int(df['Minutes played'].min()), int(df['Minutes played'].max())
        
        # Usar valores temporários salvos
        default_minutes = st.session_state.temp_minutes_range
        if default_minutes[0] < min_min or default_minutes[1] > max_min:
            default_minutes = (min_min, max_min)
            st.session_state.temp_minutes_range = default_minutes
        
        # Caixas de texto para entrada direta de valores de minutos
        st.sidebar.subheader("Minutes Played")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            min_minutes_input = st.number_input(
                "Min", 
                min_value=min_min, 
                max_value=max_min, 
                value=default_minutes[0], 
                step=50,
                key="min_minutes_input"
            )
        
        with col2:
            max_minutes_input = st.number_input(
                "Max", 
                min_value=min_min, 
                max_value=max_min, 
                value=default_minutes[1], 
                step=50,
                key="max_minutes_input"
            )
            
        # Armazenar valores temporários
        temp_minutes_range = (min_minutes_input, max_minutes_input)
        st.session_state.temp_minutes_range = temp_minutes_range
        
    except Exception as e:
        st.error(f"Error setting up minutes slider: {str(e)}")
        min_min, max_min = 0, 9000
        st.session_state.temp_minutes_range = (min_min, max_min)
    
    # Calcular minutos por jogo com tratamento adequado para divisão por zero
    try:
        # Criar cópia temporária para calcular MPG
        temp_df = df.copy()
        temp_df['Minutes per game'] = temp_df['Minutes played'] / temp_df['Matches played'].replace(0, np.nan)
        temp_df['Minutes per game'] = temp_df['Minutes per game'].fillna(0).clip(0, 120)
        
        min_mpg, max_mpg = int(temp_df['Minutes per game'].min()), int(temp_df['Minutes per game'].max())
        
        # Usar valores temporários salvos
        default_mpg = st.session_state.temp_mpg_range
        if default_mpg[0] < min_mpg or default_mpg[1] > max_mpg:
            default_mpg = (min_mpg, max_mpg)
            st.session_state.temp_mpg_range = default_mpg
        
        # Caixas de texto para entrada direta de valores de minutos por jogo
        st.sidebar.subheader("Minutes per Game")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            min_mpg_input = st.number_input(
                "Min", 
                min_value=min_mpg, 
                max_value=max_mpg, 
                value=default_mpg[0], 
                step=5,
                key="min_mpg_input"
            )
        
        with col2:
            max_mpg_input = st.number_input(
                "Max", 
                min_value=min_mpg, 
                max_value=max_mpg, 
                value=default_mpg[1], 
                step=5,
                key="max_mpg_input"
            )
            
        # Armazenar valores temporários
        temp_mpg_range = (min_mpg_input, max_mpg_input)
        st.session_state.temp_mpg_range = temp_mpg_range
        
    except Exception as e:
        st.error(f"Error setting up minutes per game slider: {str(e)}")
        min_mpg, max_mpg = 0, 120
        st.session_state.temp_mpg_range = (min_mpg, max_mpg)
    
    # Filtro de idade
    try:
        # Forçar mínimo de 15 anos para incluir jovens como Lamine Yamal
        data_min_age = int(df['Age'].min())
        min_age = min(data_min_age, 15)  # Usa 15 ou o valor mínimo dos dados, o que for menor
        max_age = int(df['Age'].max())
        
        # Usar valores temporários salvos
        default_age = st.session_state.temp_age_range
        if default_age[0] < min_age or default_age[1] > max_age:
            default_age = (min_age, max_age)
            st.session_state.temp_age_range = default_age
        
        # Caixas de texto para entrada direta de valores de idade
        st.sidebar.subheader("Age Range")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            min_age_input = st.number_input(
                "Min", 
                min_value=min_age, 
                max_value=max_age, 
                value=default_age[0], 
                step=1,
                key="min_age_input"
            )
        
        with col2:
            max_age_input = st.number_input(
                "Max", 
                min_value=min_age, 
                max_value=max_age, 
                value=default_age[1], 
                step=1,
                key="max_age_input"
            )
            
        # Armazenar valores temporários
        temp_age_range = (min_age_input, max_age_input)
        st.session_state.temp_age_range = temp_age_range
        
    except Exception as e:
        st.error(f"Error setting up age slider: {str(e)}")
        min_age, max_age = 15, 40
        st.session_state.temp_age_range = (min_age, max_age)
    
    # Coleta posições
    if 'Position' in df.columns:
        # Criar um conjunto de todas as posições disponíveis
        all_positions = set()
        for pos_list in df['Position'].astype(str).apply(lambda x: [p.strip() for p in x.split(',')]):
            all_positions.update(pos_list)
        
        all_pos = sorted(all_positions)
        
        # Verificar se há posições temporárias salvas e se elas ainda são válidas
        if 'temp_positions' not in st.session_state or not st.session_state.temp_positions:
            default_positions = all_pos
            st.session_state.temp_positions = default_positions
        
        # Usar container para posições
        st.sidebar.subheader('Positions')
        positions_container = st.sidebar.container()
        
        # Criar checkbox para selecionar/deselecionar todas as posições
        select_all = st.sidebar.checkbox(
            "Select All Positions", 
            value=len(st.session_state.temp_positions) == len(all_pos),
            key="select_all_positions"
        )
        
        if select_all:
            st.session_state.temp_positions = all_pos
        
        # Expandable para mostrar opções de posições
        with st.sidebar.expander("Show position selection", expanded=False):
            # Inicializar a lista temporária se não existir
            if 'temp_pos_checkboxes' not in st.session_state:
                st.session_state.temp_pos_checkboxes = {pos: pos in st.session_state.temp_positions for pos in all_pos}
            
            # Criar checkboxes para cada posição
            position_changed = False
            
            # Dividir posições em grupos para colocar em colunas
            half_length = (len(all_pos) + 1) // 2
            first_half = all_pos[:half_length]
            second_half = all_pos[half_length:]
            
            cols = st.columns(2)
            
            # Primeira coluna
            with cols[0]:
                for pos in first_half:
                    # Atualizar valor do checkbox se select_all mudou
                    if select_all:
                        st.session_state.temp_pos_checkboxes[pos] = True
                    
                    # Criar checkbox
                    checked = st.checkbox(
                        pos, 
                        value=st.session_state.temp_pos_checkboxes[pos],
                        key=f"pos_{pos}"
                    )
                    
                    # Atualizar estado se mudou
                    if checked != st.session_state.temp_pos_checkboxes[pos]:
                        st.session_state.temp_pos_checkboxes[pos] = checked
                        position_changed = True
            
            # Segunda coluna
            with cols[1]:
                for pos in second_half:
                    # Atualizar valor do checkbox se select_all mudou
                    if select_all:
                        st.session_state.temp_pos_checkboxes[pos] = True
                    
                    # Criar checkbox
                    checked = st.checkbox(
                        pos, 
                        value=st.session_state.temp_pos_checkboxes[pos],
                        key=f"pos_{pos}_b"
                    )
                    
                    # Atualizar estado se mudou
                    if checked != st.session_state.temp_pos_checkboxes[pos]:
                        st.session_state.temp_pos_checkboxes[pos] = checked
                        position_changed = True
            
            # Atualizar seleção baseada nos checkboxes
            temp_sel_pos = [pos for pos in all_pos if st.session_state.temp_pos_checkboxes[pos]]
            
            if not temp_sel_pos:
                st.warning("At least one position must be selected")
                temp_sel_pos = all_pos
                for pos in all_pos:
                    st.session_state.temp_pos_checkboxes[pos] = True
            
            # Armazenar valor temporário
            st.session_state.temp_positions = temp_sel_pos
    else:
        all_pos = []
        temp_sel_pos = []
        st.session_state.temp_positions = []
    
    # Botão para aplicar todos os filtros
    apply_filters = st.sidebar.button("Apply Filters", key="apply_filters_button")
    
    # Apenas processar os filtros quando o botão for clicado
    if apply_filters:
        with st.spinner("Processing filters..."):
            # Criamos dois DataFrames:
            # 1. df_percentiles - Filtrado APENAS por minutos e mpg para cálculo de percentis
            # 2. df_filtered - Filtrado por TODOS os critérios para visualização
            
            # 1. Primeiro, filtrar ambos por minutos jogados (obrigatório para ambos)
            minutes_range = st.session_state.temp_minutes_range
            df_percentiles = df[df['Minutes played'].between(*minutes_range)].copy()
            df_filtered = df_percentiles.copy()  # Começamos com o mesmo filtro
            
            # Verificar se temos jogadores após o filtro
            if df_percentiles.empty:
                st.warning("No players match the selected minutes range. Using all players.")
                df_percentiles = df.copy()
                df_filtered = df.copy()
                minutes_range = (min_min, max_min)
            
            # Calcular minutos por jogo para ambos
            df_percentiles['Minutes per game'] = df_percentiles['Minutes played'] / df_percentiles['Matches played'].replace(0, np.nan)
            df_percentiles['Minutes per game'] = df_percentiles['Minutes per game'].fillna(0).clip(0, 120)
            df_filtered['Minutes per game'] = df_percentiles['Minutes per game'].copy()
            
            # Filtrar ambos por minutos por jogo (obrigatório para percentis)
            mpg_range = st.session_state.temp_mpg_range
            df_percentiles = df_percentiles[df_percentiles['Minutes per game'].between(*mpg_range)]
            df_filtered = df_filtered[df_filtered['Minutes per game'].between(*mpg_range)]
            
            # Verificar se temos jogadores após o filtro
            if df_percentiles.empty:
                st.warning("No players match the selected minutes per game range. Using previous filter.")
                df_percentiles = df[df['Minutes played'].between(*minutes_range)].copy()
                df_filtered = df_percentiles.copy()
                mpg_range = (min_mpg, max_mpg)
                
                # Recalcular minutos por jogo
                df_percentiles['Minutes per game'] = df_percentiles['Minutes played'] / df_percentiles['Matches played'].replace(0, np.nan)
                df_percentiles['Minutes per game'] = df_percentiles['Minutes per game'].fillna(0).clip(0, 120)
                df_filtered['Minutes per game'] = df_percentiles['Minutes per game'].copy()
            
            # Salvar o DataFrame de percentis (filtrado apenas por minutos e mpg)
            # Este será usado no cálculo de percentis em vez do DataFrame totalmente filtrado
            st.session_state.df_percentiles = df_percentiles.copy()
            
            # 2. Agora, aplicar filtros adicionais APENAS ao DataFrame de visualização
            
            # Filtrar por idade (apenas df_filtered)
            age_range = st.session_state.temp_age_range
            df_age_filtered = df_filtered[df_filtered['Age'].between(*age_range)]
            
            # Verificar se temos jogadores após o filtro
            if df_age_filtered.empty:
                st.warning("No players match the selected age range. Using previous filter.")
                age_range = (min_age, max_age)
            else:
                df_filtered = df_age_filtered
            
            # Filtrar por posição (apenas df_filtered)
            sel_pos = st.session_state.temp_positions
            if 'Position' in df_filtered.columns and sel_pos:
                df_filtered['Position_split'] = df_filtered['Position'].astype(str).apply(lambda x: [p.strip() for p in x.split(',')])
                df_pos_filtered = df_filtered[df_filtered['Position_split'].apply(lambda x: any(pos in x for pos in sel_pos))]
                
                # Verificar se temos jogadores após o filtro
                if df_pos_filtered.empty:
                    st.warning("No players match the selected positions. Using previous filter.")
                else:
                    df_filtered = df_pos_filtered
            
            # Mostrar informações sobre os filtros aplicados
            st.success(f"Dataset for percentile calculation: {len(df_percentiles)} players")
            st.success(f"Dataset for visualization (with age/position filters): {len(df_filtered)} players")
            
            # Salvar o DataFrame final filtrado para visualização
            df_minutes = df_filtered
            
            # Salvar as seleções finais no session_state para persistência
            st.session_state.last_minutes_range = minutes_range
            st.session_state.last_mpg_range = mpg_range
            st.session_state.last_age_range = age_range
            st.session_state.last_positions = sel_pos
            
            # Feedback ao usuário
            st.sidebar.success(f"Filters applied: {len(df_minutes)} players match the criteria")
    else:
        # Se os filtros não foram aplicados, usar os últimos filtros válidos
        # Filtrar por minutos jogados
        minutes_range = st.session_state.last_minutes_range
        df_minutes = df[df['Minutes played'].between(*minutes_range)].copy()
        
        # Calcular minutos por jogo
        df_minutes['Minutes per game'] = df_minutes['Minutes played'] / df_minutes['Matches played'].replace(0, np.nan)
        df_minutes['Minutes per game'] = df_minutes['Minutes per game'].fillna(0).clip(0, 120)
        
        # Filtrar por minutos por jogo
        mpg_range = st.session_state.last_mpg_range
        df_filtered = df_minutes[df_minutes['Minutes per game'].between(*mpg_range)]
        if not df_filtered.empty:
            df_minutes = df_filtered
        
        # Filtrar por idade
        age_range = st.session_state.last_age_range
        df_filtered = df_minutes[df_minutes['Age'].between(*age_range)]
        if not df_filtered.empty:
            df_minutes = df_filtered
        
        # Filtrar por posição
        sel_pos = st.session_state.last_positions
        if 'Position' in df_minutes.columns and sel_pos:
            df_minutes['Position_split'] = df_minutes['Position'].astype(str).apply(lambda x: [p.strip() for p in x.split(',')])
            df_filtered = df_minutes[df_minutes['Position_split'].apply(lambda x: any(pos in x for pos in sel_pos))]
            if not df_filtered.empty:
                df_minutes = df_filtered

    # Usar df_minutes apenas para exibição e seleção de jogadores (manter todos os filtros)
    if 'Position_split' in df_minutes.columns and sel_pos:
        df_group = df_minutes[df_minutes['Position_split'].apply(
            lambda x: any(pos in x for pos in sel_pos))]
    else:
        df_group = df_minutes.copy()
    
    # Verificar se temos o dataframe de percentis criado (deve ter sido criado no bloco "apply_filters")
    if 'df_percentiles' not in st.session_state or st.session_state.df_percentiles is None:
        # Se não foi criado ainda, criamos agora como fallback
        st.warning("Creating percentile dataset using current filters (fallback mode)")
        # Cria dataframe para cálculos de grupo filtrando apenas por minutos jogados e minutos por jogo
        df_percentiles = df.copy()
        # Aplicar apenas filtros de minutos e mpg para percentis
        df_percentiles = df_percentiles[df_percentiles['Minutes played'].between(*minutes_range)].copy()
        # Calcular minutos por jogo
        df_percentiles['Minutes per game'] = df_percentiles['Minutes played'] / df_percentiles['Matches played'].replace(0, np.nan)
        df_percentiles['Minutes per game'] = df_percentiles['Minutes per game'].fillna(0).clip(0, 120)
        df_percentiles = df_percentiles[df_percentiles['Minutes per game'].between(*mpg_range)]
        # Salvar o dataframe de percentis na sessão para uso em cálculos futuros
        st.session_state.df_percentiles = df_percentiles
    
    context = get_context_info(df_minutes, minutes_range, mpg_range, age_range,
                               sel_pos)
    players = sorted(df_minutes['Player'].unique())

    # Get numeric columns for metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove some columns that aren't player metrics
    exclude_cols = [
        'Age', 'Minutes played', 'Matches played', 'Minutes per game'
    ]
    metric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Create tabs for different analyses
    # Definir nomes das abas
    tab_names = [
        'Pizza Chart', 'Bars', 'Scatter', 'Similarity', 'Correlation',
        'Composite Index (PCA)', 'Profiler', 'Custom Metrics Groups'
    ]
    
    # Callback para mudar aba
    def tab_changed():
        st.session_state.current_tab = st.session_state.tab_selector
    
    # Usar um índice de tab persistente através da sessão
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    
    # Selector de abas discreto para gerenciar qual aba está ativa
    # Colocando-o em um container vazio, ele não será visível
    with st.container():
        st.selectbox(
            "Select Tab:", 
            options=range(len(tab_names)),
            format_func=lambda i: tab_names[i],
            index=st.session_state.current_tab,
            key="tab_selector",
            on_change=tab_changed,
            label_visibility="collapsed"
        )
    
    # Criar as abas visuais
    tabs = st.tabs(tab_names)
    
    # Exibir qual aba está ativa (opcional, apenas para debugging)
    # st.sidebar.text(f"Aba ativa: {tab_names[st.session_state.current_tab]}")

    # =============================================
    # Pizza Chart (Aba 1) - Estilo Sofyan Amrabat
    # =============================================
    with tabs[0]:
        st.header('Pizza Chart')

        # Verificar se temos um benchmark carregado
        benchmark_available = st.session_state.benchmark_loaded and st.session_state.benchmark_df is not None

        # Primeiramente, decidir se queremos usar o benchmark ou não
        if benchmark_available:
            use_benchmark = st.checkbox(
                "Use benchmark for comparison",
                value=False,
                help=
                "When enabled, you can compare a player from your dataset with players from the benchmark league"
            )
        else:
            use_benchmark = False

        col1, col2 = st.columns(2)
        with col1:
            # Verificar se o jogador anteriormente selecionado ainda está disponível
            player_index = 0
            if st.session_state.saved_pizza_p1 in players:
                player_index = players.index(st.session_state.saved_pizza_p1)

            # Usar o index para manter a seleção anterior quando possível
            p1 = st.selectbox('Select Player 1',
                              players,
                              index=player_index,
                              key='pizza_p1')

            # Atualizar nossa variável de armazenamento (não o widget diretamente)
            st.session_state.saved_pizza_p1 = p1

        # A seleção do segundo jogador depende se estamos usando benchmark ou não
        with col2:
            if use_benchmark:
                # Filtrar o benchmark com os mesmos filtros aplicados à base principal
                filtered_benchmark = apply_benchmark_filter(
                    st.session_state.benchmark_df, minutes_range, mpg_range,
                    age_range, sel_pos)

                if filtered_benchmark is not None and not filtered_benchmark.empty:
                    benchmark_players = sorted(
                        filtered_benchmark['Player'].unique())

                    # Verificar se o jogador anteriormente selecionado ainda está disponível
                    benchmark_index = 0
                    if st.session_state.saved_pizza_p2 in benchmark_players:
                        benchmark_index = benchmark_players.index(
                            st.session_state.saved_pizza_p2)

                    p2 = st.selectbox('Select Benchmark Player',
                                      benchmark_players,
                                      index=benchmark_index,
                                      key='pizza_p2_benchmark')

                    # Atualizar nossa variável de armazenamento (não o widget diretamente)
                    st.session_state.saved_pizza_p2 = p2

                    # Exibir informações do jogador benchmark selecionado
                    st.info(
                        f"Benchmark: {p2} from {st.session_state.benchmark_name}"
                    )
                else:
                    st.warning(
                        "No benchmark players match the applied filters")
                    p2 = None
                    use_benchmark = False
            else:
                # Filtrar jogadores para não incluir o jogador 1
                remaining_players = [p for p in players if p != p1]

                if remaining_players:
                    # Verificar se o jogador anteriormente selecionado ainda está disponível
                    player2_index = 0
                    if st.session_state.saved_pizza_p2 in remaining_players:
                        player2_index = remaining_players.index(
                            st.session_state.saved_pizza_p2)

                    # Usar o index para manter a seleção anterior quando possível
                    p2 = st.selectbox('Select Player 2',
                                      remaining_players,
                                      index=player2_index,
                                      key='pizza_p2')

                    # Atualizar nossa variável de armazenamento (não o widget diretamente)
                    st.session_state.saved_pizza_p2 = p2
                else:
                    st.warning("No second player available")
                    p2 = None

        # Carregar grupos personalizados salvos na sessão
        if 'custom_metric_groups' in st.session_state:
            custom_groups = st.session_state.custom_metric_groups
        else:
            custom_groups = {}

        # Predefined metric groups for pizza charts
        pizza_metric_groups = {
            "Overall": [
                'Progressive runs per 90', 'Progressive passes per 90',
                'Deep completions per 90', 'Key passes per 90',
                'Accurate passes, %', 'Accurate crosses, %', 'xA per 90',
                'Defensive duels won, %', 'Aerial duels won, %',
                'PAdj Interceptions', 'PAdj Sliding tackles', 'npxG per 90',
                'npxG per Shot', 'Goal conversion, %', 'Shots on target, %'
            ],
            "Offensive": [
                'Goals per 90', 'Shots per 90', 'xG per 90', 'npxG', 'G-xG',
                'npxG per Shot', 'Box Efficiency', 'Shots on target per 90',
                'Successful dribbles per 90', 'Progressive runs per 90'
            ],
            "Passing": [
                'Passes per 90', 'Accurate passes, %', 'Forward passes per 90',
                'Progressive passes per 90', 'Key passes per 90',
                'Assists per 90', 'xA per 90'
            ],
            "Defensive": [
                'Interceptions per 90', 'Tackles per 90',
                'Defensive duels per 90', 'Aerial duels won per 90',
                'Recoveries per 90'
            ],
            "Physical": [
                'Accelerations per 90', 'Sprint distance per 90',
                'Distance covered per 90'
            ],
            "Advanced Metrics":
            ['npxG', 'G-xG', 'npxG per Shot', 'Box Efficiency'
             ],  # Incluindo Box Efficiency nas métricas avançadas
            "Pedri Metrics": [
                'Smart passes per 90', 'Key passes per 90', 'Shot assists per 90', 'xA per 90',
                'Passes per 90', 'Progressive passes per 90', 'Passes to final third per 90', 'Through passes per 90',
                'Progressive runs per 90', 'Dribbles per 90', 'Successful dribbles, %',
                'Interceptions per 90', 'Defensive duels per 90', 'Defensive duels won, %',
                'Received passes per 90'
            ]
        }
        
        # Adicionar grupos personalizados salvos na session state
        if 'custom_metric_groups' in st.session_state:
            # Adicionar cada grupo personalizado ao dicionário de grupos de métricas
            for group_name, metrics in st.session_state.custom_metric_groups.items():
                # Garantir que não sobrescreva grupos existentes acidentalmente
                if group_name not in pizza_metric_groups:
                    pizza_metric_groups[group_name] = metrics

        # Let user select metric selection mode
        metric_selection_mode = st.radio(
            "Metric Selection Mode",
            ["Custom (Select Individual Metrics)", "Use Metric Presets"],
            horizontal=True,
            key="pizza_metric_selection_mode"  # Adicionando chave única
        )

        if metric_selection_mode == "Use Metric Presets":
            # Definir valores padrão para grupos de métricas se não houver seleção prévia
            if 'temp_pizza_selected_groups' not in st.session_state:
                # Inicializar com os grupos salvos ou com valores padrão
                if 'pizza_selected_groups' in st.session_state and st.session_state.pizza_selected_groups:
                    default_groups = st.session_state.pizza_selected_groups
                else:
                    default_groups = ["Offensive", "Passing"]
                
                # Filtrar para manter apenas grupos que ainda existem
                valid_groups = [
                    g for g in default_groups
                    if g in pizza_metric_groups.keys()
                ]
                st.session_state.temp_pizza_selected_groups = valid_groups if valid_groups else ["Offensive", "Passing"]
                
            # Selecionar grupos de métricas (temporariamente)
            selected_groups = st.multiselect(
                "Select Metric Groups",
                list(pizza_metric_groups.keys()),
                default=st.session_state.temp_pizza_selected_groups,
                key="pizza_preset_groups"  # Adicionando chave única
            )
            
            # Atualizar variável temporária
            st.session_state.temp_pizza_selected_groups = selected_groups

            # Combine all metrics from selected groups
            preset_metrics = []
            for group in selected_groups:
                # Only add metrics that exist in the dataframe
                available_metrics = [
                    m for m in pizza_metric_groups[group] if m in metric_cols
                ]
                preset_metrics.extend(available_metrics)
            
            # Remover duplicatas mantendo a ordem
            preset_metrics = list(dict.fromkeys(preset_metrics))
            
            # Sempre atualizar as métricas temporárias quando os grupos são selecionados
            # Isso faz com que as métricas apareçam imediatamente na lista de métricas
            st.session_state.temp_pizza_preset_metrics = preset_metrics
            
            # Permitir ajustes manuais nas métricas (temporariamente)
            temp_preset_metrics = st.multiselect(
                'Add or Remove Individual Metrics (6-15 recommended)',
                metric_cols,
                default=st.session_state.temp_pizza_preset_metrics,
                key="pizza_preset_metrics"  # Adicionando chave única
            )
            
            # Atualizar variável temporária
            st.session_state.temp_pizza_preset_metrics = temp_preset_metrics
            
            # Botão para aplicar as métricas selecionadas
            apply_preset_metrics = st.button("Apply Selected Metrics", key="apply_pizza_preset_metrics")
            
            if apply_preset_metrics:
                # Salvar os grupos selecionados
                st.session_state.pizza_selected_groups = st.session_state.temp_pizza_selected_groups
                # Salvar as métricas selecionadas
                st.session_state.saved_pizza_metrics = st.session_state.temp_pizza_preset_metrics
                st.rerun()
        else:
            # Definir valores padrão para métricas com base na sessão ou primeiras 9 métricas
            if 'saved_pizza_metrics' in st.session_state and st.session_state.saved_pizza_metrics:
                # Filtrar para manter apenas métricas que ainda existem no dataframe
                valid_metrics = [
                    m for m in st.session_state.saved_pizza_metrics
                    if m in metric_cols
                ]
                default_metrics = valid_metrics if valid_metrics else metric_cols[:9]
            else:
                default_metrics = metric_cols[:9]
                
            # Criar variáveis temporárias para seleção de métricas
            if 'temp_pizza_metrics' not in st.session_state:
                # Verificar que os valores padrão são válidos (presentes em metric_cols)
                valid_metrics = [m for m in default_metrics if m in metric_cols]
                st.session_state.temp_pizza_metrics = valid_metrics if valid_metrics else metric_cols[:min(9, len(metric_cols))]
            
            # Garantir que todos os itens em temp_pizza_metrics estão em metric_cols para evitar erros
            valid_temp_metrics = [m for m in st.session_state.temp_pizza_metrics if m in metric_cols]
            st.session_state.temp_pizza_metrics = valid_temp_metrics if valid_temp_metrics else metric_cols[:min(9, len(metric_cols))]
                
            # Let user select metrics manually (sem aplicar imediatamente)
            temp_metrics = st.multiselect(
                'Metrics for Pizza Chart (6-15)',
                metric_cols,
                default=st.session_state.temp_pizza_metrics,
                key="pizza_temp_metrics"
            )
            
            # Atualizar a seleção temporária
            st.session_state.temp_pizza_metrics = temp_metrics
            
            # Botão para aplicar as métricas selecionadas
            apply_metrics = st.button("Apply Selected Metrics", key="apply_pizza_metrics")
            
            if apply_metrics:
                # Aplicar as métricas selecionadas
                st.session_state.saved_pizza_metrics = st.session_state.temp_pizza_metrics
                st.rerun()
            
            # Usar as métricas já confirmadas para o processamento
            if 'saved_pizza_metrics' in st.session_state:
                sel = st.session_state.saved_pizza_metrics
            else:
                sel = st.session_state.temp_pizza_metrics

        # Para ambos os modos, definir sel com base em saved_pizza_metrics
        if 'saved_pizza_metrics' in st.session_state:
            sel = st.session_state.saved_pizza_metrics
        elif metric_selection_mode == "Use Metric Presets" and 'temp_pizza_preset_metrics' in st.session_state:
            sel = st.session_state.temp_pizza_preset_metrics
        elif 'temp_pizza_metrics' in st.session_state:
            sel = st.session_state.temp_pizza_metrics
        else:
            # Fallback para evitar erros
            sel = metric_cols[:min(9, len(metric_cols))]

        if 6 <= len(sel) <= 15:
            # Dados do jogador 1 (sempre da base principal)
            d1 = df_minutes[df_minutes['Player'] == p1].iloc[0]

            # Verificar modo de cálculo de percentis (global ou por liga)
            combine_leagues = st.session_state.get('combine_leagues', True)

            # Identificar a liga do jogador 1
            player1_league = d1['Data Origin']

            # Dados do jogador 2 (pode ser do benchmark ou da base principal)
            if use_benchmark and p2 is not None:
                # Obter dados do benchmark
                d2 = filtered_benchmark[filtered_benchmark['Player'] ==
                                        p2].iloc[0]

                # Calcular percentis de acordo com a configuração
                if combine_leagues:
                    # Modo Global: usar todo o dataframe para cálculo de percentis
                    p1pct = [
                        calc_percentile(df_minutes[m], d1[m]) * 100
                        for m in sel
                    ]
                    p2pct = [
                        calc_percentile(filtered_benchmark[m], d2[m]) * 100
                        for m in sel
                    ]

                    # Média global para comparação
                    gm = {m: filtered_benchmark[m].mean() for m in sel}
                    gmpct = [
                        calc_percentile(filtered_benchmark[m], gm[m]) * 100
                        for m in sel
                    ]
                else:
                    # Modo Por Liga: usar apenas jogadores da mesma liga para cálculo de percentis
                    # Filtrar jogadores da mesma liga que o jogador 1
                    same_league_players = df_minutes[df_minutes['Data Origin']
                                                     == player1_league]

                    # Para jogador 1, calcular percentis dentro da sua própria liga
                    p1pct = [
                        calc_percentile(same_league_players[m], d1[m]) * 100
                        for m in sel
                    ]

                    # Para jogador 2 (benchmark), calcular percentis dentro do benchmark
                    p2pct = [
                        calc_percentile(filtered_benchmark[m], d2[m]) * 100
                        for m in sel
                    ]

                    # Média do benchmark para comparação
                    gm = {m: filtered_benchmark[m].mean() for m in sel}
                    gmpct = [
                        calc_percentile(filtered_benchmark[m], gm[m]) * 100
                        for m in sel
                    ]

                # Texto especial para o subtítulo
                benchmark_text = f" | Benchmark: {st.session_state.benchmark_name}"

                # Adicionar informação sobre modo de cálculo de percentis
                if not combine_leagues:
                    benchmark_text += " | Percentiles: League-specific"
            else:
                # Fluxo normal - ambos jogadores da base principal
                d2 = df_minutes[df_minutes['Player'] == p2].iloc[0]
                player2_league = d2['Data Origin']

                if combine_leagues:
                    # Modo Global: usar todo o dataframe para cálculo de percentis
                    p1pct = [
                        calc_percentile(df_minutes[m], d1[m]) * 100
                        for m in sel
                    ]
                    p2pct = [
                        calc_percentile(df_minutes[m], d2[m]) * 100
                        for m in sel
                    ]

                    # Média global para comparação
                    gm = {m: df_group[m].mean() for m in sel}
                    gmpct = [
                        calc_percentile(df_minutes[m], gm[m]) * 100
                        for m in sel
                    ]
                else:
                    # Modo Por Liga: usar apenas jogadores da mesma liga para cálculo de percentis
                    # Filtrar jogadores da mesma liga que o jogador 1
                    same_league_players1 = df_minutes[df_minutes['Data Origin']
                                                      == player1_league]
                    same_league_players2 = df_minutes[df_minutes['Data Origin']
                                                      == player2_league]

                    # Para cada jogador, calcular percentis dentro da sua própria liga
                    p1pct = [
                        calc_percentile(same_league_players1[m], d1[m]) * 100
                        for m in sel
                    ]
                    p2pct = [
                        calc_percentile(same_league_players2[m], d2[m]) * 100
                        for m in sel
                    ]

                    # Calcular média global (manter comportamento original)
                    gm = {m: df_group[m].mean() for m in sel}

                    if player1_league == player2_league:
                        # Se ambos jogadores são da mesma liga, usar essa liga para gmpct
                        gmpct = [
                            calc_percentile(same_league_players1[m], gm[m]) *
                            100 for m in sel
                        ]
                    else:
                        # Se são de ligas diferentes, usar o dataset global para gmpct
                        gmpct = [
                            calc_percentile(df_minutes[m], gm[m]) * 100
                            for m in sel
                        ]

                # Adicionar informação sobre modo de cálculo de percentis no subtítulo
                benchmark_text = "" if combine_leagues else " | Percentiles: League-specific"

            # Controle de visualização modificado para limitar a 2 elementos
            st.subheader("Display Options")

            # Opção para mostrar valores nominais nos rótulos (mantendo visualização em percentil)
            show_nominal_values = st.checkbox(
                "Show nominal values in labels (visualization remains percentile-based)",
                value=False,
                help=
                "When checked, the chart labels will show actual metric values instead of percentile ranks, but chart visualization will still use percentiles"
            )

            # Sempre mostrar jogador 1 por padrão
            show_p1 = True

            # Opções de comparação (exclusivas)
            # Se estamos usando benchmark, forçar a comparação jogador vs jogador
            if use_benchmark and p2 is not None:
                comparison_option = f"Player vs Benchmark Player ({p1} vs {p2})"
                show_p2 = True
                show_avg = False
                title = f"{p1} vs {p2} (Benchmark)"
            else:
                # Opções normais
                comparison_option = st.radio(
                    "Compare with:",
                    [
                        f"No comparison (only {p1})",
                        f"Player vs Player ({p1} vs {p2})",
                        f"Player vs Group Average ({p1} vs Avg)"
                    ],
                    index=0,
                    key="pizza_comparison_options"  # Adicionando chave única
                )

                # Configurar valores com base na opção selecionada
                # Obter posição, time e idade do jogador 1
                p1_position = d1['Position']
                p1_team = d1['Team']
                p1_age = int(d1['Age']) if isinstance(d1['Age'],
                                                      (int,
                                                       float)) else d1['Age']
                p1_display = f"{p1} ({p1_team}, {p1_position}, {p1_age})"

                # Obter posição, time e idade do jogador 2 se estiver disponível
                if p2 is not None:
                    p2_position = d2['Position']
                    p2_team = d2['Team']
                    p2_age = int(d2['Age']) if isinstance(
                        d2['Age'], (int, float)) else d2['Age']
                    p2_display = f"{p2} ({p2_team}, {p2_position}, {p2_age})"
                else:
                    p2_display = p2

                if comparison_option == f"No comparison (only {p1})":
                    show_p2 = False
                    show_avg = False
                    title = f"{p1_display}"
                elif comparison_option == f"Player vs Player ({p1} vs {p2})":
                    show_p2 = True
                    show_avg = False
                    title = f"{p1_display} vs {p2_display}"
                else:  # Player vs Group Average
                    show_p2 = False
                    show_avg = True
                    title = f"{p1_display} vs Group Average"

            # Ajustar o subtítulo com base na escolha de valores nominais ou percentis
            if show_nominal_values:
                subtitle = (
                    f"Actual Values | {context['leagues']} | "
                    f"{context['min_min']}+ mins | Position: {context['positions']}{benchmark_text}"
                )
            else:
                subtitle = (
                    f"Percentile Rank | {context['leagues']} | "
                    f"{context['min_min']}+ mins | Position: {context['positions']}{benchmark_text}"
                )

            # Armazenar os valores originais para uso posterior
            original_values_p1 = [d1[m] for m in sel] if show_p1 else None
            original_values_p2 = [d2[m] for m in sel] if show_p2 else None
            original_values_avg = [gm[m] for m in sel] if show_avg else None

            # Para a visualização, SEMPRE usamos percentis para manter consistência
            # independentemente da opção de exibição de valores
            values_p1_arg = p1pct if show_p1 else None
            values_p2_arg = p2pct if show_p2 else None
            values_avg_arg = gmpct if show_avg else None

            # Armazenar os valores nominais na sessão para uso no gráfico
            st.session_state.nominal_values_p1 = original_values_p1
            st.session_state.nominal_values_p2 = original_values_p2

            # Guardar os valores originais para exibição condicional nos rótulos
            # Se show_nominal_values é True, a função de plotagem usará esses valores nos rótulos
            if show_nominal_values:
                st.session_state.display_values_p1 = original_values_p1
                st.session_state.display_values_p2 = original_values_p2
                st.session_state.display_values_avg = original_values_avg
            else:
                st.session_state.display_values_p1 = values_p1_arg
                st.session_state.display_values_p2 = values_p2_arg
                st.session_state.display_values_avg = values_avg_arg

            # Flag para usar o chart comparativo para comparações entre jogadores ou jogador vs média
            use_comparison_chart = False
            if show_p1 and (show_p2 or show_avg):
                use_comparison_chart = True

            # Criar o Pizza Chart de acordo com a escolha
            if use_comparison_chart:
                # Usar o chart comparativo (para dois jogadores ou jogador vs média)
                fig = create_comparison_pizza_chart(params=sel,
                                                    values_p1=values_p1_arg,
                                                    values_p2=values_p2_arg,
                                                    values_avg=values_avg_arg,
                                                    title=title,
                                                    subtitle=subtitle,
                                                    p1_name=p1,
                                                    p2_name=p2)
            else:
                # Usar o chart padrão para casos com média ou apenas um jogador
                fig = create_pizza_chart(params=sel,
                                         values_p1=values_p1_arg,
                                         values_p2=values_p2_arg,
                                         values_avg=values_avg_arg,
                                         title=title,
                                         subtitle=subtitle,
                                         p1_name=p1,
                                         p2_name=p2)

            # Display pizza chart
            st.pyplot(fig)

            # Display nominal values table
            st.markdown("**Nominal Values**")

            table_data = {'Metric': sel}
            if show_p1:
                table_data[p1] = [round(d1[m], 2) for m in sel]
            if show_p2:
                table_data[p2] = [round(d2[m], 2) for m in sel]
            if show_avg:
                table_data['Group Avg'] = [round(gm[m], 2) for m in sel]

            df_table = pd.DataFrame(table_data).set_index('Metric')
            st.dataframe(df_table)

            # Export button
            if st.button('Export Pizza Chart (300 DPI)', key='export_pizza'):
                img_bytes = fig_to_bytes(fig)
                players_str = p1
                if show_p2:
                    players_str += f"_vs_{p2}"
                if show_avg and not show_p2:
                    players_str += "_vs_avg"
                st.download_button("⬇️ Download Pizza Chart",
                                   data=img_bytes,
                                   file_name=f"pizza_chart_{players_str}.png",
                                   mime="image/png")

        # =============================================
        # Bar Charts (Aba 2)
        # =============================================
        with tabs[1]:
            st.header('Bar Chart Comparison')

            # Verificar se temos um benchmark carregado
            benchmark_available = st.session_state.benchmark_loaded and st.session_state.benchmark_df is not None

            # Primeiramente, decidir se queremos usar o benchmark ou não
            if benchmark_available:
                use_benchmark = st.checkbox(
                    "Use benchmark for comparison",
                    value=False,
                    help=
                    "When enabled, you can compare a player from your dataset with players from the benchmark league",
                    key="bar_use_benchmark")
            else:
                use_benchmark = False

            col1, col2 = st.columns(2)
            with col1:
                # Verificar se o jogador anteriormente selecionado ainda está disponível
                player_index = 0
                if st.session_state.saved_bar_p1 in players:
                    player_index = players.index(st.session_state.saved_bar_p1)

                # Usar o index para manter a seleção anterior quando possível
                p1 = st.selectbox('Select Player 1',
                                  players,
                                  index=player_index,
                                  key='bar_p1')

                # Atualizar nossa variável de armazenamento (não o widget diretamente)
                st.session_state.saved_bar_p1 = p1

            # A seleção do segundo jogador depende se estamos usando benchmark ou não
            with col2:
                if use_benchmark:
                    # Filtrar o benchmark com os mesmos filtros aplicados à base principal
                    filtered_benchmark = apply_benchmark_filter(
                        st.session_state.benchmark_df, minutes_range,
                        mpg_range, age_range, sel_pos)

                    if filtered_benchmark is not None and not filtered_benchmark.empty:
                        benchmark_players = sorted(
                            filtered_benchmark['Player'].unique())

                        # Verificar se o jogador anteriormente selecionado ainda está disponível
                        benchmark_index = 0
                        if st.session_state.saved_bar_p2 in benchmark_players:
                            benchmark_index = benchmark_players.index(
                                st.session_state.saved_bar_p2)

                        p2 = st.selectbox('Select Benchmark Player',
                                          benchmark_players,
                                          index=benchmark_index,
                                          key='bar_p2_benchmark')

                        # Atualizar nossa variável de armazenamento (não o widget diretamente)
                        st.session_state.saved_bar_p2 = p2

                        # Exibir informações do jogador benchmark selecionado
                        st.info(
                            f"Benchmark: {p2} from {st.session_state.benchmark_name}"
                        )
                    else:
                        st.warning(
                            "No benchmark players match the applied filters")
                        p2 = None
                        use_benchmark = False
                else:
                    # Seleção normal do segundo jogador da mesma base
                    remaining_players = [p for p in players if p != p1]

                    if remaining_players:
                        # Verificar se o jogador anteriormente selecionado ainda está disponível
                        player2_index = 0
                        if st.session_state.saved_bar_p2 in remaining_players:
                            player2_index = remaining_players.index(
                                st.session_state.saved_bar_p2)

                        # Usar o index para manter a seleção anterior quando possível
                        p2 = st.selectbox('Select Player 2',
                                          remaining_players,
                                          index=player2_index,
                                          key='bar_p2')

                        # Atualizar nossa variável de armazenamento (não o widget diretamente)
                        st.session_state.saved_bar_p2 = p2
                    else:
                        st.warning("No second player available")
                        p2 = None
                        
            # Carregar grupos personalizados salvos na sessão
            if 'custom_metric_groups' in st.session_state:
                custom_groups = st.session_state.custom_metric_groups
            else:
                custom_groups = {}
                
            # Opção para usar grupos de métricas predefinidos
            use_presets = st.checkbox("Use metric presets", key="bar_use_presets", value=False)
            
            if use_presets:
                # Definir grupos predefinidos de métricas para comparação
                bar_metric_groups = {
                    "Offensive": [
                        'Goals per 90', 'Shots per 90', 'xG per 90', 'npxG per 90',
                        'G-xG', 'npxG per Shot', 'Box Efficiency'
                    ],
                    "Passing": [
                        'Passes per 90', 'Accurate passes, %',
                        'Forward passes per 90', 'Progressive passes per 90',
                        'Key passes per 90'
                    ],
                    "Defensive": [
                        'Interceptions per 90', 'Tackles per 90',
                        'Defensive duels per 90', 'Aerial duels won per 90'
                    ]
                }
                
                # Adicionar grupos personalizados
                for group_name, metrics in custom_groups.items():
                    if group_name not in bar_metric_groups:
                        bar_metric_groups[group_name] = metrics
                
                # Seleção de grupos de métricas
                selected_groups = st.multiselect(
                    "Select Metric Groups",
                    list(bar_metric_groups.keys()),
                    default=[list(bar_metric_groups.keys())[0]],
                    key="bar_preset_groups")
                
                # Combinar todas as métricas dos grupos selecionados
                preset_metrics = []
                for group in selected_groups:
                    # Adicionar apenas métricas que existem no dataframe
                    available_metrics = [
                        m for m in bar_metric_groups[group]
                        if m in metric_cols
                    ]
                    preset_metrics.extend(available_metrics)
                
                # Remover duplicatas e limitar ao máximo de 5 métricas
                preset_metrics = list(dict.fromkeys(preset_metrics))[:5]
                
                # Usar as métricas dos grupos para comparação
                if preset_metrics:
                    st.session_state.saved_bar_metrics = preset_metrics
            
            # Definir métricas padrão com base na session_state ou primeiras métricas
            if 'saved_bar_metrics' in st.session_state and st.session_state.saved_bar_metrics:
                # Filtrar para manter apenas métricas que ainda existem no dataframe
                valid_metrics = [
                    m for m in st.session_state.saved_bar_metrics
                    if m in metric_cols
                ]
                default_metrics = valid_metrics if valid_metrics else metric_cols[:1]
            else:
                default_metrics = metric_cols[:1]
                
            # Criar variáveis temporárias para seleção de métricas
            if 'temp_bar_metrics' not in st.session_state:
                st.session_state.temp_bar_metrics = default_metrics
            
            # Opção para colar lista de métricas
            with st.expander("📋 Paste metrics list"):
                metrics_text = st.text_area(
                    "Paste metrics (one per line or comma-separated)",
                    height=150,
                    key="bar_metrics_paste"
                )
                
                if st.button("Process pasted metrics", key="process_bar_metrics"):
                    # Processar as métricas coladas
                    valid_metrics = process_pasted_metrics(metrics_text, metric_cols)
                    if valid_metrics:
                        st.session_state.temp_bar_metrics = valid_metrics
                        st.success(f"Found {len(valid_metrics)} valid metrics!")
                        st.rerun()
                    else:
                        st.warning("No valid metrics found in the pasted text. Please check the format and try again.")
                
            # Garantir que todos os itens em temp_bar_metrics estão em metric_cols para evitar erros
            valid_temp_metrics = [m for m in st.session_state.temp_bar_metrics if m in metric_cols]
            st.session_state.temp_bar_metrics = valid_temp_metrics if valid_temp_metrics else metric_cols[:min(1, len(metric_cols))]
                
            # Let user select metrics manually (sem aplicar imediatamente)
            temp_metrics = st.multiselect(
                'Select metrics (max 5)',
                metric_cols,
                default=st.session_state.temp_bar_metrics,
                key="bar_temp_metrics"
            )
            
            # Atualizar a seleção temporária
            st.session_state.temp_bar_metrics = temp_metrics
            
            # Botão para aplicar as métricas selecionadas
            apply_metrics = st.button("Apply Selected Metrics", key="apply_bar_metrics")
            
            if apply_metrics:
                # Aplicar as métricas selecionadas
                st.session_state.saved_bar_metrics = st.session_state.temp_bar_metrics
                st.rerun()
            
            # Usar as métricas já confirmadas para o processamento
            selected_metrics = st.session_state.saved_bar_metrics

            if len(selected_metrics) > 5:
                st.error("Maximum 5 metrics allowed!")

            elif len(selected_metrics) >= 1 and p2 is not None:
                # Dados do jogador 1 (sempre da base principal)
                d1 = df_minutes[df_minutes['Player'] == p1].iloc[0]

                # Dados do jogador 2 (pode ser do benchmark ou da base principal)
                if use_benchmark:
                    # Obter dados do benchmark
                    d2 = filtered_benchmark[filtered_benchmark['Player'] ==
                                            p2].iloc[0]

                    # A média do grupo é a média do benchmark para comparação consistente
                    avg_values = [
                        filtered_benchmark[m].mean() for m in selected_metrics
                    ]

                    # Texto especial para o subtítulo
                    benchmark_text = f" | Benchmark: {st.session_state.benchmark_name}"
                else:
                    # Fluxo normal - ambos jogadores da base principal
                    d2 = df_minutes[df_minutes['Player'] == p2].iloc[0]

                    # Group average da base principal
                    avg_values = [df_group[m].mean() for m in selected_metrics]

                    benchmark_text = ""

                # Valores nominais para os jogadores
                p1_values = [d1[m] for m in selected_metrics]
                p2_values = [d2[m] for m in selected_metrics]
                
                # Formatação dos nomes com os times
                p1_team = d1['Team']
                p2_team = d2['Team']
                p1_display = f"{p1} ({p1_team})"
                p2_display = f"{p2} ({p2_team})"

                title = "Metric Comparison"
                subtitle = (
                    f"Context: {context['leagues']} ({context['seasons']}) | "
                    f"Players: {context['total_players']} | Filters: {context['min_age']}-{context['max_age']} years{benchmark_text}"
                )

                fig = create_bar_chart(metrics=selected_metrics,
                                       p1_name=p1_display,
                                       p1_values=p1_values,
                                       p2_name=p2_display,
                                       p2_values=p2_values,
                                       avg_values=avg_values,
                                       title=title,
                                       subtitle=subtitle)

                # Display bar chart
                st.pyplot(fig)

                # Export button
                if st.button('Export Bar Chart (300 DPI)', key='export_bar'):
                    img_bytes = fig_to_bytes(fig)
                    st.download_button("⬇️ Download Bar Chart",
                                       data=img_bytes,
                                       file_name=f"bar_{p1}_vs_{p2}.png",
                                       mime="image/png")

        # =============================================
        # Scatter Plot (Aba 3)
        # =============================================
        with tabs[2]:
            st.header('Scatter Plot Analysis')

            # Verificar se temos um benchmark carregado
            benchmark_available = st.session_state.benchmark_loaded and st.session_state.benchmark_df is not None

            # Primeiramente, decidir se queremos incluir o benchmark no scatter plot
            if benchmark_available:
                include_benchmark = st.checkbox(
                    "Include benchmark players in scatter plot",
                    value=False,
                    help=
                    "When enabled, players from benchmark database will be shown in a different color",
                    key="scatter_use_benchmark")
            else:
                include_benchmark = False
                
            # Carregar grupos personalizados salvos na sessão
            if 'custom_metric_groups' in st.session_state:
                custom_groups = st.session_state.custom_metric_groups
            else:
                custom_groups = {}
                
            # Define preset metric groups for scatter plot
            scatter_metric_groups = {
                "Offensive": [
                    'Goals per 90', 'Shots per 90', 'xG per 90', 'npxG per 90',
                    'Shots on target per 90', 'Successful dribbles per 90'
                ],
                "Passing": [
                    'Passes per 90', 'Accurate passes, %',
                    'Forward passes per 90', 'Progressive passes per 90',
                    'Key passes per 90', 'Assists per 90', 'xA per 90'
                ],
                "Defensive": [
                    'Interceptions per 90', 'Tackles per 90',
                    'Defensive duels per 90', 'Aerial duels won per 90',
                    'Recoveries per 90'
                ],
                "Physical": [
                    'Accelerations per 90', 'Sprint distance per 90',
                    'Distance covered per 90'
                ],
                "Advanced Metrics": [
                    'npxG', 'G-xG', 'npxG per Shot', 'Box Efficiency'
                ]
            }
            
            # Adicionar grupos personalizados ao dicionário de grupos de métricas
            for group_name, metrics in custom_groups.items():
                if group_name not in scatter_metric_groups:
                    scatter_metric_groups[group_name] = metrics

            # Opção para usar presets ou seleção manual
            metric_selection_mode = st.radio(
                "Metric Selection Mode",
                ["Direct Selection", "Use Metric Presets"],
                horizontal=True,
                key="scatter_metric_selection_mode"
            )

            if metric_selection_mode == "Use Metric Presets":
                # Selecionar grupo de métricas
                selected_group = st.selectbox(
                    "Select Metrics Group",
                    list(scatter_metric_groups.keys()),
                    key="scatter_metric_group"
                )
                
                # Mostrar métricas disponíveis no grupo selecionado
                group_metrics = [m for m in scatter_metric_groups[selected_group] if m in metric_cols]
                
                if len(group_metrics) < 2:
                    st.warning(f"The selected group '{selected_group}' has less than 2 metrics available in the current dataset. Please select another group or use direct selection.")
                else:
                    st.info(f"Available metrics in group '{selected_group}': {', '.join(group_metrics)}")
                    
                    # Selecionar métricas X e Y a partir do grupo
                    col1, col2 = st.columns(2)
                    with col1:
                        # Verificar se a métrica X anteriormente selecionada está no grupo
                        x_index = 0
                        if st.session_state.saved_scatter_x_metric in group_metrics:
                            x_index = group_metrics.index(st.session_state.saved_scatter_x_metric)
                        
                        # Selecionar métrica X mantendo a seleção anterior quando possível
                        x_metric = st.selectbox('X-Axis Metric', 
                                              group_metrics,
                                              index=x_index,
                                              key='scatter_x_preset')
                        
                        # Atualizar nossa variável de armazenamento
                        st.session_state.saved_scatter_x_metric = x_metric
                        
                    with col2:
                        # Verificar se a métrica Y anteriormente selecionada está no grupo
                        available_y_metrics = [m for m in group_metrics if m != x_metric]
                        
                        if not available_y_metrics:
                            st.warning("Not enough metrics in group for Y-axis. Please select a different group.")
                            y_metric = x_metric  # Fallback to prevent errors
                        else:
                            y_index = 0
                            if st.session_state.saved_scatter_y_metric in available_y_metrics:
                                y_index = available_y_metrics.index(st.session_state.saved_scatter_y_metric)
                            elif len(available_y_metrics) > 1:
                                y_index = 1  # Default to second metric if available
                            
                            # Selecionar métrica Y mantendo a seleção anterior quando possível
                            y_metric = st.selectbox('Y-Axis Metric',
                                                  available_y_metrics,
                                                  index=y_index,
                                                  key='scatter_y_preset')
                            
                            # Atualizar nossa variável de armazenamento
                            st.session_state.saved_scatter_y_metric = y_metric
            else:
                # Seleção direta das métricas
                col1, col2 = st.columns(2)
                with col1:
                    # Verificar se a métrica X anteriormente selecionada ainda está disponível
                    x_index = 0
                    if st.session_state.saved_scatter_x_metric in metric_cols:
                        x_index = metric_cols.index(st.session_state.saved_scatter_x_metric)

                    # Selecionar métrica X mantendo a seleção anterior quando possível
                    x_metric = st.selectbox('X-Axis Metric',
                                          metric_cols,
                                          index=x_index,
                                          key='scatter_x')

                    # Atualizar nossa variável de armazenamento (não o widget diretamente)
                    st.session_state.saved_scatter_x_metric = x_metric

                with col2:
                    # Verificar se a métrica Y anteriormente selecionada ainda está disponível
                    y_index = min(1, len(metric_cols) - 1)  # Valor padrão
                    if st.session_state.saved_scatter_y_metric in metric_cols:
                        y_index = metric_cols.index(st.session_state.saved_scatter_y_metric)

                    # Selecionar métrica Y mantendo a seleção anterior quando possível
                    y_metric = st.selectbox('Y-Axis Metric',
                                          metric_cols,
                                          index=y_index,
                                          key='scatter_y')

                    # Atualizar nossa variável de armazenamento (não o widget diretamente)
                    st.session_state.saved_scatter_y_metric = y_metric

            # Configurar o título base
            title = f"Scatter Analysis: {x_metric} vs {y_metric}"
            subtitle = f"League(s): {context['leagues']} | Season(s): {context['seasons']}"

            # Se estamos incluindo o benchmark, preparar os dados
            if include_benchmark:
                # Filtrar o benchmark com os mesmos filtros aplicados à base principal
                filtered_benchmark = apply_benchmark_filter(
                    st.session_state.benchmark_df, minutes_range, mpg_range,
                    age_range, sel_pos)

                if filtered_benchmark is not None and not filtered_benchmark.empty:
                    # Adicionar informação sobre o benchmark ao subtítulo
                    subtitle += f" | Benchmark: {st.session_state.benchmark_name}"

                    # Criar figura manualmente para combinar os dados
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plotar pontos da base principal (azul)
                    sc1 = ax.scatter(df_group[x_metric],
                                     df_group[y_metric],
                                     alpha=0.7,
                                     s=60,
                                     c=player1_color,
                                     edgecolor='white',
                                     label='Current Database')

                    # Plotar pontos do benchmark (vermelho)
                    sc2 = ax.scatter(filtered_benchmark[x_metric],
                                     filtered_benchmark[y_metric],
                                     alpha=0.7,
                                     s=60,
                                     c=player2_color,
                                     edgecolor='white',
                                     label='Benchmark Database')

                    # Adicionar rótulos para jogadores da base principal
                    for i, row in df_group.iterrows():
                        # Adicionar textos abaixo dos pontos
                        ax.annotate(row['Player'],
                                    (row[x_metric], row[y_metric]),
                                    xytext=(0, -10),
                                    textcoords='offset points',
                                    fontsize=8,
                                    ha='center',
                                    va='top',
                                    alpha=0.8,
                                    color=player1_color,
                                    bbox=dict(facecolor='white',
                                              alpha=0.7,
                                              edgecolor=player1_color,
                                              boxstyle="round,pad=0.1",
                                              linewidth=0.5),
                                    zorder=10)

                    # Adicionar rótulos para jogadores do benchmark
                    for i, row in filtered_benchmark.iterrows():
                        # Adicionar textos abaixo dos pontos
                        ax.annotate(row['Player'],
                                    (row[x_metric], row[y_metric]),
                                    xytext=(0, -10),
                                    textcoords='offset points',
                                    fontsize=8,
                                    ha='center',
                                    va='top',
                                    alpha=0.8,
                                    color=player2_color,
                                    bbox=dict(facecolor='white',
                                              alpha=0.7,
                                              edgecolor=player2_color,
                                              boxstyle="round,pad=0.1",
                                              linewidth=0.5),
                                    zorder=10)

                    # Adicionar linhas médias
                    ax.axvline(df_group[x_metric].mean(),
                               color=player1_color,
                               linestyle='--',
                               alpha=0.5,
                               label='Current DB Mean')
                    ax.axhline(df_group[y_metric].mean(),
                               color=player1_color,
                               linestyle='--',
                               alpha=0.5)

                    ax.axvline(filtered_benchmark[x_metric].mean(),
                               color=player2_color,
                               linestyle='--',
                               alpha=0.5,
                               label='Benchmark Mean')
                    ax.axhline(filtered_benchmark[y_metric].mean(),
                               color=player2_color,
                               linestyle='--',
                               alpha=0.5)

                    # Configurar rótulos e título
                    ax.set_xlabel(x_metric, fontsize=12)
                    ax.set_ylabel(y_metric, fontsize=12)
                    ax.set_title(title, fontsize=14)
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    # Adicionar subtítulo
                    plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=10)

                    # Ajustar layout
                    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                else:
                    st.warning(
                        "No benchmark players match the applied filters")

                    # Criar scatter plot normal se o benchmark não tem dados
                    fig = create_scatter_plot(df=df_group,
                                              x_metric=x_metric,
                                              y_metric=y_metric,
                                              title=title)
            else:
                # Criar scatter plot normal
                fig = create_scatter_plot(df=df_group,
                                          x_metric=x_metric,
                                          y_metric=y_metric,
                                          title=title)

            # Display scatter plot
            st.pyplot(fig)

            # Show correlation coefficient
            corr = df_group[x_metric].corr(df_group[y_metric])
            st.info(f"Correlation coefficient: {corr:.4f}")

            # Export button
            if st.button('Export Scatter Plot (300 DPI)',
                         key='export_scatter'):
                img_bytes = fig_to_bytes(fig)
                st.download_button(
                    "⬇️ Download Scatter Plot",
                    data=img_bytes,
                    file_name=f"scatter_{x_metric}_vs_{y_metric}.png",
                    mime="image/png")

        # =============================================
        # Player Similarity (Aba 4) - Advanced Similarity Model
        # =============================================
        with tabs[3]:
            st.header('Advanced Player Similarity Analysis')

            # Explicação do modelo de similaridade aprimorado
            with st.expander("ℹ️ About Player Similarity Model",
                             expanded=False):
                st.markdown("""
                This advanced player similarity model uses the approach from [SteveAQ's Player Similarity Models](https://steveaq.github.io/Player-Similarity-Models/).

                **How it works:**
                1. **PCA (Principal Component Analysis)** reduces multiple metrics to two dimensions
                2. **K-Means Clustering** groups players with similar styles
                3. **Euclidean Distance** in PCA space finds players with most similar profiles
                4. Players closer in the PCA chart have more similar playing styles

                This approach provides more intuitive and visually informative player comparisons than traditional methods.
                """)

            col1, col2 = st.columns(2)
            with col1:
                # Initialize benchmark player usage flag
                use_benchmark_player = st.session_state.get(
                    'use_benchmark_player', False)

                # Verificar opção de player benchmark e recarregar a página se necessário
                # Isso garante que a interface seja atualizada quando o usuário marca/desmarca a opção
                if st.session_state.get(
                        'use_benchmark_player') != use_benchmark_player:
                    # Atualizar o estado
                    st.session_state.use_benchmark_player = use_benchmark_player

                # Select player for similarity search based on benchmark preference
                if use_benchmark_player and st.session_state.benchmark_loaded:
                    # If using benchmark player as reference, prepare benchmark data
                    try:
                        # Get benchmark data with minimal filters
                        benchmark_min_filter = apply_benchmark_filter(
                            st.session_state.benchmark_df,
                            [0, 5000],  # Wide minutes range
                            [0, 90],  # Wide MPG range
                            [15, 45],  # Wide age range
                            ["All"]  # All positions
                        )

                        if benchmark_min_filter is not None and not benchmark_min_filter.empty:
                            st.success(
                                "Using benchmark player as reference to find similar players in main dataset"
                            )
                            # Show dropdown with benchmark players
                            benchmark_players = benchmark_min_filter[
                                'Player'].sort_values().tolist()
                            sim_player = st.selectbox(
                                'Select BENCHMARK Reference Player',
                                benchmark_players,
                                key='sim_player_benchmark')

                            # Set benchmark player flag
                            is_benchmark_player = True
                        else:
                            st.error("Could not load benchmark players")
                            sim_player = st.selectbox(
                                'Select Reference Player',
                                players,
                                key='sim_player')
                            is_benchmark_player = False
                    except Exception as bench_ref_error:
                        st.error(
                            f"Error loading benchmark players: {str(bench_ref_error)}"
                        )
                        sim_player = st.selectbox('Select Reference Player',
                                                  players,
                                                  key='sim_player')
                        is_benchmark_player = False
                else:
                    # Regular selection from main dataset
                    # Verificar se o jogador anteriormente selecionado ainda está disponível
                    player_index = 0
                    if st.session_state.saved_similarity_player in players:
                        player_index = players.index(
                            st.session_state.saved_similarity_player)

                    # Usar o index para manter a seleção anterior quando possível
                    sim_player = st.selectbox('Select Reference Player',
                                              players,
                                              index=player_index,
                                              key='sim_player')

                    # Atualizar nossa variável de armazenamento (não o widget diretamente)
                    st.session_state.saved_similarity_player = sim_player
                    is_benchmark_player = False

            with col2:
                # Verificar se o método anteriormente selecionado ainda está disponível
                sim_methods = [
                    'PCA + K-Means (Recommended)', 'Cosine Similarity',
                    'Euclidean Distance'
                ]
                method_index = 0  # Padrão para PCA + K-Means

                # Mapear o método interno para o nome de exibição
                method_display_mapping = {
                    'pca_kmeans': 'PCA + K-Means (Recommended)',
                    'cosine': 'Cosine Similarity',
                    'euclidean': 'Euclidean Distance'
                }

                # Se temos um método salvo e ele é válido, use-o como padrão
                if st.session_state.saved_similarity_method in method_display_mapping.keys(
                ):
                    display_method = method_display_mapping[
                        st.session_state.saved_similarity_method]
                    if display_method in sim_methods:
                        method_index = sim_methods.index(display_method)

                similarity_method = st.selectbox('Similarity Method',
                                                 sim_methods,
                                                 index=method_index)

            # Map the displayed method name to internal method identifier
            method_mapping = {
                'PCA + K-Means (Recommended)': 'pca_kmeans',
                'Cosine Similarity': 'cosine',
                'Euclidean Distance': 'euclidean'
            }
            method = method_mapping[similarity_method]

            # Salvar nossa escolha para preservar entre mudanças de filtro
            st.session_state.saved_similarity_method = method

            # Select metrics for similarity calculation
            st.subheader("Select Metrics for Similarity Calculation")

            # Predefined metric groups for easy selection - always using per 90 metrics
            # Carregar grupos personalizados salvos na sessão
            if 'custom_metric_groups' in st.session_state:
                custom_groups = st.session_state.custom_metric_groups
            else:
                custom_groups = {}
                
            metric_groups = {
                "Overall": [
                    'Progressive runs per 90', 'Progressive passes per 90',
                    'Deep completions per 90', 'Key passes per 90',
                    'Accurate passes, %', 'Accurate crosses, %', 'xA per 90',
                    'Defensive duels won, %', 'Aerial duels won, %',
                    'PAdj Interceptions', 'PAdj Sliding tackles',
                    'npxG per 90', 'npxG per Shot', 'Goal conversion, %',
                    'Shots on target, %'
                ],
                "Offensive": [
                    'Goals per 90', 'Shots per 90', 'xG per 90', 'npxG',
                    'G-xG', 'npxG per Shot', 'Box Efficiency',
                    'Shots on target per 90', 'Successful dribbles per 90',
                    'Progressive runs per 90'
                ],
                "Passing": [
                    'Passes per 90', 'Accurate passes, %',
                    'Forward passes per 90', 'Progressive passes per 90',
                    'Key passes per 90', 'Assists per 90', 'xA per 90'
                ],
                "Defensive": [
                    'Interceptions per 90', 'Tackles per 90',
                    'Defensive duels per 90', 'Aerial duels won per 90',
                    'Recoveries per 90'
                ],
                "Physical": [
                    'Accelerations per 90', 'Sprint distance per 90',
                    'Distance covered per 90'
                ],
                "Advanced Metrics":
                ['npxG', 'G-xG', 'npxG per Shot', 'Box Efficiency'
                 ],  # Incluindo Box Efficiency nas métricas avançadas
                "Pedri Metrics": [
                    'Smart passes per 90', 'Key passes per 90', 'Shot assists per 90', 'xA per 90',
                    'Passes per 90', 'Progressive passes per 90', 'Passes to final third per 90', 'Through passes per 90',
                    'Progressive runs per 90', 'Dribbles per 90', 'Successful dribbles, %',
                    'Interceptions per 90', 'Defensive duels per 90', 'Defensive duels won, %',
                    'Received passes per 90'
                ]
            }
            
            # Adicionar grupos personalizados ao dicionário de grupos de métricas
            for group_name, metrics in custom_groups.items():
                if group_name not in metric_groups:
                    metric_groups[group_name] = metrics

            # Let user first select a preset group or custom
            metric_selection_mode = st.radio(
                "Metric Selection Mode",
                ["Custom (Select Individual Metrics)", "Use Metric Presets"],
                horizontal=True,
                key="similarity_metric_selection_mode"  # Adicionando chave única
            )

            if metric_selection_mode == "Use Metric Presets":
                selected_groups = st.multiselect(
                    "Select Metric Groups",
                    list(metric_groups.keys()),
                    default=["Offensive", "Passing"],
                    key="similarity_preset_groups"  # Adicionando chave única
                )

                # Combine all metrics from selected groups
                preset_metrics = []
                for group in selected_groups:
                    # Only add metrics that exist in the dataframe
                    available_metrics = [
                        m for m in metric_groups[group] if m in metric_cols
                    ]
                    preset_metrics.extend(available_metrics)

                # Criar variáveis temporárias para armazenar métricas
                if 'temp_sim_metrics' not in st.session_state:
                    st.session_state.temp_sim_metrics = preset_metrics
                
                # Opção para colar lista de métricas
                with st.expander("📋 Paste metrics list"):
                    metrics_text = st.text_area(
                        "Paste metrics (one per line or comma-separated)",
                        height=150,
                        key="sim_metrics_paste"
                    )
                    
                    if st.button("Process pasted metrics", key="process_sim_metrics"):
                        # Processar as métricas coladas
                        valid_metrics = process_pasted_metrics(metrics_text, metric_cols)
                        if valid_metrics:
                            st.session_state.temp_sim_metrics = valid_metrics
                            st.success(f"Found {len(valid_metrics)} valid metrics!")
                            st.rerun()
                        else:
                            st.warning("No valid metrics found in the pasted text. Please check the format and try again.")
                
                # Allow further customization
                temp_metrics = st.multiselect(
                    'Add or Remove Individual Metrics',
                    metric_cols,
                    default=st.session_state.temp_sim_metrics,
                    key="similarity_temp_metrics"
                )
                
                # Atualizar a seleção temporária
                st.session_state.temp_sim_metrics = temp_metrics
                
                # Botão para aplicar as métricas selecionadas
                apply_metrics = st.button("Apply Selected Metrics", key="apply_sim_preset_metrics")
                
                if apply_metrics:
                    # Aplicar as métricas selecionadas
                    st.session_state.saved_similarity_metrics = st.session_state.temp_sim_metrics
                    st.rerun()
                
                # Usar as métricas já confirmadas para o processamento
                sim_metric_options = st.session_state.saved_similarity_metrics
            else:
                # Criar variáveis temporárias para armazenar métricas custom
                if 'temp_sim_custom_metrics' not in st.session_state:
                    st.session_state.temp_sim_custom_metrics = metric_cols[:min(8, len(metric_cols))]
                
                # Opção para colar lista de métricas
                with st.expander("📋 Paste metrics list"):
                    metrics_text = st.text_area(
                        "Paste metrics (one per line or comma-separated)",
                        height=150,
                        key="sim_custom_metrics_paste"
                    )
                    
                    if st.button("Process pasted metrics", key="process_sim_custom_metrics"):
                        # Processar as métricas coladas
                        valid_metrics = process_pasted_metrics(metrics_text, metric_cols)
                        if valid_metrics:
                            st.session_state.temp_sim_custom_metrics = valid_metrics
                            st.success(f"Found {len(valid_metrics)} valid metrics!")
                            st.rerun()
                        else:
                            st.warning("No valid metrics found in the pasted text. Please check the format and try again.")
                
                # Garantir que todos os itens em temp_sim_custom_metrics estão em metric_cols para evitar erros
                valid_temp_metrics = [m for m in st.session_state.temp_sim_custom_metrics if m in metric_cols]
                st.session_state.temp_sim_custom_metrics = valid_temp_metrics if valid_temp_metrics else metric_cols[:min(8, len(metric_cols))]
                
                # Let user select metrics manually
                temp_metrics = st.multiselect(
                    'Choose Individual Metrics (6-15 recommended for PCA)',
                    metric_cols,
                    default=st.session_state.temp_sim_custom_metrics,
                    key="similarity_temp_custom_metrics"
                )
                
                # Atualizar a seleção temporária
                st.session_state.temp_sim_custom_metrics = temp_metrics
                
                # Botão para aplicar as métricas selecionadas
                apply_metrics = st.button("Apply Selected Metrics", key="apply_sim_custom_metrics")
                
                if apply_metrics:
                    # Aplicar as métricas selecionadas
                    st.session_state.saved_similarity_metrics = st.session_state.temp_sim_custom_metrics
                    st.rerun()
                
                # Usar as métricas já confirmadas para o processamento
                sim_metric_options = st.session_state.saved_similarity_metrics

            # Advanced options in expander
            with st.expander("Advanced Filtering Options", expanded=False):
                # Number of similar players to find
                num_similar = st.slider('Number of similar players to show', 3,
                                        15, 8)

                # Filter similar players by position
                filter_by_position = st.checkbox(
                    'Filter similar players by reference player position',
                    True)

                # Filter similar players by age
                filter_by_age = st.checkbox(
                    'Filter similar players by age range', False)
                if filter_by_age:
                    ref_player_age = df_minutes[df_minutes['Player'] ==
                                                sim_player]['Age'].iloc[0]
                    age_diff = st.slider('Maximum age difference (years)', 0,
                                         10, 3)
                    min_age_filter = ref_player_age - age_diff
                    max_age_filter = ref_player_age + age_diff

                    # Apply age filter to dataframe
                    df_sim = df_minutes[df_minutes['Age'].between(
                        min_age_filter, max_age_filter)]
                else:
                    df_sim = df_minutes.copy()

                # Only if using PCA+K-Means, let user specify number of clusters
                if method == 'pca_kmeans':
                    num_clusters = st.slider('Number of clusters for K-Means',
                                             3, 12, 8)
                else:
                    num_clusters = 8

                # Benchmark options
                use_benchmark = False

                if st.session_state.benchmark_loaded:
                    st.info(
                        f"Benchmark dataset loaded: {st.session_state.benchmark_name}"
                    )

                    benchmark_options = st.columns(2)
                    with benchmark_options[0]:
                        use_benchmark = st.checkbox(
                            "Include benchmark players in results",
                            value=True,
                            help=
                            "Include players from benchmark dataset in similarity search results"
                        )

                    with benchmark_options[1]:
                        use_benchmark_player = st.checkbox(
                            "Use benchmark player as reference",
                            value=False,
                            help=
                            "Select a player from the benchmark dataset to find similar ones in main dataset"
                        )

                        # Verificar se o valor mudou para forçar recarregamento da página
                        old_value = st.session_state.get(
                            'use_benchmark_player', None)
                        if old_value != use_benchmark_player:
                            # Atualizar estado e forçar recarregamento da página
                            st.session_state.use_benchmark_player = use_benchmark_player
                            st.rerun()

                        # Adicionar opção para buscar similares em ambos os datasets ou apenas no principal
                        if use_benchmark_player:
                            st.session_state.search_in_benchmark = st.checkbox(
                                'Include Benchmark Players in similarity search',
                                False,
                                help=
                                "When checked, players from both the main dataset and benchmark will be considered. When unchecked, only players from the main dataset will be shown as similar."
                            )

                if use_benchmark:
                    try:
                        # Use current filter values from sidebar for benchmark
                        curr_minutes_range = st.session_state.last_minutes_range
                        curr_mpg_range = st.session_state.last_mpg_range
                        curr_age_range = st.session_state.last_age_range
                        curr_pos = st.session_state.last_positions

                        # Apply filters to benchmark data
                        filtered_benchmark = apply_benchmark_filter(
                            st.session_state.benchmark_df, curr_minutes_range,
                            curr_mpg_range, curr_age_range, curr_pos)

                        if filtered_benchmark is not None and not filtered_benchmark.empty:
                            # Apply additional age filter if needed
                            if filter_by_age:
                                # Set default age range values if not already defined
                                age_min = min_age_filter if 'min_age_filter' in locals(
                                ) else 15
                                age_max = max_age_filter if 'max_age_filter' in locals(
                                ) else 45

                                filtered_benchmark = filtered_benchmark[
                                    filtered_benchmark['Age'].between(
                                        age_min, age_max)]

                            # Get reference player position for filtering
                            if filter_by_position and 'Position_split' in df_minutes.columns:
                                ref_player_pos = df_minutes[
                                    df_minutes['Player'] ==
                                    sim_player]['Position_split'].iloc[0]

                                # Apply position filter to benchmark if possible
                                if 'Position_split' in filtered_benchmark.columns:
                                    filtered_benchmark = filtered_benchmark[
                                        filtered_benchmark['Position_split'].
                                        apply(lambda x: any(
                                            pos in ref_player_pos for pos in x
                                        ) if isinstance(x, list) else False)]
                                elif 'Position' in filtered_benchmark.columns:
                                    # Try using regular Position column if Position_split not available
                                    ref_player_pos_text = df_minutes[
                                        df_minutes['Player'] ==
                                        sim_player]['Position'].iloc[0]
                                    filtered_benchmark = filtered_benchmark[
                                        filtered_benchmark['Position'] ==
                                        ref_player_pos_text]

                            # Combine dataframes for similarity search
                            df_sim = pd.concat([df_sim, filtered_benchmark],
                                               ignore_index=True)
                            st.info(
                                f"Including {len(filtered_benchmark)} benchmark players in similarity search"
                            )
                        else:
                            st.warning(
                                "No benchmark players match the current filters"
                            )
                    except Exception as bench_error:
                        st.error(
                            f"Error applying benchmark filters: {str(bench_error)}"
                        )
                elif not st.session_state.benchmark_loaded:
                    st.info(
                        "No benchmark database loaded. Upload one in the sidebar to include players from other leagues."
                    )

                # Handle benchmark player as reference (search in main dataset only)
                if is_benchmark_player and use_benchmark_player:
                    # Get the benchmark player data
                    benchmark_player_data = None
                    try:
                        benchmark_player_data = st.session_state.benchmark_df[
                            st.session_state.benchmark_df['Player'] ==
                            sim_player].iloc[0]
                    except Exception as bench_player_error:
                        st.error(
                            f"Could not get benchmark player data: {str(bench_player_error)}"
                        )

                    if benchmark_player_data is not None:
                        # Para jogador de referência do benchmark, verificar opção do usuário
                        if st.session_state.get('search_in_benchmark', False):
                            st.info(
                                "Finding similar players to benchmark player in both datasets (main and benchmark)"
                            )
                            # Manter o dataframe combinado (df_sim já contém ambos os datasets)
                        else:
                            st.info(
                                "Finding similar players to benchmark player in the main dataset only"
                            )
                            df_sim = df_minutes.copy(
                            )  # Reset to only use main dataset

                        # Apply position filter if selected - get position from benchmark player
                        if filter_by_position:
                            if 'Position_split' in st.session_state.benchmark_df.columns and 'Position_split' in df_minutes.columns:
                                ref_player_pos = benchmark_player_data[
                                    'Position_split']
                                # Check if Position_split contains list data or float (error handling)
                                df_sim = df_sim[df_sim['Position_split'].apply(
                                    lambda x: any(pos in ref_player_pos
                                                  for pos in x)
                                    if isinstance(x, list) else False)]
                            elif 'Position' in st.session_state.benchmark_df.columns and 'Position' in df_minutes.columns:
                                ref_position = benchmark_player_data[
                                    'Position']
                                df_sim = df_sim[df_sim['Position'] ==
                                                ref_position]
                # Regular position filtering for main dataset reference player
                elif filter_by_position and 'Position_split' in df_minutes.columns:
                    ref_player_pos = df_minutes[
                        df_minutes['Player'] ==
                        sim_player]['Position_split'].iloc[0]
                    # Check if Position_split contains list data or float (error handling)
                    df_sim = df_sim[df_sim['Position_split'].apply(
                        lambda x: any(pos in ref_player_pos for pos in x)
                        if isinstance(x, list) else False)]

                # Option for PCA specific settings
                st.subheader("PCA Specific Settings")
                
                # Option to use different metrics for PCA than for detail display
                use_diff_pca_metrics = st.checkbox(
                    "Use different metrics for PCA analysis",
                    value=False,
                    help="When enabled, you can select a specific metric group to be used only for the PCA clustering and similarity calculation"
                )
                
                if use_diff_pca_metrics:
                    # Use the same pattern as with other metric groups
                    st.info("Select metrics specifically for PCA calculation")
                    
                    # Define PCA metric groups
                    pca_metric_selection_mode = st.radio(
                        "PCA Metric Selection Mode",
                        ["Custom (Select Individual Metrics)", "Use Metric Presets"],
                        horizontal=True,
                        key="pca_metric_selection_mode"
                    )
                    
                    if pca_metric_selection_mode == "Use Metric Presets":
                        # Let user select a preset group
                        pca_selected_group = st.selectbox(
                            "Select Metric Group for PCA",
                            list(metric_groups.keys()),
                            key="pca_preset_group"
                        )
                        
                        # Get metrics from the selected group
                        pca_metrics = []
                        for metric in metric_groups[pca_selected_group]:
                            if metric in metric_cols:
                                pca_metrics.append(metric)
                        
                        if len(pca_metrics) < 2:
                            st.warning(f"The selected group '{pca_selected_group}' has fewer than 2 valid metrics. Using the main metrics for PCA.")
                            pca_metrics = sim_metric_options
                        else:
                            st.success(f"Using {len(pca_metrics)} metrics from group '{pca_selected_group}' for PCA analysis")
                    else:
                        # Custom metric selection for PCA
                        pca_metrics = st.multiselect(
                            "Select metrics for PCA (6-15 recommended)",
                            metric_cols,
                            default=sim_metric_options[:min(8, len(sim_metric_options))],
                            key="pca_custom_metrics"
                        )
                        
                        if len(pca_metrics) < 2:
                            st.warning("At least 2 metrics are required for PCA. Using the main metrics.")
                            pca_metrics = sim_metric_options
                else:
                    # Use the same metrics as selected for the similarity analysis
                    pca_metrics = sim_metric_options
                
                # Compute similar players with progress indicator
                if len(sim_metric_options) >= 2 and len(df_sim) > 1:
                    try:
                        with st.spinner(
                                f"Finding players similar to {sim_player} using {method} method..."
                        ):
                            # Special handling for benchmark player as reference
                            if is_benchmark_player and use_benchmark_player:
                                # We need to create a temporary combined dataframe just for similarity calculation
                                # that includes the benchmark player
                                st.info(
                                    "Using benchmark player as reference - special processing"
                                )

                                # Get benchmark player data
                                benchmark_player_row = st.session_state.benchmark_df[
                                    st.session_state.benchmark_df['Player'] ==
                                    sim_player]

                                if not benchmark_player_row.empty:
                                    # Create a temporary dataframe with the benchmark player added to main dataframe
                                    temp_df_sim = pd.concat(
                                        [df_sim, benchmark_player_row],
                                        ignore_index=True)

                                    # Verify player is in the dataframe
                                    if sim_player not in temp_df_sim[
                                            'Player'].values:
                                        st.error(
                                            f"Could not add benchmark player to temporary dataframe!"
                                        )
                                    else:
                                        st.success(
                                            f"Successfully added benchmark player to comparison dataset"
                                        )
                                        # Use this temporary dataframe for similarity calculations
                                        df_sim = temp_df_sim
                                else:
                                    st.error(
                                        f"Could not find benchmark player '{sim_player}' data!"
                                    )
                            # Regular check for player in dataframe (non-benchmark reference)
                            elif sim_player not in df_sim['Player'].values:
                                st.error(
                                    f"Player '{sim_player}' not found in the filtered dataset!"
                                )
                                # Try using alternative method
                                method = 'euclidean'
                                st.warning(
                                    f"Switching to alternative method: {method}"
                                )

                            # Show information about the dataset
                            st.info(
                                f"Analyzing {len(df_sim)} players with {len(pca_metrics)} metrics for PCA"
                            )

                            # Check if we have all necessary metrics for PCA
                            missing_metrics = [
                                m for m in pca_metrics
                                if m not in df_sim.columns
                            ]
                            if missing_metrics:
                                st.warning(
                                    f"Metrics not found for PCA: {missing_metrics}")
                                # Remove missing metrics
                                pca_metrics = [
                                    m for m in pca_metrics
                                    if m not in missing_metrics
                                ]
                                st.info(
                                    f"Using only available metrics for PCA: {pca_metrics}"
                                )

                            # If using PCA+K-Means, pass the number of clusters
                            if method == 'pca_kmeans':
                                try:
                                    # Create PCA+K-Means DataFrame first to display PCA chart separately
                                    st.info(
                                        "Applying PCA and K-Means clustering..."
                                    )
                                    pca_df = create_pca_kmeans_df(
                                        df_sim,
                                        pca_metrics,
                                        n_clusters=num_clusters)

                                    if pca_df is None:
                                        st.error(
                                            "Could not create the PCA+K-Means model. Using alternative method."
                                        )
                                        method = 'euclidean'

                                    # Get similar players
                                    similar_players = compute_player_similarity(
                                        df=df_sim,
                                        player=sim_player,
                                        metrics=pca_metrics,
                                        n=num_similar,
                                        method=method)
                                except Exception as pca_error:
                                    st.error(
                                        f"Error in PCA+K-Means model: {str(pca_error)}"
                                    )
                                    st.warning(
                                        "Using alternative similarity method..."
                                    )
                                    method = 'euclidean'
                                    similar_players = compute_player_similarity(
                                        df=df_sim,
                                        player=sim_player,
                                        metrics=sim_metric_options,
                                        n=num_similar,
                                        method=method)
                            else:
                                # Use vector-based method
                                similar_players = compute_player_similarity(
                                    df=df_sim,
                                    player=sim_player,
                                    metrics=sim_metric_options,
                                    n=num_similar,
                                    method=method)
                    except Exception as e:
                        st.error(f"Error computing similarity: {str(e)}")
                        similar_players = []

                    # Display results in tabs
                    if similar_players:
                        result_tabs = st.tabs(["Similarity Table", "Raw Data"])

                        # Tab 1: Similarity Table
                        with result_tabs[0]:
                            st.subheader(
                                f"Players Most Similar to {sim_player}")

                            # Create a more detailed dataframe
                            detailed_data = []
                            for player_name, similarity in similar_players:
                                # Skip reference player when using benchmark reference
                                if is_benchmark_player and player_name == sim_player:
                                    continue

                                # For benchmark player reference, verificar se deve incluir benchmark ou não
                                if is_benchmark_player and use_benchmark_player and not st.session_state.get(
                                        'search_in_benchmark', False):
                                    # Checar se o jogador está no dataset principal (não no benchmark)
                                    if player_name not in df_minutes[
                                            'Player'].values:
                                        continue  # Pular jogadores do benchmark nos resultados

                                    # Use main dataset for info
                                    player_info = df_minutes[
                                        df_minutes['Player'] == player_name]
                                else:
                                    # Regular case - use full dataset
                                    player_info = df_sim[df_sim['Player'] ==
                                                         player_name]

                                if not player_info.empty:
                                    player_row = {
                                        'Player':
                                        player_name,
                                        'Team':
                                        player_info.iloc[0].get('Team', 'N/A'),
                                        'Age':
                                        player_info.iloc[0].get('Age', 'N/A'),
                                        'Position':
                                        player_info.iloc[0].get(
                                            'Position', 'N/A'),
                                        'Minutes':
                                        player_info.iloc[0].get(
                                            'Minutes played', 'N/A'),
                                        'Similarity':
                                        f"{similarity:.1f}%"
                                    }

                                    # Add metric data
                                    for metric in sim_metric_options:
                                        if metric in player_info.columns:
                                            player_row[
                                                metric] = player_info.iloc[0][
                                                    metric]

                                    detailed_data.append(player_row)

                            if detailed_data:
                                sim_df = pd.DataFrame(detailed_data)

                                # Create two display options: standard table and styled PIQ-like table
                                table_display_option = st.radio(
                                    "Display format:", [
                                        "Standard Table",
                                        "PIQ-style Similarity Table"
                                    ],
                                    horizontal=True)

                                if table_display_option == "Standard Table":
                                    # Show standard table with all metrics
                                    st.dataframe(sim_df,
                                                 use_container_width=True)
                                else:
                                    # Create a styled table similar to PIQ reference
                                    # Get reference player values
                                    ref_player_values = df_sim[df_sim['Player']
                                                               == sim_player]

                                    if not ref_player_values.empty:
                                        # Create dataframe for PIQ-style table
                                        # Use all available metrics for the PIQ style table
                                        display_metrics = sim_metric_options

                                        # Create columns for reference and delta values
                                        piq_cols = [
                                            'Player', 'Position', 'Similarity'
                                        ]

                                        # Add metrics and delta columns
                                        for metric in display_metrics:
                                            piq_cols.extend(
                                                [metric, f"Δ% {metric}"])

                                        # Create PIQ style dataframe
                                        piq_data = []
                                        ref_values = {
                                            metric:
                                            ref_player_values.iloc[0].get(
                                                metric, 0)
                                            for metric in display_metrics
                                        }

                                        for player_name, similarity in similar_players:
                                            # Skip reference player when using benchmark reference
                                            if is_benchmark_player and player_name == sim_player:
                                                continue

                                            # For benchmark player reference, verificar se deve incluir benchmark ou não
                                            if is_benchmark_player and use_benchmark_player and not st.session_state.get(
                                                    'search_in_benchmark',
                                                    False):
                                                # Checar se o jogador está no dataset principal (não no benchmark)
                                                if player_name not in df_minutes[
                                                        'Player'].values:
                                                    continue  # Pular jogadores do benchmark nos resultados

                                                # Use main dataset for info
                                                player_info = df_minutes[
                                                    df_minutes['Player'] ==
                                                    player_name]
                                            else:
                                                # Regular case - use full dataset
                                                player_info = df_sim[
                                                    df_sim['Player'] ==
                                                    player_name]

                                            if not player_info.empty:
                                                row_data = {
                                                    'Player':
                                                    player_name,
                                                    'Position':
                                                    player_info.iloc[0].get(
                                                        'Position', 'N/A'),
                                                    'Similarity':
                                                    f"{similarity:.1f}%"
                                                }

                                                # Calculate metrics and deltas
                                                for metric in display_metrics:
                                                    metric_value = player_info.iloc[
                                                        0].get(metric, 0)
                                                    row_data[
                                                        metric] = f"{metric_value:.2f}"

                                                    # Calculate percentage difference
                                                    if ref_values[metric] != 0:
                                                        delta_pct = (
                                                            (metric_value -
                                                             ref_values[metric]
                                                             ) /
                                                            abs(ref_values[
                                                                metric])) * 100
                                                        # Format with sign and one decimal place for PIQ-style
                                                        if delta_pct > 0:
                                                            delta_str = f"+{delta_pct:.1f}"
                                                        else:
                                                            delta_str = f"{delta_pct:.1f}"
                                                    else:
                                                        delta_str = "N/A"

                                                    row_data[
                                                        f"Δ% {metric}"] = delta_str

                                                piq_data.append(row_data)

                                        # Create styled dataframe
                                        if piq_data:
                                            piq_df = pd.DataFrame(piq_data)

                                            # Apply styling - similar to PIQ reference image with more precise control
                                            def color_delta_cells(val):
                                                try:
                                                    if isinstance(
                                                            val, str
                                                    ) and val not in ['N/A']:
                                                        # Handle the "+" sign in values for proper comparison
                                                        if "+" in val:
                                                            val_float = float(
                                                                val.replace(
                                                                    "+", ""))
                                                        else:
                                                            val_float = float(
                                                                val)

                                                        # More nuanced color intensity based on value magnitude - PIQ-style
                                                        if val_float > 20:  # Strong positive
                                                            return 'background-color: #9be0a5; color: #1a4d1f; font-weight: bold;'
                                                        elif val_float > 10:  # Positive
                                                            return 'background-color: #c1efc8; color: #1a4d1f;'
                                                        elif val_float > 0:  # Slight positive
                                                            return 'background-color: #e6f7e9; color: #1a4d1f;'
                                                        elif val_float > -10:  # Slight negative
                                                            return 'background-color: #fbebee; color: #871924;'
                                                        elif val_float > -20:  # Negative
                                                            return 'background-color: #f6ccd1; color: #871924;'
                                                        else:  # Strong negative
                                                            return 'background-color: #ee9ca4; color: #871924; font-weight: bold;'
                                                except:
                                                    pass
                                                return ''

                                            # Updated version for newer pandas
                                            def color_delta_cells_elem(x):
                                                try:
                                                    if isinstance(
                                                            x,
                                                            str) and x not in [
                                                                'N/A'
                                                            ]:
                                                        # Handle the "+" sign in values for proper comparison
                                                        if "+" in x:
                                                            val_float = float(
                                                                x.replace(
                                                                    "+", ""))
                                                        else:
                                                            val_float = float(
                                                                x)

                                                        # More nuanced color intensity based on value magnitude - PIQ-style
                                                        if val_float > 20:  # Strong positive
                                                            return 'background-color: #9be0a5; color: #1a4d1f; font-weight: bold;'
                                                        elif val_float > 10:  # Positive
                                                            return 'background-color: #c1efc8; color: #1a4d1f;'
                                                        elif val_float > 0:  # Slight positive
                                                            return 'background-color: #e6f7e9; color: #1a4d1f;'
                                                        elif val_float > -10:  # Slight negative
                                                            return 'background-color: #fbebee; color: #871924;'
                                                        elif val_float > -20:  # Negative
                                                            return 'background-color: #f6ccd1; color: #871924;'
                                                        else:  # Strong negative
                                                            return 'background-color: #ee9ca4; color: #871924; font-weight: bold;'
                                                except:
                                                    pass
                                                return ''

                                            # Apply styles and display
                                            # Create a header similar to PIQ image
                                            st.markdown(
                                                f"### {sim_player} - PIQ Similarity Model Results"
                                            )

                                            # Add subtitle explaining model and metrics
                                            metric_names = ", ".join([
                                                m.replace("_", " ").title()
                                                for m in display_metrics
                                            ])
                                            st.markdown(
                                                f"**{metric_names}** metrics via {method.upper()} model"
                                            )

                                            # Explain what the statistics mean
                                            st.markdown(
                                                "Similarity shows overall profile match. Δ% columns show percentage difference from reference player."
                                            )
                                            st.markdown(
                                                "🟢 Green = better value | 🟡 Yellow = similar value | 🔴 Red = worse value"
                                            )

                                            # Create better styling with team logos and bars for similarity scores
                                            # First add similarity score bars
                                            def bar_similarity(val):
                                                if isinstance(
                                                        val,
                                                        str) and '%' in val:
                                                    try:
                                                        score = float(
                                                            val.replace(
                                                                '%', ''))
                                                        width = min(
                                                            100, max(0, score))
                                                        return f'background: linear-gradient(90deg, #0066cc {width}%, transparent {width}%); color: white; font-weight: bold;'
                                                    except:
                                                        pass
                                                return ''

                                            # Apply more PIQ-like styling - updated for newer pandas
                                            styled_df = piq_df.style\
                                                .map(color_delta_cells_elem, subset=[col for col in piq_df.columns if col.startswith('Δ%')])\
                                                .format({'Similarity': lambda x: x})\
                                                .set_properties(**{'text-align': 'center'}, subset=piq_df.columns[2:])\
                                                .set_properties(**{'font-weight': 'bold'}, subset=['Player'])\
                                                .hide(axis='index')

                                            st.dataframe(
                                                styled_df,
                                                use_container_width=True)
                                        else:
                                            st.warning(
                                                "Could not generate PIQ-style table with available data"
                                            )
                                    else:
                                        st.error(
                                            f"Reference player '{sim_player}' data not found for table creation"
                                        )

                                # Export as CSV
                                csv = sim_df.to_csv(
                                    index=False).encode('utf-8')
                                st.download_button(
                                    "⬇️ Download Similarity Table as CSV",
                                    csv,
                                    f"similarity_{sim_player}_{method}.csv",
                                    "text/csv",
                                    key="download_similarity_csv")

                        # Tab 3: Raw Data
                        with result_tabs[1]:
                            st.subheader("Raw Player Data")
                            # Create list of players to display
                            players_to_show = [sim_player] + [
                                p[0] for p in similar_players
                            ]

                            # Para referência de jogador do benchmark, verificar opção do usuário
                            if is_benchmark_player and use_benchmark_player:
                                # Adicionar o jogador do benchmark como referência
                                benchmark_player_data = st.session_state.benchmark_df[
                                    st.session_state.benchmark_df['Player'] ==
                                    sim_player]

                                if st.session_state.get(
                                        'search_in_benchmark', False):
                                    # Mostrar jogadores do dataset principal e do benchmark
                                    # Já estão todos em df_sim, apenas filtrar por nome
                                    raw_data = df_sim[df_sim['Player'].isin(
                                        players_to_show)]
                                    st.info(
                                        f"Showing benchmark player '{sim_player}' and similar players from both datasets"
                                    )

                                    # Usar coluna para highlight
                                    raw_data['Data Source'] = raw_data[
                                        'Player'].apply(
                                            lambda p: "Benchmark"
                                            if p != sim_player and p not in
                                            df_minutes['Player'].values else
                                            "Main Dataset")
                                else:
                                    # Obter apenas jogadores do dataset principal (excluindo o jogador benchmark)
                                    main_players = [
                                        p for p in players_to_show
                                        if p != sim_player
                                        and p in df_minutes['Player'].values
                                    ]
                                    main_data = df_minutes[df_minutes['Player']
                                                           .isin(main_players)]

                                    # Combine reference player with main dataset players (if we have benchmark player data)
                                    if not benchmark_player_data.empty:
                                        raw_data = pd.concat(
                                            [benchmark_player_data, main_data],
                                            ignore_index=True)
                                        st.info(
                                            f"Showing benchmark player '{sim_player}' and similar players from main dataset"
                                        )
                                    else:
                                        raw_data = main_data
                                        st.warning(
                                            f"Benchmark player '{sim_player}' data not found, showing only similar players"
                                        )
                            else:
                                # Regular case - use combined dataset
                                raw_data = df_sim[df_sim['Player'].isin(
                                    players_to_show)]

                            # Show selected metrics only for better readability
                            columns_to_show = [
                                'Player', 'Team', 'Age', 'Position',
                                'Minutes played'
                            ] + sim_metric_options

                            # Make sure we only show columns that exist
                            available_columns = [
                                col for col in columns_to_show
                                if col in raw_data.columns
                            ]
                            st.dataframe(raw_data[available_columns],
                                         use_container_width=True)
                    else:
                        st.warning(
                            f"No similar players found for {sim_player}. Try adjusting filters or selecting different metrics."
                        )
                else:
                    if len(sim_metric_options) < 2:
                        st.warning(
                            "Please select at least 2 metrics for similarity calculation"
                        )
                    else:
                        st.warning(
                            "Not enough players matching the filter criteria")

            # =============================================
            # Correlation Matrix (Aba 5)
            # =============================================
            with tabs[4]:
                st.header('Metric Correlation Analysis')

                # Select metrics for correlation
                corr_metrics = st.multiselect(
                    'Select metrics to analyze correlations (2-10)',
                    metric_cols,
                    default=metric_cols[:5],
                    key='corr_metrics')

                if len(corr_metrics) < 2:
                    st.warning("Please select at least 2 metrics")
                elif len(corr_metrics) > 10:
                    st.warning(
                        "Too many metrics selected. Please limit to 10 or fewer"
                    )
                else:
                    # Calculate correlation matrix
                    corr_matrix = df_group[corr_metrics].corr()

                    # Create correlation heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(corr_matrix,
                                   cmap='coolwarm',
                                   vmin=-1,
                                   vmax=1)

                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Correlation Coefficient')

                    # Add labels
                    ax.set_xticks(np.arange(len(corr_metrics)))
                    ax.set_yticks(np.arange(len(corr_metrics)))
                    ax.set_xticklabels(corr_metrics, rotation=45, ha="right")
                    ax.set_yticklabels(corr_metrics)

                    # Loop over data dimensions and create text annotations
                    for i in range(len(corr_metrics)):
                        for j in range(len(corr_metrics)):
                            text = ax.text(
                                j,
                                i,
                                f"{corr_matrix.iloc[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color="white" if abs(
                                    corr_matrix.iloc[i, j]) > 0.5 else "black")

                    ax.set_title("Correlation Matrix")
                    fig.tight_layout()

                    # Display correlation heatmap
                    st.pyplot(fig)

                    # Export button
                    if st.button('Export Correlation Matrix (300 DPI)',
                                 key='export_corr'):
                        img_bytes = fig_to_bytes(fig)
                        st.download_button("⬇️ Download Correlation Matrix",
                                           data=img_bytes,
                                           file_name="correlation_matrix.png",
                                           mime="image/png")

                    # Show correlation table
                    st.subheader("Correlation Values")
                    st.dataframe(
                        corr_matrix.style.format("{:.2f}").background_gradient(
                            cmap='coolwarm', axis=None))

            # =============================================
            # Composite Index - PCA (Aba 6)
            # =============================================
            with tabs[5]:
                st.header('Composite Index Analysis (PCA)')

                # Select metrics for PCA
                pca_metrics = st.multiselect(
                    'Select metrics for composite index (3-10 recommended)',
                    metric_cols,
                    default=metric_cols[:5],
                    key='pca_metrics')

                if len(pca_metrics) < 3:
                    st.warning("Please select at least 3 metrics")
                elif len(pca_metrics) > 15:
                    st.warning(
                        "Too many metrics selected. Please limit to 15 or fewer"
                    )
                else:
                    # Standardize data
                    X = df_group[pca_metrics].values
                    X_scaled = StandardScaler().fit_transform(X)

                    # Perform PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(X_scaled)

                    # Criar dataframe com resultados do PCA
                    pca_df = pd.DataFrame(data=pca_result,
                                          columns=['PC1', 'PC2'])

                    # Adicionar informações dos jogadores
                    pca_df['Player'] = df_group['Player'].values
                    pca_df['Team'] = df_group['Team'].values
                    if 'Position' in df_group.columns:
                        pca_df['Position'] = df_group['Position'].values
                    if 'Age' in df_group.columns:
                        pca_df['Age'] = df_group['Age'].values

                    # Adicionar outros dados relevantes
                    for metric in pca_metrics:
                        if metric in df_group.columns:
                            pca_df[metric] = df_group[metric].values

                    # Create PCA plot
                    fig, ax = plt.subplots(figsize=(12, 8))

                    # Definir cores consistentes
                    player1_color = "#1A78CF"  # Azul real para jogador 1
                    player2_color = "#E41A1C"  # Vermelho para jogador 2

                    # Plot all players com pontos menores para melhor visualização (maior nitidez)
                    sc = ax.scatter(pca_result[:, 0],
                                    pca_result[:, 1],
                                    alpha=0.7,
                                    s=30,
                                    c='gray')

                    # Adicionar nomes de todos os jogadores (com texto pequeno e discreto)
                    max_labels = min(
                        50, len(df_group)
                    )  # Limitar número de labels para legibilidade

                    # Adicionar nomes abaixo dos pontos para todos os jogadores
                    for i, player in enumerate(
                            df_group['Player'].values[:max_labels]):
                        # Adicionar textos abaixo dos pontos usando um estilo mais nítido
                        ax.annotate(
                            player,
                            (pca_result[i, 0], pca_result[i, 1]),
                            xytext=(0, -7),  # Posição abaixo do ponto
                            textcoords='offset points',
                            fontsize=7,  # Fonte pequena
                            ha='center',  # Centralizado
                            va='top',  # Alinhado ao topo do texto
                            alpha=0.8,  # Maior nitidez
                            color='#333333'
                        )  # Cor mais escura para melhor visibilidade

                    # Highlight selected players
                    p1_idx = df_group[df_group['Player'] == p1].index
                    p2_idx = df_group[df_group['Player'] == p2].index

                    if len(p1_idx) > 0 and p1 != 'None':
                        p1_idx = p1_idx[0]
                        p1_idx_in_group = df_group.index.get_indexer([p1_idx
                                                                      ])[0]
                        if p1_idx_in_group >= 0:  # Player is in the filtered group
                            ax.scatter(pca_result[p1_idx_in_group, 0],
                                       pca_result[p1_idx_in_group, 1],
                                       s=100,
                                       c=player1_color,
                                       edgecolor='black',
                                       label=p1,
                                       zorder=10)
                            ax.annotate(
                                p1,
                                (pca_result[p1_idx_in_group, 0],
                                 pca_result[p1_idx_in_group, 1]),
                                xytext=(0, -9),  # Posição abaixo do ponto
                                textcoords='offset points',
                                fontsize=9,  # Fonte um pouco maior
                                fontweight='bold',
                                ha='center',  # Centralizado
                                va='top',  # Alinhado ao topo do texto
                                color=player1_color,  # Cor do jogador
                                bbox=dict(facecolor='white',
                                          alpha=0.7,
                                          edgecolor=player1_color,
                                          boxstyle="round,pad=0.1",
                                          linewidth=0.5),
                                zorder=11)

                    if len(p2_idx) > 0 and p2 != 'None':
                        p2_idx = p2_idx[0]
                        p2_idx_in_group = df_group.index.get_indexer([p2_idx
                                                                      ])[0]
                        if p2_idx_in_group >= 0:  # Player is in the filtered group
                            ax.scatter(pca_result[p2_idx_in_group, 0],
                                       pca_result[p2_idx_in_group, 1],
                                       s=100,
                                       c=player2_color,
                                       edgecolor='black',
                                       label=p2,
                                       zorder=10)
                            ax.annotate(
                                p2,
                                (pca_result[p2_idx_in_group, 0],
                                 pca_result[p2_idx_in_group, 1]),
                                xytext=(0, -9),  # Posição abaixo do ponto
                                textcoords='offset points',
                                fontsize=9,  # Fonte um pouco maior
                                fontweight='bold',
                                ha='center',  # Centralizado
                                va='top',  # Alinhado ao topo do texto
                                color=player2_color,  # Cor do jogador
                                bbox=dict(facecolor='white',
                                          alpha=0.7,
                                          edgecolor=player2_color,
                                          boxstyle="round,pad=0.1",
                                          linewidth=0.5),
                                zorder=11)

                    # Plot feature vectors
                    coeff = pca.components_.T
                    feat_xs = coeff[:, 0]
                    feat_ys = coeff[:, 1]

                    # Scale feature vectors for better visibility
                    scale_factor = 5

                    for i, (x, y) in enumerate(zip(feat_xs, feat_ys)):
                        plt.arrow(0,
                                  0,
                                  x * scale_factor,
                                  y * scale_factor,
                                  head_width=0.15,
                                  head_length=0.2,
                                  fc='red',
                                  ec='red')
                        plt.text(x * scale_factor * 1.1,
                                 y * scale_factor * 1.1,
                                 pca_metrics[i],
                                 color='red')

                    # Add explanations
                    explained_var = pca.explained_variance_ratio_
                    ax.set_xlabel(f"PC1 ({explained_var[0]:.2%} variance)",
                                  fontsize=12)
                    ax.set_ylabel(f"PC2 ({explained_var[1]:.2%} variance)",
                                  fontsize=12)

                    ax.set_title("Principal Component Analysis", fontsize=14)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

                    # Add legend
                    if p1 != 'None' or p2 != 'None':
                        ax.legend(loc='best')

                    # Display PCA plot
                    st.pyplot(fig)

                    # Show explained variance
                    st.info(
                        f"Total explained variance: {sum(explained_var):.2%}")

                    # Export button
                    if st.button('Export PCA Analysis (300 DPI)',
                                 key='export_pca'):
                        img_bytes = fig_to_bytes(fig)
                        st.download_button("⬇️ Download PCA Analysis",
                                           data=img_bytes,
                                           file_name="pca_analysis.png",
                                           mime="image/png")

                    # Show loadings
                    st.subheader("PCA Loadings (Feature Importance)")
                    loadings = pd.DataFrame(
                        pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(2)],
                        index=pca_metrics)
                    st.dataframe(loadings.style.format("{:.4f}"))

                    # Adicionar botão para exportar resultados do PCA como CSV
                    st.subheader("Export PCA Results with Player Data")

                    # Converter o dataframe para CSV
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                    @st.cache_data
                    def save_metrics_group_to_csv(group_name, metrics_list):
                        """
                        Salva um grupo de métricas como CSV para download
                        
                        Args:
                            group_name: Nome do grupo de métricas
                            metrics_list: Lista de métricas no grupo
                            
                        Returns:
                            CSV string
                        """
                        # Criar DataFrame com o nome do grupo e as métricas
                        metrics_df = pd.DataFrame({
                            'group_name': [group_name] * len(metrics_list),
                            'metric_name': metrics_list
                        })
                        
                        return metrics_df.to_csv(index=False).encode('utf-8')
                    
                    def load_metrics_from_csv(uploaded_file):
                        """
                        Carrega métricas de um arquivo CSV
                        
                        Args:
                            uploaded_file: Arquivo CSV carregado via st.file_uploader
                            
                        Returns:
                            tuple: (nome do grupo, lista de métricas)
                        """
                        try:
                            metrics_df = pd.read_csv(uploaded_file)
                            if 'group_name' in metrics_df.columns and 'metric_name' in metrics_df.columns:
                                group_name = metrics_df['group_name'].iloc[0]
                                metrics_list = metrics_df['metric_name'].tolist()
                                return group_name, metrics_list
                            else:
                                return None, []
                        except Exception as e:
                            st.error(f"Error loading metrics file: {e}")
                            return None, []

                    csv = convert_df_to_csv(pca_df)

                    # Botão para download do CSV
                    st.download_button("⬇️ Download PCA Results as CSV",
                                       csv,
                                       "pca_player_results.csv",
                                       "text/csv",
                                       key='download-pca-csv')

                    # Mostrar prévia dos dados
                    with st.expander("Preview PCA Results DataFrame"):
                        st.dataframe(pca_df)

        # =============================================
        # Profiler: Filtro por Percentis (Nova aba)
        # =============================================
        with tabs[6]:
            st.header('Player Profiler')
            st.subheader('Filter players by metric percentiles')

            # Interface de duas colunas para os filtros
            col1, col2 = st.columns(2)

            with col1:
                # Nota explicativa sobre uso dos filtros
                st.info(
                    "Note: Use os filtros na barra lateral para ajustar minutos, idade e posições."
                )

                # Verificar se temos um benchmark carregado
                benchmark_available = st.session_state.benchmark_loaded and st.session_state.benchmark_df is not None

                # Opção para usar benchmark como fonte de dados para profiling
                if benchmark_available:
                    use_benchmark_profile = st.checkbox(
                        "Use benchmark database for profiling",
                        value=False,
                        help=
                        "When enabled, you'll profile players from the benchmark database instead",
                        key="profiler_use_benchmark")

                    if use_benchmark_profile:
                        # Filtrar o benchmark com os mesmos filtros aplicados à base principal
                        filtered_benchmark = apply_benchmark_filter(
                            st.session_state.benchmark_df, minutes_range,
                            mpg_range, age_range, sel_pos)

                        if filtered_benchmark is not None and not filtered_benchmark.empty:
                            # Usar o benchmark filtrado para profiling
                            profiler_df = filtered_benchmark
                            st.success(
                                f"Profiling using benchmark: {st.session_state.benchmark_name} ({len(profiler_df)} players)"
                            )
                        else:
                            st.warning(
                                "No benchmark players match the applied filters. Using your database instead."
                            )
                            profiler_df = df_minutes
                            use_benchmark_profile = False
                    else:
                        profiler_df = df_minutes
                else:
                    profiler_df = df_minutes
                    use_benchmark_profile = False

            with col2:
                # Carregar grupos personalizados salvos na sessão
                if 'custom_metric_groups' in st.session_state:
                    custom_groups = st.session_state.custom_metric_groups
                else:
                    custom_groups = {}
                    
                # Definir presets de métricas para o profiler
                profiler_metric_groups = {
                    "Overall": [
                        'Progressive runs per 90', 'Progressive passes per 90',
                        'Deep completions per 90', 'Key passes per 90',
                        'Accurate passes, %', 'Accurate crosses, %',
                        'xA per 90', 'Defensive duels won, %',
                        'Aerial duels won, %', 'PAdj Interceptions',
                        'PAdj Sliding tackles', 'npxG per 90', 'npxG per Shot',
                        'Goal conversion, %', 'Shots on target, %'
                    ],
                    "Offensive": [
                        'Goals per 90', 'Shots per 90', 'xG per 90', 'npxG',
                        'G-xG', 'npxG per Shot', 'Shots on target per 90',
                        'Box Efficiency'
                    ],
                    "Passing": [
                        'Passes per 90', 'Accurate passes, %',
                        'Forward passes per 90', 'Progressive passes per 90',
                        'Key passes per 90'
                    ],
                    "Defensive": [
                        'Interceptions per 90', 'Tackles per 90',
                        'Defensive duels per 90', 'Aerial duels won per 90'
                    ],
                    "Physical": [
                        'Accelerations per 90', 'Sprint distance per 90',
                        'Distance covered per 90'
                    ],
                    "Advanced Metrics":
                    ['npxG', 'G-xG', 'npxG per Shot', 'Box Efficiency'],
                    "Pedri Metrics": [
                        'Smart passes per 90', 'Key passes per 90', 'Shot assists per 90', 'xA per 90',
                        'Passes per 90', 'Progressive passes per 90', 'Passes to final third per 90', 'Through passes per 90',
                        'Progressive runs per 90', 'Dribbles per 90', 'Successful dribbles, %',
                        'Interceptions per 90', 'Defensive duels per 90', 'Defensive duels won, %',
                        'Received passes per 90'
                    ]
                }
                
                # Adicionar grupos personalizados salvos na session state
                if 'custom_metric_groups' in st.session_state:
                    # Adicionar cada grupo personalizado ao dicionário de grupos de métricas
                    for group_name, metrics in custom_groups.items():
                        # Garantir que não sobrescreva grupos existentes acidentalmente
                        if group_name not in profiler_metric_groups:
                            profiler_metric_groups[group_name] = metrics

                # Opção para selecionar por grupo ou personalizar
                metric_selection = st.radio(
                    "Metric Selection Mode",
                    ["Select Individual Metrics", "Use Metric Presets"],
                    horizontal=True,
                    key="profiler_metric_selection")

                if metric_selection == "Use Metric Presets":
                    selected_groups = st.multiselect(
                        "Select Metric Groups",
                        list(profiler_metric_groups.keys()),
                        default=["Offensive"],
                        key="profiler_preset_groups")

                    # Combinar todas as métricas dos grupos selecionados
                    preset_metrics = []
                    for group in selected_groups:
                        # Adicionar apenas métricas que existem no dataframe
                        available_metrics = [
                            m for m in profiler_metric_groups[group]
                            if m in metric_cols
                        ]
                        preset_metrics.extend(available_metrics)

                    # Permitir personalização adicional
                    profile_metrics = st.multiselect(
                        'Add or Remove Individual Metrics (1-15 recommended)',
                        metric_cols,
                        default=preset_metrics,  # Carrega todas as métricas do grupo sem limitação
                        key="profiler_preset_metrics")
                else:
                    # Seleção manual de métricas
                    profile_metrics = st.multiselect(
                        'Select metrics for profiling (1-15)',
                        metric_cols,
                        default=metric_cols[:min(3, len(metric_cols))],
                        key="profiler_metrics")

                # Para cada métrica selecionada, adicionar um slider para o percentil mínimo
                percentile_filters = {}
                for metric in profile_metrics:
                    min_percentile = st.slider(
                        f"Minimum percentile for {metric}",
                        0,
                        100,
                        65,  # Padrão em 65º percentil
                        key=f"percentile_{metric}")
                    percentile_filters[metric] = min_percentile

            # Aplicar filtros
            if not profile_metrics:
                st.warning("Please select at least one metric for profiling")
            else:
                try:
                    # Usar a fonte de dados selecionada (normal ou benchmark)
                    df_filtered = profiler_df.copy()

                    # Filtro por percentis para cada métrica
                    for metric, min_percentile in percentile_filters.items():
                        if min_percentile > 0:
                            # Calcular o valor correspondente ao percentil na distribuição da fonte de dados selecionada
                            percentile_value = np.percentile(
                                profiler_df[metric].dropna(), min_percentile)
                            df_filtered = df_filtered[df_filtered[metric] >=
                                                      percentile_value].copy()

                    # Mostrar resultados com indicação da fonte de dados
                    source_text = f" from {st.session_state.benchmark_name}" if use_benchmark_profile else ""
                    st.subheader(
                        f"Players matching criteria: {len(df_filtered)}{source_text}"
                    )

                    if len(df_filtered) > 0:
                        # Ordenar os jogadores por uma métrica de performance global (média dos percentis)
                        df_filtered['Overall Score'] = df_filtered[
                            profile_metrics].apply(lambda row: np.mean([
                                calc_percentile(profiler_df[m], row[m]) * 100
                                for m in profile_metrics
                            ]),
                                                   axis=1)

                        # Ordenar por "Overall Score" descendente
                        df_sorted = df_filtered.sort_values(
                            'Overall Score',
                            ascending=False).reset_index(drop=True)

                        # Adicionar colunas de percentil para cada métrica
                        for metric in profile_metrics:
                            df_sorted[f"{metric} (percentile)"] = df_sorted[
                                metric].apply(
                                    lambda x:
                                    f"{calc_percentile(profiler_df[metric], x)*100:.1f}%"
                                )

                        # Selecionar colunas para exibição
                        display_cols = [
                            'Player', 'Team', 'Age', 'Position',
                            'Minutes played', 'Overall Score'
                        ]
                        display_cols.extend(
                            [metric for metric in profile_metrics])
                        display_cols.extend([
                            f"{metric} (percentile)"
                            for metric in profile_metrics
                        ])

                        # Mostrar os resultados com formatação de percentil
                        st.dataframe(df_sorted[display_cols].style.format({
                            'Overall Score':
                            '{:.1f}',
                            'Age':
                            '{:.1f}',
                            'Minutes played':
                            '{:.0f}'
                        }))

                        # Exportar para CSV
                        csv = convert_df_to_csv(df_sorted[display_cols])
                        st.download_button(
                            "⬇️ Download Profile Results as CSV",
                            csv,
                            "player_profile_results.csv",
                            "text/csv",
                            key='download-profile-csv')

                        # Visualização dos jogadores filtrados
                        if len(profile_metrics) >= 2 and st.checkbox(
                                "Show scatter plot of filtered players",
                                value=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                x_metric = st.selectbox('X-Axis Metric',
                                                        profile_metrics,
                                                        index=0,
                                                        key="profile_x_metric")
                            with col2:
                                y_metric = st.selectbox(
                                    'Y-Axis Metric',
                                    profile_metrics,
                                    index=min(1,
                                              len(profile_metrics) - 1),
                                    key="profile_y_metric")

                            # Criar scatter plot
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Plotar pontos
                            sc = ax.scatter(df_sorted[x_metric],
                                            df_sorted[y_metric],
                                            alpha=0.7,
                                            s=60,
                                            c=player1_color,
                                            edgecolor='white')

                            # Adicionar rótulos para todos os jogadores
                            for i, row in df_sorted.iterrows():
                                # Adicionar textos abaixo dos pontos
                                ax.annotate(row['Player'],
                                            (row[x_metric], row[y_metric]),
                                            xytext=(0, -10),
                                            textcoords='offset points',
                                            fontsize=8,
                                            ha='center',
                                            va='top',
                                            alpha=0.9,
                                            color=player1_color,
                                            bbox=dict(facecolor='white',
                                                      alpha=0.7,
                                                      edgecolor=player1_color,
                                                      boxstyle="round,pad=0.1",
                                                      linewidth=0.5),
                                            zorder=10)

                            # Adicionar linhas médias usando a fonte de dados selecionada
                            ax.axvline(profiler_df[x_metric].mean(),
                                       color='gray',
                                       linestyle='--',
                                       alpha=0.5)
                            ax.axhline(profiler_df[y_metric].mean(),
                                       color='gray',
                                       linestyle='--',
                                       alpha=0.5)

                            # Configurar rótulos e título
                            ax.set_xlabel(x_metric, fontsize=12)
                            ax.set_ylabel(y_metric, fontsize=12)

                            # Adicionar informação sobre a fonte de dados no título
                            source_name = f" ({st.session_state.benchmark_name})" if use_benchmark_profile else ""
                            ax.set_title(
                                f"Profile Results: {x_metric} vs {y_metric}{source_name}",
                                fontsize=14)
                            ax.grid(True, alpha=0.3)

                            # Exibir o gráfico
                            st.pyplot(fig)

                            # Exportar gráfico
                            if st.button('Export Scatter Plot (300 DPI)',
                                         key='export_profile_scatter'):
                                img_bytes = fig_to_bytes(fig)
                                st.download_button(
                                    "⬇️ Download Scatter Plot",
                                    data=img_bytes,
                                    file_name=
                                    f"profile_scatter_{x_metric}_vs_{y_metric}.png",
                                    mime="image/png")
                    else:
                        st.warning(
                            "No players match the selected criteria. Try relaxing some filters."
                        )

                except Exception as e:
                    st.error(f"Error in profiler: {str(e)}")
                    st.exception(e)
        
        # =============================================
        # Custom Metrics Groups (Aba 8)
        # =============================================
        with tabs[7]:
            st.header('Custom Metrics Groups')
            st.subheader('Create, save and load your own metric groups')
            
            # Explicação sobre a funcionalidade
            st.info("""
            This tab allows you to create your own custom metric groups that can be saved as CSV files and loaded later.
            Custom metric groups can be used in Pizza Charts, Bar Charts, and all other visualizations just like the pre-defined groups.
            """)
            
            # Divisão em duas colunas
            col1, col2 = st.columns(2)
            
            # Coluna 1: Criar e salvar grupos personalizados
            with col1:
                st.subheader("Create Custom Metric Group")
                
                # Nome do grupo personalizado
                custom_group_name = st.text_input(
                    "Enter a name for your custom metric group",
                    value="My Custom Group",
                    key="custom_group_name"
                )
                
                # Selecionar métricas para o grupo
                custom_metrics = st.multiselect(
                    "Select metrics for your custom group",
                    metric_cols,
                    key="custom_metrics_selection"
                )
                
                # Botão para salvar o grupo como CSV
                if st.button("Save Metric Group as CSV", key="save_metric_group"):
                    if not custom_metrics:
                        st.error("Please select at least one metric for your custom group.")
                    else:
                        # Salvar grupo como CSV para download
                        # Criar DataFrame com o nome do grupo e as métricas
                        metrics_df = pd.DataFrame({
                            'group_name': [custom_group_name] * len(custom_metrics),
                            'metric_name': custom_metrics
                        })
                        
                        # Converter para CSV
                        csv_data = metrics_df.to_csv(index=False).encode('utf-8')
                        file_name = f"{custom_group_name.replace(' ', '_').lower()}_metrics.csv"
                        
                        st.download_button(
                            "⬇️ Download Metrics Group",
                            csv_data,
                            file_name,
                            "text/csv",
                            key="download_metrics_group"
                        )
                        
                        st.success(f"Metric group '{custom_group_name}' with {len(custom_metrics)} metrics is ready for download.")
            
            # Coluna 2: Carregar grupos personalizados
            with col2:
                st.subheader("Load Custom Metric Group")
                
                # Upload de arquivo CSV
                uploaded_file = st.file_uploader(
                    "Upload a metrics group CSV file",
                    type=["csv"],
                    key="metrics_group_uploader"
                )
                
                if uploaded_file is not None:
                    # Processar arquivo carregado
                    try:
                        metrics_df = pd.read_csv(uploaded_file)
                        if 'group_name' in metrics_df.columns and 'metric_name' in metrics_df.columns:
                            group_name = metrics_df['group_name'].iloc[0]
                            metrics_list = metrics_df['metric_name'].tolist()
                            
                            # Mostrar informações do grupo carregado
                            st.success(f"Successfully loaded '{group_name}' with {len(metrics_list)} metrics.")
                            
                            # Verificar quais métricas estão disponíveis no dataset atual
                            available_metrics = [m for m in metrics_list if m in metric_cols]
                            unavailable_metrics = [m for m in metrics_list if m not in metric_cols]
                            
                            if len(unavailable_metrics) > 0:
                                st.warning(f"{len(unavailable_metrics)} metrics from this group are not available in the current dataset and will be ignored.")
                            
                            # Mostrar as métricas do grupo
                            st.subheader(f"Metrics in '{group_name}'")
                            
                            # Criar duas colunas para exibir métricas disponíveis e indisponíveis
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.markdown("**Available Metrics:**")
                                for m in available_metrics:
                                    st.markdown(f"- {m}")
                            
                            if unavailable_metrics:
                                with metrics_col2:
                                    st.markdown("**Unavailable Metrics:**")
                                    for m in unavailable_metrics:
                                        st.markdown(f"- {m}")
                            
                            # Botão para aplicar o grupo às métricas selecionadas
                            if st.button("Apply This Metrics Group", key="apply_loaded_group"):
                                if available_metrics:
                                    # Adicionar o grupo personalizado carregado a todos os dicionários de grupos de métricas
                                    # 1. Pizza Metric Groups
                                    pizza_metric_groups[group_name] = available_metrics
                                    
                                    # 2. Buscar outros grupos de métricas no aplicativo e atualizar
                                    # Profiler metric groups (aba Profiler)
                                    if 'profiler_metric_groups' in locals():
                                        profiler_metric_groups[group_name] = available_metrics
                                    
                                    # Salvamos o grupo na session_state para persistência
                                    if 'custom_metric_groups' not in st.session_state:
                                        st.session_state.custom_metric_groups = {}
                                    
                                    # Armazenar o grupo na session_state para garantir persistência
                                    st.session_state.custom_metric_groups[group_name] = available_metrics
                                    
                                    # Atualizar as variáveis para o modo de seleção de métricas
                                    st.session_state.temp_pizza_preset_metrics = available_metrics
                                    st.session_state.saved_pizza_metrics = available_metrics
                                    
                                    # Se houver variáveis relacionadas a outras abas, atualizamos também
                                    if 'saved_bar_metrics' in st.session_state:
                                        st.session_state.saved_bar_metrics = available_metrics
                                        
                                    if 'saved_scatter_metrics' in st.session_state:
                                        st.session_state.saved_scatter_metrics = available_metrics
                                    
                                    # Mostrar mensagem de sucesso
                                    st.success(f"Group '{group_name}' has been added to available metric groups and can now be used in visualizations!")
                                    
                                    # Instruções para usar o grupo
                                    st.info("Go to the Pizza Chart or Bars tab, select 'Use Metric Presets' and you will see your custom group in the dropdown list.")
                                else:
                                    st.error("No available metrics in this group for the current dataset.")
                        else:
                            st.error("Invalid metrics group file. Please upload a valid CSV file with 'group_name' and 'metric_name' columns.")
                    except Exception as e:
                        st.error(f"Error loading metrics file: {e}")
                
            # Preview dos grupos existentes
            st.subheader("Preview Existing Metric Groups")
            
            # Exibir grupos existentes em formato de tabela para referência
            groups_data = []
            for group_name, metrics in pizza_metric_groups.items():
                groups_data.append({
                    "Group Name": group_name,
                    "Number of Metrics": len(metrics),
                    "Example Metrics": ", ".join(metrics[:3]) + ("..." if len(metrics) > 3 else "")
                })
                
            groups_df = pd.DataFrame(groups_data)
            st.dataframe(groups_df)

else:
    st.info("👈 Please upload Wyscout Excel files to begin analysis")

    # Show example data format
    with st.expander("📋 Expected Data Format"):
        st.markdown("""
        Your Excel files should contain the following columns:
        - **Player**: Player name
        - **Team**: Team name
        - **Age**: Player age
        - **Position**: Player position(s)
        - **Minutes played**: Total minutes played
        - **Matches played**: Number of matches

        Plus various performance metrics like:
        - Goals
        - Assists
        - Passes
        - Tackles
        - etc.

        Files should be in Excel (.xlsx) format exported from Wyscout or similar platforms.
        """)
