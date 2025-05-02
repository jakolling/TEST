# Football Analytics App - Versão Completa Corrigida
# Código integral sem abreviações e com todas as correções

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
left_column, center_column, right_column = st.columns([1, 3, 1])
with center_column:
    st.image('vif_logo.png', width=400)

st.title('Technical Scouting Department')
st.subheader('Football Analytics Dashboard')
st.caption("Created by João Alberto Kolling | Player Analysis System v3.0")

# Guia do Usuário
with st.expander("📘 User Guide & Instructions", expanded=False):
    st.markdown("""
    **⚠️ Requirements:**  
    1. Install dependencies:  
    `pip install kaleido==0.2.1.post1 xlsxwriter`  
    2. Data must contain columns: Player, Age, Position, Metrics, Team  
    
    **Main Features:**  
    - Player comparison with radar and bar charts  
    - Metric correlation analysis  
    - Advanced filtering system  
    - Professional chart exports (300 DPI)  
    """)

# =============================================
# Funções Principais
# =============================================
if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}

def load_and_clean_data(uploaded_files):
    dataframes_list = []
    for file in uploaded_files[:15]:
        dataframe = pd.read_excel(file)
        dataframe.dropna(how="all", inplace=True)
        dataframe = dataframe.loc[:, dataframe.columns.notnull()]
        dataframe.columns = [str(column).strip() for column in dataframe.columns]
        
        metadata = st.session_state.file_metadata[file.name]
        dataframe['Data Origin'] = metadata['label']
        dataframe['Season'] = metadata['season']
        
        dataframes_list.append(dataframe)
    return pd.concat(dataframes_list, ignore_index=True)

@st.cache_data
def calculate_percentile(data_series, value):
    return (data_series <= value).sum() / len(data_series)

def get_context_information(dataframe, minutes_range, minutes_per_game_range, age_range, selected_positions):
    return {
        'leagues': ', '.join(dataframe['Data Origin'].unique()),
        'seasons': ', '.join(dataframe['Season'].unique()),
        'total_players': len(dataframe),
        'min_minutes': minutes_range[0],
        'max_minutes': minutes_range[1],
        'min_minutes_per_game': minutes_per_game_range[0],
        'max_minutes_per_game': minutes_per_game_range[1],
        'min_age': age_range[0],
        'max_age': age_range[1],
        'positions': ', '.join(selected_positions) if selected_positions else 'All'
    }

# =============================================
# Filtros da Barra Lateral
# =============================================
st.sidebar.header('Filters')
with st.sidebar.expander("⚙️ Advanced Filters", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload up to 15 Wyscout Excel files", 
        type=["xlsx"], 
        accept_multiple_files=True
    )

if uploaded_files:
    # Coleta de Metadados
    new_files = [file for file in uploaded_files if file.name not in st.session_state.file_metadata]
    
    for file in new_files:
        with st.form(key=f'metadata_{file.name}'):
            st.subheader(f"Metadata for: {file.name}")
            label = st.text_input("Data origin label (e.g., Bundesliga 2)", key=f"label_{file.name}")
            
            seasons = [f"{year}/{year+1}" for year in range(2020, 2026)] + [str(year) for year in range(2020, 2026)]
            season = st.selectbox("Season", seasons, key=f"season_{file.name}")
            
            if st.form_submit_button("Confirm"):
                st.session_state.file_metadata[file.name] = {'label': label, 'season': season}
                st.rerun()

    if missing_metadata := [file.name for file in uploaded_files if file.name not in st.session_state.file_metadata]:
        st.warning("Please provide metadata for all uploaded files")
        st.stop()

    try:
        full_dataframe = load_and_clean_data(uploaded_files)

        # Filtros Principais
        min_minutes, max_minutes = int(full_dataframe['Minutes played'].min()), int(full_dataframe['Minutes played'].max())
        minutes_range = st.sidebar.slider('Minutes Played', min_minutes, max_minutes, (min_minutes, max_minutes))
        filtered_minutes_dataframe = full_dataframe[full_dataframe['Minutes played'].between(*minutes_range)].copy()

        filtered_minutes_dataframe['Minutes per game'] = (
            filtered_minutes_dataframe['Minutes played'] / 
            filtered_minutes_dataframe['Matches played'].replace(0, np.nan)
        )
        filtered_minutes_dataframe['Minutes per game'] = filtered_minutes_dataframe['Minutes per game'].fillna(0).clip(0, 120)
        
        min_mpg, max_mpg = int(filtered_minutes_dataframe['Minutes per game'].min()), int(filtered_minutes_dataframe['Minutes per game'].max())
        minutes_per_game_range = st.sidebar.slider('Minutes per Game', min_mpg, max_mpg, (min_mpg, max_mpg))
        filtered_minutes_dataframe = filtered_minutes_dataframe[filtered_minutes_dataframe['Minutes per game'].between(*minutes_per_game_range)]

        min_age, max_age = int(filtered_minutes_dataframe['Age'].min()), int(filtered_minutes_dataframe['Age'].max())
        age_range = st.sidebar.slider('Age Range', min_age, max_age, (min_age, max_age))

        if 'Position' in filtered_minutes_dataframe.columns:
            filtered_minutes_dataframe['Position_split'] = (
                filtered_minutes_dataframe['Position']
                .astype(str)
                .apply(lambda positions: [position.strip() for position in positions.split(',')])
            )
            all_positions = sorted({position for position_list in filtered_minutes_dataframe['Position_split'] for position in position_list})
            selected_positions = st.sidebar.multiselect('Positions', all_positions, default=all_positions)
        else:
            selected_positions = []

        context = get_context_information(filtered_minutes_dataframe, minutes_range, minutes_per_game_range, age_range, selected_positions)
        players = sorted(filtered_minutes_dataframe['Player'].unique())
        selected_player_1 = st.sidebar.selectbox('Select Player 1', players)
        selected_player_2 = st.sidebar.selectbox('Select Player 2', [player for player in players if player != selected_player_1])

        numeric_columns = filtered_minutes_dataframe.select_dtypes(include=[np.number]).columns.tolist()
        analysis_tabs = st.tabs(['Radar', 'Bars', 'Scatter', 'Profiler', 'Correlation', 'Composite Index (PCA)'])

        # =============================================
        # Radar Chart Corrigido (Aba 1)
        # =============================================
        with analysis_tabs[0]:
            st.header('Radar Chart')
            selected_metrics = st.multiselect('Metrics for Radar (6–12)', numeric_columns, default=numeric_columns[:6])
            
            if 6 <= len(selected_metrics) <= 12:
                player_1_data = filtered_minutes_dataframe[filtered_minutes_dataframe['Player'] == selected_player_1].iloc[0]
                player_2_data = filtered_minutes_dataframe[filtered_minutes_dataframe['Player'] == selected_player_2].iloc[0]
                
                player_1_percentiles = {
                    metric: calculate_percentile(filtered_minutes_dataframe[metric], player_1_data[metric]) 
                    for metric in selected_metrics
                }
                player_2_percentiles = {
                    metric: calculate_percentile(filtered_minutes_dataframe[metric], player_2_data[metric]) 
                    for metric in selected_metrics
                }

                if selected_positions:
                    position_filtered_group = filtered_minutes_dataframe[
                        filtered_minutes_dataframe['Position_split'].apply(
                            lambda positions: any(position in positions for position in selected_positions)
                        )
                    ]
                else:
                    position_filtered_group = filtered_minutes_dataframe

                group_means = {metric: position_filtered_group[metric].mean() for metric in selected_metrics}
                group_mean_percentiles = {
                    metric: calculate_percentile(filtered_minutes_dataframe[metric], group_means[metric]) 
                    for metric in selected_metrics
                }

                comparison_dataframe = pd.DataFrame({
                    'Metric': selected_metrics,
                    selected_player_1: [player_1_data[metric] for metric in selected_metrics],
                    selected_player_2: [player_2_data[metric] for metric in selected_metrics],
                    'Group Average': [group_means[metric] for metric in selected_metrics]
                }).set_index('Metric').round(2)

                show_group_average = st.checkbox('Show Group Average', True)
                radar_figure = go.Figure()
                
                radar_figure.add_trace(go.Scatterpolar(
                    r=[player_1_percentiles[metric] * 100 for metric in selected_metrics],
                    theta=selected_metrics,
                    fill='toself',
                    name=selected_player_1
                ))
                
                radar_figure.add_trace(go.Scatterpolar(
                    r=[player_2_percentiles[metric] * 100 for metric in selected_metrics],
                    theta=selected_metrics,
                    fill='toself',
                    name=selected_player_2
                ))
                
                if show_group_average:
                    radar_figure.add_trace(go.Scatterpolar(
                        r=[group_mean_percentiles[metric] * 100 for metric in selected_metrics],
                        theta=selected_metrics,
                        fill='toself',
                        name='Group Average'
                    ))
                
                chart_title = (
                    f"<b>{selected_player_1} vs {selected_player_2}</b><br>"
                    f"<sup>Leagues: {context['leagues']} | Seasons: {context['seasons']}<br>"
                    f"Filters: {context['min_minutes']}-{context['max_minutes']} mins | "
                    f"{context['min_minutes_per_game']}-{context['max_minutes_per_game']} min/game | "
                    f"Age {context['min_age']}-{context['max_age']} | Positions: {context['positions']}</sup>"
                )
                
                radar_figure.update_layout(
                    title=dict(text=chart_title, x=0.03, xanchor='left', font=dict(size=18)),
                    polar=dict(radialaxis=dict(range=[0, 100])),
                    template='plotly_white',
                    margin=dict(t=200, b=100, l=100, r=100),
                    height=700
                )
                st.plotly_chart(radar_figure)
                
                st.subheader('Nominal Values Comparison')
                st.dataframe(
                    comparison_dataframe.style
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]
                    }])
                    .format(precision=2)
                )
                
                if st.button('Export Radar Chart (300 DPI)', key='export_radar'):
                    radar_figure.update_layout(margin=dict(t=250))
                    image_bytes = radar_figure.to_image(format="png", width=1600, height=900, scale=3)
                    st.download_button(
                        "⬇️ Download Radar Chart", 
                        data=image_bytes, 
                        file_name=f"radar_{selected_player_1}_vs_{selected_player_2}.png", 
                        mime="image/png"
                    )

        # =============================================
        # Gráficos de Barras (Aba 2) - Correção Aplicada
        # =============================================
        with analysis_tabs[1]:
            st.header('Bar Chart Comparison')
            selected_metrics = st.multiselect('Select metrics (max 5)', numeric_columns, default=numeric_columns[:1])
            
            if len(selected_metrics) > 5:
                st.error("Maximum 5 metrics allowed!")
                st.stop()
            
            if len(selected_metrics) >= 1:
                figure = make_subplots(
                    rows=len(selected_metrics),
                    cols=1,
                    subplot_titles=selected_metrics,
                    vertical_spacing=0.15
                )
                
                for index, metric in enumerate(selected_metrics, 1):
                    player_1_value = filtered_minutes_dataframe[filtered_minutes_dataframe['Player'] == selected_player_1][metric].iloc[0]
                    player_2_value = filtered_minutes_dataframe[filtered_minutes_dataframe['Player'] == selected_player_2][metric].iloc[0]
                    
                    if selected_positions:
                        position_filtered_group = filtered_minutes_dataframe[
                            filtered_minutes_dataframe['Position_split'].apply(
                                lambda positions: any(position in positions for position in selected_positions)
                            )
                        ]
                    else:
                        position_filtered_group = filtered_minutes_dataframe
                    
                    average_value = position_filtered_group[metric].mean()
                    
                    figure.add_trace(go.Bar(
                        y=[selected_player_1], 
                        x=[player_1_value], 
                        orientation='h',
                        name=selected_player_1, 
                        marker_color='#1f77b4', 
                        showlegend=(index == 1)
                    ), row=index, col=1)
                    
                    figure.add_trace(go.Bar(
                        y=[selected_player_2], 
                        x=[player_2_value], 
                        orientation='h',
                        name=selected_player_2, 
                        marker_color='#ff7f0e', 
                        showlegend=(index == 1)
                    ), row=index, col=1)
                    
                    figure.add_vline(
                        x=average_value, 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text="Group Avg", 
                        row=index, 
                        col=1
                    )
                
                chart_title = (
                    f"<b>Metric Comparison</b><br>"
                    f"<sup>Context: {context['leagues']} ({context['seasons']}) | "
                    f"Players: {context['total_players']} | Filters: {context['min_age']}-{context['max_age']} years</sup>"
                )
                
                figure.update_layout(
                    title=dict(text=chart_title, x=0.03, xanchor='left', font=dict(size=18)),
                    height=300*len(selected_metrics),
                    width=800,
                    template='plotly_white',
                    barmode='group',
                    margin=dict(t=200, b=100, l=100, r=100)
                )
                st.plotly_chart(figure)
                
                if st.button('Export Bar Charts (300 DPI)', key='export_bar'):
                    figure.update_layout(margin=dict(t=250))
                    image_bytes = figure.to_image(format="png", width=1600, height=300*len(selected_metrics)+300, scale=3)
                    st.download_button(
                        "⬇️ Download Charts", 
                        data=image_bytes, 
                        file_name="bar_charts.png", 
                        mime="image/png"
                    )

        # =============================================
        # Scatter Plot (Aba 3)
        # =============================================
        with analysis_tabs[2]:
            st.header('Scatter Plot')
            x_metric = st.selectbox('X metric', numeric_columns)
            y_metric = st.selectbox('Y metric', numeric_columns)
            highlighted_players = st.multiselect('Highlight up to 5 players', players, default=[selected_player_1, selected_player_2])[:5]
            
            filtered_data = filtered_minutes_dataframe[filtered_minutes_dataframe['Age'].between(*age_range)]
            if selected_positions:
                filtered_data = filtered_data[filtered_data['Position_split'].apply(lambda x: any(pos in x for pos in selected_positions)]
            
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=filtered_data[x_metric], 
                y=filtered_data[y_metric], 
                mode='markers', 
                marker=dict(color='cornflowerblue', opacity=0.5, size=8), 
                text=filtered_data['Player'], 
                hoverinfo='text', 
                name='All'
            ))
            
            colors = ['red','blue','green','orange','purple']
            for index, player in enumerate(highlighted_players):
                player_data = filtered_data[filtered_data['Player']==player]
                if not player_data.empty:
                    figure.add_trace(go.Scatter(
                        x=player_data[x_metric], 
                        y=player_data[y_metric], 
                        text=player_data['Player'], 
                        mode='markers+text', 
                        marker=dict(size=12, color=colors[index]), 
                        name=player
                    ))
            
            chart_title = (
                f"<b>{x_metric} vs {y_metric}</b><br>"
                f"<sup>Data Source: {context['leagues']} ({context['seasons']})<br>"
                f"Filters: {context['total_players']} players | "
                f"{context['min_minutes_per_game']}+ min/game | {context['positions']}</sup>"
            )
            
            figure.update_layout(
                title=dict(text=chart_title, x=0.03, xanchor='left', font=dict(size=18)),
                width=1000, 
                height=700,
                template='plotly_dark',
                margin=dict(t=200, b=100, l=100, r=100)
            )
            st.plotly_chart(figure)
            
            if st.button('Export Scatter Plot (300 DPI)', key='export_scatter'):
                figure.update_layout(margin=dict(t=250))
                image_bytes = figure.to_image(format="png", width=1800, height=1200, scale=3)
                st.download_button(
                    "⬇️ Download Scatter Plot", 
                    data=image_bytes, 
                    file_name=f"scatter_{x_metric}_vs_{y_metric}.png", 
                    mime="image/png"
                )

        # =============================================
        # Profiler (Aba 4)
        # =============================================
        with analysis_tabs[3]:
            st.header('Profiler')
            selected_metrics = st.multiselect('Select 4–12 metrics', numeric_columns)
            
            if 4 <= len(selected_metrics) <= 12:
                percentiles = {metric: filtered_minutes_dataframe[metric].rank(pct=True) for metric in selected_metrics}
                min_percentiles = {metric: st.slider(f'Min % for {metric}', 0,100,50) for metric in selected_metrics}
                filter_mask = np.logical_and.reduce([percentiles[metric]*100 >= min_percentiles[metric] for metric in selected_metrics])
                st.dataframe(filtered_minutes_dataframe.loc[filter_mask, ['Player', 'Team'] + selected_metrics].reset_index(drop=True))
            else:
                st.warning('Select between 4 and 12 metrics.')

        # =============================================
        # Correlation Matrix (Aba 5)
        # =============================================
        with analysis_tabs[4]:
            st.header('Correlation Matrix')
            selected_metrics = st.multiselect('Metrics to correlate', numeric_columns, default=numeric_columns)
            
            if len(selected_metrics) >= 2:
                correlation_matrix = filtered_minutes_dataframe[selected_metrics].corr()
                figure = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values, 
                    x=selected_metrics, 
                    y=selected_metrics, 
                    zmin=-1, 
                    zmax=1, 
                    colorscale='Viridis'
                ))
                
                chart_title = (
                    f"<b>Metric Relationships</b><br>"
                    f"<sup>Dataset: {context['leagues']} ({context['seasons']})<br>"
                    f"Players: {context['total_players']} | Min. Minutes: {context['min_minutes']}+</sup>"
                )
                
                figure.update_layout(
                    title=dict(text=chart_title, x=0.03, xanchor='left', font=dict(size=18)),
                    template='plotly_dark',
                    margin=dict(t=200, b=100, l=100, r=100)
                )
                st.plotly_chart(figure)
                
                if st.button('Export Correlation Matrix (300 DPI)', key='export_corr'):
                    figure.update_layout(margin=dict(t=250))
                    image_bytes = figure.to_image(format="png", width=1400, height=1400, scale=3)
                    st.download_button(
                        "⬇️ Download Correlation Matrix", 
                        data=image_bytes, 
                        file_name="correlation_matrix.png", 
                        mime="image/png"
                    )

        # =============================================
        # Composite PCA Index (Aba 6)
        # =============================================
        with analysis_tabs[5]:
            st.header('Composite PCA Index + Excel Export')
            performance_metrics = [col for col in numeric_columns if col not in ['Age','Height','Country','Minutes played','Position']]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                kernel_type = st.selectbox('Kernel Type',['linear','rbf'], index=1)
            with col2:
                gamma_value = st.number_input('Gamma', value=0.1, min_value=0.0, step=0.1, disabled=(kernel_type=='linear'))
            with col3:
                correlation_threshold = st.slider(
                    'Correlation Threshold', 
                    0.0, 1.0, 0.5, 0.05,
                    help='Minimum average correlation for feature inclusion',
                    disabled=st.session_state.get('manual_weights', False)
                )
            with col4:
                manual_weights = st.checkbox('Manual Weights', key='manual_weights')

            selected_metrics = st.multiselect('Select performance metrics', performance_metrics)
            
            if len(selected_metrics)<2:
                st.warning('Select at least two performance metrics.')
                st.stop()

            if manual_weights:
                st.subheader('Manual Weight Adjustment')
                weight_sliders = {}
                cols = st.columns(3)
                for idx, metric in enumerate(selected_metrics):
                    with cols[idx%3]:
                        weight_sliders[metric] = st.slider(f'Weight for {metric}', 0.0, 1.0, 0.5, key=f'weight_{metric}')
                weights = pd.Series(weight_sliders)
                excluded_metrics = []
            else:
                @st.cache_data
                def calculate_weights(dataframe, features, threshold):
                    cm = dataframe[features].corr().abs()
                    ac = cm.mean(axis=1)
                    return ac.where(ac>threshold, 0)
                weights = calculate_weights(filtered_minutes_dataframe, selected_metrics, correlation_threshold)
                excluded_metrics = weights[weights==0].index.tolist()

            if excluded_metrics and not manual_weights:
                st.warning(f'Excluded metrics (low correlation): {", ".join(excluded_metrics)}')
                selected_metrics = [metric for metric in selected_metrics if metric not in excluded_metrics]

            class WeightedKPCA:
                def __init__(self, kernel='rbf', gamma=None):
                    self.kernel = kernel
                    self.gamma = gamma
                    self.scaler = StandardScaler()
                
                def fit_transform(self, X, weights):
                    X_scaled = self.scaler.fit_transform(X)
                    X_weighted = X_scaled * weights.values
                    self.kpca = KernelPCA(
                        n_components=1,
                        kernel=self.kernel,
                        gamma=self.gamma,
                        random_state=42
                    )
                    return self.kpca.fit_transform(X_weighted).flatten()

            if len(selected_metrics)>=2:
                try:
                    kpca = WeightedKPCA(kernel=kernel_type, gamma=(None if kernel_type=='linear' else gamma_value))
                    df_sel = filtered_minutes_dataframe[selected_metrics].dropna()
                    scores = kpca.fit_transform(df_sel, weights)
                    idx = df_sel.index
                    
                    df_pca = pd.DataFrame({
                        'Player': filtered_minutes_dataframe.loc[idx, 'Player'],
                        'Team': filtered_minutes_dataframe.loc[idx, 'Team'],
                        'PCA Score': scores,
                        'Age': filtered_minutes_dataframe.loc[idx, 'Age'],
                        'Position': filtered_minutes_dataframe.loc[idx, 'Position'],
                        'Data Origin': filtered_minutes_dataframe.loc[idx, 'Data Origin'],
                        'Season': filtered_minutes_dataframe.loc[idx, 'Season']
                    })

                    st.write('**Feature Weights**')
                    wdf = pd.DataFrame({
                        'Metric': weights.index,
                        'Weight': weights.values
                    }).sort_values('Weight', ascending=False)
                    st.dataframe(wdf.style.format({'Weight':'{:.2f}'}))

                    age_filter = df_pca['Age'].between(*age_range)
                    position_filter = (
                        df_pca['Position'].astype(str).apply(lambda x: any(pos in x for pos in selected_positions)) 
                        if selected_positions 
                        else pd.Series(True, index=df_pca.index)
                    )
                    df_filtered = df_pca[age_filter & position_filter]

                    if not df_filtered.empty:
                        min_score, max_score = df_filtered['PCA Score'].min(), df_filtered['PCA Score'].max()
                        score_range = st.slider(
                            'Filter PCA Score range',
                            min_value=float(min_score),
                            max_value=float(max_score),
                            value=(float(min_score), float(max_score))
                        )
                        
                        final_df = df_filtered[df_filtered['PCA Score'].between(*score_range)]
                        if final_df.empty:
                            st.warning('No players in the selected PCA score range.')
                        else:
                            st.write(f'**Matching Players ({len(final_df)})**')
                            st.dataframe(final_df.sort_values('PCA Score', ascending=False).reset_index(drop=True))
                            
                            st.write('**Score Distribution**')
                            fig_pca = go.Figure(data=[go.Bar(x=final_df['Player'], y=final_df['PCA Score'])])
                            
                            chart_title = (
                                f"<b>PCA Scores</b><br>"
                                f"<sup>Context: {context['leagues']} ({context['seasons']})<br>"
                                f"Filters: Age {context['min_age']}-{context['max_age']} | "
                                f"{context['positions']} | Metrics: {len(selected_metrics)} selected</sup>"
                            )
                            
                            fig_pca.update_layout(
                                title=dict(text=chart_title, x=0.03, xanchor='left', font=dict(size=18)),
                                template='plotly_dark',
                                margin=dict(t=200, b=100, l=100, r=100)
                            )
                            st.plotly_chart(fig_pca)
                            
                            if st.button('Export PCA Scores (300 DPI)', key='export_pca'):
                                fig_pca.update_layout(margin=dict(t=250))
                                image_bytes = fig_pca.to_image(format="png", width=1600, height=900, scale=3)
                                st.download_button(
                                    "⬇️ Download PCA Chart", 
                                    data=image_bytes, 
                                    file_name="pca_scores.png", 
                                    mime="image/png"
                                )
                            
                            # Exportar Excel
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                final_df.to_excel(writer, sheet_name='PCA Results', index=False)
                            buffer.seek(0)
                            st.download_button(
                                '📥 Download Results as Excel',
                                data=buffer,
                                file_name='pca_results.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                    else:
                        st.warning('No players match the current filters.')
                except Exception as error:
                    st.error(f'PCA calculation error: {str(error)}')
            else:
                st.error('Insufficient valid metrics after filtering.')

    except Exception as error:
        st.error(f'Error: {error}')
else:
    st.info('Please upload up to 15 Wyscout Excel files to start the analysis')
    st.warning("⚠️ Required: `pip install kaleido==0.2.1.post1`")
