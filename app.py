# Football Analytics App - Versão Completa Definitiva
# Autor: João Alberto Kolling

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from io import BytesIO
import kaleido
from datetime import datetime

# =============================================
# Configuração Inicial
# =============================================
st.set_page_config(
    page_title='Football Analytics',
    layout='wide',
    page_icon="⚽"
)

# Header com logo
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image('vif_logo.png.jpg', width=400)

st.title('Technical Scouting Department')
st.subheader('Football Analytics Dashboard')
st.caption("Created by João Alberto Kolling | Player Analysis System v7.0")

# =============================================
# Funções Principais
# =============================================
if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}

def load_and_clean(files):
    dfs = []
    for file in files[:15]:
        try:
            df = pd.read_excel(file)
            df.dropna(how="all", inplace=True)
            df = df.loc[:, df.columns.notnull()]
            df.columns = [str(c).strip() for c in df.columns]
            
            metadata = st.session_state.file_metadata[file.name]
            df['Data Origin'] = metadata['label']
            df['Season'] = metadata['season']
            
            dfs.append(df)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

@st.cache_data
def calc_percentile(series, value):
    return (series <= value).sum() / len(series)

def get_context_info(df, minutes_range, mpg_range, age_range, sel_pos):
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

class WeightedKPCA:
    def __init__(self, kern='rbf', gamma=None):
        self.kernel = kern
        self.gamma = gamma
        self.scaler = StandardScaler()
    
    def fit_transform(self, X, weights):
        Xs = self.scaler.fit_transform(X)
        Xw = Xs * weights.values
        self.kpca = KernelPCA(
            n_components=1,
            kernel=self.kernel,
            gamma=self.gamma,
            random_state=42
        )
        return self.kpca.fit_transform(Xw).flatten()

# =============================================
# Interface Principal
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
    new_files = [f for f in uploaded_files if f.name not in st.session_state.file_metadata]
    
    for file in new_files:
        with st.form(key=f'metadata_{file.name}'):
            st.subheader(f"Metadata for: {file.name}")
            label = st.text_input("Data origin label (e.g., Bundesliga 2)", key=f"label_{file.name}")
            
            seasons = [f"{y}/{y+1}" for y in range(2020, 2026)] + [str(y) for y in range(2020, 2026)]
            season = st.selectbox("Season", seasons, key=f"season_{file.name}")
            
            if st.form_submit_button("Confirm"):
                st.session_state.file_metadata[file.name] = {'label': label, 'season': season}
                st.rerun()

    if missing_metadata := [f.name for f in uploaded_files if f.name not in st.session_state.file_metadata]:
        st.warning("Please provide metadata for all uploaded files")
        st.stop()

    try:
        df = load_and_clean(uploaded_files)

        # =============================================
        # Filtros Principais
        # =============================================
        min_min, max_min = int(df['Minutes played'].min()), int(df['Minutes played'].max())
        minutes_range = st.sidebar.slider('Minutes Played', min_min, max_min, (min_min, max_min))
        df_minutes = df[df['Minutes played'].between(*minutes_range)].copy()

        df_minutes['Minutes per game'] = df_minutes['Minutes played'] / df_minutes['Matches played'].replace(0, np.nan)
        df_minutes['Minutes per game'] = df_minutes['Minutes per game'].fillna(0).clip(0, 120)
        
        min_mpg, max_mpg = int(df_minutes['Minutes per game'].min()), int(df_minutes['Minutes per game'].max())
        mpg_range = st.sidebar.slider('Minutes per Game', min_mpg, max_mpg, (min_mpg, max_mpg))
        df_minutes = df_minutes[df_minutes['Minutes per game'].between(*mpg_range)]

        min_age, max_age = int(df_minutes['Age'].min()), int(df_minutes['Age'].max())
        age_range = st.sidebar.slider('Age Range', min_age, max_age, (min_age, max_age))

        if 'Position' in df_minutes.columns:
            df_minutes['Position_split'] = df_minutes['Position'].astype(str).apply(lambda x: [p.strip() for p in x.split(',')])
            all_pos = sorted({p for lst in df_minutes['Position_split'] for p in lst})
            sel_pos = st.sidebar.multiselect('Positions', all_pos, default=all_pos)
            df_minutes = df_minutes[df_minutes['Position_split'].apply(lambda x: any(pos in x for pos in sel_pos))] if sel_pos else df_minutes
        else:
            sel_pos = []

        context = get_context_info(df_minutes, minutes_range, mpg_range, age_range, sel_pos)
        players = sorted(df_minutes['Player'].unique())
        p1 = st.sidebar.selectbox('Select Player 1', players)
        p2 = st.sidebar.selectbox('Select Player 2', [p for p in players if p != p1])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        tabs = st.tabs(['Radar', 'Bars', 'Scatter', 'Profiler', 'Correlation', 'Composite Index (PCA)'])

        # =============================================
        # Radar Chart (Implementação Completa)
        # =============================================
        with tabs[0]:
            st.header('Radar Chart')
            sel = st.multiselect('Metrics for Radar (6–12)', numeric_cols, default=numeric_cols[:6])
            
            if 6 <= len(sel) <= 12:
                d1 = df_minutes[df_minutes['Player']==p1].iloc[0]
                d2 = df_minutes[df_minutes['Player']==p2].iloc[0]
                
                # Cálculo do Grupo de Referência
                if sel_pos:
                    group_df = df_minutes.copy()
                else:
                    p1_pos = [p.strip() for p in d1['Position'].split(',')]
                    p2_pos = [p.strip() for p in d2['Position'].split(',')]
                    group_df = df_minutes[df_minutes['Position_split'].apply(lambda x: any(pos in x for pos in p1_pos + p2_pos))]
                
                # Cálculos Estatísticos
                p1pct = {m: calc_percentile(group_df[m], d1[m]) for m in sel}
                p2pct = {m: calc_percentile(group_df[m], d2[m]) for m in sel}
                group_avg = group_df[sel].mean()
                gmpct = {m: calc_percentile(group_df[m], group_avg[m]) for m in sel}
                
                show_avg = st.checkbox('Show Group Average', True)
                
                # Construção do Gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[p1pct[m]*100 for m in sel],
                    theta=sel,
                    fill='toself',
                    name=p1
                ))
                fig.add_trace(go.Scatterpolar(
                    r=[p2pct[m]*100 for m in sel],
                    theta=sel,
                    fill='toself',
                    name=p2
                ))
                
                if show_avg:
                    fig.add_trace(go.Scatterpolar(
                        r=[gmpct[m]*100 for m in sel],
                        theta=sel,
                        fill='toself',
                        name='Group Avg'
                    ))
                
                # Texto Descritivo
                caption_text = f"""
                **Metodologia:**  
                • Percentis calculados sobre {len(group_df)} jogadores  
                • Filtros aplicados:  
                  - Minutos jogados: {context['min_min']}-{context['max_min']}  
                  - Minutos por jogo: {context['min_mpg']}-{context['max_mpg']}  
                  - Idade: {context['min_age']}-{context['max_age']}  
                  - Posições: {context['positions']}  
                • Dados: {context['leagues']} ({context['seasons']})  
                • Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}  
                """
                
                # Layout Final
                fig.update_layout(
                    polar=dict(radialaxis=dict(range=[0,100])),
                    template='plotly_white',
                    margin=dict(t=50, b=250, l=50, r=50),
                    height=700,
                    title=f"{p1} vs {p2}"
                )
                
                st.plotly_chart(fig)
                st.caption(caption_text)
                
                # Exportação
                if st.button('Export Radar Chart (300 DPI)', key='export_radar'):
                    fig.add_annotation(
                        x=0.5,
                        y=-0.3,
                        text=caption_text,
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        align="left",
                        font=dict(size=10)
                    )
                    img_bytes = fig.to_image(format="png", width=1400, height=900, scale=3)
                    st.download_button("Download Radar Chart", data=img_bytes, file_name="radar_chart.png", mime="image/png")

        # =============================================
        # Bar Charts (Implementação Completa)
        # =============================================
        with tabs[1]:
            st.header('Bar Chart Comparison')
            selected_metrics = st.multiselect('Select metrics (max 5)', numeric_cols, default=numeric_cols[:1])
            
            if len(selected_metrics) > 5:
                st.error("Maximum 5 metrics allowed!")
                st.stop()
            
            if len(selected_metrics) >= 1:
                fig = make_subplots(
                    rows=len(selected_metrics),
                    cols=1,
                    subplot_titles=selected_metrics,
                    vertical_spacing=0.15
                )
                
                for idx, metric in enumerate(selected_metrics, 1):
                    p1_val = df_minutes[df_minutes['Player'] == p1][metric].iloc[0]
                    p2_val = df_minutes[df_minutes['Player'] == p2][metric].iloc[0]
                    avg_val = df_minutes[metric].mean()
                    
                    fig.add_trace(go.Bar(y=[p1], x=[p1_val], orientation='h', name=p1, marker_color='#1f77b4', showlegend=(idx == 1)), row=idx, col=1)
                    fig.add_trace(go.Bar(y=[p2], x=[p2_val], orientation='h', name=p2, marker_color='#ff7f0e', showlegend=(idx == 1)), row=idx, col=1)
                    fig.add_vline(x=avg_val, line_dash="dash", line_color="green", annotation_text="Group Avg", row=idx, col=1)
                
                caption_text = f"""
                **Contexto:**  
                • População de referência: {len(df_minutes)} jogadores  
                • Filtros aplicados:  
                  - Minutos totais: {context['min_min']}+  
                  - Minutos por jogo: {context['min_mpg']}+  
                  - Faixa etária: {context['min_age']}-{context['max_age']}  
                  - Posições: {context['positions']}  
                • Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}  
                """
                
                fig.update_layout(
                    height=300*len(selected_metrics),
                    width=800,
                    template='plotly_white',
                    barmode='group',
                    margin=dict(t=50, b=150)
                )
                
                st.plotly_chart(fig)
                st.caption(caption_text)

        # =============================================
        # Scatter Plot (Implementação Completa)
        # =============================================
        with tabs[2]:
            st.header('Scatter Plot')
            x_metric = st.selectbox('X metric', numeric_cols)
            y_metric = st.selectbox('Y metric', numeric_cols)
            highlight_players = st.multiselect('Highlight up to 5 players', players, default=[p1, p2])[:5]
            
            df_filtered = df_minutes.copy()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtered[x_metric], 
                y=df_filtered[y_metric], 
                mode='markers',
                marker=dict(color='rgba(100,149,237,0.5)', size=8),
                name='All Players'
            ))
            
            colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080']
            for i, player in enumerate(highlight_players):
                player_data = df_filtered[df_filtered['Player'] == player]
                if not player_data.empty:
                    fig.add_trace(go.Scatter(
                        x=player_data[x_metric],
                        y=player_data[y_metric],
                        mode='markers+text',
                        marker=dict(color=colors[i], size=12),
                        name=player
                    ))
            
            caption_text = f"""
            **Configuração:**  
            • Eixo X: {x_metric}  
            • Eixo Y: {y_metric}  
            • População: {len(df_filtered)} jogadores  
            • Filtros ativos:  
              - {context['positions']}  
              - {context['min_age']}-{context['max_age']} anos  
              - {context['min_min']}+ minutos jogados  
            • Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}  
            """
            
            fig.update_layout(
                title=f"{x_metric} vs {y_metric}",
                template='plotly_white',
                margin=dict(t=50, b=150)
            )
            
            st.plotly_chart(fig)
            st.caption(caption_text)

        # =============================================
        # Profiler (Implementação Completa)
        # =============================================
        with tabs[3]:
            st.header('Player Profiler')
            selected_profiler_metrics = st.multiselect('Select 4–12 metrics', numeric_cols)
            
            if 4 <= len(selected_profiler_metrics) <= 12:
                thresholds = {m: st.slider(f'Minimum percentile for {m}', 0, 100, 50) for m in selected_profiler_metrics}
                
                filtered_players = df_minutes.copy()
                for metric, threshold in thresholds.items():
                    filtered_players = filtered_players[filtered_players[metric] >= filtered_players[metric].quantile(threshold/100)]
                
                st.dataframe(filtered_players[['Player', 'Team', 'Age', 'Position'] + selected_profiler_metrics])
            else:
                st.warning('Select between 4 and 12 metrics.')

        # =============================================
        # Correlation Matrix (Implementação Completa)
        # =============================================
        with tabs[4]:
            st.header('Correlation Matrix')
            corr_metrics = st.multiselect('Select metrics to correlate', numeric_cols, default=numeric_cols[:5])
            
            if len(corr_metrics) >= 2:
                corr_matrix = df_minutes[corr_metrics].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                
                caption_text = f"""
                **Parâmetros:**  
                • Métricas incluídas: {', '.join(corr_metrics)}  
                • Número de jogadores: {len(df_minutes)}  
                • Método: Pearson Correlation  
                • Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}  
                """
                
                fig.update_layout(
                    title='Correlation Matrix',
                    template='plotly_white',
                    margin=dict(t=50, b=150)
                )
                
                st.plotly_chart(fig)
                st.caption(caption_text)

        # =============================================
        # Composite PCA Index (Implementação Completa)
        # =============================================
        with tabs[5]:
            st.header('Composite PCA Index + Excel Export')
            performance_cols = [col for col in numeric_cols if col not in ['Age','Height','Country','Minutes played','Position']]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                kernel_type = st.selectbox('Kernel Type', ['linear', 'rbf'], index=1)
            with col2:
                gamma = st.number_input('Gamma', value=0.1, min_value=0.0, step=0.1, disabled=(kernel_type == 'linear'))
            with col3:
                corr_threshold = st.slider('Correlation Threshold', 0.0, 1.0, 0.5, 0.05)
            with col4:
                manual_weights = st.checkbox('Manual Weights', key='manual_weights')

            sel = st.multiselect('Select performance metrics', performance_cols)
            
            if len(sel) >= 2:
                try:
                    if manual_weights:
                        st.subheader('Manual Weight Adjustment')
                        weight_sliders = {}
                        cols = st.columns(3)
                        for idx, m in enumerate(sel):
                            with cols[idx % 3]:
                                weight_sliders[m] = st.slider(f'Weight for {m}', 0.0, 1.0, 0.5, key=f'weight_{m}')
                        weights = pd.Series(weight_sliders)
                    else:
                        @st.cache_data
                        def calculate_weights(_df, features, threshold):
                            cm = _df[features].corr().abs()
                            ac = cm.mean(axis=1)
                            return ac.where(ac > threshold, 0)
                        weights = calculate_weights(df_minutes, sel, corr_threshold)

                    kp = WeightedKPCA(kern=kernel_type, gamma=(None if kernel_type == 'linear' else gamma))
                    df_sel = df_minutes[sel].dropna()
                    scores = kp.fit_transform(df_sel, weights)
                    
                    df_pca = pd.DataFrame({
                        'Player': df_minutes.loc[df_sel.index, 'Player'],
                        'Team': df_minutes.loc[df_sel.index, 'Team'],
                        'Age': df_minutes.loc[df_sel.index, 'Age'],
                        'Position': df_minutes.loc[df_sel.index, 'Position'],
                        'PCA Score': scores
                    })

                    st.write('**Feature Weights**')
                    wdf = pd.DataFrame({'Metric': weights.index, 'Weight': weights.values}).sort_values('Weight', ascending=False)
                    st.dataframe(wdf.style.format({'Weight':'{:.2f}'}))

                    st.write('**PCA Results**')
                    st.dataframe(df_pca.sort_values('PCA Score', ascending=False))
                    
                    bio = BytesIO()
                    with pd.ExcelWriter(bio, engine='xlsxwriter') as writer:
                        df_pca.to_excel(writer, index=False)
                    bio.seek(0)
                    
                    st.download_button(
                        '📥 Download PCA Results',
                        data=bio,
                        file_name='pca_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                except Exception as e:
                    st.error(f'PCA calculation error: {str(e)}')

    except Exception as e:
        st.error(f'Application Error: {str(e)}')
else:
    st.info('Please upload Wyscout Excel files to begin analysis')
    st.warning("Required dependencies: pip install kaleido xlsxwriter")
