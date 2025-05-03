# Football Analytics App - Versão Completa
# Todos os componentes + Filtros de Exibição por Idade - v3.5

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

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image('vif_logo.png.jpg', width=400)

st.title('Technical Scouting Department')
st.subheader('Football Analytics Dashboard')
st.caption("Created by João Alberto Kolling | Player Analysis System v3.5")

with st.expander("📘 User Guide & Instructions", expanded=False):
    st.markdown("""
    **⚠️ Requirements:**  
    1. Install dependencies:  
    `pip install kaleido==0.2.1.post1 xlsxwriter`  
    2. Data must contain columns: Player, Age, Position, Metrics, Team  
    
    **Key Features:**  
    - Age filters affect only results display  
    - Statistical calculations use full dataset  
    - Professional 300 DPI exports  
    - Advanced PCA analysis  
    - Dynamic metric filtering  
    """)

# =============================================
# Funções Principais
# =============================================
if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}

def load_and_clean(files):
    dfs = []
    for file in files[:15]:
        df = pd.read_excel(file)
        df.dropna(how="all", inplace=True)
        df = df.loc[:, df.columns.notnull()]
        df.columns = [str(c).strip() for c in df.columns]
        
        metadata = st.session_state.file_metadata[file.name]
        df['Data Origin'] = metadata['label']
        df['Season'] = metadata['season']
        
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

@st.cache_data
def calc_percentile(series, value):
    return (series <= value).sum() / len(series)

def get_context_info(df, minutes_range, mpg_range, sel_pos):
    return {
        'leagues': ', '.join(df['Data Origin'].unique()),
        'seasons': ', '.join(df['Season'].unique()),
        'total_players': len(df),
        'min_min': minutes_range[0],
        'max_min': minutes_range[1],
        'min_mpg': mpg_range[0],
        'max_mpg': mpg_range[1],
        'positions': ', '.join(sel_pos) if sel_pos else 'All'
    }

# =============================================
# Filtros da Barra Lateral
# =============================================
st.sidebar.header('Core Filters')
with st.sidebar.expander("⚙️ Data Settings", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload up to 15 Wyscout Excel files", 
        type=["xlsx"], 
        accept_multiple_files=True
    )

if uploaded_files:
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

        # Filtros Principais (Sem Idade)
        min_min, max_min = int(df['Minutes played'].min()), int(df['Minutes played'].max())
        minutes_range = st.sidebar.slider('Minutes Played', min_min, max_min, (min_min, max_min))
        df_minutes = df[df['Minutes played'].between(*minutes_range)].copy()

        df_minutes['Minutes per game'] = df_minutes['Minutes played'] / df_minutes['Matches played'].replace(0, np.nan)
        df_minutes['Minutes per game'] = df_minutes['Minutes per game'].fillna(0).clip(0, 120)
        
        min_mpg, max_mpg = int(df_minutes['Minutes per game'].min()), int(df_minutes['Minutes per game'].max())
        mpg_range = st.sidebar.slider('Minutes per Game', min_mpg, max_mpg, (min_mpg, max_mpg))
        df_minutes = df_minutes[df_minutes['Minutes per game'].between(*mpg_range)]

        # Filtro de Posição (Não Altera DataFrame)
        if 'Position' in df_minutes.columns:
            df_minutes['Position_split'] = df_minutes['Position'].astype(str).apply(lambda x: [p.strip() for p in x.split(',')])
            all_pos = sorted({p for lst in df_minutes['Position_split'] for p in lst})
            sel_pos = st.sidebar.multiselect('Positions', all_pos, default=all_pos)
        else:
            sel_pos = []

        context = get_context_info(df_minutes, minutes_range, mpg_range, sel_pos)
        players = sorted(df_minutes['Player'].unique())
        p1 = st.sidebar.selectbox('Select Player 1', players)
        p2 = st.sidebar.selectbox('Select Player 2', [p for p in players if p != p1])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        tabs = st.tabs(['Radar', 'Bars', 'Scatter', 'Profiler', 'Correlation', 'Composite Index (PCA)'])

        # =============================================
        # Radar Chart (Aba 1)
        # =============================================
        with tabs[0]:
            st.header('Radar Chart')
            col1, col2 = st.columns([3, 1])
            with col1:
                sel = st.multiselect('Metrics for Radar (6–12)', numeric_cols, default=numeric_cols[:6])
            
            if 6 <= len(sel) <= 12:
                d1 = df_minutes[df_minutes['Player']==p1].iloc[0]
                d2 = df_minutes[df_minutes['Player']==p2].iloc[0]
                
                with col2:
                    show_avg = st.checkbox('Show Group Average', True)
                    export_quality = st.selectbox('Export Quality', ['Screen (72 DPI)', 'Print (300 DPI)'], index=0)

                p1pct = {m: calc_percentile(df_minutes[m], d1[m]) for m in sel}
                p2pct = {m: calc_percentile(df_minutes[m], d2[m]) for m in sel}
                gm = {m: df_minutes[m].mean() for m in sel}
                gmpct = {m: calc_percentile(df_minutes[m], gm[m]) for m in sel}
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[p1pct[m]*100 for m in sel],
                    theta=sel,
                    fill='toself',
                    name=p1,
                    line_color='#1f77b4'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=[p2pct[m]*100 for m in sel],
                    theta=sel,
                    fill='toself',
                    name=p2,
                    line_color='#ff7f0e'
                ))
                
                if show_avg:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[gmpct[m]*100 for m in sel],
                        theta=sel,
                        fill='toself',
                        name='Group Avg',
                        line_color='#2ca02c'
                    ))
                
                title_text = (f"<b>{p1} vs {p2}</b><br>"
                             f"<sup>Context: {context['leagues']} ({context['seasons']}) | "
                             f"{context['positions']} | {len(df_minutes)} players</sup>")
                
                fig_radar.update_layout(
    title=dict(text=title_text, x=0.03, xanchor='left', font=dict(size=18)),
    polar=dict(
        radialaxis=dict(range=[0,100]),
        angularaxis=dict(rotation=90)
    ),
    template='plotly_white',
    height=700,
    margin=dict(t=150)
)
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Exportação
                if st.button('Export Radar Chart'):
                    scale = 3 if export_quality == 'Print (300 DPI)' else 1
                    img_bytes = fig_radar.to_image(format="png", width=1600, height=1200, scale=scale)
                    st.download_button(
                        "⬇️ Download Image",
                        data=img_bytes,
                        file_name=f"radar_{p1}_vs_{p2}.png",
                        mime="image/png"
                    )
                
                # Tabela de Valores
                st.subheader("Metric Values Comparison")
                df_table = pd.DataFrame({
                    'Metric': sel,
                    p1: [round(d1[m], 2) for m in sel],
                    p2: [round(d2[m], 2) for m in sel],
                    'Group Avg': [round(gm[m], 2) for m in sel]
                }).set_index('Metric')
                st.dataframe(df_table.style.format(precision=2).highlight_max(axis=1, color='#90EE90'))

        # =============================================
        # Bar Charts (Aba 2)
        # =============================================
        with tabs[1]:
            st.header('Comparative Analysis - Bar Charts')
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_metrics = st.multiselect('Select Metrics (max 5)', numeric_cols, default=numeric_cols[:3])
            with col2:
                export_quality = st.selectbox('Export Quality', ['Screen (72 DPI)', 'Print (300 DPI)'], index=0, key='bar_quality')
            
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
                    
                    fig.add_trace(go.Bar(
                        y=[p1], 
                        x=[p1_val], 
                        orientation='h',
                        name=p1, 
                        marker_color='#1f77b4', 
                        showlegend=(idx == 1)
                    ), row=idx, col=1)
                    
                    fig.add_trace(go.Bar(
                        y=[p2], 
                        x=[p2_val], 
                        orientation='h',
                        name=p2, 
                        marker_color='#ff7f0e', 
                        showlegend=(idx == 1)
                    ), row=idx, col=1)
                    
                    fig.add_vline(
                        x=avg_val, 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text="Group Avg", 
                        row=idx, 
                        col=1
                    )
                
                fig.update_layout(
                    title=dict(text=f"<b>Metric Comparison</b><br><sup>{context['leagues']} | {context['seasons']}</sup>", x=0.03),
                    height=300*len(selected_metrics),
                    template='plotly_white',
                    barmode='group',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Exportação
                if st.button('Export Bar Charts'):
                    scale = 3 if export_quality == 'Print (300 DPI)' else 1
                    img_bytes = fig.to_image(format="png", width=1600, height=300*len(selected_metrics)+300, scale=scale)
                    st.download_button(
                        "⬇️ Download Charts", 
                        data=img_bytes, 
                        file_name="bar_comparison.png", 
                        mime="image/png"
                    )

        # =============================================
        # Scatter Plot (Aba 3)
        # =============================================
        with tabs[2]:
            st.header('Scatter Plot Analysis')
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                x = st.selectbox('X Axis Metric', numeric_cols, index=0)
            with col2:
                y = st.selectbox('Y Axis Metric', numeric_cols, index=1)
            with col3:
                age_min, age_max = int(df_minutes['Age'].min()), int(df_minutes['Age'].max())
                age_range = st.slider('Age Range', age_min, age_max, (age_min, age_max))
                export_quality = st.selectbox('Export Quality', ['Screen', 'Print'], key='scatter_quality')
            
            df_display = df_minutes[df_minutes['Age'].between(*age_range)]
            if sel_pos:
                df_display = df_display[df_display['Position_split'].apply(lambda x: any(pos in x for pos in sel_pos))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_display[x], 
                y=df_display[y], 
                mode='markers',
                marker=dict(
                    color=df_display['Age'],
                    colorscale='Viridis',
                    size=8,
                    showscale=True
                ),
                text=df_display['Player'],
                hoverinfo='text+x+y'
            ))
            
            # Highlight selected players
            for player in [p1, p2]:
                p_data = df_display[df_display['Player'] == player]
                if not p_data.empty:
                    fig.add_trace(go.Scatter(
                        x=p_data[x],
                        y=p_data[y],
                        mode='markers+text',
                        marker=dict(size=12, color='red'),
                        name=player,
                        text=player,
                        textposition='top center'
                    ))
            
            fig.update_layout(
                title=f"{x} vs {y} - Age {age_range[0]}-{age_range[1]}",
                xaxis_title=x,
                yaxis_title=y,
                template='plotly_white',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Exportação
            if st.button('Export Scatter Plot'):
                scale = 3 if export_quality == 'Print' else 1
                img_bytes = fig.to_image(format="png", width=1600, height=1200, scale=scale)
                st.download_button(
                    "⬇️ Download Plot", 
                    data=img_bytes, 
                    file_name=f"scatter_{x}_vs_{y}.png", 
                    mime="image/png"
                )

        # =============================================
        # Profiler (Aba 4)
        # =============================================
        with tabs[3]:
            st.header('Advanced Player Profiler')
            col1, col2 = st.columns([3, 1])
            with col1:
                sel = st.multiselect('Performance Metrics', numeric_cols, default=numeric_cols[:5])
            with col2:
                age_min, age_max = int(df_minutes['Age'].min()), int(df_minutes['Age'].max())
                age_range = st.slider('Age Filter', age_min, age_max, (age_min, age_max))
                min_percentile = st.slider('Minimum Percentile', 0, 100, 50)
            
            if len(sel) >= 1:
                df_display = df_minutes[df_minutes['Age'].between(*age_range)]
                if sel_pos:
                    df_display = df_display[df_display['Position_split'].apply(lambda x: any(pos in x for pos in sel_pos))]
                
                # Cálculos de percentil
                percentiles = df_display[sel].rank(pct=True) * 100
                mask = (percentiles >= min_percentile).all(axis=1)
                
                # Resultados
                results = df_display.loc[mask, ['Player', 'Team', 'Age'] + sel]
                st.dataframe(
                    results.sort_values(sel, ascending=False)
                    .style.format({m: "{:.2f}" for m in sel})
                    .background_gradient(cmap='Blues', subset=sel),
                    height=600
                )

        # =============================================
        # Correlation Matrix (Aba 5)
        # =============================================
        with tabs[4]:
            st.header('Metric Correlation Analysis')
            selected_corr = st.multiselect('Select Metrics', numeric_cols, default=numeric_cols[:10])
            
            if len(selected_corr) >= 2:
                corr_matrix = df_minutes[selected_corr].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=selected_corr,
                    y=selected_corr,
                    zmin=-1,
                    zmax=1,
                    colorscale='RdBu',
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Metric Correlation Matrix',
                    height=800,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Exportação
                if st.button('Export Correlation Matrix'):
                    img_bytes = fig.to_image(format="png", width=1600, height=1600, scale=3)
                    st.download_button(
                        "⬇️ Download Matrix", 
                        data=img_bytes, 
                        file_name="correlation_matrix.png", 
                        mime="image/png"
                    )

        # =============================================
        # Composite PCA Index (Aba 6)
        # =============================================
        with tabs[5]:
            st.header('PCA Composite Index Analysis')
            col1, col2 = st.columns([3, 1])
            with col1:
                pca_metrics = st.multiselect('PCA Metrics', numeric_cols, default=numeric_cols[:8])
            with col2:
                age_min, age_max = int(df_minutes['Age'].min()), int(df_minutes['Age'].max())
                age_range = st.slider('Age Filter', age_min, age_max, (age_min, age_max))
                n_components = st.slider('Components', 2, 5, 2)
            
            if len(pca_metrics) >= 2:
                df_display = df_minutes[df_minutes['Age'].between(*age_range)]
                
                # Pré-processamento
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df_display[pca_metrics])
                
                # PCA
                pca = PCA(n_components=n_components)
                principal_components = pca.fit_transform(scaled_data)
                
                # Visualização
                fig = go.Figure()
                for i in range(n_components-1):
                    fig.add_trace(go.Scatter(
                        x=principal_components[:, i],
                        y=principal_components[:, i+1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=df_display['Age'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=df_display['Player'],
                        hoverinfo='text+x+y'
                    ))
                
                fig.update_layout(
                    title=f'PCA Analysis ({n_components} Components)',
                    xaxis_title='Component 1',
                    yaxis_title='Component 2',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Exportação
                if st.button('Export PCA Results'):
                    bio = BytesIO()
                    with pd.ExcelWriter(bio, engine='xlsxwriter') as writer:
                        pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)]).to_excel(writer, index=False)
                    bio.seek(0)
                    st.download_button(
                        "📥 Download PCA Data",
                        data=bio,
                        file_name="pca_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    except Exception as e:
        st.error(f'Error: {e}')
else:
    st.info('Please upload up to 15 Wyscout Excel files to start the analysis')
