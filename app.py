# Football Analytics App - Versão Completa
# Todos os componentes + Filtros de Exibição por Idade - v3.4

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
st.caption("Created by João Alberto Kolling | Player Analysis System v3.4")

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
            sel = st.multiselect('Metrics for Radar (6–12)', numeric_cols, default=numeric_cols[:6])
            
            if 6 <= len(sel) <= 12:
                d1 = df_minutes[df_minutes['Player']==p1].iloc[0]
                d2 = df_minutes[df_minutes['Player']==p2].iloc[0]
                
                p1pct = {m: calc_percentile(df_minutes[m], d1[m]) for m in sel}
                p2pct = {m: calc_percentile(df_minutes[m], d2[m]) for m in sel}
                gm = {m: df_minutes[m].mean() for m in sel}
                
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
                
                title_text = (f"<b>{p1} vs {p2}</b><br>"
                             f"<sup>Full Dataset: {context['leagues']} | {context['seasons']}</sup>")
                
                fig_radar.update_layout(
                    title=dict(text=title_text, x=0.03, xanchor='left', font=dict(size=18)),
                    polar=dict(radialaxis=dict(range=[0,100])),
                    template='plotly_white',
                    height=700
                )
                st.plotly_chart(fig_radar)
                
                # Tabela de Valores Nominais
                st.markdown("**Nominal Values**")
                df_table = pd.DataFrame({
                    'Metric': sel,
                    p1: [round(d1[m], 2) for m in sel],
                    p2: [round(d2[m], 2) for m in sel],
                    'Group Avg': [round(gm[m], 2) for m in sel]
                }).set_index('Metric')
                st.dataframe(df_table.style.format(precision=2))

        # =============================================
        # Bar Charts (Aba 2)
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
                    title=dict(text=f"<b>Metric Comparison</b><br><sup>Full Dataset Analysis</sup>", x=0.03),
                    height=300*len(selected_metrics),
                    template='plotly_white',
                    barmode='group'
                )
                st.plotly_chart(fig)

        # =============================================
        # Scatter Plot (Aba 3)
        # =============================================
        with tabs[2]:
            st.header('Scatter Plot')
            
            # Filtro de Exibição
            age_min, age_max = int(df_minutes['Age'].min()), int(df_minutes['Age'].max())
            age_range = st.slider('Display Age Range', age_min, age_max, (age_min, age_max))
            df_display = df_minutes[df_minutes['Age'].between(*age_range)]
            
            x = st.selectbox('X metric', numeric_cols)
            y = st.selectbox('Y metric', numeric_cols)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_display[x], 
                y=df_display[y], 
                mode='markers', 
                marker=dict(color='cornflowerblue', opacity=0.5, size=8), 
                text=df_display['Player'], 
                name='Players'
            ))
            
            title_text = (f"<b>{x} vs {y}</b><br>"
                         f"<sup>Displaying: Age {age_range[0]}-{age_range[1]} | {len(df_display)} players</sup>")
            
            fig.update_layout(
                title=dict(text=title_text, x=0.03, xanchor='left'),
                template='plotly_white',
                height=600
            )
            st.plotly_chart(fig)

        # =============================================
        # Profiler (Aba 4)
        # =============================================
        with tabs[3]:
            st.header('Profiler')
            
            # Filtro de Exibição
            age_min, age_max = int(df_minutes['Age'].min()), int(df_minutes['Age'].max())
            age_range = st.slider('Display Age Range', age_min, age_max, (age_min, age_max), key='profiler_age')
            df_display = df_minutes[df_minutes['Age'].between(*age_range)]
            
            sel = st.multiselect('Select 4–12 metrics', numeric_cols)
            
            if 4 <= len(sel) <= 12:
                # Cálculos no dataset completo
                pct = {m: df_minutes[m].rank(pct=True) for m in sel}
                
                # Filtragem para exibição
                mins = {m: st.slider(f'Min % for {m}', 0,100,50) for m in sel}
                mask = np.logical_and.reduce([pct[m]*100 >= mins[m] for m in sel])
                filtered_players = df_display.loc[mask, ['Player', 'Team', 'Age']+sel]
                
                st.dataframe(
                    filtered_players.sort_values(sel, ascending=False).reset_index(drop=True),
                    height=400
                )

        # =============================================
        # Correlation Matrix (Aba 5)
        # =============================================
        with tabs[4]:
            st.header('Correlation Matrix')
            sel = st.multiselect('Metrics to correlate', numeric_cols, default=numeric_cols)
            
            if len(sel) >= 2:
                corr = df_minutes[sel].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values, 
                    x=sel, 
                    y=sel, 
                    zmin=-1, 
                    zmax=1, 
                    colorscale='Viridis'
                ))
                
                fig.update_layout(
                    title=dict(text="<b>Metric Relationships</b><br><sup>Full Dataset Analysis</sup>", x=0.03),
                    height=600
                )
                st.plotly_chart(fig)

        # =============================================
        # Composite PCA Index (Aba 6)
        # =============================================
        with tabs[5]:
            st.header('Composite PCA Index + Excel Export')
            
            # Filtro de Exibição
            age_min, age_max = int(df_minutes['Age'].min()), int(df_minutes['Age'].max())
            age_range = st.slider('Display Age Range', age_min, age_max, (age_min, age_max), key='pca_age')
            df_display = df_minutes[df_minutes['Age'].between(*age_range)]
            
            performance_cols = [col for col in numeric_cols if col not in ['Age','Height','Country','Minutes played','Position']]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                kernel_type = st.selectbox('Kernel Type',['linear','rbf'], index=1)
            with col2:
                gamma = st.number_input('Gamma', value=0.1, min_value=0.0, step=0.1, disabled=(kernel_type=='linear'))
            with col3:
                corr_threshold = st.slider('Correlation Threshold', 0.0, 1.0, 0.5, 0.05)
            with col4:
                manual_weights = st.checkbox('Manual Weights')

            sel = st.multiselect('Select performance metrics', performance_cols)
            
            if len(sel)>=2:
                try:
                    # Cálculos no dataset completo
                    scaler = StandardScaler()
                    X = scaler.fit_transform(df_minutes[sel])
                    kpca = KernelPCA(
                        n_components=2,
                        kernel=kernel_type,
                        gamma=gamma,
                        random_state=42
                    )
                    principalComponents = kpca.fit_transform(X)
                    
                    # Aplica ao dataset filtrado
                    df_pca = df_display.copy()
                    df_pca[['PC1', 'PC2']] = principalComponents[df_display.index]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_pca['PC1'],
                        y=df_pca['PC2'],
                        mode='markers',
                        text=df_pca['Player'],
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f"PCA - Displaying {len(df_pca)} players (Age {age_range[0]}-{age_range[1]})",
                        xaxis_title='Principal Component 1',
                        yaxis_title='Principal Component 2',
                        height=600
                    )
                    st.plotly_chart(fig)
                    
                    # Export Excel
                    bio = BytesIO()
                    with pd.ExcelWriter(bio, engine='xlsxwriter') as writer:
                        df_pca.to_excel(writer, sheet_name='PCA Results', index=False)
                    bio.seek(0)
                    st.download_button(
                        '📥 Download Results as Excel',
                        data=bio,
                        file_name='pca_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                    
                except Exception as e:
                    st.error(f"PCA Error: {str(e)}")

    except Exception as e:
        st.error(f'Error: {e}')
else:
    st.info('Please upload up to 15 Wyscout Excel files to start the analysis')
