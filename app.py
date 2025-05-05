# Football Analytics App - Complete Version
# Todos os componentes incluídos - v3.0

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
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image('vif_logo.png.jpg', width=400)

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
    
    **Key Features:**  
    - Player comparison with radar/barcharts  
    - Metric correlation analysis  
    - Advanced filtering system  
    - Professional 300 DPI exports  
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

st.sidebar.markdown("---")
st.sidebar.subheader("Dataframe Filters")

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

        # Filtros Principais
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
        df_minutes = df_minutes[df_minutes['Age'].between(*age_range)]

        # Coleta posições
        if 'Position' in df_minutes.columns:
            df_minutes['Position_split'] = df_minutes['Position'].astype(str).apply(lambda x: [p.strip() for p in x.split(',')])
            all_pos = sorted({p for lst in df_minutes['Position_split'] for p in lst})
            sel_pos = st.sidebar.multiselect('Positions', all_pos, default=all_pos)
        else:
            sel_pos = []

        # Cria dataframe para cálculos de grupo
        if 'Position_split' in df_minutes.columns and sel_pos:
            df_group = df_minutes[df_minutes['Position_split'].apply(lambda x: any(pos in x for pos in sel_pos))]
        else:
            df_group = df_minutes.copy()

        context = get_context_info(df_minutes, minutes_range, mpg_range, age_range, sel_pos)
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
                gm = {m: df_group[m].mean() for m in sel}
                gmpct = {m: calc_percentile(df_minutes[m], gm[m]) for m in sel}
                
                show_avg = st.checkbox('Show Group Average', True)
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
                             f"<sup>Leagues: {context['leagues']} | Seasons: {context['seasons']}<br>"
                             f"Filters: {context['min_min']}-{context['max_min']} mins | "
                             f"{context['min_mpg']}-{context['max_mpg']} min/game | "
                             f"Age {context['min_age']}-{context['max_age']} | Positions: {context['positions']}</sup>")
                
                
fig_radar.update_layout(
    template='plotly_dark',
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            showline=False,
            gridcolor="white",
            gridwidth=1,
            tickfont=dict(size=10, color='white'),
            ticks='outside',
        ),
        angularaxis=dict(
            tickfont=dict(size=12, color='white'),
            gridcolor="gray"
        ),
        bgcolor='rgba(0,0,0,0)'
    ),
    showlegend=True,
    legend=dict(font=dict(color='white')),
    margin=dict(t=80, b=80, l=80, r=80),
    height=750,
    title=dict(
        text=title_text,
        x=0.03,
        xanchor='left',
        font=dict(size=18, color='white')
    )
)

                
                st.plotly_chart(fig_radar)
                
                st.markdown("**Nominal Values**")
                df_table = pd.DataFrame({
                    'Metric': sel,
                    p1: [round(d1[m], 2) for m in sel],
                    p2: [round(d2[m], 2) for m in sel],
                    'Group Avg': [round(gm[m], 2) for m in sel]
                }).set_index('Metric')
                
                st.dataframe(
                    df_table.style
                    .format(precision=2)
                    .set_properties(**{'background-color': 'white', 'color': 'black'})
                )
                
                if st.button('Export Radar Chart (300 DPI)', key='export_radar'):
                    img_bytes = fig_radar.to_image(
                        format="png", 
                        width=1600, 
                        height=1400, 
                        scale=3
                    )
                    st.download_button(
                        "⬇️ Download Radar Chart", 
                        data=img_bytes, 
                        file_name=f"radar_{p1}_vs_{p2}.png", 
                        mime="image/png"
                    )

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
                    avg_val = df_group[metric].mean()
                    
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
                
                title_text = (f"<b>Metric Comparison</b><br>"
                             f"<sup>Context: {context['leagues']} ({context['seasons']}) | "
                             f"Players: {context['total_players']} | Filters: {context['min_age']}-{context['max_age']} years</sup>")
                
                fig.update_layout(
                    title=dict(text=title_text, x=0.03, xanchor='left', font=dict(size=18)),
                    height=300*len(selected_metrics),
                    width=800,
                    template='plotly_white',
                    barmode='group',
                    margin=dict(t=200, b=100, l=100, r=100)
                )
                st.plotly_chart(fig)
                
                if st.button('Export Bar Charts (300 DPI)', key='export_bar'):
                    fig.update_layout(margin=dict(t=250))
                    img_bytes = fig.to_image(format="png", width=1600, height=300*len(selected_metrics)+300, scale=3)
                    st.download_button(
                        "⬇️ Download Charts", 
                        data=img_bytes, 
                        file_name="bar_charts.png", 
                        mime="image/png"
                    )

        # =============================================
        # Scatter Plot (Aba 3)
        # =============================================
        with tabs[2]:
            st.header('Scatter Plot')
            x = st.selectbox('X metric', numeric_cols)
            y = st.selectbox('Y metric', numeric_cols)
            highlight_players = st.multiselect('Highlight up to 5 players', players, default=[p1, p2])[:5]
            
            df_filtered = df_minutes[df_minutes['Age'].between(*age_range)]
            if sel_pos:
                df_filtered = df_filtered[df_filtered['Position_split'].apply(lambda x: any(pos in x for pos in sel_pos))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtered[x], 
                y=df_filtered[y], 
                mode='markers', 
                marker=dict(color='cornflowerblue', opacity=0.5, size=8), 
                text=df_filtered['Player'], 
                hoverinfo='text', 
                name='All'
            ))
            
            colors = ['red','blue','green','orange','purple']
            for i,p in enumerate(highlight_players):
                pdata = df_filtered[df_filtered['Player']==p]
                if not pdata.empty:
                    fig.add_trace(go.Scatter(
                        x=pdata[x], 
                        y=pdata[y], 
                        text=pdata['Player'], 
                        mode='markers+text', 
                        marker=dict(size=12, color=colors[i]), 
                        name=p
                    ))
            
            title_text = (f"<b>{x} vs {y}</b><br>"
                         f"<sup>Data Source: {context['leagues']} ({context['seasons']})<br>"
                         f"Filters: {context['total_players']} players | "
                         f"{context['min_mpg']}+ min/game | {context['positions']}</sup>")
            
            fig.update_layout(
                title=dict(text=title_text, x=0.03, xanchor='left', font=dict(size=18)),
                width=1000, 
                height=700,
                template='plotly_dark',
                margin=dict(t=200, b=100, l=100, r=100)
            )
            st.plotly_chart(fig)
            
            if st.button('Export Scatter Plot (300 DPI)', key='export_scatter'):
                fig.update_layout(margin=dict(t=250))
                img_bytes = fig.to_image(format="png", width=1800, height=1200, scale=3)
                st.download_button(
                    "⬇️ Download Scatter Plot", 
                    data=img_bytes, 
                    file_name=f"scatter_{x}_vs_{y}.png", 
                    mime="image/png"
                )

        # =============================================
        # Profiler (Aba 4)
        # =============================================
        with tabs[3]:
            st.header('Profiler')
            sel = st.multiselect('Select 4–12 metrics', numeric_cols)
            
            age_min_profiler, age_max_profiler = st.slider(
                'Age Range (Profiler)', 
                min_value=int(df_minutes['Age'].min()), 
                max_value=int(df_minutes['Age'].max()), 
                value=(int(df_minutes['Age'].min()), int(df_minutes['Age'].max()))
            )
            
            if 4 <= len(sel) <= 12:
                pct = {m: df_minutes[m].rank(pct=True) for m in sel}
                mins = {m: st.slider(f'Min % for {m}', 0,100,50) for m in sel}
                mask = np.logical_and.reduce([pct[m]*100 >= mins[m] for m in sel])
                
                df_profiler_filtered = df_minutes.loc[mask].copy()
                df_profiler_filtered = df_profiler_filtered[
                    df_profiler_filtered['Age'].between(age_min_profiler, age_max_profiler)
                ]
                
                st.dataframe(df_profiler_filtered[['Player', 'Team']+sel].reset_index(drop=True))
            else:
                st.warning('Select between 4 and 12 metrics.')

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
                
                title_text = (f"<b>Metric Relationships</b><br>"
                             f"<sup>Dataset: {context['leagues']} ({context['seasons']})<br>"
                             f"Players: {context['total_players']} | Min. Minutes: {context['min_min']}+</sup>")
                
                fig.update_layout(
                    title=dict(text=title_text, x=0.03, xanchor='left', font=dict(size=18)),
                    template='plotly_dark',
                    margin=dict(t=200, b=100, l=100, r=100)
                )
                st.plotly_chart(fig)
                
                if st.button('Export Correlation Matrix (300 DPI)', key='export_corr'):
                    fig.update_layout(margin=dict(t=250))
                    img_bytes = fig.to_image(format="png", width=1400, height=1400, scale=3)
                    st.download_button(
                        "⬇️ Download Correlation Matrix", 
                        data=img_bytes, 
                        file_name="correlation_matrix.png", 
                        mime="image/png"
                    )

        # =============================================
        # Composite PCA Index (Aba 6) - SEÇÃO CORRIGIDA
        # =============================================
        with tabs[5]:
            st.header('Composite PCA Index + Excel Export')
            performance_cols = [col for col in numeric_cols if col not in ['Age','Height','Country','Minutes played','Position']]
            
            age_min_pca, age_max_pca = st.slider(
                'Age Range (PCA)', 
                min_value=int(df_minutes['Age'].min()), 
                max_value=int(df_minutes['Age'].max()), 
                value=(int(df_minutes['Age'].min()), int(df_minutes['Age'].max()))
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                kernel_type = st.selectbox('Kernel Type',['linear','rbf'], index=1)
            with col2:
                gamma = st.number_input('Gamma', value=0.1, min_value=0.0, step=0.1, disabled=(kernel_type=='linear'))
            with col3:
                corr_threshold = st.slider(
                    'Correlation Threshold', 
                    0.0, 1.0, 0.5, 0.05,
                    help='Minimum average correlation for feature inclusion',
                    disabled=st.session_state.get('manual_weights', False)
                )
            with col4:
                manual_weights = st.checkbox('Manual Weights', key='manual_weights')

            sel = st.multiselect('Select performance metrics', performance_cols)
            
            if len(sel)<2:
                st.warning('Select at least two performance metrics.')
                st.stop()

            if manual_weights:
                st.subheader('Manual Weight Adjustment')
                weight_sliders = {}
                cols2 = st.columns(3)
                for idx, m in enumerate(sel):
                    with cols2[idx%3]:
                        weight_sliders[m] = st.slider(f'Weight for {m}', 0.0, 1.0, 0.5, key=f'weight_{m}')
                weights = pd.Series(weight_sliders)
                excluded = []
            else:
                @st.cache_data
                def calculate_weights(_df, features, threshold):
                    cm = _df[features].corr().abs()
                    ac = cm.mean(axis=1)
                    return ac.where(ac>threshold, 0)
                weights = calculate_weights(df_minutes, sel, corr_threshold)
                excluded = weights[weights==0].index.tolist()

            if excluded and not manual_weights:
                st.warning(f'Excluded metrics (low correlation): {", ".join(excluded)}')
                sel = [m for m in sel if m not in excluded]

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

            if len(sel)>=2:
                try:
                    kp = WeightedKPCA(kern=kernel_type, gamma=(None if kernel_type=='linear' else gamma))
                    df_sel = df_minutes[sel].dropna()
                    scores = kp.fit_transform(df_sel, weights)
                    idx = df_sel.index
                    
                    df_pca = pd.DataFrame({
                        'Player': df_minutes.loc[idx, 'Player'],
                        'Team': df_minutes.loc[idx, 'Team'],
                        'PCA Score': scores,
                        'Age': df_minutes.loc[idx, 'Age'],
                        'Position': df_minutes.loc[idx, 'Position'],
                        'Data Origin': df_minutes.loc[idx, 'Data Origin'],
                        'Season': df_minutes.loc[idx, 'Season']
                    })

                    st.write('**Feature Weights**')
                    wdf = pd.DataFrame({
                        'Metric': weights.index,
                        'Weight': weights.values
                    }).sort_values('Weight', ascending=False)
                    st.dataframe(wdf.style.format({'Weight':'{:.2f}'}))

                    # Linha corrigida com parêntese fechado
                    af = df_pca['Age'].between(age_min_pca, age_max_pca)
                    pf = (df_pca['Position'].astype(str).apply(lambda x: any(pos in x for pos in sel_pos))) if sel_pos else pd.Series(True, index=df_pca.index)
                    df_f = df_pca[af & pf]

                    if not df_f.empty:
                        mn, mx = df_f['PCA Score'].min(), df_f['PCA Score'].max()
                        sr = st.slider(
                            'Filter PCA Score range',
                            min_value=float(mn),
                            max_value=float(mx),
                            value=(float(mn), float(mx))
                        )
                        
                        df_final = df_f[df_f['PCA Score'].between(*sr)]
                        if df_final.empty:
                            st.warning('No players in the selected PCA score range.')
                        else:
                            st.write(f'**Matching Players ({len(df_final)})**')
                            st.dataframe(df_final.sort_values('PCA Score', ascending=False).reset_index(drop=True))
                            
                            st.write('**Score Distribution**')
                            fig_pca = go.Figure(data=[go.Bar(x=df_final['Player'], y=df_final['PCA Score'])])
                            
                            title_text = (f"<b>PCA Scores</b><br>"
                                         f"<sup>Context: {context['leagues']} ({context['seasons']})<br>"
                                         f"Filters: Age {context['min_age']}-{context['max_age']} | "
                                         f"{context['positions']} | Metrics: {len(sel)} selected</sup>")
                            
                            fig_pca.update_layout(
                                title=dict(text=title_text, x=0.03, xanchor='left', font=dict(size=18)),
                                template='plotly_dark',
                                margin=dict(t=200, b=100, l=100, r=100)
                            )
                            st.plotly_chart(fig_pca)
                            
                            if st.button('Export PCA Scores (300 DPI)', key='export_pca'):
                                fig_pca.update_layout(margin=dict(t=250))
                                img_bytes = fig_pca.to_image(format="png", width=1600, height=900, scale=3)
                                st.download_button(
                                    "⬇️ Download PCA Chart", 
                                    data=img_bytes, 
                                    file_name="pca_scores.png", 
                                    mime="image/png"
                                )
                            
                            # Export Excel
                            bio = BytesIO()
                            with pd.ExcelWriter(bio, engine='xlsxwriter') as writer:
                                df_final.to_excel(writer, sheet_name='PCA Results', index=False)
                            bio.seek(0)
                            st.download_button(
                                '📥 Download Results as Excel',
                                data=bio,
                                file_name='pca_results.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                    else:
                        st.warning('No players match the current filters.')
                except Exception as e:
                    st.error(f'PCA calculation error: {str(e)}')
            else:
                st.error('Insufficient valid metrics after filtering.')

    except Exception as e:
        st.error(f'Error: {e}')
else:
    st.info('Please upload up to 15 Wyscout Excel files to start the analysis')
    st.warning("⚠️ Required: `pip install kaleido==0.2.1.post1`")
