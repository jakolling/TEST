# Football Analytics App - Enhanced Version with mplsoccer
# Player Comparison and Similarity Analysis System - v4.0

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import io
from io import BytesIO
import random
from mplsoccer import Radar, FontManager, grid, PyPizza
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cdist
import base64
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

# =============================================
# Configuration
# =============================================
st.set_page_config(
    page_title='Football Analytics',
    layout='wide',
    page_icon="⚽"
)

roboto_condensed = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/roboto/'
                              'Roboto%20Condensed.ttf')

plt.rcParams['font.family'] = 'sans-serif'
LOGO_PATH = "Vålerenga_Oslo_logo.svg.png"

# =============================================
# Helper Functions
# =============================================
if 'file_metadata' not in st.session_state:
    st.session_state.file_metadata = {}

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
    
def safe_operation(func, error_msg, fallback=None, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"{error_msg}: {str(e)}")
        return fallback

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

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def create_pizza_chart(params=None, values_p1=None, values_p2=None, values_avg=None, 
                       title=None, subtitle=None, p1_name="Player 1", p2_name="Player 2"):
    try:
        if params is None:
            params = [
                "xG per 90",
                "Shots on target, %",
                "Goal conversion, %",
                "Touches in box per 90",
                "Progressive runs per 90",
                "Explosive Acceleration to Sprint P90",
                "M/min P90",
                "Sprint Count P90",
                "Accurate passes, %",
                "Deep completions per 90",
                "Progressive passes per 90",
                "Key passes per 90"
            ]
            if values_p1 is None:
                values_p1 = [random.randint(50, 95) for _ in range(len(params))]
        
        if values_p1 is not None and len(params) != len(values_p1):
            raise ValueError(f"Parâmetros ({len(params)}) ≠ Valores ({len(values_p1)})")
        
        if values_p1 is not None:
            values_p1 = [round(v) for v in values_p1]
        if values_p2 is not None:
            values_p2 = [round(v) for v in values_p2]
        if values_avg is not None:
            values_avg = [round(v) for v in values_avg]
        
        player1_color = "#0052CC"
        player2_color = "#00A3FF"
        avg_color = "#B3CFFF"
        text_color = "#000000"
        background_color = "#F5F5F5"
        
        min_values = [0] * len(params)
        max_values = [100] * len(params)
        
        baker = PyPizza(
            params=params,
            background_color=background_color,
            straight_line_color="#CCCCCC",
            straight_line_lw=1,
            last_circle_lw=1,
            other_circle_lw=1,
            other_circle_ls="-",
            inner_circle_size=30
        )
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=background_color, subplot_kw={"projection": "polar"})
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
        ax.set_ylim(0, 0.9)
        
        circles = [0.25, 0.5, 0.75]
        for circle in circles:
            ax.plot(np.linspace(0, 2*np.pi, 100), [circle] * 100, 
                    color='#AAAAAA', linestyle='-', linewidth=0.8, zorder=1, alpha=0.7)
        
        baker = PyPizza(
            params=params,
            min_range=min_values,
            max_range=max_values,
            background_color=background_color,
            straight_line_color="#999999",
            straight_line_lw=1.2,
            last_circle_lw=1.5,
            other_circle_lw=1,
            other_circle_ls="-",
            inner_circle_size=30
        )
        
        baker.make_pizza(
            values_p1,
            ax=ax,
            color_blank_space="same",
            slice_colors=[player1_color] * len(params),
            value_colors=["#FF0000"] * len(params),
            value_bck_colors=["#FFFFFF"] * len(params),
            blank_alpha=0.4,
            kwargs_slices=dict(
                edgecolor="#F2F2F2", zorder=2, linewidth=1
            ),
            kwargs_params=dict(
                color="#000000",
                fontsize=8,
                fontweight="normal",
                va="center",
                zorder=3,
                fontproperties=roboto_condensed.prop
            ),
            kwargs_values=dict(
                color="#FF0000",
                fontsize=8,
                fontweight="bold",
                zorder=5,
                bbox=dict(
                    edgecolor="#000000",
                    facecolor="#FFFFFF",
                    boxstyle="round,pad=0.1",
                    lw=0.5,
                    alpha=0.9
                )
            )
        )
        
        if values_p2 is not None:
            for i, value in enumerate(values_p2):
                angle = (i / len(params)) * 2 * np.pi
                ax.plot([angle, angle], [0, value/100], color=player2_color, 
                        linewidth=2.5, linestyle='-', zorder=10)
                
                if value > 25:
                    radius = value / 100
                    ax.text(angle, radius + 0.05, f"{value}", color='#FF0000', 
                            fontsize=9, ha='center', va='center', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='#FFFFFF', 
                                    alpha=0.9, edgecolor='#000000', linewidth=1))
        
        if values_avg is not None:
            for i, value in enumerate(values_avg):
                angle = (i / len(params)) * 2 * np.pi
                ax.plot([angle, angle], [0, value/100], color=avg_color, 
                        linewidth=2, linestyle='--', zorder=5, alpha=0.7)
        
        if title:
            title_text = title
        else:
            title_text = f"{p1_name}" + (f" vs {p2_name}" if values_p2 is not None else "")
        
        fig.text(
            0.5, 0.97, title_text, 
            size=16, ha="center", fontweight="bold", color="#000000"
        )
        
        if subtitle:
            fig.text(
                0.5, 0.93, subtitle,
                size=12, ha="center", color="#666666"
            )
        
        fig.text(
            0.5, 0.02, "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
            size=10, ha="center", color="#666666"
        )
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if values_p2 is not None or values_avg is not None:
            legend_elements = []
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=player1_color, 
                              edgecolor='white', label=p1_name)
            )
            if values_p2 is not None:
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=player2_color, 
                                  edgecolor='white', label=p2_name)
                )
            if values_avg is not None:
                legend_elements.append(
                    plt.Line2D([0], [0], color=avg_color, linewidth=2, 
                              linestyle='--', label='Média do Grupo')
                )
            ax.legend(
                handles=legend_elements,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.1),
                ncol=len(legend_elements),
                frameon=True,
                facecolor='white',
                edgecolor='#CCCCCC'
            )
        
        try:
            drawing = svg2rlg(LOGO_PATH)
            renderPM.drawToFile(drawing, "temp_logo.png", fmt="PNG")
            logo = plt.imread("temp_logo.png")
            imagebox = OffsetImage(logo, zoom=0.08)
            ab = AnnotationBbox(imagebox, (0,0), frameon=False, xycoords='data', pad=0)
            ax.add_artist(ab)
        except Exception as e:
            st.error(f"Erro ao carregar o logo: {str(e)}")
    
    except Exception as e:
        st.error(f"Erro detalhado: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5, 0.5, f"Erro ao criar pizza chart: {str(e)}", 
                ha='center', va='center', fontsize=12, color='#333333')
        ax.axis('off')
    
    return fig

def create_comparison_pizza_chart(params, values_p1, values_p2=None, values_avg=None,
                       title=None, subtitle=None, p1_name="Player 1", p2_name="Player 2"):
    try:
        if values_p1 is not None and len(params) != len(values_p1):
            raise ValueError(f"Parâmetros ({len(params)}) ≠ Valores ({len(values_p1)})")
        
        compare_values = None
        compare_name = p2_name
        if values_p2 is not None and len(values_p2) == len(params):
            compare_values = values_p2
        elif values_avg is not None and len(values_avg) == len(params):
            compare_values = values_avg
            compare_name = "Group Average"
        
        if compare_values is None:
            raise ValueError("Valores de comparação ausentes")
        
        values_p1 = [round(v) for v in values_p1]
        compare_values = [round(v) for v in compare_values]
        
        player1_color = "#1A78CF"
        player2_color = "#E41A1C"
        avg_color = "#E41A1C"
        text_color = "#000000"
        background_color = "#F5F5F5"
        
        compare_color = player2_color
        
        min_range = [0] * len(params)
        max_range = [100] * len(params)
        
        baker = PyPizza(
            params=params,
            min_range=min_range,
            max_range=max_range,
            background_color=background_color,
            straight_line_color="#999999",
            straight_line_lw=1.2,
            last_circle_lw=1.5,
            other_circle_lw=1,
            other_circle_ls="-",
            inner_circle_size=30
        )
        
        fig, ax = baker.make_pizza(
            values_p1,
            compare_values=compare_values,
            figsize=(10, 10),
            color_blank_space="same",
            blank_alpha=0.4,
            param_location=110,
            kwargs_slices=dict(
                facecolor=player1_color, edgecolor="#F2F2F2",
                zorder=2, linewidth=1
            ),
            kwargs_compare=dict(
                facecolor=compare_color, edgecolor="#000000", 
                zorder=3, linewidth=1, alpha=0.8
            ),
            kwargs_params=dict(
                color="#000000",
                fontsize=8,
                fontweight="normal",
                va="center",
                zorder=3,
                fontproperties=roboto_condensed.prop
            ),
            kwargs_values=dict(
                color=player1_color,
                fontsize=8,
                fontweight="bold",
                zorder=5,
                bbox=dict(
                    edgecolor="#000000",
                    facecolor="#FFFFFF",
                    boxstyle="round,pad=0.1",
                    lw=1,
                    alpha=0.9
                )
            ),
            kwargs_compare_values=dict(
                color=compare_color,
                fontsize=8,
                fontweight="bold",
                zorder=6,
                bbox=dict(
                    edgecolor="#000000",
                    facecolor="#FFFFFF",
                    boxstyle="round,pad=0.1",
                    lw=1,
                    alpha=0.9
                )
            )
        )
        
        baker.adjust_texts([True]*len(params), offset=-0.25)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)
        
        if title:
            title_text = title
        else:
            title_text = f"{p1_name}" + (f" vs {compare_name}" if compare_values is not None else "")
        
        fig.text(
            0.5, 0.97, title_text, 
            size=16, ha="center", fontweight="bold", color="#000000"
        )
        
        if subtitle:
            fig.text(
                0.5, 0.93, subtitle,
                size=12, ha="center", color="#666666"
            )
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        legend_elements = []
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=player1_color, 
                         edgecolor='white', label=p1_name)
        )
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=compare_color, 
                         edgecolor='white', alpha=0.8, label=compare_name)
        )
        
        ax.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(legend_elements),
            frameon=True,
            facecolor='white',
            edgecolor='#CCCCCC'
        )
        
        fig.text(
            0.5, 0.02, "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
            size=10, ha="center", color="#666666"
        )
        
        try:
            drawing = svg2rlg(LOGO_PATH)
            renderPM.drawToFile(drawing, "temp_logo.png", fmt="PNG")
            logo = plt.imread("temp_logo.png")
            imagebox = OffsetImage(logo, zoom=0.08)
            ab = AnnotationBbox(imagebox, (0,0), frameon=False, xycoords='data', pad=0)
            ax.add_artist(ab)
        except Exception as e:
            st.error(f"Erro ao carregar o logo: {str(e)}")
    
    except Exception as e:
        st.error(f"Erro na criação do pizza chart comparativo: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5, 0.5, f"Erro ao criar pizza chart: {str(e)}", 
                ha='center', va='center', fontsize=12, color='#333333')
        ax.axis('off')
    
    return fig

def create_bar_chart(metrics, p1_name, p1_values, p2_name, p2_values, avg_values,
                    title=None, subtitle=None):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)))
    
    if len(metrics) == 1:
        axes = [axes]
    
    player1_color = "#1A78CF"
    player2_color = "#E41A1C"
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        y_pos = [0, 1]
        ax.barh(y_pos[0], p1_values[i], color=player1_color, height=0.6)
        ax.barh(y_pos[1], p2_values[i], color=player2_color, height=0.6)
        
        ax.axvline(x=avg_values[i], color='#2ca02c', linestyle='--', alpha=0.7)
        ax.text(avg_values[i], 0.5, f'Group Avg', 
                va='center', ha='right', rotation=90, color='#2ca02c',
                backgroundcolor='white', alpha=0.8)
        
        ax.set_title(metrics[i], fontsize=12, pad=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([p1_name, p2_name])
        
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        ax.text(p1_values[i], y_pos[0], f' {p1_values[i]:.2f}', va='center', 
               color=player1_color, fontweight='bold')
        ax.text(p2_values[i], y_pos[1], f' {p2_values[i]:.2f}', va='center', 
               color=player2_color, fontweight='bold')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    if subtitle:
        plt.figtext(0.5, 0.99, subtitle, ha='center', fontsize=10, wrap=True)
        
    plt.tight_layout()
    return fig

def create_scatter_plot(df, x_metric, y_metric, highlight_players=None, title=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    player1_color = "#1A78CF"
    player2_color = "#E41A1C"
    
    player_names = {}
    
    all_players = df['Player'].values
    sc = ax.scatter(df[x_metric], df[y_metric], alpha=0.5, s=50, c='gray')
    
    for i, player in enumerate(all_players):
        if i < len(df[x_metric]) and i < len(df[y_metric]):
            player_names[i] = player
    
    if highlight_players:
        colors = [player1_color, player2_color]
        
        for idx, (player, color) in enumerate(highlight_players.items()):
            player_data = df[df['Player'] == player]
            if not player_data.empty:
                if idx < 2:
                    use_color = colors[idx]
                else:
                    use_color = color
                
                ax.scatter(player_data[x_metric], player_data[y_metric], 
                          s=100, c=use_color, edgecolor='black', label=player)
                
                ax.annotate(player, 
                           (player_data[x_metric].iloc[0], player_data[y_metric].iloc[0]),
                           xytext=(10, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color=use_color)
    
    import mpld3
    from mpld3 import plugins
    
    tooltip = plugins.PointHTMLTooltip(sc, 
                                       labels=[f"<b>{p}</b><br>x: {df[x_metric].iloc[i]:.2f}<br>y: {df[y_metric].iloc[i]:.2f}" 
                                               for i, p in player_names.items()],
                                       voffset=10, hoffset=10)
    mpld3.plugins.connect(fig, tooltip)
    
    ax.axvline(df[x_metric].mean(), color='#333333', linestyle='--', alpha=0.5)
    ax.axhline(df[y_metric].mean(), color='#333333', linestyle='--', alpha=0.5)
    
    ax.set_xlabel(x_metric, fontsize=12)
    ax.set_ylabel(y_metric, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    if highlight_players:
        ax.legend(loc='best')
    
    ax.grid(True, alpha=0.3)
    
    return fig

def create_similarity_viz(selected_player, similar_players, metrics, df):
    if not similar_players:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No similar players found for {selected_player}",
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    try:
        player_data = df[df['Player'] == selected_player].iloc[0]
        
        player1_color = "#1A78CF"
        player2_color = "#E41A1C"
        
        num_players = len(similar_players)
        num_players = min(num_players, 5)
        fig = plt.figure(figsize=(12, num_players * 2.5))
        
        fig.suptitle(f"Players Similar to {selected_player}", fontsize=18, y=0.98)
        plt.figtext(0.5, 0.96, f"Based on metrics: {', '.join(metrics)}", 
                   ha='center', fontsize=12, fontstyle='italic')
        
        gs = fig.add_gridspec(num_players, 2, height_ratios=[2.5]*num_players, 
                             width_ratios=[1, 1.5], hspace=0.4, wspace=0.15)
        
        processed_players = 0
        
        for i, (sim_player, sim_score) in enumerate(similar_players[:num_players]):
            if sim_player not in df['Player'].values:
                continue
                
            sim_data = df[df['Player'] == sim_player].iloc[0]
            processed_players += 1
            
            try:
                values_selected = [calc_percentile(df[m], player_data[m])*100 for m in metrics]
                values_similar = [calc_percentile(df[m], sim_data[m])*100 for m in metrics]
            except Exception as e:
                st.warning(f"Erro ao calcular percentis: {str(e)}")
                continue
            
            radar_ax = fig.add_subplot(gs[i, 0], polar=True)
            
            radar = Radar(
                metrics, 
                min_range=[0]*len(metrics), 
                max_range=[100]*len(metrics),
                round_int=[True]*len(metrics),
                num_rings=4,
                ring_width=1,
                center_circle_radius=1
            )
            
            radar_ax = radar.setup_axis(ax=radar_ax)
            rings_inner = radar.draw_circles(ax=radar_ax, facecolor='#f9f9f9', edgecolor='#c5c5c5')
            
            try:
                radar_poly1, rings_outer1, vertices1 = radar.draw_radar(
                    values_selected, ax=radar_ax, 
                    kwargs_radar={'facecolor': player1_color, 'alpha': 0.6, 'edgecolor': player1_color, 'linewidth': 1.5},
                    kwargs_rings={'facecolor': player1_color, 'alpha': 0.1}
                )
                
                radar_poly2, rings_outer2, vertices2 = radar.draw_radar(
                    values_similar, ax=radar_ax,
                    kwargs_radar={'facecolor': player2_color, 'alpha': 0.6, 'edgecolor': player2_color, 'linewidth': 1.5},
                    kwargs_rings={'facecolor': player2_color, 'alpha': 0.1}
                )
                
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=player1_color, 
                            markersize=10, label=selected_player),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=player2_color, 
                            markersize=10, label=sim_player)
                ]
                radar_ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1.05), fontsize=9)
                
            except Exception as radar_error:
                radar_ax.clear()
                radar_ax.text(0.5, 0.5, f"Não foi possível criar o radar: {str(radar_error)}", 
                            ha='center', va='center', transform=radar_ax.transAxes,
                            fontsize=10, wrap=True)
                radar_ax.axis('off')
            
            info_ax = fig.add_subplot(gs[i, 1])
            info_ax.axis('off')
            
            info_text = f"""
            Player: {sim_player}
            Team: {sim_data.get('Team', 'N/A')}
            Age: {sim_data.get('Age', 'N/A')}
            Position: {sim_data.get('Position', 'N/A')}
            Similarity Score: {sim_score:.2f}
            
            Minutes played: {sim_data.get('Minutes played', 'N/A')}
            """
            
            if 'Minutes per game' in sim_data:
                info_text += f"Minutes per game: {sim_data['Minutes per game']:.1f}\n"
            
            info_text += "\nMetric comparison:\n"
            for metric in metrics[:10]:
                p1_value = player_data.get(metric, 0)
                p2_value = sim_data.get(metric, 0)
                info_text += f"{metric}: {p2_value:.2f} (Selected: {p1_value:.2f})\n"
            
            if len(metrics) > 10:
                info_text += "... and more metrics"
            
            info_ax.text(0, 1, info_text, va='top', ha='left', fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', 
                               boxstyle="round,pad=1", linewidth=1))
        
        if processed_players == 0:
            raise ValueError("Nenhum jogador similar pôde ser processado")
            
        fig.text(
            0.5, 0.01, "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
            size=9, ha="center", color="#666666"
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Não foi possível gerar a visualização de similaridade:\n{str(e)}", 
               ha='center', va='center', fontsize=14, wrap=True)
        ax.axis('off')
    
    return fig

def compute_player_similarity(df, player, metrics, n=5, method='cosine'):
    if player not in df['Player'].values:
        st.error(f"Player '{player}' not found!")
        return []
    
    valid_metrics = [m for m in metrics if m in df.columns]
    if len(valid_metrics) != len(metrics):
        missing = set(metrics) - set(valid_metrics)
        st.warning(f"Missing metrics: {missing}")
        if not valid_metrics:
            return []
    
    X = df[valid_metrics].values
    
    if np.isnan(X).any():
        st.warning("Filling missing values with column means")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    player_data = df[df['Player'] == player]
    if player_data.empty:
        st.error(f"Player '{player}' not found")
        return []
    
    player_idx = player_data.index[0]
    
    if player_idx >= X_scaled.shape[0]:
        st.error(f"Player index out of bounds")
        return []
    
    try:
        if method == 'cosine':
            similarities = cosine_similarity([X_scaled[player_idx]], X_scaled)[0]
            similarities = (similarities + 1) / 2
        else:
            distances = cdist([X_scaled[player_idx]], X_scaled, 'euclidean')[0]
            max_dist = np.max(distances)
            similarities = 1 - (distances / max_dist)
        
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = [i for i in similar_indices if i != player_idx][:n]
        
        similar_players = [(df.iloc[i]['Player'], similarities[i]) for i in similar_indices]
        return similar_players
    
    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return []

# =============================================
# Main Application
# =============================================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("⚽ Football Analytics")

st.header('Technical Scouting Department')
st.subheader('Football Analytics Dashboard')
st.caption("Created by João Alberto Kolling | Enhanced Player Analysis System v4.0")

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
        df = safe_operation(
            load_and_clean, 
            "Error loading and processing files",
            pd.DataFrame(),
            uploaded_files
        )
        
        if df.empty:
            st.error("Failed to load data from uploaded files")
            st.stop()
            
        required_cols = ['Player', 'Minutes played', 'Matches played', 'Age']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            st.stop()

        try:
            min_min, max_min = int(df['Minutes played'].min()), int(df['Minutes played'].max())
            minutes_range = st.sidebar.slider('Minutes Played', min_min, max_min, (min_min, max_min))
            df_minutes = df[df['Minutes played'].between(*minutes_range)].copy()
            
            if df_minutes.empty:
                st.warning("No players match the minutes range")
                df_minutes = df.copy()
                minutes_range = (min_min, max_min)
        except Exception as e:
            st.error(f"Error filtering by minutes: {str(e)}")
            df_minutes = df.copy()
            minutes_range = (df['Minutes played'].min(), df['Minutes played'].max())
        
        try:
            df_minutes['Minutes per game'] = df_minutes['Minutes played'] / df_minutes['Matches played'].replace(0, np.nan)
            df_minutes['Minutes per game'] = df_minutes['Minutes per game'].fillna(0).clip(0, 120)
            
            min_mpg, max_mpg = int(df_minutes['Minutes per game'].min()), int(df_minutes['Minutes per game'].max())
            mpg_range = st.sidebar.slider('Minutes per Game', min_mpg, max_mpg, (min_mpg, max_mpg))
            df_filtered = df_minutes[df_minutes['Minutes per game'].between(*mpg_range)]
            
            if df_filtered.empty:
                st.warning("No players match the minutes per game range")
                df_filtered = df_minutes
                mpg_range = (min_mpg, max_mpg)
            else:
                df_minutes = df_filtered
        except Exception as e:
            st.error(f"Error calculating minutes per game: {str(e)}")
            mpg_range = (0, 120)

        try:
            min_age, max_age = int(df_minutes['Age'].min()), int(df_minutes['Age'].max())
            age_range = st.sidebar.slider('Age Range', min_age, max_age, (min_age, max_age))
            df_filtered = df_minutes[df_minutes['Age'].between(*age_range)]
            
            if df_filtered.empty:
                st.warning("No players match the age range")
                age_range = (min_age, max_age)
            else:
                df_minutes = df_filtered
        except Exception as e:
            st.error(f"Error filtering by age: {str(e)}")
            age_range = (df_minutes['Age'].min(), df_minutes['Age'].max())

        if 'Position' in df_minutes.columns:
            df_minutes['Position_split'] = df_minutes['Position'].astype(str).apply(lambda x: [p.strip() for p in x.split(',')])
            all_pos = sorted({p for lst in df_minutes['Position_split'] for p in lst})
            sel_pos = st.sidebar.multiselect('Positions', all_pos, default=all_pos)
        else:
            sel_pos = []

        if 'Position_split' in df_minutes.columns and sel_pos:
            df_group = df_minutes[df_minutes['Position_split'].apply(lambda x: any(pos in x for pos in sel_pos))]
        else:
            df_group = df_minutes.copy()

        context = get_context_info(df_minutes, minutes_range, mpg_range, age_range, sel_pos)
        players = sorted(df_minutes['Player'].unique())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Age', 'Minutes played', 'Matches played', 'Minutes per game']
        metric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        tabs = st.tabs(['Pizza Chart', 'Bars', 'Scatter', 'Similarity', 'Correlation', 'Composite Index (PCA)'])

        with tabs[0]:
            st.header('Pizza Chart')
            
            col1, col2 = st.columns(2)
            with col1:
                p1 = st.selectbox('Select Player 1', players)
            with col2:
                p2 = st.selectbox('Select Player 2', [p for p in players if p != p1])
                
            sel = st.multiselect('Metrics for Pizza Chart (6-15)', metric_cols, default=metric_cols[:9])
            
            if 6 <= len(sel) <= 15:
                d1 = df_minutes[df_minutes['Player']==p1].iloc[0]
                d2 = df_minutes[df_minutes['Player']==p2].iloc[0]
                
                p1pct = [calc_percentile(df_minutes[m], d1[m])*100 for m in sel]
                p2pct = [calc_percentile(df_minutes[m], d2[m])*100 for m in sel]
                
                gm = {m: df_group[m].mean() for m in sel}
                gmpct = [calc_percentile(df_minutes[m], gm[m])*100 for m in sel]
                
                st.subheader("Display Options")
                
                comparison_option = st.radio(
                    "Compare with:", 
                    [f"No comparison (only {p1})", 
                     f"Player vs Player ({p1} vs {p2})", 
                     f"Player vs Group Average ({p1} vs Avg)"],
                    index=0
                )
                
                if comparison_option == f"No comparison (only {p1})":
                    show_p2 = False
                    show_avg = False
                    title = f"{p1}"
                elif comparison_option == f"Player vs Player ({p1} vs {p2})":
                    show_p2 = True
                    show_avg = False
                    title = f"{p1} vs {p2}"
                else:
                    show_p2 = False
                    show_avg = True
                    title = f"{p1} vs Group Average"
                
                subtitle = (f"Percentile Rank | {context['leagues']} | "
                          f"{context['min_min']}+ mins | Position: {context['positions']}")
                
                values_p1_arg = p1pct if show_p1 else None
                values_p2_arg = p2pct if show_p2 else None
                values_avg_arg = gmpct if show_avg else None
                
                use_comparison_chart = False
                if show_p1 and (show_p2 or show_avg):
                    use_comparison_chart = True
                
                if use_comparison_chart:
                    fig = create_comparison_pizza_chart(
                        params=sel,
                        values_p1=values_p1_arg, 
                        values_p2=values_p2_arg,
                        values_avg=values_avg_arg,
                        title=title,
                        subtitle=subtitle,
                        p1_name=p1,
                        p2_name=p2
                    )
                else:
                    fig = create_pizza_chart(
                        params=sel,
                        values_p1=values_p1_arg, 
                        values_p2=values_p2_arg,
                        values_avg=values_avg_arg,
                        title=title,
                        subtitle=subtitle,
                        p1_name=p1,
                        p2_name=p2
                    )
                
                st.pyplot(fig)
                
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
                
                if st.button('Export Pizza Chart (300 DPI)', key='export_pizza'):
                    img_bytes = fig_to_bytes(fig)
                    players_str = p1
                    if show_p2:
                        players_str += f"_vs_{p2}"
                    if show_avg and not show_p2:
                        players_str += "_vs_avg"
                    st.download_button(
                        "⬇️ Download Pizza Chart", 
                        data=img_bytes, 
                        file_name=f"pizza_chart_{players_str}.png", 
                        mime="image/png"
                    )

        with tabs[1]:
            st.header('Bar Chart Comparison')
            
            col1, col2 = st.columns(2)
            with col1:
                if 'p1' not in locals():
                    p1 = st.selectbox('Select Player 1', players, key='bar_p1')
                else:
                    p1 = st.selectbox('Select Player 1', players, index=players.index(p1), key='bar_p1')
            with col2:
                if 'p2' not in locals():
                    p2 = st.selectbox('Select Player 2', [p for p in players if p != p1], key='bar_p2')
                else:
                    remaining_players = [p for p in players if p != p1]
                    if p2 in remaining_players:
                        p2 = st.selectbox('Select Player 2', remaining_players, 
                                         index=remaining_players.index(p2), key='bar_p2')
                    else:
                        p2 = st.selectbox('Select Player 2', remaining_players, key='bar_p2')
            
            selected_metrics = st.multiselect('Select metrics (max 5)', metric_cols, 
                                            default=metric_cols[:1], key='bar_metrics')
            
            if len(selected_metrics) > 5:
                st.error("Maximum 5 metrics allowed!")
            
            elif len(selected_metrics) >= 1:
                d1 = df_minutes[df_minutes['Player']==p1].iloc[0]
                d2 = df_minutes[df_minutes['Player']==p2].iloc[0]
                
                p1_values = [d1[m] for m in selected_metrics]
                p2_values = [d2[m] for m in selected_metrics]
                avg_values = [df_group[m].mean() for m in selected_metrics]
                
                title = "Metric Comparison"
                subtitle = (f"Context: {context['leagues']} ({context['seasons']}) | "
                          f"Players: {context['total_players']} | Filters: {context['min_age']}-{context['max_age']} years")
                
                fig = create_bar_chart(
                    metrics=selected_metrics,
                    p1_name=p1,
                    p1_values=p1_values,
                    p2_name=p2,
                    p2_values=p2_values,
                    avg_values=avg_values,
                    title=title,
                    subtitle=subtitle
                )
                
                st.pyplot(fig)
                
                if st.button('Export Bar Chart (300 DPI)', key='export_bar'):
                    img_bytes = fig_to_bytes(fig)
                    st.download_button(
                        "⬇️ Download Bar Chart", 
                        data=img_bytes, 
                        file_name=f"bar_{p1}_vs_{p2}.png", 
                        mime="image/png"
                    )

        with tabs[2]:
            st.header('Scatter Plot Analysis')
            
            col1, col2 = st.columns(2)
            with col1:
                x_metric = st.selectbox('X-Axis Metric', metric_cols, index=0)
            with col2:
                y_metric = st.selectbox('Y-Axis Metric', metric_cols, index=min(1, len(metric_cols)-1))
            
            col1, col2 = st.columns(2)
            with col1:
                if 'p1' not in locals():
                    p1 = st.selectbox('Highlight Player 1', ['None'] + players, key='scatter_p1')
                else:
                    p1 = st.selectbox('Highlight Player 1', ['None'] + players, 
                                     index=players.index(p1)+1 if p1 in players else 0, key='scatter_p1')
            with col2:
                if 'p2' not in locals():
                    p2 = st.selectbox('Highlight Player 2', ['None'] + [p for p in players if p != p1], key='scatter_p2')
                else:
                    remaining_players = ['None'] + [p for p in players if p != p1]
                    if p2 in players and p2 != p1:
                        p2 = st.selectbox('Highlight Player 2', remaining_players, 
                                         index=remaining_players.index(p2), key='scatter_p2')
                    else:
                        p2 = st.selectbox('Highlight Player 2', remaining_players, key='scatter_p2')
            
            highlight_players = {}
            if p1 != 'None':
                highlight_players[p1] = '#1f77b4'
            if p2 != 'None':
                highlight_players[p2] = '#ff7f0e'
            
            title = f"Scatter Analysis: {x_metric} vs {y_metric}"
            subtitle = f"League(s): {context['leagues']} | Season(s): {context['seasons']}"
            
            fig = create_scatter_plot(
                df=df_group,
                x_metric=x_metric,
                y_metric=y_metric,
                highlight_players=highlight_players,
                title=title
            )
            
            st.pyplot(fig)
            
            corr = df_group[x_metric].corr(df_group[y_metric])
            st.info(f"Correlation coefficient: {corr:.4f}")
            
            if st.button('Export Scatter Plot (300 DPI)', key='export_scatter'):
                img_bytes = fig_to_bytes(fig)
                st.download_button(
                    "⬇️ Download Scatter Plot", 
                    data=img_bytes, 
                    file_name=f"scatter_{x_metric}_vs_{y_metric}.png", 
                    mime="image/png"
                )

        with tabs[3]:
            st.header('Player Similarity Analysis')
            
            col1, col2 = st.columns(2)
            with col1:
                sim_player = st.selectbox('Select Reference Player', players, key='sim_player')
            with col2:
                similarity_method = st.selectbox('Similarity Method', 
                                              ['Cosine Similarity', 'Euclidean Distance'], 
                                              index=0)
            
            st.subheader("Select Metrics for Similarity Calculation")
            sim_metric_options = st.multiselect(
                'Choose metrics that define player style (3-8 recommended)', 
                metric_cols, 
                default=metric_cols[:4]
            )
            
            num_similar = st.slider('Number of similar players to show', 1, 10, 5)
            
            filter_by_position = st.checkbox('Filter similar players by reference player position', True)
            
            filter_by_age = st.checkbox('Filter similar players by age range', False)
            if filter_by_age:
                ref_player_age = df_minutes[df_minutes['Player'] == sim_player]['Age'].iloc[0]
                age_diff = st.slider('Maximum age difference (years)', 0, 10, 3)
                min_age_filter = ref_player_age - age_diff
                max_age_filter = ref_player_age + age_diff
                
                df_sim = df_minutes[df_minutes['Age'].between(min_age_filter, max_age_filter)]
            else:
                df_sim = df_minutes.copy()
            
            if filter_by_position and 'Position_split' in df_minutes.columns:
                ref_player_pos = df_minutes[df_minutes['Player'] == sim_player]['Position_split'].iloc[0]
                df_sim = df_sim[df_sim['Position_split'].apply(
                    lambda x: any(pos in ref_player_pos for pos in x))]
            
            if len(sim_metric_options) >= 2 and len(df_sim) > 1:
                method = 'cosine' if similarity_method == 'Cosine Similarity' else 'euclidean'
                similar_players = compute_player_similarity(
                    df=df_sim,
                    player=sim_player,
                    metrics=sim_metric_options,
                    n=num_similar,
                    method=method
                )
                
                st.subheader("Most Similar Players")
                sim_df = pd.DataFrame(similar_players, columns=['Player', 'Similarity Score'])
                sim_df['Similarity Score'] = sim_df['Similarity Score'].apply(lambda x: f"{x:.2f}")
                st.dataframe(sim_df, use_container_width=True)
                
                st.subheader("Similarity Visualization")
                fig = create_similarity_viz(
                    selected_player=sim_player,
                    similar_players=similar_players,
                    metrics=sim_metric_options,
                    df=df_minutes
                )
                
                st.pyplot(fig)
                
                if st.button('Export Similarity Analysis (300 DPI)', key='export_similarity'):
                    img_bytes = fig_to_bytes(fig)
                    st.download_button(
                        "⬇️ Download Similarity Analysis", 
                        data=img_bytes, 
                        file_name=f"similarity_{sim_player}.png", 
                        mime="image/png"
                    )
            else:
                if len(sim_metric_options) < 2:
                    st.warning("Please select at least 2 metrics")
                else:
                    st.warning("Not enough players matching the filter criteria")

        with tabs[4]:
            st.header('Metric Correlation Analysis')
            
            corr_metrics = st.multiselect(
                'Select metrics to analyze correlations (2-10)', 
                metric_cols, 
                default=metric_cols[:5],
                key='corr_metrics'
            )
            
            if len(corr_metrics) < 2:
                st.warning("Please select at least 2 metrics")
            elif len(corr_metrics) > 10:
                st.warning("Too many metrics selected. Please limit to 10 or fewer")
            else:
                corr_matrix = df_group[corr_metrics].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient')
                
                ax.set_xticks(np.arange(len(corr_metrics)))
                ax.set_yticks(np.arange(len(corr_metrics)))
                ax.set_xticklabels(corr_metrics, rotation=45, ha="right")
                ax.set_yticklabels(corr_metrics)
                
                for i in range(len(corr_metrics)):
                    for j in range(len(corr_metrics)):
                        text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                    ha="center", va="center", 
                                    color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                
                ax.set_title("Correlation Matrix")
                fig.tight_layout()
                
                st.pyplot(fig)
                
                if st.button('Export Correlation Matrix (300 DPI)', key='export_corr'):
                    img_bytes = fig_to_bytes(fig)
                    st.download_button(
                        "⬇️ Download Correlation Matrix", 
                        data=img_bytes, 
                        file_name="correlation_matrix.png", 
                        mime="image/png"
                    )
                
                st.subheader("Correlation Values")
                st.dataframe(corr_matrix.style.format("{:.2f}").background_gradient(cmap='coolwarm', axis=None))

        with tabs[5]:
            st.header('Composite Index Analysis (PCA)')
            
            pca_metrics = st.multiselect(
                'Select metrics for composite index (3-10 recommended)', 
                metric_cols, 
                default=metric_cols[:5],
                key='pca_metrics'
            )
            
            if len(pca_metrics) < 3:
                st.warning("Please select at least 3 metrics")
            elif len(pca_metrics) > 15:
                st.warning("Too many metrics selected. Please limit to 15 or fewer")
            else:
                X = df_group[pca_metrics].values
                X_scaled = StandardScaler().fit_transform(X)
                
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X_scaled)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=50, c='gray')
                
                p1_idx = df_group[df_group['Player'] == p1].index
                p2_idx = df_group[df_group['Player'] == p2].index
                
                if len(p1_idx) > 0 and p1 != 'None':
                    p1_idx = p1_idx[0]
                    p1_idx_in_group = df_group.index.get_indexer([p1_idx])[0]
                    if p1_idx_in_group >= 0:
                        ax.scatter(pca_result[p1_idx_in_group, 0], pca_result[p1_idx_in_group, 1], 
                                  s=100, c='#1f77b4', edgecolor='black', label=p1)
                        ax.annotate(p1, 
                                  (pca_result[p1_idx_in_group, 0], pca_result[p1_idx_in_group, 1]),
                                  xytext=(10, 5), textcoords='offset points',
                                  fontsize=10, fontweight='bold')
                
                if len(p2_idx) > 0 and p2 != 'None':
                    p2_idx = p2_idx[0]
                    p2_idx_in_group = df_group.index.get_indexer([p2_idx])[0]
                    if p2_idx_in_group >= 0:
                        ax.scatter(pca_result[p2_idx_in_group, 0], pca_result[p2_idx_in_group, 1], 
                                  s=100, c='#ff7f0e', edgecolor='black', label=p2)
                        ax.annotate(p2, 
                                  (pca_result[p2_idx_in_group, 0], pca_result[p2_idx_in_group, 1]),
                                  xytext=(10, 5), textcoords='offset points',
                                  fontsize=10, fontweight='bold')
                
                coeff = pca.components_.T
                feat_xs = coeff[:, 0]
                feat_ys = coeff[:, 1]
                
                scale_factor = 5
                
                for i, (x, y) in enumerate(zip(feat_xs, feat_ys)):
                    plt.arrow(0, 0, x*scale_factor, y*scale_factor, head_width=0.15, head_length=0.2, fc='red', ec='red')
                    plt.text(x*scale_factor*1.1, y*scale_factor*1.1, pca_metrics[i], color='red')
                
                explained_var = pca.explained_variance_ratio_
                ax.set_xlabel(f"PC1 ({explained_var[0]:.2%} variance)", fontsize=12)
                ax.set_ylabel(f"PC2 ({explained_var[1]:.2%} variance)", fontsize=12)
                
                ax.set_title("Principal Component Analysis", fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                
                if p1 != 'None' or p2 != 'None':
                    ax.legend(loc='best')
                
                st.pyplot(fig)
                
                st.info(f"Total explained variance: {sum(explained_var):.2%}")
                
                if st.button('Export PCA Analysis (300 DPI)', key='export_pca'):
                    img_bytes = fig_to_bytes(fig)
                    st.download_button(
                        "⬇️ Download PCA Analysis", 
                        data=img_bytes, 
                        file_name="pca_analysis.png", 
                        mime="image/png"
                    )
                
                st.subheader("PCA Loadings (Feature Importance)")
                loadings = pd.DataFrame(
                    pca.components_.T, 
                    columns=[f'PC{i+1}' for i in range(2)],
                    index=pca_metrics
                )
                st.dataframe(loadings.style.format("{:.4f}"))

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)
else:
    st.info("👈 Please upload Wyscout Excel files to begin analysis")
    
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
