import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title='Football Analytics',
                   layout='wide',
                   page_icon="⚽")

# Import other libraries after set_page_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
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
import re
import tempfile
from PIL import Image
import pathlib
import zipfile
import unicodedata
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import KernelPCA
import streamlit.components.v1 as components

# Set default matplotlib style
plt.style.use('dark_background')
mpl.rcParams['axes.facecolor'] = '#0E1117'
mpl.rcParams['figure.facecolor'] = '#0E1117'
mpl.rcParams['grid.color'] = 'white'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.alpha'] = 0.3

# Initialize session state variables
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'skillcorner'
if 'selected_leagues_skillcorner' not in st.session_state:
    st.session_state.selected_leagues_skillcorner = []
if 'selected_leagues_wyscout' not in st.session_state:
    st.session_state.selected_leagues_wyscout = []
if 'combine_leagues' not in st.session_state:
    st.session_state.combine_leagues = True
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
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
if 'benchmark_data' not in st.session_state:
    st.session_state.benchmark_data = None
if 'metrics_groups' not in st.session_state:
    st.session_state.metrics_groups = {}
if 'global_percentile_mode' not in st.session_state:
    st.session_state.global_percentile_mode = False
if 'cached_similarity_results' not in st.session_state:
    st.session_state.cached_similarity_results = {}
if 'selected_metrics_preset' not in st.session_state:
    st.session_state.selected_metrics_preset = 'Custom'

# Utility Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_available_leagues(data_source='skillcorner'):
    """
    Load available leagues based on data source
    """
    if data_source == 'skillcorner':
        return {
            "England - Premier League": "England - Premier League",
            "Spain - LaLiga": "Spain - LaLiga",
            "Germany - Bundesliga": "Germany - Bundesliga",
            "Italy - Serie A": "Italy - Serie A", 
            "France - Ligue 1": "France - Ligue 1",
            "Netherlands - Eredivisie": "Netherlands - Eredivisie",
            "Portugal - Liga Portugal": "Portugal - Liga Portugal"
        }
    elif data_source == 'wyscout':
        return {
            "All Leagues": "All Leagues",
            "Custom Upload": "Custom Upload"
        }
    return {}

@st.cache_data
def load_league_data(selected_leagues):
    """
    Load data for selected leagues
    This is a placeholder function - in a real app this would load data from files/APIs
    """
    # In a real implementation, this would load actual football data
    # For now, we'll just return empty DataFrames
    df = pd.DataFrame()
    league_dfs = {}
    
    return df, league_dfs

def safe_operation(func, error_msg, fallback=None, *args, **kwargs):
    """Safely execute a function and return a fallback value on error"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.warning(f"{error_msg}: {str(e)}")
        return fallback

def calculate_offensive_metrics(df):
    """Calculate advanced offensive metrics"""
    # Add npxG (non-penalty xG)
    if 'xG' in df.columns and 'Penalty goals' in df.columns:
        df['npxG'] = df['xG'] - df['Penalty goals']
    elif 'xG' in df.columns:
        df['npxG'] = df['xG']
    
    # Add G-xG (Goals minus Expected Goals)
    if 'Goals' in df.columns and 'xG' in df.columns:
        df['G-xG'] = df['Goals'] - df['xG']
    
    # Add npxG per Shot (non-penalty xG per shot)
    if 'npxG' in df.columns and 'Shots' in df.columns:
        df['npxG per Shot'] = df['npxG'] / df['Shots'].where(df['Shots'] > 0, 0)
    
    return df

def load_and_clean(files):
    """Load and clean data from Excel files"""
    all_dfs = []
    file_metadata = {}
    
    # This is a simplified placeholder - in actual implementation,
    # we'd read Excel files and process the data
    
    # Return placeholders
    df_combined = pd.DataFrame()
    file_metadata_with_stats = {}
    
    return df_combined, file_metadata_with_stats

def calc_percentile(series, value, benchmark_series=None):
    """Calculate percentile of a value in a series"""
    if series.empty:
        return 0
    
    # Use benchmark series if provided, otherwise use the original series
    if benchmark_series is not None:
        calc_series = benchmark_series
    else:
        calc_series = series
    
    # Handle edge cases
    if calc_series.nunique() <= 1:
        return 50  # Default to 50% if all values are the same
    
    # Calculate percentile 
    return round(sum(calc_series <= value) / len(calc_series) * 100)

def apply_benchmark_filter(benchmark_df, 
                          minutes_range, 
                          mpg_range, 
                          age_range, 
                          positions):
    """
    Apply the same filters to benchmark database as applied to main database
    """
    filtered_df = benchmark_df.copy()
    
    # This is a simplified placeholder - in actual implementation,
    # we'd apply the filters to the benchmark dataframe
    
    return filtered_df

def get_context_info(df, minutes_range, mpg_range, age_range, sel_pos):
    """Get context information for current filters"""
    # Count players
    num_players = len(df)
    
    # Get age range
    min_age = df['Age'].min() if 'Age' in df.columns else 'N/A'
    max_age = df['Age'].max() if 'Age' in df.columns else 'N/A'
    
    # Get leagues
    leagues = df['League'].unique() if 'League' in df.columns else ['N/A']
    
    return {
        'num_players': num_players,
        'age_range': f"{min_age:.1f}-{max_age:.1f}" if isinstance(min_age, (int, float)) else 'N/A',
        'leagues': ', '.join(leagues),
        'positions': sel_pos
    }

def abbreviate_metric_name(metric):
    """Abbreviate long metric names for better display in charts"""
    # Common abbreviations
    abbrev = {
        'Progressive passes': 'Prog passes',
        'Progressive runs': 'Prog runs',
        'Successful defensive actions': 'Def actions',
        'Defensive duels won, %': 'Def duels %',
        'Offensive duels won, %': 'Off duels %',
        'Aerial duels won, %': 'Aerial duels %',
        'Progressive passes per 90': 'Prog passes p90',
        'Successful defensive actions per 90': 'Def actions p90',
        'npxG per 90': 'npxG p90',
        'npxG per Shot': 'npxG/Shot'
    }
    
    if metric in abbrev:
        return abbrev[metric]
    
    # If not in predefined list, keep it short
    if len(metric) > 15:
        words = metric.split()
        if len(words) > 2:
            return ' '.join(w[0] for w in words[:-1]) + ' ' + words[-1]
        else:
            return metric[:15] + '...'
    
    return metric

def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for Streamlit"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def create_pizza_chart(params=None,
                      values_p1=None,
                      values_p2=None,
                      values_avg=None,
                      title=None,
                      subtitle=None,
                      p1_name="Player 1",
                      p2_name="Player 2"):
    """
    Creates a professional pizza chart using PyPizza with team logo in the center
    """
    try:
        # Default metrics if not provided
        if params is None:
            params = [
                "xG per 90", "Shots on target, %", "Goal conversion, %",
                "Touches in box per 90", "Progressive runs per 90",
                "Sprint Count P90", "Accurate passes, %",
                "Progressive passes per 90", "Key passes per 90"
            ]
            # Create random values for demonstration if no data
            if values_p1 is None:
                values_p1 = [
                    random.randint(50, 95) for _ in range(len(params))
                ]

        # Safety checks
        if values_p1 is not None and len(params) != len(values_p1):
            raise ValueError(
                f"Number of parameters ({len(params)}) doesn't match number of values ({len(values_p1)})"
            )

        # Round values to integers
        if values_p1 is not None:
            values_p1 = [round(v) for v in values_p1]
        if values_p2 is not None:
            values_p2 = [round(v) for v in values_p2]
        if values_avg is not None:
            values_avg = [round(v) for v in values_avg]

        # Color scheme
        player1_color = "#0052CC"  # Royal blue
        player2_color = "#00A3FF"  # Light blue
        avg_color = "#B3CFFF"      # Very light blue for average
        text_color = "#000000"     # Black for text
        background_color = "#F5F5F5"  # Light gray for background

        # Min and max for each parameter
        min_values = [0] * len(params)
        max_values = [100] * len(params)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 8),
                               facecolor=background_color,
                               subplot_kw={"projection": "polar"},
                               dpi=100)

        # Set higher DPI for export, but smaller size for display
        fig.set_dpi(300)

        # Center and adjust figure with more space for legend
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)

        # Limit chart size (reduce radius)
        ax.set_ylim(0, 0.9)

        # Values to plot
        values = values_p1

        # Same colors for all slices
        slice_colors = [player1_color] * len(params)
        text_colors = ["#FF0000"] * len(params)  # Red text for values

        # Improve grid
        # Create more defined reference circles (25%, 50%, 75%)
        circles = [0.25, 0.5, 0.75]
        for circle in circles:
            ax.plot(np.linspace(0, 2 * np.pi, 100), [circle] * 100,
                    color='#AAAAAA',
                    linestyle='-',
                    linewidth=0.8,
                    zorder=1,
                    alpha=0.7)

        # Abbreviate parameter labels for better display
        abbreviated_params = [
            abbreviate_metric_name(param) for param in params
        ]

        # Create PyPizza instance
        baker = PyPizza(
            params=abbreviated_params,
            min_range=min_values,
            max_range=max_values,
            background_color=background_color,
            straight_line_color="#FFFFFF",
            straight_line_lw=1.5,
            last_circle_lw=1.5,
            other_circle_color="#FFFFFF",
            other_circle_lw=1.5,
            other_circle_ls="-",
            inner_circle_size=10
        )

        # Create the pizza for player 1
        baker.make_pizza(
            values,
            ax=ax,
            color_blank_space="same",
            slice_colors=slice_colors,
            value_colors=text_colors,
            value_bck_colors=["#FFFFFF"] * len(params),
            blank_alpha=0.4,
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

        # Add player 2 if provided
        if values_p2 is not None:
            if len(values_p2) != len(params):
                raise ValueError(
                    f"Number of values for player 2 ({len(values_p2)}) doesn't match number of parameters ({len(params)})"
                )

            # Add lines for player 2
            for i, value in enumerate(values_p2):
                angle = (i / len(params)) * 2 * np.pi
                ax.plot([angle, angle], [0, value / 100],
                        color=player2_color,
                        linewidth=2.5,
                        linestyle='-',
                        zorder=10)

                # Add value in box for player 2
                if value > 25:  # Show only relevant values
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

        # Add group average if provided
        if values_avg is not None:
            if len(values_avg) != len(params):
                raise ValueError(
                    f"Number of average values ({len(values_avg)}) doesn't match number of parameters ({len(params)})"
                )

            # Add lines for average
            for i, value in enumerate(values_avg):
                angle = (i / len(params)) * 2 * np.pi
                ax.plot([angle, angle], [0, value / 100],
                        color=avg_color,
                        linewidth=2,
                        linestyle='--',
                        zorder=5,
                        alpha=0.7)

        # Add centered title
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

        # Add centered subtitle
        if subtitle:
            fig.text(0.5,
                     0.93,
                     subtitle,
                     size=12,
                     ha="center",
                     color="#666666")

        # Add credits in bottom right in italic
        fig.text(0.95,
                 0.02,
                 "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
                 size=8,
                 ha="right",
                 color="#666666",
                 style='italic')

        # Remove grid and ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add team logo to center of the chart AFTER creating the pizza
        try:
            print(
                f"Tentando adicionar logo do Vålerenga ao gráfico pizza (DEPOIS DO PLOT)"
            )

            # Load the image file directly
            logo_file = "attached_assets/Vålerenga_Oslo_logo.svg.png"

            if os.path.exists(logo_file):
                # Draw a white circle in the center as background for the logo
                center_circle = Circle((0, 0),
                                     0.08,
                                     facecolor='white',
                                     edgecolor='none',
                                     zorder=20)
                ax.add_patch(center_circle)

                # Load logo
                logo_img = plt.imread(logo_file)

                # Create a new axes in the center with appropriate size
                logo_ax = fig.add_axes((0.46, 0.46, 0.08, 0.08), zorder=30)

                # Show logo and remove axes
                logo_ax.imshow(logo_img)
                logo_ax.axis('off')

                print(f"Logo adicionado com sucesso DEPOIS da pizza")
            else:
                print(f"Arquivo de logo não encontrado: {logo_file}")
        except Exception as e:
            print(f"Erro ao adicionar logo: {str(e)}")

        # Add legend if needed
        if values_p2 is not None or values_avg is not None:
            legend_elements = []

            # Player 1
            legend_elements.append(
                patches.Rectangle((0, 0),
                              1,
                              1,
                              facecolor=player1_color,
                              edgecolor='white',
                              label=p1_name))

            # Player 2
            if values_p2 is not None:
                legend_elements.append(
                    patches.Rectangle((0, 0),
                                  1,
                                  1,
                                  facecolor=player2_color,
                                  edgecolor='white',
                                  label=p2_name))

            # Average
            if values_avg is not None:
                legend_elements.append(
                    Line2D([0], [0],
                         color=avg_color,
                         linewidth=2,
                         linestyle='--',
                         label='Group Average'))

            # Position legend centered below the chart
            ax.legend(handles=legend_elements,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.1),
                    ncol=len(legend_elements),
                    frameon=True,
                    facecolor='white',
                    edgecolor='#CCCCCC')

    except Exception as e:
        # In case of error, create a chart with error message
        st.error(f"Detailed error: {str(e)}")

        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5,
                0.5,
                f"Error creating pizza chart: {str(e)}",
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
    Creates a pizza chart for comparison between two players or player vs average,
    using the style of the standard chart but with direct overlap of slices.
    Includes logo in the center of the chart.
    """
    try:
        # Safety checks
        if values_p1 is not None and len(params) != len(values_p1):
            raise ValueError(
                f"Number of parameters ({len(params)}) doesn't match number of values ({len(values_p1)})"
            )

        # Determine which values to use for comparison (values_p2 or values_avg)
        compare_values = None
        compare_name = p2_name
        if values_p2 is not None and len(values_p2) == len(params):
            compare_values = values_p2
        elif values_avg is not None and len(values_avg) == len(params):
            compare_values = values_avg
            compare_name = "Group Average"

        if compare_values is None:
            raise ValueError(
                "Comparison values not provided (Player 2 or Group Average)"
            )

        # Just convert to the standard pizza chart function to maintain consistency
        return create_pizza_chart(params, values_p1, values_p2, values_avg, 
                                 title, subtitle, p1_name, p2_name)
                                 
    except Exception as e:
        # In case of error, create a chart with error message
        st.error(f"Detailed error: {str(e)}")

        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5,
                0.5,
                f"Error creating comparison pizza chart: {str(e)}",
                ha='center',
                va='center',
                fontsize=12,
                color='#333333')
        ax.axis('off')

    return fig

def create_bar_chart(metrics,
                    values_p1,
                    values_p2=None,
                    values_avg=None,
                    title=None,
                    subtitle=None,
                    p1_name="Player 1",
                    p2_name="Player 2"):
    """
    Creates a bar chart for comparing metrics between players
    """
    try:
        fig = plt.figure(figsize=(12, 6), facecolor='#0E1117')
        ax = fig.add_subplot(111, facecolor='#0E1117')
        
        # Set up the bar positions
        x = np.arange(len(metrics))
        width = 0.25
        
        # Colors
        player1_color = "#0052CC"  # Royal blue
        player2_color = "#00A3FF"  # Light blue  
        avg_color = "#B3CFFF"      # Very light blue
        
        # Abbreviate metric names for better display
        abbreviated_metrics = [abbreviate_metric_name(m) for m in metrics]
        
        # Plot bars for player 1
        bars1 = ax.bar(x - width, values_p1, width, color=player1_color, label=p1_name)
        
        # Initialize bars2 and bars3
        bars2 = None
        bars3 = None
        
        # Plot bars for player 2 if provided
        if values_p2 is not None:
            bars2 = ax.bar(x, values_p2, width, color=player2_color, label=p2_name)
            
        # Plot bars for average if provided
        if values_avg is not None:
            avg_label = "Group Average"
            bars3 = ax.bar(x + width, values_avg, width, color=avg_color, label=avg_label)
        
        # Add some text for labels, title and axes
        ax.set_ylabel('Percentile', fontsize=12, color='white')
        if title:
            ax.set_title(title, fontsize=16, color='white')
        else:
            comparison_title = f"{p1_name} vs {p2_name}" if values_p2 is not None else p1_name
            ax.set_title(comparison_title, fontsize=16, color='white')
            
        ax.set_xticks(x)
        ax.set_xticklabels(abbreviated_metrics, rotation=45, ha='right', fontsize=10, color='white')
        
        # Y-axis limits for percentiles
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        def add_labels(bars):
            if bars is not None:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                color='white', fontsize=9)
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        # Add a horizontal line at the 50th percentile
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        # Add grid with white lines for better visibility
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='white')
        
        # Adjust layout to make room for rotated labels
        plt.tight_layout()
        
        # Add credit
        fig.text(0.95, 0.02, 
                "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
                size=8, ha="right", color='gray', style='italic')
        
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")
        
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5, 0.5, 
                f"Error creating bar chart: {str(e)}",
                ha='center', va='center',
                fontsize=12, color='#333333')
        ax.axis('off')
        
    return fig

def create_scatter_plot(df, x_metric, y_metric, title=None):
    """
    Creates a scatter plot for two metrics with player names as annotations
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        
        # Safety checks
        if x_metric not in df.columns or y_metric not in df.columns:
            missing = []
            if x_metric not in df.columns:
                missing.append(x_metric)
            if y_metric not in df.columns:
                missing.append(y_metric)
            raise ValueError(f"Missing metrics in data: {', '.join(missing)}")
            
        # Extract data
        x_data = df[x_metric]
        y_data = df[y_metric]
        
        # Create scatter plot
        scatter = ax.scatter(x_data, y_data, 
                            c='#4285F4',  # Google blue
                            s=80, 
                            alpha=0.6,
                            edgecolors='white')
        
        # Add player names as annotations with small font
        for i, player in enumerate(df['Player']):
            ax.annotate(player, 
                       (x_data.iloc[i], y_data.iloc[i]),
                       fontsize=8,
                       ha='center', 
                       va='bottom',
                       color='white',
                       xytext=(0, 5),
                       textcoords='offset points')
        
        # Add quadrant lines (median values)
        x_median = x_data.median()
        y_median = y_data.median()
        
        ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(x_data.max() - (x_data.max() - x_data.min()) * 0.05, 
                y_data.max() - (y_data.max() - y_data.min()) * 0.05, 
                "High " + y_metric + "\nHigh " + x_metric, 
                ha='right', va='top', color='white', alpha=0.7)
                
        ax.text(x_data.min() + (x_data.max() - x_data.min()) * 0.05, 
                y_data.max() - (y_data.max() - y_data.min()) * 0.05, 
                "High " + y_metric + "\nLow " + x_metric, 
                ha='left', va='top', color='white', alpha=0.7)
                
        ax.text(x_data.max() - (x_data.max() - x_data.min()) * 0.05, 
                y_data.min() + (y_data.max() - y_data.min()) * 0.05, 
                "Low " + y_metric + "\nHigh " + x_metric, 
                ha='right', va='bottom', color='white', alpha=0.7)
                
        ax.text(x_data.min() + (x_data.max() - x_data.min()) * 0.05, 
                y_data.min() + (y_data.max() - y_data.min()) * 0.05, 
                "Low " + y_metric + "\nLow " + x_metric, 
                ha='left', va='bottom', color='white', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(x_metric, fontsize=12, color='white')
        ax.set_ylabel(y_metric, fontsize=12, color='white')
        
        if title:
            ax.set_title(title, fontsize=14, color='white')
        else:
            ax.set_title(f"{y_metric} vs {x_metric}", fontsize=14, color='white')
        
        # Style adjustments
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', alpha=0.3, color='white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Add credits
        fig.text(0.95, 0.02, 
                "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
                size=8, ha="right", color='gray', style='italic')
        
    except Exception as e:
        st.error(f"Error creating scatter plot: {str(e)}")
        
        fig = plt.figure(figsize=(10, 8), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5, 0.5, 
                f"Error creating scatter plot: {str(e)}",
                ha='center', va='center',
                fontsize=12, color='#333333')
        ax.axis('off')
        
    return fig
    
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
        # Using n_init as int, not string
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
                # Calculate distances from reference player to all other players
                distances = np.array([euclidean(X_scaled[player_idx], X_scaled[i]) for i in range(len(X_scaled))])
                
                # Convert to similarity score (0 to 1, higher is better)
                max_dist = np.max(distances)
                similarities = 1 - (distances / max_dist if max_dist > 0 else distances)

            # Convert to percentages and format as (player, similarity) tuples
            player_similarities = []
            for i in range(len(df)):
                if i != player_idx:
                    player_name = df.iloc[i]['Player']
                    sim_value = float(similarities[i]) * 100
                    player_similarities.append((player_name, sim_value))

            # Sort by similarity (descending) and return top n
            player_similarities.sort(key=lambda x: x[1], reverse=True)
            return player_similarities[:n]

    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return []

def create_similarity_viz(selected_player,
                         similar_players,
                         all_players_df,
                         metrics,
                         selected_metrics_for_display=None):
    """
    Creates a visualization for player similarity with:
    1. Scatter plot showing player positions in PCA space
    2. Table with top similar players and similarity scores
    3. Radar chart comparing the selected player with the most similar player
    """
    try:
        # Process data
        if all_players_df is None or all_players_df.empty:
            st.error("No player data available")
            return None
            
        if selected_metrics_for_display is None:
            # If no specific metrics for display, use the first 4 from the list
            selected_metrics_for_display = metrics[:min(4, len(metrics))]
        
        # Reduce dimensionality with PCA for visualization
        pca_result = create_pca_kmeans_df(all_players_df, metrics)
        if pca_result is None:
            st.error("Failed to create PCA visualization")
            return None
            
        # Sort similar players by similarity score (descending)
        similar_players.sort(key=lambda x: x[1], reverse=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10), facecolor='#0E1117', constrained_layout=True)
        gs = fig.add_gridspec(4, 6)
        
        # PCA scatter plot
        ax_pca = fig.add_subplot(gs[0:4, 0:3], facecolor='#0E1117')
        
        # Get reference player data
        ref_player = pca_result[pca_result['Player'] == selected_player]
        if ref_player.empty:
            st.error(f"Selected player '{selected_player}' not found in PCA result")
            return None
            
        ref_player_x = ref_player['x'].values[0]
        ref_player_y = ref_player['y'].values[0]
        ref_player_cluster = ref_player['cluster'].values[0]
        
        # Plot all players with cluster colors
        clusters = pca_result['cluster'].unique()
        cmap = plt.cm.get_cmap('tab10', len(clusters))
        
        for i, cluster in enumerate(clusters):
            cluster_df = pca_result[pca_result['cluster'] == cluster]
            ax_pca.scatter(cluster_df['x'], 
                          cluster_df['y'], 
                          c=[cmap(i)], 
                          s=50, 
                          alpha=0.5, 
                          label=f'Cluster {cluster+1}')
                          
        # Highlight selected player with a star marker
        ax_pca.scatter(ref_player_x, 
                      ref_player_y, 
                      c='gold', 
                      s=200, 
                      marker='*', 
                      edgecolors='black',
                      linewidth=1.5,
                      zorder=10,
                      label=selected_player)
        
        # Highlight similar players
        for player_name, similarity in similar_players[:5]:  # Top 5 for visibility
            player_data = pca_result[pca_result['Player'] == player_name]
            if not player_data.empty:
                player_x = player_data['x'].values[0]
                player_y = player_data['y'].values[0]
                
                # Highlight with a different marker
                ax_pca.scatter(player_x, 
                              player_y, 
                              c='white', 
                              s=100, 
                              marker='o', 
                              edgecolors='red',
                              linewidth=1.5,
                              alpha=0.8,
                              zorder=9)
                
                # Add player names below points with small font
                ax_pca.annotate(player_name, 
                               (player_x, player_y),
                               fontsize=8,
                               ha='center', 
                               va='bottom',
                               color='white',
                               xytext=(0, -10),
                               textcoords='offset points')
                               
        # Add selected player name with larger font
        ax_pca.annotate(selected_player, 
                       (ref_player_x, ref_player_y),
                       fontsize=10,
                       fontweight='bold',
                       ha='center', 
                       va='bottom',
                       color='white',
                       xytext=(0, 10),
                       textcoords='offset points')
        
        # Set labels and title
        ax_pca.set_xlabel('Principal Component 1', fontsize=12, color='white')
        ax_pca.set_ylabel('Principal Component 2', fontsize=12, color='white')
        ax_pca.set_title('Player Similarity Map (PCA)', fontsize=14, color='white')
        
        # Style adjustments
        ax_pca.spines['bottom'].set_color('white')
        ax_pca.spines['top'].set_color('white')
        ax_pca.spines['left'].set_color('white')
        ax_pca.spines['right'].set_color('white')
        ax_pca.tick_params(colors='white')
        ax_pca.grid(True, linestyle='--', alpha=0.3, color='white')
        
        # Add legend
        ax_pca.legend(fontsize=8, 
                     loc='upper right',
                     framealpha=0.7, 
                     facecolor='#0E1117',
                     edgecolor='white')
        
        # Top similar players table
        ax_table = fig.add_subplot(gs[0:2, 3:6])
        ax_table.axis('off')
        
        # Create table data
        table_data = []
        table_data.append(['Rank', 'Player', 'Team', 'Position', 'Age', 'Similarity'])
        
        for i, (player_name, similarity) in enumerate(similar_players[:10]):  # Top 10
            player_data = all_players_df[all_players_df['Player'] == player_name]
            if not player_data.empty:
                team = player_data['Team'].values[0] if 'Team' in player_data.columns else 'N/A'
                position = player_data['Position'].values[0] if 'Position' in player_data.columns else 'N/A'
                age = player_data['Age'].values[0] if 'Age' in player_data.columns else 'N/A'
                
                # Format age to 1 decimal place if it's a number
                age_str = f"{age:.1f}" if isinstance(age, (int, float)) else str(age)
                
                table_data.append([
                    f"#{i+1}",
                    player_name,
                    team,
                    position,
                    age_str,
                    f"{similarity:.1f}%"
                ])
        
        # Create table
        if table_data:
            table = ax_table.table(
                cellText=table_data,
                cellLoc='center',
                loc='center',
                cellColours=[['#0E1117'] * len(table_data[0])] * len(table_data)
            )
            
            # Table styling
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Style header row and add color gradient for similarity
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4285F4')
                else:
                    # Set text color to white
                    cell.set_text_props(color='white')
                    
                    # Add color gradient for similarity column
                    if j == 5 and i > 0:  # Similarity column, after header
                        similarity_value = float(table_data[i][j].strip('%'))
                        # Create a color gradient: higher similarity = deeper blue
                        alpha = similarity_value / 100
                        cell.set_facecolor((0, alpha*0.8, alpha, 0.5))
            
            ax_table.set_title('Top Similar Players', fontsize=14, color='white')
        
        # Metric comparison chart
        ax_metrics = fig.add_subplot(gs[2:4, 3:6])
        
        # Find most similar player
        if similar_players:
            most_similar_player = similar_players[0][0]
            
            # Create bar chart for selected metric comparison
            ref_player_values = []
            similar_player_values = []
            
            for metric in selected_metrics_for_display:
                if metric in all_players_df.columns:
                    ref_value = all_players_df.loc[all_players_df['Player'] == selected_player, metric].values[0]
                    similar_value = all_players_df.loc[all_players_df['Player'] == most_similar_player, metric].values[0]
                    
                    ref_player_values.append(ref_value)
                    similar_player_values.append(similar_value)
            
            # Create bar chart for metric comparison
            x = np.arange(len(selected_metrics_for_display))
            width = 0.35
            
            # Create bars
            player1_color = "#0052CC"  # Royal blue  
            player2_color = "#00A3FF"  # Light blue
            
            ax_metrics.bar(x - width/2, ref_player_values, width, color=player1_color, label=selected_player)
            ax_metrics.bar(x + width/2, similar_player_values, width, color=player2_color, label=most_similar_player)
            
            # Customize the chart
            ax_metrics.set_facecolor('#0E1117')
            ax_metrics.set_ylabel('Value', color='white')
            ax_metrics.set_title('Metric Comparison with Most Similar Player', fontsize=14, color='white')
            ax_metrics.set_xticks(x)
            ax_metrics.set_xticklabels([abbreviate_metric_name(m) for m in selected_metrics_for_display], rotation=45, ha='right', color='white')
            
            # Style adjustments
            ax_metrics.spines['bottom'].set_color('white')
            ax_metrics.spines['top'].set_color('white')
            ax_metrics.spines['left'].set_color('white')
            ax_metrics.spines['right'].set_color('white')
            ax_metrics.tick_params(colors='white')
            ax_metrics.grid(True, axis='y', linestyle='--', alpha=0.3, color='white')
            
            # Add legend
            ax_metrics.legend(fontsize=10)
            
        # Add credits
        fig.text(0.95, 0.02, 
                "made by Joao Alberto Kolling\ndata via WyScout/SkillCorner",
                size=8, ha="right", color='gray', style='italic')
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating similarity visualization: {str(e)}")
        
        fig = plt.figure(figsize=(15, 10), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        ax.text(0.5, 0.5, 
                f"Error creating similarity visualization: {str(e)}",
                ha='center', va='center',
                fontsize=12, color='#333333')
        ax.axis('off')
        
        return fig

# Title and description
st.title("Football Analytics App")
st.markdown("""
This application provides advanced football analytics tools. Upload your data to start exploring player performance metrics and comparisons.
""")

# App is divided into main sections - Sidebar for filters, main area for visualizations
st.sidebar.header("Data Selection")

# Data source selector
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["Upload Files (Wyscout)", "SkillCorner Database"],
    index=0,
    key="data_source_select"
)

# Placeholder for file upload
uploaded_files = st.sidebar.file_uploader("Upload Wyscout Excel files", 
                                         type=["xlsx"], 
                                         accept_multiple_files=True,
                                         key="file_uploader")

# Placeholder - Process uploaded files
if uploaded_files:
    st.write("Files uploaded, processing data...")
    # In a real implementation, we'd process the files here
    # df_combined, file_metadata = load_and_clean(uploaded_files)
    # st.session_state.processed_data = df_combined
    # st.session_state.file_metadata = file_metadata

# Main tabs interface
tabs = st.tabs([
    'Pizza Chart', 'Bars', 'Scatter', 'Similarity', 'Correlation',
    'Composite Index (PCA)', 'Profiler'
])

# Pizza Chart Tab - First Tab
with tabs[0]:
    st.header("Pizza Chart Visualization")
    
    # Player selection section
    st.subheader("Player Selection")
    
    # Mock data for demonstration purposes 
    # In a real implementation, we'd use the uploaded data
    if st.session_state.processed_data is None:
        # Create sample data for demo
        sample_metrics = [
            "xG per 90", "Shots on target, %", "Goal conversion, %",
            "Touches in box per 90", "Progressive runs per 90",
            "Sprint Count P90", "Accurate passes, %",
            "Progressive passes per 90", "Key passes per 90"
        ]
        
        # Generate sample players
        sample_players = ["Erling Haaland", "Kevin De Bruyne", "Mohamed Salah", 
                         "Harry Kane", "Bruno Fernandes", "Jude Bellingham"]
        
        # Metric selection
        st.write("Select metrics to visualize:")
        selected_metrics = st.multiselect(
            "Choose metrics (recommended 8-12 for best visualization):",
            sample_metrics,
            default=sample_metrics[:6]
        )
        
        # Player selection for pizza charts
        col1, col2 = st.columns(2)
        
        with col1:
            selected_player1 = st.selectbox(
                "Select Player 1:",
                sample_players,
                index=0
            )
            
        with col2:    
            selected_player2 = st.selectbox(
                "Select Player 2 (optional):",
                ["None"] + sample_players,
                index=0
            )
            
        selected_player2 = None if selected_player2 == "None" else selected_player2
        
        # Add option to show group average
        show_average = st.checkbox("Show Group Average on Chart", value=False)
        
        # Create pizza chart button
        if st.button("Generate Pizza Chart"):
            if not selected_metrics:
                st.warning("Please select at least one metric to visualize.")
            else:
                try:
                    # Generate random values between 30-95 for demonstration
                    # In a real implementation we would fetch actual values from the data
                    p1_values = [random.randint(30, 95) for _ in range(len(selected_metrics))]
                    p2_values = [random.randint(30, 95) for _ in range(len(selected_metrics))] if selected_player2 else None
                    avg_values = [random.randint(40, 70) for _ in range(len(selected_metrics))] if show_average else None
                    
                    # Create and display pizza chart
                    pizza_fig = create_pizza_chart(
                        params=selected_metrics,
                        values_p1=p1_values,
                        values_p2=p2_values,
                        values_avg=avg_values,
                        p1_name=selected_player1,
                        p2_name=selected_player2 if selected_player2 else "Player 2"
                    )
                    
                    st.pyplot(pizza_fig)
                    
                    # Option to download chart
                    buf = fig_to_bytes(pizza_fig)
                    st.download_button(
                        label="Download Chart as PNG",
                        data=buf,
                        file_name=f"{selected_player1}_pizza_chart.png",
                        mime="image/png"
                    )
                
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
        
        # Helper text for users without data
        st.info("""
        This is a demonstration using random data. 
        
        To analyze real players:
        1. Upload Wyscout Excel files using the file uploader in the sidebar
        2. Select leagues, players and metrics to generate actual performance charts
        """)
    
    else:
        # This section would handle real data once uploaded
        st.write("Process and visualize real player data here.")
        # Actual implementation would use the dataframe in st.session_state.processed_data

# Footer with credits
st.markdown("---")
st.markdown("Made by Joao Alberto Kolling | Data via WyScout/SkillCorner")
