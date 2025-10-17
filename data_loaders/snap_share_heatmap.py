"""
Generate a heatmap visualization of offensive snap share data by week.

Displays skill position players (QB, WR, RB, TE) with weeks on x-axis and
team-position-player on y-axis, colored by snap share percentage.
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_and_filter_data(csv_path: str, min_games: int = 3) -> pl.DataFrame:
    """
    Load opportunity report CSV and filter for skill positions.
    
    Args:
        csv_path: Path to the opportunity report CSV file
        min_games: Minimum number of games a player must appear in
        
    Returns:
        Filtered polars DataFrame
    """
    # Load CSV with polars
    df = pl.read_csv(csv_path)
    
    # Filter for skill positions only
    skill_positions = ['QB', 'WR', 'RB', 'TE']
    df = df.filter(pl.col('position').is_in(skill_positions))
    
    # Select relevant columns
    df = df.select([
        'week',
        'team',
        'position',
        'player_name',
        'offense_snap_share',
        'targets',
        'percent_share_of_intended_air_yards'
    ])
    
    # Remove rows with null snap share
    df = df.filter(pl.col('offense_snap_share').is_not_null())
    
    # Filter out players with too few appearances
    player_game_counts = df.group_by('player_name').agg(
        pl.col('week').count().alias('games_played')
    )
    
    valid_players = player_game_counts.filter(
        pl.col('games_played') >= min_games
    )['player_name']
    
    df = df.filter(pl.col('player_name').is_in(valid_players))
    
    return df


def create_composite_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create player labels and prepare for hierarchical sorting.
    Players will be organized by team, then position (QB, RB, WR, TE), then player name.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with composite_label and sorting columns added
    """
    # Use player name as label since grouping is visual
    df = df.with_columns(
        pl.col('player_name').alias('composite_label')
    )
    
    # Add position order for sorting (QB, RB, WR, TE)
    df = df.with_columns(
        pl.col('position').map_elements(
            lambda x: {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}.get(x, 99),
            return_dtype=pl.Int32
        ).alias('pos_order')
    )
    
    return df


def pivot_to_matrix(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int], list[dict]]:
    """
    Transform data into a 2D matrix for heatmap.
    
    Args:
        df: Input DataFrame with composite_label
        
    Returns:
        Tuple of (snap_matrix, target_matrix, target_share_matrix, row_labels, col_labels, grouping_info)
    """
    # Get unique weeks and sort them
    weeks = sorted(df['week'].unique().to_list())
    
    # Calculate average snap share per player for sorting
    # Sort by: team, then position (QB, RB, WR, TE), then avg snap share descending
    player_stats = df.group_by(['composite_label', 'player_name']).agg([
        pl.col('offense_snap_share').mean().alias('avg_snap'),
        pl.col('pos_order').first().alias('pos_order'),
        pl.col('team').first().alias('team'),
        pl.col('position').first().alias('position')
    ]).sort(['team', 'pos_order', 'avg_snap'], descending=[False, False, True])
    
    players = player_stats['composite_label'].to_list()
    
    # Create grouping info for visual separators
    grouping_info = []
    for row in player_stats.iter_rows(named=True):
        grouping_info.append({
            'player': row['composite_label'],
            'team': row['team'],
            'position': row['position']
        })
    
    # Create matrices filled with NaN
    snap_matrix = np.full((len(players), len(weeks)), np.nan)
    target_matrix = np.full((len(players), len(weeks)), np.nan)
    target_share_matrix = np.full((len(players), len(weeks)), np.nan)
    
    # Fill matrices with values
    for i, player in enumerate(players):
        player_data = df.filter(pl.col('composite_label') == player)
        for j, week in enumerate(weeks):
            week_data = player_data.filter(pl.col('week') == week)
            if len(week_data) > 0:
                snap_matrix[i, j] = week_data['offense_snap_share'][0]
                # Only store target data for WRs
                if grouping_info[i]['position'] == 'WR':
                    targets = week_data['targets'][0]
                    target_share = week_data['percent_share_of_intended_air_yards'][0]
                    if targets is not None and not np.isnan(targets):
                        target_matrix[i, j] = targets
                    if target_share is not None and not np.isnan(target_share):
                        target_share_matrix[i, j] = target_share
    
    return snap_matrix, target_matrix, target_share_matrix, players, weeks, grouping_info


def create_heatmap(
    snap_matrix: np.ndarray,
    target_matrix: np.ndarray,
    target_share_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[int],
    grouping_info: list[dict],
    output_path: str,
    output_format: str = 'png',
    dpi: int = 300
):
    """
    Create and save the heatmap visualization.
    
    Args:
        snap_matrix: 2D numpy array with snap share values
        target_matrix: 2D numpy array with target counts for WRs
        target_share_matrix: 2D numpy array with target share for WRs
        row_labels: Player labels for y-axis
        col_labels: Week numbers for x-axis
        grouping_info: List of dicts with team/position info for each player
        output_path: Path to save the output file
        output_format: File format (png, jpg, pdf)
        dpi: DPI for output image
    """
    # Expand matrices to include target rows for WRs
    expanded_snap_matrix, expanded_target_share, expanded_targets, expanded_labels, expanded_grouping = \
        expand_for_wr_targets(snap_matrix, target_matrix, target_share_matrix, row_labels, grouping_info)
    
    # Calculate figure size based on data dimensions
    n_rows, n_cols = expanded_snap_matrix.shape
    fig_width = max(12, n_cols * 0.6)
    fig_height = max(8, n_rows * 0.20)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create combined matrix for visualization using pcolormesh with custom colors
    # This allows us to use different colormaps for different rows
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    # Create RGB array for the heatmap
    rgba_array = np.zeros((n_rows, n_cols, 4))
    
    buGn_cmap = plt.colormaps['BuGn']
    reds_cmap = plt.colormaps['Reds']
    
    # Find max target count for normalization
    max_targets = np.nanmax(expanded_targets)
    if np.isnan(max_targets) or max_targets == 0:
        max_targets = 15  # Default max for normalization
    
    for i in range(n_rows):
        is_target_row = expanded_grouping[i].get('is_target_row', False)
        for j in range(n_cols):
            if is_target_row:
                # Use Reds colormap based on target COUNT (not target share)
                target_count = expanded_targets[i, j]
                if not np.isnan(target_count):
                    # Normalize target count to 0-1 scale for colormap
                    normalized_value = min(target_count / max_targets, 1.0)
                    rgba_array[i, j, :] = reds_cmap(normalized_value)
                else:
                    rgba_array[i, j, :] = [1, 1, 1, 1]  # White for NaN
            else:
                # Use BuGn colormap for snap share
                value = expanded_snap_matrix[i, j]
                if not np.isnan(value):
                    rgba_array[i, j, :] = buGn_cmap(value)
                else:
                    rgba_array[i, j, :] = [1, 1, 1, 1]  # White for NaN
    
    # Display the custom colored array
    im = ax.imshow(rgba_array, aspect='auto', interpolation='nearest')
    
    # Set tick positions and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(expanded_labels)))
    ax.set_xticklabels([f'Week {w}' for w in col_labels], rotation=45, ha='right')
    ax.set_yticklabels(expanded_labels, fontsize=7)
    
    # Color-code player name backgrounds by position
    color_code_player_labels(ax, expanded_grouping)
    
    # Add visual separators between teams and positions
    add_grouping_lines(ax, expanded_grouping, n_cols)
    
    # Add title
    ax.set_title('2025 Offensive Snap Share by Week - Skill Positions (WR Target Rows Below Each WR)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add annotations - percentages for snap share, target counts for target rows
    annotate_heatmap_with_targets(ax, expanded_snap_matrix, expanded_targets, expanded_grouping)
    
    # Configure layout
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_path}.{output_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
    plt.close()


def expand_for_wr_targets(
    snap_matrix: np.ndarray,
    target_matrix: np.ndarray,
    target_share_matrix: np.ndarray,
    row_labels: list[str],
    grouping_info: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict]]:
    """
    Expand matrices to include target rows for WRs.
    
    Returns:
        Tuple of (expanded_snap_matrix, expanded_target_share, expanded_targets, expanded_labels, expanded_grouping)
    """
    n_rows, n_cols = snap_matrix.shape
    wr_count = sum(1 for info in grouping_info if info['position'] == 'WR')
    new_n_rows = n_rows + wr_count
    
    # Create expanded matrices
    expanded_snap = np.full((new_n_rows, n_cols), np.nan)
    expanded_target_share = np.full((new_n_rows, n_cols), np.nan)
    expanded_targets = np.full((new_n_rows, n_cols), np.nan)
    expanded_labels = []
    expanded_grouping = []
    
    new_row_idx = 0
    for i in range(n_rows):
        # Add the main player row
        expanded_snap[new_row_idx, :] = snap_matrix[i, :]
        expanded_labels.append(row_labels[i])
        expanded_grouping.append(grouping_info[i].copy())
        new_row_idx += 1
        
        # If this is a WR, add a target row
        if grouping_info[i]['position'] == 'WR':
            expanded_target_share[new_row_idx, :] = target_share_matrix[i, :]
            expanded_targets[new_row_idx, :] = target_matrix[i, :]
            expanded_labels.append(f"  → Targets")
            expanded_grouping.append({
                'player': f"{grouping_info[i]['player']}_targets",
                'team': grouping_info[i]['team'],
                'position': 'WR',
                'is_target_row': True
            })
            new_row_idx += 1
    
    return expanded_snap, expanded_target_share, expanded_targets, expanded_labels, expanded_grouping


def add_grouping_lines(ax, grouping_info: list[dict], n_cols: int):
    """
    Add horizontal lines to separate teams and positions.
    
    Args:
        ax: Matplotlib axes
        grouping_info: List of dicts with team/position info for each player
        n_cols: Number of columns in the heatmap
    """
    prev_team = None
    prev_position = None
    
    for i, info in enumerate(grouping_info):
        # Add thick line between teams
        if prev_team is not None and info['team'] != prev_team:
            ax.axhline(y=i - 0.5, color='black', linewidth=2.5, linestyle='-')
        # Add thin line between positions within same team
        elif prev_position is not None and info['position'] != prev_position:
            ax.axhline(y=i - 0.5, color='gray', linewidth=1, linestyle='--', alpha=0.6)
        
        prev_team = info['team']
        prev_position = info['position']


def color_code_player_labels(ax, grouping_info: list[dict]):
    """
    Color-code player name backgrounds by position using pastel colors.
    
    Args:
        ax: Matplotlib axes
        grouping_info: List of dicts with team/position info for each player
    """
    # Define pastel colors for each position
    position_colors = {
        'QB': '#AED6F1',   # Pastel blue
        'RB': '#FFDAB9',   # Pastel orange (peach)
        'WR': '#FFB6C1',   # Pastel red (light pink)
        'TE': '#DDA0DD'    # Pastel purple (plum)
    }
    
    # Color each y-tick label background
    for i, info in enumerate(grouping_info):
        position = info['position']
        
        # Use lighter color for target rows
        if info.get('is_target_row', False):
            color = '#FFE4E1'  # Very light pink for target rows
        else:
            color = position_colors.get(position, 'white')
        
        # Get the y-tick label and add background color
        ytick_label = ax.get_yticklabels()[i]
        ytick_label.set_bbox(dict(boxstyle='round,pad=0.3', 
                                  facecolor=color, 
                                  edgecolor='none', 
                                  alpha=0.7))


def annotate_heatmap_with_targets(ax, snap_data: np.ndarray, target_data: np.ndarray, 
                                   grouping_info: list[dict], threshold: float = 0.5):
    """
    Add text annotations to heatmap cells with automatic color adjustment.
    Shows percentages for snap share rows and target counts for target rows.
    
    Args:
        ax: Matplotlib axes
        snap_data: 2D numpy array with snap share values
        target_data: 2D numpy array with target counts
        grouping_info: List of dicts with player/position info
        threshold: Threshold for switching text color (0-1)
    """
    texts = []
    for i in range(snap_data.shape[0]):
        is_target_row = grouping_info[i].get('is_target_row', False)
        
        for j in range(snap_data.shape[1]):
            if is_target_row:
                # For target rows, show target count
                value = target_data[i, j]
                if np.isnan(value):
                    continue
                text_str = f'{int(value)}'
                # Red gradient - darker for more targets
                text_color = 'white' if value > 5 else 'black'
            else:
                # For snap share rows, show percentage
                value = snap_data[i, j]
                if np.isnan(value):
                    continue
                text_str = f'{value * 100:.0f}%'
                # BuGn gradient
                text_color = 'white' if value > threshold else 'black'
            
            # Add text to cell
            text = ax.text(j, i, text_str,
                          ha='center', va='center',
                          color=text_color, fontsize=6,
                          fontweight='bold')
            texts.append(text)
    
    return texts


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate offensive snap share heatmap for skill positions'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='reports/opportunity_report_2025.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/snap_share_heatmap_2025',
        help='Output path (without extension)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'jpg', 'pdf'],
        help='Output file format'
    )
    parser.add_argument(
        '--min-games',
        type=int,
        default=3,
        help='Minimum number of games a player must appear in'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output image'
    )
    
    args = parser.parse_args()
    
    # Load and process data
    print(f"Loading data from {args.input}...")
    df = load_and_filter_data(args.input, min_games=args.min_games)
    print(f"Loaded {len(df)} records for {df['player_name'].n_unique()} players")
    
    # Create composite labels
    print("Creating composite labels...")
    df = create_composite_labels(df)
    
    # Pivot to matrix
    print("Pivoting data to matrix...")
    snap_matrix, target_matrix, target_share_matrix, row_labels, col_labels, grouping_info = pivot_to_matrix(df)
    print(f"Matrix shape: {snap_matrix.shape[0]} players × {snap_matrix.shape[1]} weeks")
    
    # Create heatmap
    print("Generating heatmap...")
    create_heatmap(
        snap_matrix, target_matrix, target_share_matrix,
        row_labels, col_labels, grouping_info,
        args.output, args.format, args.dpi
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
