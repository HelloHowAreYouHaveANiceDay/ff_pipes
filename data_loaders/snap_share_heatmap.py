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
from matplotlib.backends.backend_pdf import PdfPages


def load_and_filter_data(csv_path: str, min_games: int = 1) -> pl.DataFrame:
    """
    Load opportunity report CSV and filter for skill positions.
    
    Args:
        csv_path: Path to the opportunity report CSV file
        min_games: Minimum number of games a player-team combination must appear in
        
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
        'percent_share_of_intended_air_yards',
        'avg_yac',
        'rush_first_downs',
        'receiving_first_downs',
        'completion_percentage'
    ])
    
    # Remove rows with null snap share
    df = df.filter(pl.col('offense_snap_share').is_not_null())
    
    # Filter out player-team combinations with too few appearances
    # This handles players who change teams during the season
    player_team_game_counts = df.group_by(['player_name', 'team']).agg(
        pl.col('week').count().alias('games_played')
    )
    
    valid_player_teams = player_team_game_counts.filter(
        pl.col('games_played') >= min_games
    ).select(['player_name', 'team'])
    
    df = df.join(valid_player_teams, on=['player_name', 'team'], how='inner')
    
    return df


def create_composite_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create player labels and prepare for hierarchical sorting.
    Players will be organized by team, then position (QB, RB, WR, TE), then player name.
    For players who changed teams, they appear separately for each team.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with composite_label and sorting columns added
    """
    # Create composite label: player_name for single-team players, player_name (team) for multi-team
    # First, identify players who played for multiple teams
    player_team_counts = df.group_by('player_name').agg(
        pl.col('team').n_unique().alias('team_count')
    )
    
    # Join back to add team_count column
    df = df.join(player_team_counts, on='player_name', how='left')
    
    # Create composite label: add team abbreviation for players on multiple teams
    df = df.with_columns(
        pl.when(pl.col('team_count') > 1)
        .then(pl.concat_str([pl.col('player_name'), pl.lit(' ('), pl.col('team'), pl.lit(')')]))
        .otherwise(pl.col('player_name'))
        .alias('composite_label')
    )
    
    # Add position order for sorting (QB, RB, WR, TE)
    df = df.with_columns(
        pl.col('position').map_elements(
            lambda x: {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}.get(x, 99),
            return_dtype=pl.Int32
        ).alias('pos_order')
    )
    
    return df


def pivot_to_matrix(df: pl.DataFrame) -> dict:
    """
    Transform data into a dictionary of 2D matrices grouped by team.
    
    Args:
        df: Input DataFrame with composite_label
        
    Returns:
        Dictionary mapping team names to tuples of (snap_matrix, target_matrix, target_share_matrix, 
        avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, 
        row_labels, col_labels, grouping_info)
    """
    # Get unique weeks and sort them
    weeks = sorted(df['week'].unique().to_list())
    
    # Get unique teams
    teams = sorted(df['team'].unique().to_list())
    
    team_data = {}
    
    for team in teams:
        # Filter data for this team
        team_df = df.filter(pl.col('team') == team)
        
        # Calculate average snap share per player for sorting
        # Sort by: position (QB, RB, WR, TE), then avg snap share descending
        player_stats = team_df.group_by(['composite_label', 'player_name']).agg([
            pl.col('offense_snap_share').mean().alias('avg_snap'),
            pl.col('pos_order').first().alias('pos_order'),
            pl.col('position').first().alias('position')
        ]).sort(['pos_order', 'avg_snap'], descending=[False, True])
        
        players = player_stats['composite_label'].to_list()
        
        # Create grouping info for visual separators
        grouping_info = []
        for i, row in enumerate(player_stats.iter_rows(named=True)):
            grouping_info.append({
                'player': row['composite_label'],
                'team': team,
                'position': row['position']
            })
        
        # Create matrices filled with NaN
        snap_matrix = np.full((len(players), len(weeks)), np.nan)
        target_matrix = np.full((len(players), len(weeks)), np.nan)
        target_share_matrix = np.full((len(players), len(weeks)), np.nan)
        avg_yac_matrix = np.full((len(players), len(weeks)), np.nan)
        rush_first_downs_matrix = np.full((len(players), len(weeks)), np.nan)
        receiving_first_downs_matrix = np.full((len(players), len(weeks)), np.nan)
        completion_pct_matrix = np.full((len(players), len(weeks)), np.nan)
        
        # Fill matrices with values
        for i, player in enumerate(players):
            player_data = team_df.filter(pl.col('composite_label') == player)
            for j, week in enumerate(weeks):
                week_data = player_data.filter(pl.col('week') == week)
                if len(week_data) > 0:
                    snap_matrix[i, j] = week_data['offense_snap_share'][0]
                    position = grouping_info[i]['position']
                    
                    # Store target data for WRs, RBs, and TEs
                    if position in ['WR', 'RB', 'TE']:
                        targets = week_data['targets'][0]
                        target_share = week_data['percent_share_of_intended_air_yards'][0]
                        if targets is not None and not np.isnan(targets):
                            target_matrix[i, j] = targets
                        if target_share is not None and not np.isnan(target_share):
                            target_share_matrix[i, j] = target_share
                    
                    # Store avg_yac for WRs and TEs
                    if position in ['WR', 'TE']:
                        avg_yac = week_data['avg_yac'][0]
                        if avg_yac is not None and not np.isnan(avg_yac):
                            avg_yac_matrix[i, j] = avg_yac
                    
                    # Store receiving_first_downs for WRs, RBs, and TEs
                    if position in ['WR', 'RB', 'TE']:
                        rec_fd = week_data['receiving_first_downs'][0]
                        if rec_fd is not None and not np.isnan(rec_fd):
                            receiving_first_downs_matrix[i, j] = rec_fd
                    
                    # Store rush_first_downs for RBs
                    if position == 'RB':
                        rush_fd = week_data['rush_first_downs'][0]
                        if rush_fd is not None and not np.isnan(rush_fd):
                            rush_first_downs_matrix[i, j] = rush_fd
                    
                    # Store completion_percentage for QBs
                    if position == 'QB':
                        comp_pct = week_data['completion_percentage'][0]
                        if comp_pct is not None and not np.isnan(comp_pct):
                            completion_pct_matrix[i, j] = comp_pct
        
        team_data[team] = (
            snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, 
            rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix,
            players, weeks, grouping_info
        )
    
    return team_data


def create_heatmap(
    team_data: dict,
    output_path: str,
    output_format: str = 'pdf',
    dpi: int = 300
):
    """
    Create and save the heatmap visualization with one page per team.
    
    Args:
        team_data: Dictionary mapping team names to tuples of matrices and labels
        output_path: Path to save the output file
        output_format: File format (png, jpg, pdf)
        dpi: DPI for output image
    """
    output_file = f"{output_path}.{output_format}"
    
    if output_format == 'pdf':
        # Create multi-page PDF
        with PdfPages(output_file) as pdf:
            for team_name in sorted(team_data.keys()):
                snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, \
                    rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, \
                    row_labels, col_labels, grouping_info = team_data[team_name]
                
                create_team_page(
                    team_name, snap_matrix, target_matrix, target_share_matrix, 
                    avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix,
                    completion_pct_matrix, row_labels, col_labels, grouping_info, pdf, dpi
                )
        print(f"Multi-page PDF heatmap saved to: {output_file}")
    else:
        # For non-PDF formats, save each team as a separate file
        for team_name in sorted(team_data.keys()):
            snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, \
                rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, \
                row_labels, col_labels, grouping_info = team_data[team_name]
            
            team_output_file = f"{output_path}_{team_name}.{output_format}"
            create_team_page(
                team_name, snap_matrix, target_matrix, target_share_matrix, 
                avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix,
                completion_pct_matrix, row_labels, col_labels, grouping_info, None, dpi,
                team_output_file
            )
            print(f"Team heatmap saved to: {team_output_file}")


def create_team_page(
    team_name: str,
    snap_matrix: np.ndarray,
    target_matrix: np.ndarray,
    target_share_matrix: np.ndarray,
    avg_yac_matrix: np.ndarray,
    rush_first_downs_matrix: np.ndarray,
    receiving_first_downs_matrix: np.ndarray,
    completion_pct_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[int],
    grouping_info: list[dict],
    pdf: PdfPages = None,
    dpi: int = 300,
    output_file: str = None
):
    """
    Create a single heatmap page for one team.
    
    Args:
        team_name: Name of the team
        snap_matrix: 2D numpy array with snap share values
        target_matrix: 2D numpy array with target counts for WRs, RBs, TEs
        target_share_matrix: 2D numpy array with target share for WRs, RBs, TEs
        avg_yac_matrix: 2D numpy array with avg yards after catch for WRs, TEs
        rush_first_downs_matrix: 2D numpy array with rush first downs for RBs
        receiving_first_downs_matrix: 2D numpy array with receiving first downs for WRs, RBs, TEs
        completion_pct_matrix: 2D numpy array with completion percentage for QBs
        row_labels: Player labels for y-axis
        col_labels: Week numbers for x-axis
        grouping_info: List of dicts with team/position info for each player
        pdf: PdfPages object if saving to multi-page PDF
        dpi: DPI for output image
        output_file: Path to save file (if not using pdf object)
    """
    # Expand matrices to include metric rows for all positions
    expanded_snap_matrix, expanded_targets, expanded_target_share, expanded_avg_yac, expanded_rush_fd, expanded_rec_fd, expanded_comp_pct, expanded_labels, expanded_grouping = \
        expand_for_position_metrics(snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, row_labels, grouping_info)
    
    # Calculate figure size based on data dimensions
    n_rows, n_cols = expanded_snap_matrix.shape
    fig_width = max(8, n_cols * 0.3)  # Reduced from 0.6 to 0.3 (2x narrower)
    fig_height = max(6, n_rows * 0.20)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create combined matrix for visualization using pcolormesh with custom colors
    # This allows us to use different colormaps for different rows
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    import matplotlib.cm as cm
    
    # Create custom colormaps with gradients from low to high saturation
    # All start from white/light and go to the target color
    
    # snap_share: #50514F (dark gray) - low saturation to high saturation
    snap_colors = ['#F5F5F5', '#D0D0CF', '#ABABAA', '#868785', '#50514F']
    snap_cmap = LinearSegmentedColormap.from_list('snap_gradient', snap_colors, N=256)
    
    # targets: #F25F5C (red) - low saturation to high saturation
    target_colors = ['#FEFEFE', '#FCCECD', '#FA9D9B', '#F66D69', '#F25F5C']
    target_cmap = LinearSegmentedColormap.from_list('target_gradient', target_colors, N=256)
    
    # receiving_first_downs: #70C1B3 (teal) - low saturation to high saturation
    rec_fd_colors = ['#FEFEFE', '#D9F0EC', '#B3E1D9', '#8CD1C6', '#70C1B3']
    rec_fd_cmap = LinearSegmentedColormap.from_list('rec_fd_gradient', rec_fd_colors, N=256)
    
    # rush_first_downs: #247BA0 (blue) - low saturation to high saturation
    rush_fd_colors = ['#FEFEFE', '#C8DDE8', '#91BBD2', '#5A99BB', '#247BA0']
    rush_fd_cmap = LinearSegmentedColormap.from_list('rush_fd_gradient', rush_fd_colors, N=256)
    
    # avg_yac: #EDCB96 (tan/beige) - low saturation to high saturation
    yac_colors = ['#FEFEFE', '#F9F0E5', '#F6E5CB', '#F3DAB1', '#EDCB96']
    yac_cmap = LinearSegmentedColormap.from_list('yac_gradient', yac_colors, N=256)
    
    # For completion percentage (QB), keep a reasonable color scheme
    comp_colors = ['#FEFEFE', '#C8DDE8', '#91BBD2', '#5A99BB', '#247BA0']
    comp_cmap = LinearSegmentedColormap.from_list('comp_gradient', comp_colors, N=256)
    
    # Create RGB array for the heatmap
    rgba_array = np.zeros((n_rows, n_cols, 4))
    
    # Find max values for normalization
    max_targets = np.nanmax(expanded_targets)
    if np.isnan(max_targets) or max_targets == 0:
        max_targets = 15
    
    max_yac = np.nanmax(expanded_avg_yac)
    if np.isnan(max_yac) or max_yac == 0:
        max_yac = 10
    
    max_rush_fd = np.nanmax(expanded_rush_fd)
    if np.isnan(max_rush_fd) or max_rush_fd == 0:
        max_rush_fd = 5
    
    max_rec_fd = np.nanmax(expanded_rec_fd)
    if np.isnan(max_rec_fd) or max_rec_fd == 0:
        max_rec_fd = 5
    
    for i in range(n_rows):
        metric_type = expanded_grouping[i].get('metric_type', None)
        for j in range(n_cols):
            if metric_type == 'targets':
                # Use custom red gradient based on target COUNT
                target_count = expanded_targets[i, j]
                if not np.isnan(target_count):
                    normalized_value = min(target_count / max_targets, 1.0)
                    rgba_array[i, j, :] = target_cmap(normalized_value)
                else:
                    rgba_array[i, j, :] = [1, 1, 1, 1]  # White for NaN
            elif metric_type == 'avg_yac':
                # Use custom tan/beige gradient for avg YAC
                yac_value = expanded_avg_yac[i, j]
                if not np.isnan(yac_value):
                    normalized_value = min(yac_value / max_yac, 1.0)
                    rgba_array[i, j, :] = yac_cmap(normalized_value)
                else:
                    rgba_array[i, j, :] = [1, 1, 1, 1]
            elif metric_type == 'rush_first_downs':
                # Use custom blue gradient for rush first downs
                rush_fd = expanded_rush_fd[i, j]
                if not np.isnan(rush_fd):
                    normalized_value = min(rush_fd / max_rush_fd, 1.0)
                    rgba_array[i, j, :] = rush_fd_cmap(normalized_value)
                else:
                    rgba_array[i, j, :] = [1, 1, 1, 1]
            elif metric_type == 'receiving_first_downs':
                # Use custom teal gradient for receiving first downs
                rec_fd = expanded_rec_fd[i, j]
                if not np.isnan(rec_fd):
                    normalized_value = min(rec_fd / max_rec_fd, 1.0)
                    rgba_array[i, j, :] = rec_fd_cmap(normalized_value)
                else:
                    rgba_array[i, j, :] = [1, 1, 1, 1]
            elif metric_type == 'completion_percentage':
                # Use custom gradient for completion percentage (already 0-100)
                comp_pct = expanded_comp_pct[i, j]
                if not np.isnan(comp_pct):
                    # Normalize from 0-100 to 0-1, but use 40-100 range for better contrast
                    normalized_value = max(0, min((comp_pct - 40) / 60, 1.0))
                    rgba_array[i, j, :] = comp_cmap(normalized_value)
                else:
                    rgba_array[i, j, :] = [1, 1, 1, 1]
            else:
                # Use custom dark gray gradient for snap share
                value = expanded_snap_matrix[i, j]
                if not np.isnan(value):
                    rgba_array[i, j, :] = snap_cmap(value)
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
    
    # Add visual separators between positions
    add_grouping_lines(ax, expanded_grouping, n_cols)
    
    # Add title with team name
    ax.set_title(f'{team_name} - 2025 Offensive Snap Share by Week\n(QB: Comp%, RB: Targets/Rush 1st Down/Rec 1st Down, WR/TE: Targets/Avg YAC/Rec 1st Down)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add annotations - percentages for snap share, metric values for metric rows
    annotate_heatmap_with_metrics(ax, expanded_snap_matrix, expanded_targets, expanded_avg_yac, expanded_rush_fd, expanded_rec_fd, expanded_comp_pct, expanded_grouping)
    
    # Configure layout
    plt.tight_layout()
    
    # Save figure
    if pdf is not None:
        # Save to multi-page PDF
        pdf.savefig(fig, dpi=dpi, bbox_inches='tight')
    elif output_file is not None:
        # Save to individual file
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    
    plt.close()


def expand_for_position_metrics(
    snap_matrix: np.ndarray,
    target_matrix: np.ndarray,
    target_share_matrix: np.ndarray,
    avg_yac_matrix: np.ndarray,
    rush_first_downs_matrix: np.ndarray,
    receiving_first_downs_matrix: np.ndarray,
    completion_pct_matrix: np.ndarray,
    row_labels: list[str],
    grouping_info: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[dict]]:
    """
    Expand matrices to include metric rows for all positions.
    - QB: completion_percentage row
    - RB: targets row, rush_first_downs row, receiving_first_downs row
    - WR: targets row, avg_yac row, receiving_first_downs row
    - TE: targets row, avg_yac row, receiving_first_downs row
    
    Returns:
        Tuple of (expanded_snap_matrix, expanded_targets, expanded_target_share, expanded_avg_yac, 
                  expanded_rush_fd, expanded_rec_fd, expanded_comp_pct, expanded_labels, expanded_grouping)
    """
    n_rows, n_cols = snap_matrix.shape
    
    # Count additional rows needed for each position
    extra_rows = 0
    for info in grouping_info:
        pos = info['position']
        if pos == 'QB':
            extra_rows += 1  # completion_percentage
        elif pos == 'RB':
            extra_rows += 3  # targets, rush_first_downs, receiving_first_downs
        elif pos in ['WR', 'TE']:
            extra_rows += 3  # targets, avg_yac, receiving_first_downs
    
    new_n_rows = n_rows + extra_rows
    
    # Create expanded matrices
    expanded_snap = np.full((new_n_rows, n_cols), np.nan)
    expanded_targets = np.full((new_n_rows, n_cols), np.nan)
    expanded_target_share = np.full((new_n_rows, n_cols), np.nan)
    expanded_avg_yac = np.full((new_n_rows, n_cols), np.nan)
    expanded_rush_fd = np.full((new_n_rows, n_cols), np.nan)
    expanded_rec_fd = np.full((new_n_rows, n_cols), np.nan)
    expanded_comp_pct = np.full((new_n_rows, n_cols), np.nan)
    expanded_labels = []
    expanded_grouping = []
    
    new_row_idx = 0
    for i in range(n_rows):
        pos = grouping_info[i]['position']
        
        # Add the main player row
        expanded_snap[new_row_idx, :] = snap_matrix[i, :]
        expanded_labels.append(row_labels[i])
        expanded_grouping.append(grouping_info[i].copy())
        new_row_idx += 1
        
        # Add position-specific metric rows
        if pos == 'QB':
            # Add completion_percentage row
            expanded_comp_pct[new_row_idx, :] = completion_pct_matrix[i, :]
            expanded_labels.append(f"  → Comp %")
            expanded_grouping.append({
                'player': f"{grouping_info[i]['player']}_comp_pct",
                'team': grouping_info[i]['team'],
                'position': 'QB',
                'metric_type': 'completion_percentage'
            })
            new_row_idx += 1
            
        elif pos == 'RB':
            # Add targets row
            expanded_targets[new_row_idx, :] = target_matrix[i, :]
            expanded_target_share[new_row_idx, :] = target_share_matrix[i, :]
            expanded_labels.append(f"  → Targets")
            expanded_grouping.append({
                'player': f"{grouping_info[i]['player']}_targets",
                'team': grouping_info[i]['team'],
                'position': 'RB',
                'metric_type': 'targets'
            })
            new_row_idx += 1
            
            # Add rush_first_downs row
            expanded_rush_fd[new_row_idx, :] = rush_first_downs_matrix[i, :]
            expanded_labels.append(f"  → Rush 1st D")
            expanded_grouping.append({
                'player': f"{grouping_info[i]['player']}_rush_fd",
                'team': grouping_info[i]['team'],
                'position': 'RB',
                'metric_type': 'rush_first_downs'
            })
            new_row_idx += 1
            
            # Add receiving_first_downs row
            expanded_rec_fd[new_row_idx, :] = receiving_first_downs_matrix[i, :]
            expanded_labels.append(f"  → Rec 1st D")
            expanded_grouping.append({
                'player': f"{grouping_info[i]['player']}_rec_fd",
                'team': grouping_info[i]['team'],
                'position': 'RB',
                'metric_type': 'receiving_first_downs'
            })
            new_row_idx += 1
            
        elif pos in ['WR', 'TE']:
            # Add targets row
            expanded_targets[new_row_idx, :] = target_matrix[i, :]
            expanded_target_share[new_row_idx, :] = target_share_matrix[i, :]
            expanded_labels.append(f"  → Targets")
            expanded_grouping.append({
                'player': f"{grouping_info[i]['player']}_targets",
                'team': grouping_info[i]['team'],
                'position': pos,
                'metric_type': 'targets'
            })
            new_row_idx += 1
            
            # Add avg_yac row
            expanded_avg_yac[new_row_idx, :] = avg_yac_matrix[i, :]
            expanded_labels.append(f"  → Avg YAC")
            expanded_grouping.append({
                'player': f"{grouping_info[i]['player']}_avg_yac",
                'team': grouping_info[i]['team'],
                'position': pos,
                'metric_type': 'avg_yac'
            })
            new_row_idx += 1
            
            # Add receiving_first_downs row
            expanded_rec_fd[new_row_idx, :] = receiving_first_downs_matrix[i, :]
            expanded_labels.append(f"  → Rec 1st D")
            expanded_grouping.append({
                'player': f"{grouping_info[i]['player']}_rec_fd",
                'team': grouping_info[i]['team'],
                'position': pos,
                'metric_type': 'receiving_first_downs'
            })
            new_row_idx += 1
    
    return expanded_snap, expanded_targets, expanded_target_share, expanded_avg_yac, expanded_rush_fd, expanded_rec_fd, expanded_comp_pct, expanded_labels, expanded_grouping


def add_grouping_lines(ax, grouping_info: list[dict], n_cols: int):
    """
    Add horizontal lines to separate positions within a team.
    
    Args:
        ax: Matplotlib axes
        grouping_info: List of dicts with team/position info for each player
        n_cols: Number of columns in the heatmap
    """
    prev_position = None
    
    for i, info in enumerate(grouping_info):
        # Add thin line between positions
        if prev_position is not None and info['position'] != prev_position:
            ax.axhline(y=i - 0.5, color='gray', linewidth=1.5, linestyle='-', alpha=0.7)
        
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
        
        # Use lighter color for metric rows
        metric_type = info.get('metric_type', None)
        if metric_type:
            color = '#FFE4E1'  # Very light pink for metric rows
        else:
            color = position_colors.get(position, 'white')
        
        # Get the y-tick label and add background color
        ytick_label = ax.get_yticklabels()[i]
        ytick_label.set_bbox(dict(boxstyle='round,pad=0.3', 
                                  facecolor=color, 
                                  edgecolor='none', 
                                  alpha=0.7))


def annotate_heatmap_with_metrics(ax, snap_data: np.ndarray, target_data: np.ndarray, 
                                   avg_yac_data: np.ndarray, rush_fd_data: np.ndarray,
                                   rec_fd_data: np.ndarray, comp_pct_data: np.ndarray, 
                                   grouping_info: list[dict], threshold: float = 0.5):
    """
    Add text annotations to heatmap cells with automatic color adjustment.
    Shows percentages for snap share rows and metric values for metric rows.
    
    Args:
        ax: Matplotlib axes
        snap_data: 2D numpy array with snap share values
        target_data: 2D numpy array with target counts
        avg_yac_data: 2D numpy array with avg yards after catch
        rush_fd_data: 2D numpy array with rush first downs
        rec_fd_data: 2D numpy array with receiving first downs
        comp_pct_data: 2D numpy array with completion percentage
        grouping_info: List of dicts with player/position info
        threshold: Threshold for switching text color (0-1)
    """
    texts = []
    for i in range(snap_data.shape[0]):
        metric_type = grouping_info[i].get('metric_type', None)
        
        for j in range(snap_data.shape[1]):
            if metric_type == 'targets':
                # For target rows, show target count
                value = target_data[i, j]
                if np.isnan(value):
                    continue
                text_str = f'{int(value)}'
                text_color = 'white' if value > 5 else 'black'
            elif metric_type == 'avg_yac':
                # For avg YAC rows, show value with 1 decimal
                value = avg_yac_data[i, j]
                if np.isnan(value):
                    continue
                text_str = f'{value:.1f}'
                text_color = 'white' if value > 5 else 'black'
            elif metric_type == 'rush_first_downs':
                # For rush first downs, show count
                value = rush_fd_data[i, j]
                if np.isnan(value):
                    continue
                text_str = f'{int(value)}'
                text_color = 'white' if value > 2 else 'black'
            elif metric_type == 'receiving_first_downs':
                # For receiving first downs, show count
                value = rec_fd_data[i, j]
                if np.isnan(value):
                    continue
                text_str = f'{int(value)}'
                text_color = 'white' if value > 2 else 'black'
            elif metric_type == 'completion_percentage':
                # For completion percentage, show percentage
                value = comp_pct_data[i, j]
                if np.isnan(value):
                    continue
                text_str = f'{value:.0f}%'
                text_color = 'white' if value > 65 else 'black'
            else:
                # For snap share rows, show percentage
                value = snap_data[i, j]
                if np.isnan(value):
                    continue
                text_str = f'{value * 100:.0f}%'
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
        default='pdf',
        choices=['png', 'jpg', 'pdf'],
        help='Output file format'
    )
    parser.add_argument(
        '--min-games',
        type=int,
        default=1,
        help='Minimum number of games a player-team combination must appear in'
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
    
    # Pivot to matrix (grouped by team)
    print("Pivoting data to matrices grouped by team...")
    team_data = pivot_to_matrix(df)
    print(f"Generated data for {len(team_data)} teams")
    
    # Create heatmap (multi-page PDF or separate files)
    print("Generating heatmaps...")
    create_heatmap(
        team_data,
        args.output, args.format, args.dpi
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
