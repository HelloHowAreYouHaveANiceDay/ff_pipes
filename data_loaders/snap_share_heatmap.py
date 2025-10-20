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
import xml.etree.ElementTree as ET
import matplotlib.cm as cm
import matplotlib.colors as colors
import subprocess
import shutil


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
        'percent_share_of_intended_air_yards',
        'avg_yac',
        'rush_first_downs',
        'receiving_first_downs',
        'completion_percentage'
    ])
    
    # Remove rows with null snap share
    df = df.filter(pl.col('offense_snap_share').is_not_null())
    
    # Filter out players with too few appearances
    player_game_counts = df.group_by('player_name').agg(
        pl.col('week').count().alias('games_played')
    )
    
    valid_players = player_game_counts.filter(
        pl.col('games_played') >= min_games
    )['player_name'].to_list()
    
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


def pivot_to_matrix(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[int], list[dict]]:
    """
    Transform data into a 2D matrix for heatmap.
    
    Args:
        df: Input DataFrame with composite_label
        
    Returns:
        Tuple of (snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, row_labels, col_labels, grouping_info)
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
    avg_yac_matrix = np.full((len(players), len(weeks)), np.nan)
    rush_first_downs_matrix = np.full((len(players), len(weeks)), np.nan)
    receiving_first_downs_matrix = np.full((len(players), len(weeks)), np.nan)
    completion_pct_matrix = np.full((len(players), len(weeks)), np.nan)
    
    # Fill matrices with values
    for i, player in enumerate(players):
        player_data = df.filter(pl.col('composite_label') == player)
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
    
    return snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, players, weeks, grouping_info


def create_svg_heatmap(
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
    output_path: str
):
    """
    Create an optimized SVG heatmap with per-cell accessibility metadata.
    
    Args:
        snap_matrix: 2D numpy array with snap share values
        target_matrix: 2D numpy array with target counts
        target_share_matrix: 2D numpy array with target share
        avg_yac_matrix: 2D numpy array with avg yards after catch
        rush_first_downs_matrix: 2D numpy array with rush first downs
        receiving_first_downs_matrix: 2D numpy array with receiving first downs
        completion_pct_matrix: 2D numpy array with completion percentage
        row_labels: Player labels for y-axis
        col_labels: Week numbers for x-axis
        grouping_info: List of dicts with team/position info
        output_path: Path to save the SVG file
    """
    # Expand matrices for position metrics
    expanded_snap_matrix, expanded_targets, expanded_target_share, expanded_avg_yac, expanded_rush_fd, expanded_rec_fd, expanded_comp_pct, expanded_labels, expanded_grouping = \
        expand_for_position_metrics(snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, row_labels, grouping_info)
    
    n_rows, n_cols = expanded_snap_matrix.shape
    
    # Define cell dimensions for better readability
    cell_width = 50
    cell_height = 20
    margin_left = 150
    margin_top = 80
    margin_right = 20
    margin_bottom = 60
    
    svg_width = margin_left + (n_cols * cell_width) + margin_right
    svg_height = margin_top + (n_rows * cell_height) + margin_bottom
    
    # Create SVG root element with accessibility attributes
    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f'0 0 {svg_width} {svg_height}',
        'preserveAspectRatio': 'xMidYMid meet',
        'role': 'img',
        'aria-labelledby': 'title desc'
    })
    
    # Add title and description for accessibility
    title = ET.SubElement(svg, 'title', {'id': 'title'})
    title.text = '2025 Offensive Snap Share Heatmap - Skill Positions'
    
    desc = ET.SubElement(svg, 'desc', {'id': 'desc'})
    desc.text = 'Heatmap showing offensive snap share percentages by week for NFL skill position players (QB, RB, WR, TE). Each cell is colored by snap share percentage, with darker colors indicating higher snap share. Additional metrics shown include targets, yards after catch, first downs, and completion percentage.'
    
    # Define color gradients (matching the matplotlib implementation)
    snap_colors = ['#F5F5F5', '#D0D0CF', '#ABABAA', '#868785', '#50514F']
    target_colors = ['#FEFEFE', '#FCCECD', '#FA9D9B', '#F66D69', '#F25F5C']
    rec_fd_colors = ['#FEFEFE', '#D9F0EC', '#B3E1D9', '#8CD1C6', '#70C1B3']
    rush_fd_colors = ['#FEFEFE', '#C8DDE8', '#91BBD2', '#5A99BB', '#247BA0']
    yac_colors = ['#FEFEFE', '#F9F0E5', '#F6E5CB', '#F3DAB1', '#EDCB96']
    comp_colors = ['#FEFEFE', '#C8DDE8', '#91BBD2', '#5A99BB', '#247BA0']
    
    # Create colormaps
    snap_cmap = colors.LinearSegmentedColormap.from_list('snap_gradient', snap_colors, N=256)
    target_cmap = colors.LinearSegmentedColormap.from_list('target_gradient', target_colors, N=256)
    rec_fd_cmap = colors.LinearSegmentedColormap.from_list('rec_fd_gradient', rec_fd_colors, N=256)
    rush_fd_cmap = colors.LinearSegmentedColormap.from_list('rush_fd_gradient', rush_fd_colors, N=256)
    yac_cmap = colors.LinearSegmentedColormap.from_list('yac_gradient', yac_colors, N=256)
    comp_cmap = colors.LinearSegmentedColormap.from_list('comp_gradient', comp_colors, N=256)
    
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
    
    # Create a group for the heatmap cells
    cells_group = ET.SubElement(svg, 'g', {'id': 'heatmap-cells'})
    
    # Draw cells
    for i in range(n_rows):
        metric_type = expanded_grouping[i].get('metric_type', None)
        player = expanded_grouping[i]['player']
        position = expanded_grouping[i]['position']
        
        for j in range(n_cols):
            week = col_labels[j]
            x = margin_left + (j * cell_width)
            y = margin_top + (i * cell_height)
            
            # Determine color and value based on metric type
            if metric_type == 'targets':
                target_count = expanded_targets[i, j]
                if not np.isnan(target_count):
                    normalized_value = min(target_count / max_targets, 1.0)
                    rgba = target_cmap(normalized_value)
                    value_text = f'{int(target_count)}'
                    description = f'Week {week}: {player} - {int(target_count)} targets'
                else:
                    rgba = [1, 1, 1, 1]
                    value_text = ''
                    description = f'Week {week}: {player} - No data'
            elif metric_type == 'avg_yac':
                yac_value = expanded_avg_yac[i, j]
                if not np.isnan(yac_value):
                    normalized_value = min(yac_value / max_yac, 1.0)
                    rgba = yac_cmap(normalized_value)
                    value_text = f'{yac_value:.1f}'
                    description = f'Week {week}: {player} - {yac_value:.1f} avg yards after catch'
                else:
                    rgba = [1, 1, 1, 1]
                    value_text = ''
                    description = f'Week {week}: {player} - No data'
            elif metric_type == 'rush_first_downs':
                rush_fd = expanded_rush_fd[i, j]
                if not np.isnan(rush_fd):
                    normalized_value = min(rush_fd / max_rush_fd, 1.0)
                    rgba = rush_fd_cmap(normalized_value)
                    value_text = f'{int(rush_fd)}'
                    description = f'Week {week}: {player} - {int(rush_fd)} rush first downs'
                else:
                    rgba = [1, 1, 1, 1]
                    value_text = ''
                    description = f'Week {week}: {player} - No data'
            elif metric_type == 'receiving_first_downs':
                rec_fd = expanded_rec_fd[i, j]
                if not np.isnan(rec_fd):
                    normalized_value = min(rec_fd / max_rec_fd, 1.0)
                    rgba = rec_fd_cmap(normalized_value)
                    value_text = f'{int(rec_fd)}'
                    description = f'Week {week}: {player} - {int(rec_fd)} receiving first downs'
                else:
                    rgba = [1, 1, 1, 1]
                    value_text = ''
                    description = f'Week {week}: {player} - No data'
            elif metric_type == 'completion_percentage':
                comp_pct = expanded_comp_pct[i, j]
                if not np.isnan(comp_pct):
                    normalized_value = max(0, min((comp_pct - 40) / 60, 1.0))
                    rgba = comp_cmap(normalized_value)
                    value_text = f'{comp_pct:.0f}%'
                    description = f'Week {week}: {player} - {comp_pct:.0f}% completion percentage'
                else:
                    rgba = [1, 1, 1, 1]
                    value_text = ''
                    description = f'Week {week}: {player} - No data'
            else:
                # Snap share
                value = expanded_snap_matrix[i, j]
                if not np.isnan(value):
                    rgba = snap_cmap(value)
                    value_text = f'{value * 100:.0f}%'
                    description = f'Week {week}: {player} ({position}) - {value * 100:.0f}% snap share'
                else:
                    rgba = [1, 1, 1, 1]
                    value_text = ''
                    description = f'Week {week}: {player} - No data'
            
            hexcol = colors.to_hex(rgba)
            
            # Create rect element with accessibility attributes
            rect = ET.SubElement(cells_group, 'rect', {
                'x': str(x),
                'y': str(y),
                'width': str(cell_width),
                'height': str(cell_height),
                'fill': hexcol,
                'stroke': '#CCCCCC',
                'stroke-width': '0.5',
                'id': f'cell-{i}-{j}',
                'data-player': player,
                'data-week': str(week),
                'data-position': position,
                'data-value': value_text.strip('%'),
                'aria-label': description
            })
            
            # Add text annotation if there's a value
            if value_text:
                # Determine text color for readability
                if metric_type == 'targets':
                    text_color = 'white' if normalized_value > 0.4 else 'black'
                elif metric_type == 'avg_yac':
                    text_color = 'white' if normalized_value > 0.5 else 'black'
                elif metric_type in ['rush_first_downs', 'receiving_first_downs']:
                    text_color = 'white' if normalized_value > 0.4 else 'black'
                elif metric_type == 'completion_percentage':
                    text_color = 'white' if normalized_value > 0.5 else 'black'
                else:
                    # Snap share
                    text_color = 'white' if expanded_snap_matrix[i, j] > 0.5 else 'black'
                
                text = ET.SubElement(cells_group, 'text', {
                    'x': str(x + cell_width / 2),
                    'y': str(y + cell_height / 2),
                    'text-anchor': 'middle',
                    'dominant-baseline': 'middle',
                    'fill': text_color,
                    'font-size': '10',
                    'font-weight': 'bold',
                    'font-family': 'Arial, sans-serif'
                })
                text.text = value_text
    
    # Add column labels (weeks)
    labels_group = ET.SubElement(svg, 'g', {'id': 'column-labels'})
    for j, week in enumerate(col_labels):
        x = margin_left + (j * cell_width) + (cell_width / 2)
        y = margin_top - 10
        text = ET.SubElement(labels_group, 'text', {
            'x': str(x),
            'y': str(y),
            'text-anchor': 'middle',
            'fill': 'black',
            'font-size': '12',
            'font-family': 'Arial, sans-serif'
        })
        text.text = f'Week {week}'
    
    # Add row labels (players)
    row_labels_group = ET.SubElement(svg, 'g', {'id': 'row-labels'})
    position_colors = {
        'QB': '#AED6F1',
        'RB': '#FFDAB9',
        'WR': '#FFB6C1',
        'TE': '#DDA0DD'
    }
    
    for i, label in enumerate(expanded_labels):
        x = margin_left - 5
        y = margin_top + (i * cell_height) + (cell_height / 2)
        position = expanded_grouping[i]['position']
        metric_type = expanded_grouping[i].get('metric_type', None)
        
        # Use lighter color for metric rows
        if metric_type:
            bg_color = '#FFE4E1'
        else:
            bg_color = position_colors.get(position, 'white')
        
        # Add background rectangle
        bg_rect = ET.SubElement(row_labels_group, 'rect', {
            'x': str(x - 145),
            'y': str(y - 8),
            'width': '140',
            'height': '16',
            'fill': bg_color,
            'opacity': '0.7',
            'rx': '3'
        })
        
        text = ET.SubElement(row_labels_group, 'text', {
            'x': str(x),
            'y': str(y),
            'text-anchor': 'end',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'font-size': '9',
            'font-family': 'Arial, sans-serif'
        })
        text.text = label
    
    # Add title
    title_text = ET.SubElement(svg, 'text', {
        'x': str(svg_width / 2),
        'y': '30',
        'text-anchor': 'middle',
        'fill': 'black',
        'font-size': '16',
        'font-weight': 'bold',
        'font-family': 'Arial, sans-serif'
    })
    title_text.text = '2025 Offensive Snap Share by Week - Skill Positions'
    
    subtitle_text = ET.SubElement(svg, 'text', {
        'x': str(svg_width / 2),
        'y': '50',
        'text-anchor': 'middle',
        'fill': 'black',
        'font-size': '10',
        'font-family': 'Arial, sans-serif'
    })
    subtitle_text.text = '(QB: Comp%, RB: Targets/Rush 1st Down/Rec 1st Down, WR/TE: Targets/Avg YAC/Rec 1st Down)'
    
    # Write to file
    tree = ET.ElementTree(svg)
    ET.indent(tree, space='  ')
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"SVG heatmap saved to: {output_path}")
    
    # Get file size before optimization
    file_size_before = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Size: {file_size_before:.2f} MB")
    
    # Optimize with SVGO if available
    if shutil.which('svgo'):
        print(f"Optimizing SVG with SVGO...")
        try:
            # Create temporary config file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write("""export default {
  multipass: true,
  plugins: [
    'cleanupIds',
    'removeMetadata',
    'collapseGroups',
    'removeEmptyContainers',
    {
      name: 'removeTitle',
      active: false
    },
    {
      name: 'removeDesc',
      active: false
    }
  ]
};""")
                config_path = f.name
            
            # Run SVGO
            result = subprocess.run(
                ['svgo', '--config', config_path, output_path, '-o', output_path],
                capture_output=True,
                text=True
            )
            
            # Clean up config file
            Path(config_path).unlink()
            
            if result.returncode == 0:
                file_size_after = Path(output_path).stat().st_size / (1024 * 1024)
                reduction = ((file_size_before - file_size_after) / file_size_before) * 100
                print(f"  Optimized: {file_size_after:.2f} MB ({reduction:.1f}% reduction)")
            else:
                print(f"  SVGO optimization failed: {result.stderr}")
        except Exception as e:
            print(f"  SVGO optimization error: {e}")
    else:
        print(f"  SVGO not found. Install with: npm install -g svgo")


def create_heatmap(
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
    output_path: str,
    output_format: str = 'png',
    dpi: int = 300
):
    """
    Create and save the heatmap visualization.
    
    Args:
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
        output_path: Path to save the output file
        output_format: File format (png, jpg, pdf)
        dpi: DPI for output image
    """
    # Expand matrices to include metric rows for all positions
    expanded_snap_matrix, expanded_targets, expanded_target_share, expanded_avg_yac, expanded_rush_fd, expanded_rec_fd, expanded_comp_pct, expanded_labels, expanded_grouping = \
        expand_for_position_metrics(snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, row_labels, grouping_info)
    
    # Calculate figure size based on data dimensions
    n_rows, n_cols = expanded_snap_matrix.shape
    fig_width = max(8, n_cols * 0.3)  # Reduced from 0.6 to 0.3 (2x narrower)
    fig_height = max(8, n_rows * 0.20)
    
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
    
    # Add visual separators between teams and positions
    add_grouping_lines(ax, expanded_grouping, n_cols)
    
    # Add title
    ax.set_title('2025 Offensive Snap Share by Week - Skill Positions\n(QB: Comp%, RB: Targets/Rush 1st Down/Rec 1st Down, WR/TE: Targets/Avg YAC/Rec 1st Down)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add annotations - percentages for snap share, metric values for metric rows
    annotate_heatmap_with_metrics(ax, expanded_snap_matrix, expanded_targets, expanded_avg_yac, expanded_rush_fd, expanded_rec_fd, expanded_comp_pct, expanded_grouping)
    
    # Configure layout
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_path}.{output_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
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
    Add horizontal lines to separate teams and positions with team labels.
    
    Args:
        ax: Matplotlib axes
        grouping_info: List of dicts with team/position info for each player
        n_cols: Number of columns in the heatmap
    """
    prev_team = None
    prev_position = None
    
    for i, info in enumerate(grouping_info):
        # Add thick line and team label between teams
        if prev_team is not None and info['team'] != prev_team:
            ax.axhline(y=i - 0.5, color='black', linewidth=3.5, linestyle='-')
            
            # Add team label on the left side of the plot
            # Position it at the line between teams
            ax.text(-0.5, i - 0.5, f" {prev_team} ", 
                   ha='right', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='black', linewidth=1.5))
        # Add thin line between positions within same team
        elif prev_position is not None and info['position'] != prev_position:
            ax.axhline(y=i - 0.5, color='gray', linewidth=1, linestyle='--', alpha=0.6)
        
        prev_team = info['team']
        prev_position = info['position']
    
    # Add label for the last team at the bottom
    if prev_team is not None:
        ax.text(-0.5, len(grouping_info) - 0.5, f" {prev_team} ", 
               ha='right', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                       edgecolor='black', linewidth=1.5))


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
        default='png',
        choices=['png', 'jpg', 'pdf', 'svg'],
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
    snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix, row_labels, col_labels, grouping_info = pivot_to_matrix(df)
    print(f"Matrix shape: {snap_matrix.shape[0]} players × {snap_matrix.shape[1]} weeks")
    
    # Create heatmap
    print("Generating heatmap...")
    if args.format == 'svg':
        # Generate SVG
        svg_output = f"{args.output}.svg"
        create_svg_heatmap(
            snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, 
            rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix,
            row_labels, col_labels, grouping_info,
            svg_output
        )
    else:
        # Generate raster format (PNG/JPG/PDF)
        create_heatmap(
            snap_matrix, target_matrix, target_share_matrix, avg_yac_matrix, 
            rush_first_downs_matrix, receiving_first_downs_matrix, completion_pct_matrix,
            row_labels, col_labels, grouping_info,
            args.output, args.format, args.dpi
        )
    
    print("Done!")


if __name__ == '__main__':
    main()
