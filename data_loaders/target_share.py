# Per-Player, Per-Game Receiving Metrics + Weekly Target Share Report

import argparse
import sys
from pathlib import Path

import nflreadpy
import polars as pl


def calculate_team_dropbacks(pbp_data):
    """
    Calculate team dropbacks from play-by-play data.
    
    Team dropbacks = pass attempts + sacks + scrambles
    
    Args:
        pbp_data: Polars DataFrame with PBP data
        
    Returns:
        Polars DataFrame with columns: season, week, team, team_dropbacks
    """
    print("  Calculating team dropbacks from PBP data...")
    
    dropbacks = (
        pbp_data
        .group_by(['season', 'week', 'posteam'])
        .agg([
            pl.col('pass_attempt').sum().alias('pass_attempts'),
            pl.col('sack').sum().alias('sacks'),
            pl.col('qb_scramble').sum().alias('scrambles')
        ])
        .with_columns([
            (pl.col('pass_attempts') + pl.col('sacks') + pl.col('scrambles'))
                .alias('team_dropbacks')
        ])
        .rename({'posteam': 'team'})
        .select(['season', 'week', 'team', 'team_dropbacks'])
    )
    
    print(f"  Calculated dropbacks for {len(dropbacks)} team-week combinations")
    return dropbacks


def generate_receiving_opportunity_report(seasons):
    """
    Generate receiving opportunity reports with target share metrics.
    
    Args:
        seasons: int, list of ints, or True for all available seasons
        
    Returns:
        dict: Dictionary mapping season to DataFrame
    """
    print(f"Loading data for seasons: {seasons}")
    
    # 1. Load player stats (weekly)
    print("\n1. Loading player stats...")
    player_data = nflreadpy.load_player_stats(seasons)
    
    if player_data.is_empty():
        print("No player data loaded. Exiting.")
        return {}
    
    print(f"  Loaded {len(player_data)} player stat records")
    
    # Filter to players with targets (receiving-relevant players)
    player_data = player_data.filter(pl.col('targets') > 0)
    print(f"  Filtered to {len(player_data)} records with targets > 0")
    
    # Verify columns exist, adjust if needed
    available_cols = player_data.columns
    if 'player_display_name' in available_cols:
        player_name_col = 'player_display_name'
    elif 'player_name' in available_cols:
        player_name_col = 'player_name'
    else:
        print("Error: Could not find player name column")
        return {}
    
    if 'opponent_team' in available_cols:
        opponent_col = 'opponent_team'
    elif 'opponent' in available_cols:
        opponent_col = 'opponent'
    else:
        opponent_col = None
    
    # Build selection list with actual column names
    select_cols = ['season', 'week', player_name_col, 'player_id',
                   'position', 'team']
    if opponent_col:
        select_cols.append(opponent_col)
    select_cols.extend(['receptions', 'targets'])
    
    player_data = (
        player_data
        .select(select_cols)
        .rename({
            player_name_col: 'player',
            'player_id': 'pfr_player_id',
            opponent_col if opponent_col else 'team': 'opponent'
        })
    )
    
    # 2. Load team stats (weekly)
    print("\n2. Loading team stats...")
    team_data = nflreadpy.load_team_stats(seasons)
    
    if team_data.is_empty():
        print("No team data loaded. Exiting.")
        return {}
    
    print(f"  Loaded {len(team_data)} team stat records")
    
    team_pass_attempts = (
        team_data
        .select(['season', 'week', 'team', 'attempts'])
        .rename({'attempts': 'team_pass_attempts'})
    )
    
    # 3. Load and process PBP for dropbacks
    print("\n3. Loading play-by-play data...")
    pbp_data = nflreadpy.load_pbp(seasons)
    
    if pbp_data.is_empty():
        print("No PBP data loaded. Exiting.")
        return {}
    
    print(f"  Loaded {len(pbp_data)} play records")
    
    # Calculate team dropbacks
    team_dropbacks = calculate_team_dropbacks(pbp_data)
    
    # Extract game_id mapping (game_id, season, week, team/posteam)
    # We need to map teams to their game_ids
    game_ids = (
        pbp_data
        .select(['game_id', 'season', 'week', 'posteam'])
        .unique()
        .rename({'posteam': 'team'})
    )
    
    print(f"  Extracted {len(game_ids)} unique team-week-game combinations")
    
    # 4. Join all data
    print("\n4. Joining datasets...")
    joined_data = (
        player_data
        .join(game_ids, on=['season', 'week', 'team'], how='left')
        .join(team_pass_attempts, on=['season', 'week', 'team'], how='left')
        .join(team_dropbacks, on=['season', 'week', 'team'], how='left')
    )
    
    print(f"  Joined data: {len(joined_data)} rows")
    
    # 5. Calculate derived metrics with safe division
    print("\n5. Calculating derived metrics...")
    joined_data = (
        joined_data
        .with_columns([
            # target_share = targets / team_pass_attempts
            pl.when(pl.col('team_pass_attempts') > 0)
                .then(pl.col('targets') / pl.col('team_pass_attempts'))
                .otherwise(None)
                .alias('target_share'),
            # targets_per_dropback = targets / team_dropbacks
            pl.when(pl.col('team_dropbacks') > 0)
                .then(pl.col('targets') / pl.col('team_dropbacks'))
                .otherwise(None)
                .alias('targets_per_dropback')
        ])
    )
    
    # 6. Calculate WoW changes
    print("\n6. Calculating week-over-week changes...")
    joined_data = (
        joined_data
        .sort(['pfr_player_id', 'team', 'week'])
        .with_columns([
            pl.col('target_share')
                .shift(1)
                .over(['pfr_player_id', 'team', 'season'])
                .alias('prev_target_share')
        ])
        .with_columns([
            (pl.col('target_share') - pl.col('prev_target_share'))
                .alias('wow_target_share_change')
        ])
    )
    
    # Get unique seasons from the data
    unique_seasons = joined_data['season'].unique().sort()
    print(f"\nProcessing seasons: {unique_seasons.to_list()}")
    
    results = {}
    
    for season in unique_seasons:
        print(f"\nFinalizing season {season}...")
        season_data = joined_data.filter(pl.col('season') == season)
        
        # 7. Select and order columns
        final_report = (
            season_data
            .select([
                'season',
                'week',
                'game_id',
                'player',
                'pfr_player_id',
                'position',
                'team',
                'opponent',
                'receptions',
                'targets',
                'team_pass_attempts',
                'team_dropbacks',
                'target_share',
                'targets_per_dropback',
                'wow_target_share_change'
            ])
            .sort(['week', 'team', 'position', 'player'])
        )
        
        results[season] = final_report
        print(f"  Season {season}: {len(final_report)} rows in final report")
    
    return results


def write_reports_to_csv(reports, output_dir='./out'):
    """
    Write receiving opportunity reports to CSV files.
    
    Args:
        reports: dict mapping season to DataFrame
        output_dir: directory to write CSV files to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for season, df in reports.items():
        filename = output_path / f'receiving_opportunity_{season}.csv'
        df.write_csv(filename)
        print(f"Wrote {filename} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate weekly receiving opportunity reports with target share metrics'
    )
    parser.add_argument(
        'seasons',
        nargs='?',
        default='2024',
        help='Season(s) to process. Use "all" for all available, comma-separated list for multiple (e.g., "2023,2024"), or single year. Default: 2024'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='./out',
        help='Output directory for CSV files. Default: ./out'
    )
    
    args = parser.parse_args()
    
    # Parse seasons argument
    if args.seasons.lower() == 'all':
        seasons = True
    elif ',' in args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(',')]
    else:
        seasons = int(args.seasons)
    
    try:
        # Generate reports
        reports = generate_receiving_opportunity_report(seasons)
        
        if not reports:
            print("No reports generated.")
            return 1
        
        # Write to CSV
        print("\n" + "="*60)
        print("Writing CSV files...")
        print("="*60)
        write_reports_to_csv(reports, args.output_dir)
        
        print(f"\n✓ Successfully generated {len(reports)} report(s)")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())