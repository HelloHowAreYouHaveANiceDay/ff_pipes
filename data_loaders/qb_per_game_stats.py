# per-game quarterback performance report
# calculates per-player per-game quarterback stats and efficiency metrics

# inputs:
# - player_stats - https://nflreadr.nflverse.com/articles/dictionary_player_stats.html
# - pbp (optional for advanced metrics) - https://nflreadr.nflverse.com/articles/dictionary_pbp.html
# - team_stats (for team-level aggregates) - https://nflreadr.nflverse.com/articles/dictionary_team_stats.html

# output:
# one csv per season with per-quarterback per-game performance and efficiency metrics
# columns per specification in file header comments

import argparse
import sys
from pathlib import Path

import nflreadpy
import polars as pl


def calculate_team_passing_totals(player_data):
    """
    Calculate team passing totals from player stats.
    
    Args:
        player_data: Polars DataFrame with player stats
        
    Returns:
        Polars DataFrame with columns: season, week, team, team_pass_attempts, team_pass_yards
    """
    print("  Calculating team passing totals...")
    
    team_totals = (
        player_data
        .group_by(['season', 'week', 'team'])
        .agg([
            pl.col('attempts').sum().alias('team_pass_attempts'),
            pl.col('passing_yards').sum().alias('team_pass_yards')
        ])
    )
    
    print(f"  Calculated totals for {len(team_totals)} team-week combinations")
    return team_totals


def calculate_qb_pbp_metrics(pbp_data):
    """
    Calculate QB-specific metrics from play-by-play data.
    
    Calculates:
    - epa_per_dropback
    - success_rate (% of dropbacks with positive EPA)
    - cpoe (completion percentage over expected)
    - air_yards_per_attempt
    - big_play_rate (% of pass attempts gaining 20+ yards)
    - pressure_rate (% of dropbacks under pressure)
    
    Args:
        pbp_data: Polars DataFrame with PBP data
        
    Returns:
        Polars DataFrame with QB metrics per game
    """
    print("  Calculating QB metrics from PBP data...")
    
    # Filter to QB dropbacks (pass attempts + sacks)
    qb_plays = pbp_data.filter(
        (pl.col('pass_attempt') == 1) | (pl.col('sack') == 1)
    )
    
    # Calculate metrics per QB per game
    qb_metrics = (
        qb_plays
        .group_by(['game_id', 'season', 'week', 'passer_player_id', 'posteam'])
        .agg([
            # EPA metrics
            pl.col('qb_epa').mean().alias('epa_per_dropback'),
            pl.col('qb_epa').filter(pl.col('qb_epa') > 0).count().alias('success_plays'),
            pl.col('qb_epa').count().alias('total_dropbacks_pbp'),
            
            # CPOE
            pl.col('cpoe').mean().alias('cpoe'),
            
            # Air yards
            pl.col('air_yards').filter(pl.col('pass_attempt') == 1).mean().alias('air_yards_per_attempt'),
            
            # Big plays (20+ yards)
            pl.col('yards_gained').filter(
                (pl.col('pass_attempt') == 1) & (pl.col('yards_gained') >= 20)
            ).count().alias('big_plays'),
            pl.col('pass_attempt').sum().alias('pass_attempts_pbp'),
            
            # Pressure
            pl.col('qb_hit').filter(pl.col('qb_hit') == 1).count().alias('pressures'),
        ])
        .with_columns([
            # Success rate
            pl.when(pl.col('total_dropbacks_pbp') > 0)
                .then(pl.col('success_plays') / pl.col('total_dropbacks_pbp'))
                .otherwise(None)
                .alias('success_rate'),
            
            # Big play rate
            pl.when(pl.col('pass_attempts_pbp') > 0)
                .then(pl.col('big_plays') / pl.col('pass_attempts_pbp'))
                .otherwise(None)
                .alias('big_play_rate'),
            
            # Pressure rate
            pl.when(pl.col('total_dropbacks_pbp') > 0)
                .then(pl.col('pressures') / pl.col('total_dropbacks_pbp'))
                .otherwise(None)
                .alias('pressure_rate')
        ])
        .rename({
            'passer_player_id': 'player_id',
            'posteam': 'team'
        })
        .select([
            'game_id', 'season', 'week', 'player_id', 'team',
            'epa_per_dropback', 'success_rate', 'cpoe', 
            'air_yards_per_attempt', 'big_play_rate', 'pressure_rate'
        ])
    )
    
    print(f"  Calculated PBP metrics for {len(qb_metrics)} QB-game combinations")
    return qb_metrics


def generate_qb_per_game_report(seasons):
    """
    Generate per-game quarterback performance reports with efficiency metrics.
    
    Args:
        seasons: int, list of ints, or True for all available seasons
        
    Returns:
        dict: Dictionary mapping season to DataFrame
    """
    print(f"Loading data for seasons: {seasons}")
    
    # 1. Load player stats
    print("\n1. Loading player stats...")
    player_data = nflreadpy.load_player_stats(seasons)
    
    if player_data.is_empty():
        print("No player data loaded. Exiting.")
        return {}
    
    print(f"  Loaded {len(player_data)} player stat records")
    
    # Filter to QBs with pass attempts
    qb_data = player_data.filter(
        (pl.col('position') == 'QB') & (pl.col('attempts') > 0)
    )
    print(f"  Filtered to {len(qb_data)} QB records with attempts > 0")
    
    if qb_data.is_empty():
        print("No QB data found. Exiting.")
        return {}
    
    # Handle column name variations
    available_cols = qb_data.columns
    if 'player_display_name' in available_cols:
        player_name_col = 'player_display_name'
    elif 'player_name' in available_cols:
        player_name_col = 'player_name'
    else:
        raise ValueError("Could not find player name column")
    
    if 'opponent_team' in available_cols:
        opponent_col = 'opponent_team'
    elif 'opponent' in available_cols:
        opponent_col = 'opponent'
    else:
        opponent_col = None
    
    # Map column names from player_stats to our spec
    # player_stats columns: passing_interceptions, sacks_suffered, sack_yards_lost, carries, rushing_yards, rushing_tds
    qb_data = (
        qb_data
        .rename({
            player_name_col: 'player',
            'passing_interceptions': 'interceptions',
            'sacks_suffered': 'sacks',
            'sack_yards_lost': 'sack_yards',
            'carries': 'rush_attempts',
            'rushing_yards': 'rush_yards',
            'rushing_tds': 'rush_tds',
            'rushing_fumbles_lost': 'rush_fumbles_lost'
        })
    )
    
    if opponent_col:
        qb_data = qb_data.rename({opponent_col: 'opponent'})
    else:
        qb_data = qb_data.with_columns(pl.lit('').alias('opponent'))
    
    # Select needed columns (don't select game_id yet - we'll get it from PBP or generate it)
    select_cols = [
        'season', 'week', 'player', 'player_id', 'team', 'opponent', 'position',
        'completions', 'attempts', 'passing_yards', 'passing_tds', 
        'interceptions', 'sacks', 'sack_yards',
        'rush_attempts', 'rush_yards', 'rush_tds',
        'sack_fumbles_lost', 'rush_fumbles_lost'
    ]
    
    qb_data = qb_data.select(select_cols).rename({'player_id': 'pfr_player_id'})
    
    # 2. Calculate team passing totals
    print("\n2. Calculating team passing totals...")
    team_totals = calculate_team_passing_totals(
        player_data.rename({
            'passing_interceptions': 'interceptions',
            'sacks_suffered': 'sacks',
            'sack_yards_lost': 'sack_yards'
        })
    )
    
    # 3. Load and process PBP data
    print("\n3. Loading play-by-play data...")
    try:
        pbp_data = nflreadpy.load_pbp(seasons)
        
        if not pbp_data.is_empty():
            print(f"  Loaded {len(pbp_data)} play records")
            
            # Extract game_id mapping
            game_ids = (
                pbp_data
                .select(['game_id', 'season', 'week', 'posteam'])
                .unique()
                .rename({'posteam': 'team'})
            )
            print(f"  Extracted {len(game_ids)} unique team-week-game combinations")
            
            # Calculate QB metrics
            qb_pbp_metrics = calculate_qb_pbp_metrics(pbp_data)
        else:
            print("  No PBP data loaded, will skip advanced metrics and game_id")
            game_ids = None
            qb_pbp_metrics = None
    except Exception as e:
        print(f"  Warning: Could not load PBP data: {e}")
        print("  Will skip advanced metrics and game_id")
        game_ids = None
        qb_pbp_metrics = None
    
    # 4. Join all data
    print("\n4. Joining datasets...")
    joined_data = qb_data
    
    # Join game_id if available
    if game_ids is not None:
        joined_data = joined_data.join(game_ids, on=['season', 'week', 'team'], how='left')
    else:
        # Create a placeholder game_id
        joined_data = joined_data.with_columns(pl.lit('').alias('game_id'))
    
    # Join team totals
    joined_data = joined_data.join(team_totals, on=['season', 'week', 'team'], how='left')
    
    # Join PBP metrics if available
    if qb_pbp_metrics is not None:
        # Rename player_id to pfr_player_id in pbp_metrics to match our main data
        qb_pbp_metrics = qb_pbp_metrics.rename({'player_id': 'pfr_player_id'})
        joined_data = joined_data.join(
            qb_pbp_metrics,
            on=['game_id', 'season', 'week', 'pfr_player_id', 'team'],
            how='left'
        )
    else:
        # Add null columns for PBP metrics
        joined_data = joined_data.with_columns([
            pl.lit(None).alias('epa_per_dropback'),
            pl.lit(None).alias('success_rate'),
            pl.lit(None).alias('cpoe'),
            pl.lit(None).alias('air_yards_per_attempt'),
            pl.lit(None).alias('big_play_rate'),
            pl.lit(None).alias('pressure_rate')
        ])
    
    print(f"  Joined data: {len(joined_data)} rows")
    
    # 5. Calculate derived metrics
    print("\n5. Calculating derived metrics...")
    joined_data = (
        joined_data
        .with_columns([
            # Basic QB metrics
            # completion_pct = completions / attempts
            pl.when(pl.col('attempts') > 0)
                .then(pl.col('completions') / pl.col('attempts'))
                .otherwise(None)
                .alias('completion_pct'),
            
            # yards_per_attempt = passing_yards / attempts
            pl.when(pl.col('attempts') > 0)
                .then(pl.col('passing_yards') / pl.col('attempts'))
                .otherwise(None)
                .alias('yards_per_attempt'),
            
            # dropbacks = attempts + sacks
            (pl.col('attempts') + pl.col('sacks')).alias('dropbacks'),
            
            # turnovers = interceptions + sack_fumbles_lost + rush_fumbles_lost
            (pl.col('interceptions') + 
             pl.col('sack_fumbles_lost').fill_null(0) + 
             pl.col('rush_fumbles_lost').fill_null(0)).alias('turnovers'),
        ])
        .with_columns([
            # net_yards_per_attempt = (passing_yards - sack_yards) / attempts
            pl.when(pl.col('attempts') > 0)
                .then((pl.col('passing_yards') - pl.col('sack_yards')) / pl.col('attempts'))
                .otherwise(None)
                .alias('net_yards_per_attempt'),
            
            # adjusted_net_yards_per_attempt = (pass_yds + 20*TD - 45*INT - sack_yds) / (att + sacks)
            pl.when(pl.col('dropbacks') > 0)
                .then(
                    (pl.col('passing_yards') + 
                     20 * pl.col('passing_tds') - 
                     45 * pl.col('interceptions') - 
                     pl.col('sack_yards')) / pl.col('dropbacks')
                )
                .otherwise(None)
                .alias('adjusted_net_yards_per_attempt'),
            
            # yards_per_dropback = (passing_yards - sack_yards) / dropbacks
            pl.when(pl.col('dropbacks') > 0)
                .then((pl.col('passing_yards') - pl.col('sack_yards')) / pl.col('dropbacks'))
                .otherwise(None)
                .alias('yards_per_dropback'),
            
            # td_int_ratio = passing_tds / interceptions
            pl.when(pl.col('interceptions') > 0)
                .then(pl.col('passing_tds') / pl.col('interceptions'))
                .otherwise(None)
                .alias('td_int_ratio'),
            
            # sack_rate = sacks / dropbacks
            pl.when(pl.col('dropbacks') > 0)
                .then(pl.col('sacks') / pl.col('dropbacks'))
                .otherwise(None)
                .alias('sack_rate'),
            
            # Total metrics
            (pl.col('passing_yards') + pl.col('rush_yards').fill_null(0)).alias('total_yards'),
            (pl.col('passing_tds') + pl.col('rush_tds').fill_null(0)).alias('total_tds'),
            (pl.col('dropbacks') + pl.col('rush_attempts').fill_null(0)).alias('total_plays'),
        ])
        .with_columns([
            # Team share metrics
            pl.when(pl.col('team_pass_attempts') > 0)
                .then(pl.col('attempts') / pl.col('team_pass_attempts'))
                .otherwise(None)
                .alias('team_pass_attempt_share'),
            
            pl.when(pl.col('team_pass_yards') > 0)
                .then(pl.col('passing_yards') / pl.col('team_pass_yards'))
                .otherwise(None)
                .alias('team_pass_yard_share'),
        ])
    )
    
    # 6. Calculate passer rating (NFL formula)
    print("\n6. Calculating passer rating...")
    joined_data = (
        joined_data
        .with_columns([
            # Passer rating components (clamped 0-2.375)
            # a = ((completions / attempts) - 0.3) * 5
            pl.when(pl.col('attempts') > 0)
                .then(
                    ((pl.col('completions') / pl.col('attempts')) - 0.3) * 5
                )
                .otherwise(0)
                .clip(0, 2.375)
                .alias('pr_a'),
            
            # b = ((passing_yards / attempts) - 3) * 0.25
            pl.when(pl.col('attempts') > 0)
                .then(
                    ((pl.col('passing_yards') / pl.col('attempts')) - 3) * 0.25
                )
                .otherwise(0)
                .clip(0, 2.375)
                .alias('pr_b'),
            
            # c = (passing_tds / attempts) * 20
            pl.when(pl.col('attempts') > 0)
                .then(
                    (pl.col('passing_tds') / pl.col('attempts')) * 20
                )
                .otherwise(0)
                .clip(0, 2.375)
                .alias('pr_c'),
            
            # d = 2.375 - ((interceptions / attempts) * 25)
            pl.when(pl.col('attempts') > 0)
                .then(
                    2.375 - ((pl.col('interceptions') / pl.col('attempts')) * 25)
                )
                .otherwise(2.375)
                .clip(0, 2.375)
                .alias('pr_d'),
        ])
        .with_columns([
            # passer_rating = ((a + b + c + d) / 6) * 100
            ((pl.col('pr_a') + pl.col('pr_b') + pl.col('pr_c') + pl.col('pr_d')) / 6 * 100)
                .alias('passer_rating')
        ])
    )
    
    # 7. Calculate fantasy points
    print("\n7. Calculating fantasy points...")
    joined_data = (
        joined_data
        .with_columns([
            # Standard scoring: pass_yds*0.04 + pass_td*4 - int*2 + rush_yds*0.1 + rush_td*6
            (pl.col('passing_yards') * 0.04 +
             pl.col('passing_tds') * 4 -
             pl.col('interceptions') * 2 +
             pl.col('rush_yards').fill_null(0) * 0.1 +
             pl.col('rush_tds').fill_null(0) * 6)
            .alias('fantasy_points_standard'),
        ])
        .with_columns([
            # PPR same as standard for QBs
            pl.col('fantasy_points_standard').alias('fantasy_points_ppr')
        ])
    )
    
    # 8. Calculate week-over-week changes
    print("\n8. Calculating week-over-week changes...")
    joined_data = (
        joined_data
        .sort(['pfr_player_id', 'team', 'season', 'week'])
        .with_columns([
            # Previous week values
            pl.col('passing_yards').shift(1).over(['pfr_player_id', 'team', 'season']).alias('prev_passing_yards'),
            pl.col('total_tds').shift(1).over(['pfr_player_id', 'team', 'season']).alias('prev_total_tds'),
            pl.col('epa_per_dropback').shift(1).over(['pfr_player_id', 'team', 'season']).alias('prev_epa_per_dropback'),
        ])
        .with_columns([
            # WoW changes
            (pl.col('passing_yards') - pl.col('prev_passing_yards')).alias('wow_yards_change'),
            (pl.col('total_tds') - pl.col('prev_total_tds')).alias('wow_td_change'),
            (pl.col('epa_per_dropback') - pl.col('prev_epa_per_dropback')).alias('wow_epa_change'),
        ])
    )
    
    # Get unique seasons
    unique_seasons = joined_data['season'].unique().sort()
    print(f"\nProcessing seasons: {unique_seasons.to_list()}")
    
    results = {}
    
    for season in unique_seasons:
        print(f"\nFinalizing season {season}...")
        season_data = joined_data.filter(pl.col('season') == season)
        
        # 9. Select and order columns per specification
        final_report = (
            season_data
            .select([
                'season',
                'week',
                'game_id',
                'player',
                'pfr_player_id',
                'team',
                'opponent',
                'position',
                'attempts',
                'completions',
                'completion_pct',
                'passing_yards',
                'passing_tds',
                'interceptions',
                'sacks',
                'sack_yards',
                'yards_per_attempt',
                'net_yards_per_attempt',
                'adjusted_net_yards_per_attempt',
                'passer_rating',
                'dropbacks',
                'yards_per_dropback',
                'total_plays',
                'rush_attempts',
                'rush_yards',
                'rush_tds',
                'total_yards',
                'total_tds',
                'turnovers',
                'td_int_ratio',
                'team_pass_attempt_share',
                'team_pass_yard_share',
                'epa_per_dropback',
                'success_rate',
                'cpoe',
                'air_yards_per_attempt',
                'big_play_rate',
                'pressure_rate',
                'sack_rate',
                'fantasy_points_standard',
                'fantasy_points_ppr',
                'wow_yards_change',
                'wow_td_change',
                'wow_epa_change'
            ])
            .rename({'attempts': 'pass_attempts'})
            .sort(['week', 'team', 'player'])
        )
        
        results[season] = final_report
        print(f"  Season {season}: {len(final_report)} rows in final report")
    
    return results


def write_reports_to_csv(reports, output_dir='./out'):
    """
    Write QB performance reports to CSV files.
    
    Args:
        reports: dict mapping season to DataFrame
        output_dir: directory to write CSV files to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for season, df in reports.items():
        filename = output_path / f'qb_per_game_stats_{season}.csv'
        df.write_csv(filename)
        print(f"Wrote {filename} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-game quarterback performance reports with efficiency metrics'
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
        reports = generate_qb_per_game_report(seasons)
        
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
