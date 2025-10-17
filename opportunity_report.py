# Player Opportunity Report
# Wide table reporting all of a player's relevant dimensions and stats for that week
# Table is unique on season-week-team-gsis_id

# inputs:
# - players - https://nflreadr.nflverse.com/articles/dictionary_players.html
# - snap_counts - https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html
# - pbp (play by play) - https://nflreadr.nflverse.com/articles/dictionary_pbp.html
# - combine - https://nflreadr.nflverse.com/articles/dictionary_combine.html
# - nextgen_stats - https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html
# - scoring.json (local)

# output:
# one csv per season with per-player per-week opportunity metrics
# see opportunity_report.md for full column specification

import argparse
import json
import sys
from pathlib import Path

import nflreadpy
import polars as pl


def load_scoring_config():
    """Load fantasy scoring configuration from scoring.json"""
    scoring_path = Path(__file__).parent / 'scoring.json'
    with open(scoring_path, 'r') as f:
        return json.load(f)


def load_player_mapping():
    """
    Load players table to create ID mapping between gsis_id and pfr_id.
    
    Returns:
        Polars DataFrame with player metadata and ID mappings
    """
    print("\n1. Loading players table for ID mapping...")
    players = nflreadpy.load_players()
    
    if players.is_empty():
        print("Error: No players data loaded.")
        return None
    
    print(f"  Loaded {len(players)} players")
    
    # Select relevant columns for mapping and profile data
    player_cols = [
        'gsis_id', 'pfr_id', 'display_name', 'position',
        'draft_year', 'draft_round', 'draft_ovr',
        'height', 'weight'
    ]
    
    # Check which columns exist
    available_cols = players.columns
    cols_to_select = [col for col in player_cols if col in available_cols]
    
    players_mapped = players.select(cols_to_select)
    
    # Rename for consistency
    rename_map = {}
    if 'display_name' in cols_to_select:
        rename_map['display_name'] = 'player_name'
    if 'draft_ovr' in cols_to_select:
        rename_map['draft_ovr'] = 'draft_pick'
    if 'height' in cols_to_select:
        rename_map['height'] = 'player_height'
    if 'weight' in cols_to_select:
        rename_map['weight'] = 'player_weight'
    
    if rename_map:
        players_mapped = players_mapped.rename(rename_map)
    
    print(f"  Player mapping ready with {len(players_mapped)} records")
    return players_mapped


def process_snap_counts(seasons, players_df):
    """
    Load snap counts and calculate week-over-week changes.
    
    Args:
        seasons: Season(s) to load
        players_df: Players DataFrame for ID mapping
        
    Returns:
        Polars DataFrame with snap data and WoW changes
    """
    print("\n2. Loading snap counts...")
    snap_data = nflreadpy.load_snap_counts(seasons)
    
    if snap_data.is_empty():
        print("  No snap count data loaded")
        return None
    
    print(f"  Loaded {len(snap_data)} snap count records")
    
    # Join with players to get gsis_id
    snap_with_ids = snap_data.join(
        players_df.select(['gsis_id', 'pfr_id']),
        left_on='pfr_player_id',
        right_on='pfr_id',
        how='left'
    )
    
    print(f"  Joined with players: {len(snap_with_ids)} records")
    
    # Calculate week-over-week snap share changes
    print("  Calculating week-over-week snap share changes...")
    snap_with_wow = (
        snap_with_ids
        .sort(['gsis_id', 'team', 'season', 'week'])
        .with_columns([
            pl.col('offense_pct').shift(1).over(['gsis_id', 'team', 'season']).alias('prev_offense_pct'),
            pl.col('defense_pct').shift(1).over(['gsis_id', 'team', 'season']).alias('prev_defense_pct'),
            pl.col('st_pct').shift(1).over(['gsis_id', 'team', 'season']).alias('prev_st_pct'),
        ])
        .with_columns([
            (pl.col('offense_pct') - pl.col('prev_offense_pct')).alias('wow_offense_snap_share_change'),
            (pl.col('defense_pct') - pl.col('prev_defense_pct')).alias('wow_defense_snap_share_change'),
            (pl.col('st_pct') - pl.col('prev_st_pct')).alias('wow_special_teams_snap_share_change'),
        ])
    )
    
    # Select and rename columns
    snap_final = snap_with_wow.select([
        'gsis_id',
        'season',
        'week',
        'team',
        'opponent',
        'game_id',
        'offense_snaps',
        pl.col('offense_pct').alias('offense_snap_share'),
        'wow_offense_snap_share_change',
        'defense_snaps',
        pl.col('defense_pct').alias('defense_snap_share'),
        'wow_defense_snap_share_change',
        pl.col('st_snaps').alias('special_teams_snaps'),
        pl.col('st_pct').alias('special_teams_snap_share'),
        'wow_special_teams_snap_share_change',
    ])
    
    print(f"  Snap counts processed: {len(snap_final)} records")
    return snap_final


def aggregate_pbp_stats(seasons, players_df):
    """
    Aggregate play-by-play data to get per-player per-game stats.
    
    Args:
        seasons: Season(s) to load
        players_df: Players DataFrame for ID mapping
        
    Returns:
        Polars DataFrame with aggregated stats by player-game
    """
    print("\n3. Loading and aggregating play-by-play data...")
    pbp_data = nflreadpy.load_pbp(seasons)
    
    if pbp_data.is_empty():
        print("  No PBP data loaded")
        return None
    
    print(f"  Loaded {len(pbp_data)} play records")
    
    # === PASSING STATS ===
    print("  Aggregating passing stats...")
    passing_plays = pbp_data.filter(
        pl.col('passer_player_id').is_not_null()
    )
    
    passing_stats = (
        passing_plays
        .group_by(['game_id', 'season', 'week', 'passer_player_id', 'posteam'])
        .agg([
            pl.col('pass_attempt').sum().alias('pass_attempts'),
            pl.col('complete_pass').sum().alias('pass_complete'),
            pl.col('yards_gained').filter(pl.col('pass_attempt') == 1).sum().alias('passing_yards'),
            pl.col('pass_touchdown').sum().alias('pass_touchdowns'),
            pl.col('interception').sum().alias('interceptions'),
            pl.col('first_down_pass').sum().alias('pass_first_downs'),
            pl.col('two_point_conv_result').filter(
                (pl.col('two_point_conv_result') == 'success') & 
                (pl.col('pass_attempt') == 1)
            ).count().alias('pass_2pt_conversions'),
        ])
        .rename({
            'passer_player_id': 'gsis_id',
            'posteam': 'team'
        })
    )
    
    print(f"    {len(passing_stats)} passing records")
    
    # === RECEIVING STATS ===
    print("  Aggregating receiving stats...")
    receiving_plays = pbp_data.filter(
        pl.col('receiver_player_id').is_not_null()
    )
    
    receiving_stats = (
        receiving_plays
        .group_by(['game_id', 'season', 'week', 'receiver_player_id', 'posteam'])
        .agg([
            pl.col('complete_pass').sum().alias('receptions'),
            pl.col('pass_attempt').sum().alias('targets'),
            pl.col('yards_gained').filter(pl.col('complete_pass') == 1).sum().alias('receiving_yards'),
            pl.col('air_yards').filter(pl.col('pass_attempt') == 1).sum().alias('receiving_air_yards'),
            pl.col('yards_after_catch').filter(pl.col('complete_pass') == 1).sum().alias('receiving_yac'),
            pl.col('pass_touchdown').sum().alias('receiving_touchdowns'),
            pl.col('first_down_pass').filter(pl.col('complete_pass') == 1).sum().alias('receiving_first_downs'),
            pl.col('two_point_conv_result').filter(
                (pl.col('two_point_conv_result') == 'success') & 
                (pl.col('complete_pass') == 1)
            ).count().alias('receiving_2pt_conversions'),
        ])
        .rename({
            'receiver_player_id': 'gsis_id',
            'posteam': 'team'
        })
    )
    
    print(f"    {len(receiving_stats)} receiving records")
    
    # === RUSHING STATS ===
    print("  Aggregating rushing stats...")
    rushing_plays = pbp_data.filter(
        pl.col('rusher_player_id').is_not_null()
    )
    
    rushing_stats = (
        rushing_plays
        .group_by(['game_id', 'season', 'week', 'rusher_player_id', 'posteam'])
        .agg([
            pl.col('rush_attempt').sum().alias('rush_attempts'),
            pl.col('yards_gained').filter(pl.col('rush_attempt') == 1).sum().alias('rushing_yards'),
            pl.col('rush_touchdown').sum().alias('rush_touchdowns'),
            pl.col('first_down_rush').sum().alias('rush_first_downs'),
        ])
        .rename({
            'rusher_player_id': 'gsis_id',
            'posteam': 'team'
        })
    )
    
    print(f"    {len(rushing_stats)} rushing records")
    
    # === COMBINE ALL STATS ===
    print("  Combining all stats...")
    
    # Full outer join all stat types
    combined = (
        passing_stats
        .join(
            receiving_stats,
            on=['game_id', 'season', 'week', 'gsis_id', 'team'],
            how='full',
            coalesce=True
        )
        .join(
            rushing_stats,
            on=['game_id', 'season', 'week', 'gsis_id', 'team'],
            how='full',
            coalesce=True
        )
    )
    
    print(f"  Combined PBP stats: {len(combined)} player-game records")
    return combined


def extract_game_opponent_mapping(pbp_data):
    """
    Extract game_id to team/opponent mapping from PBP data.
    
    Args:
        pbp_data: Play-by-play DataFrame
        
    Returns:
        Polars DataFrame with game_id, team, opponent mappings
    """
    print("\n4. Extracting game-opponent mappings...")
    
    # Get unique game info
    game_info = (
        pbp_data
        .select(['game_id', 'season', 'week', 'home_team', 'away_team'])
        .unique()
    )
    
    # Create two records per game (one for each team)
    home_games = game_info.select([
        'game_id',
        'season',
        'week',
        pl.col('home_team').alias('team'),
        pl.col('away_team').alias('opponent'),
    ])
    
    away_games = game_info.select([
        'game_id',
        'season',
        'week',
        pl.col('away_team').alias('team'),
        pl.col('home_team').alias('opponent'),
    ])
    
    game_mappings = pl.concat([home_games, away_games])
    
    print(f"  Extracted {len(game_mappings)} game-team combinations")
    return game_mappings


def load_nextgen_stats(seasons, players_df):
    """
    Load NextGen Stats for passing, receiving, and rushing.
    
    Args:
        seasons: Season(s) to load
        players_df: Players DataFrame for ID mapping
        
    Returns:
        Tuple of (passing_ngs, receiving_ngs, rushing_ngs) DataFrames
    """
    print("\n5. Loading NextGen Stats...")
    
    # Load passing NextGen stats
    print("  Loading passing NextGen stats...")
    try:
        passing_ngs = nflreadpy.load_nextgen_stats(seasons, stat_type='passing')
        if not passing_ngs.is_empty():
            passing_ngs = passing_ngs.select([
                pl.col('player_gsis_id').alias('gsis_id'),
                'season',
                'week',
                'avg_time_to_throw',
                'avg_completed_air_yards',
                'avg_intended_air_yards',
                'avg_air_yards_differential',
                'aggressiveness',
                'max_completed_air_distance',
                'avg_air_yards_to_sticks',
                'completion_percentage',
                'expected_completion_percentage',
                'completion_percentage_above_expectation',
            ])
            print(f"    Loaded {len(passing_ngs)} passing NGS records")
        else:
            passing_ngs = None
            print("    No passing NGS data")
    except Exception as e:
        print(f"    Error loading passing NGS: {e}")
        passing_ngs = None
    
    # Load receiving NextGen stats
    print("  Loading receiving NextGen stats...")
    try:
        receiving_ngs = nflreadpy.load_nextgen_stats(seasons, stat_type='receiving')
        if not receiving_ngs.is_empty():
            # Map actual column names from NGS receiving data
            receiving_cols = ['player_gsis_id', 'season', 'week']
            rename_map = {'player_gsis_id': 'gsis_id'}
            
            # Map NGS columns to our spec names
            ngs_col_mapping = {
                'avg_cushion': 'avg_cushion',
                'avg_separation': 'avg_separation',
                'avg_intended_air_yards': 'avg_air_distance',  # NGS uses this for ADOT
                'percent_share_of_intended_air_yards': 'percent_share_of_intended_air_yards',
                'catch_percentage': 'catch_percentage',
                'avg_yac': 'avg_yac',
                'avg_expected_yac': 'avg_expected_yac',
                'avg_yac_above_expectation': 'avg_yac_above_expectation',
            }
            
            # Add available columns
            for ngs_col, our_col in ngs_col_mapping.items():
                if ngs_col in receiving_ngs.columns:
                    receiving_cols.append(ngs_col)
                    if ngs_col != our_col:
                        rename_map[ngs_col] = our_col
            
            receiving_ngs = receiving_ngs.select(receiving_cols).rename(rename_map)
            
            # Note: max_air_distance not available in NGS, will be null
            # avg_air_distance comes from avg_intended_air_yards
            
            print(f"    Loaded {len(receiving_ngs)} receiving NGS records")
        else:
            receiving_ngs = None
            print("    No receiving NGS data")
    except Exception as e:
        print(f"    Error loading receiving NGS: {e}")
        receiving_ngs = None
    
    # Load rushing NextGen stats
    print("  Loading rushing NextGen stats...")
    try:
        rushing_ngs = nflreadpy.load_nextgen_stats(seasons, stat_type='rushing')
        if not rushing_ngs.is_empty():
            rushing_ngs = rushing_ngs.select([
                pl.col('player_gsis_id').alias('gsis_id'),
                'season',
                'week',
                'efficiency',
                'percent_attempts_gte_eight_defenders',
                'avg_time_to_los',
                'expected_rush_yards',
                'rush_yards_over_expected',
                'avg_rush_yards',
                'rush_yards_over_expected_per_att',
                'rush_pct_over_expected',
            ])
            print(f"    Loaded {len(rushing_ngs)} rushing NGS records")
        else:
            rushing_ngs = None
            print("    No rushing NGS data")
    except Exception as e:
        print(f"    Error loading rushing NGS: {e}")
        rushing_ngs = None
    
    return passing_ngs, receiving_ngs, rushing_ngs


def load_combine_data(players_df):
    """
    Load combine data and join with players.
    
    Args:
        players_df: Players DataFrame for ID mapping
        
    Returns:
        Polars DataFrame with combine metrics by gsis_id
    """
    print("\n6. Loading combine data...")
    try:
        combine = nflreadpy.load_combine()
        
        if combine.is_empty():
            print("  No combine data loaded")
            return None
        
        print(f"  Loaded {len(combine)} combine records")
        
        # Select relevant columns and join with players
        combine_cols = [
            'pfr_id',
            pl.col('forty').alias('fourty'),  # Match spec spelling
            'bench',
            'vertical',
            'broad_jump',
            'cone',
            'shuttle',
        ]
        
        # Check which columns exist
        available_cols = combine.columns
        cols_to_select = ['pfr_id']
        
        if 'forty' in available_cols:
            cols_to_select.append('forty')
        if 'bench' in available_cols:
            cols_to_select.append('bench')
        if 'vertical' in available_cols:
            cols_to_select.append('vertical')
        if 'broad_jump' in available_cols:
            cols_to_select.append('broad_jump')
        if 'cone' in available_cols:
            cols_to_select.append('cone')
        if 'shuttle' in available_cols:
            cols_to_select.append('shuttle')
        
        combine_selected = combine.select(cols_to_select)
        
        # Rename forty to fourty to match spec
        if 'forty' in cols_to_select:
            combine_selected = combine_selected.rename({'forty': 'fourty'})
        
        # Join with players to get gsis_id
        combine_with_gsis = combine_selected.join(
            players_df.select(['gsis_id', 'pfr_id']),
            on='pfr_id',
            how='inner'
        ).select([col for col in ['gsis_id', 'fourty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle'] if col in combine_selected.columns or col == 'gsis_id'])
        
        print(f"  Combine data mapped to {len(combine_with_gsis)} players")
        return combine_with_gsis
        
    except Exception as e:
        print(f"  Error loading combine data: {e}")
        return None


def calculate_fantasy_points(df, scoring_config):
    """
    Calculate fantasy points based on scoring configuration.
    
    Args:
        df: DataFrame with player stats
        scoring_config: Scoring configuration from scoring.json
        
    Returns:
        DataFrame with fantasy_points column added
    """
    print("\n7. Calculating fantasy points...")
    
    scoring = scoring_config['Scoring']
    
    # Build fantasy points calculation
    fantasy_expr = (
        # Passing
        (pl.col('passing_yards').fill_null(0) * scoring['Passing']['PassingYards_PY']) +
        (pl.col('pass_touchdowns').fill_null(0) * scoring['Passing']['TouchdownPass_PTD']) +
        (pl.col('interceptions').fill_null(0) * scoring['Passing']['InterceptionsThrown_INT']) +
        (pl.col('pass_2pt_conversions').fill_null(0) * scoring['Passing']['TwoPointPassingConversion_2PC']) +
        # Rushing
        (pl.col('rushing_yards').fill_null(0) * scoring['Rushing']['RushingYards_RY']) +
        (pl.col('rush_touchdowns').fill_null(0) * scoring['Rushing']['TouchdownRush_RTD']) +
        # Receiving
        (pl.col('receiving_yards').fill_null(0) * scoring['Receiving']['ReceivingYards_REY']) +
        (pl.col('receptions').fill_null(0) * scoring['Receiving']['Reception_REC']) +
        (pl.col('receiving_touchdowns').fill_null(0) * scoring['Receiving']['TouchdownReception_RETD']) +
        (pl.col('receiving_2pt_conversions').fill_null(0) * scoring['Receiving']['TwoPointReceivingConversion_2PRE'])
    ).alias('fantasy_points')
    
    df_with_fantasy = df.with_columns([fantasy_expr])
    
    print("  Fantasy points calculated")
    return df_with_fantasy


def calculate_position_ranks(df):
    """
    Calculate fantasy position ranks for players with at least 1 snap.
    
    Args:
        df: DataFrame with fantasy_points
        
    Returns:
        DataFrame with f_position_rank column added
    """
    print("\n8. Calculating fantasy position ranks...")
    
    # Filter to players with at least 1 offensive snap
    # Calculate rank within each season-week-position group
    df_with_rank = (
        df
        .with_columns([
            pl.when(pl.col('offense_snaps').fill_null(0) >= 1)
                .then(
                    pl.col('fantasy_points').rank(method='dense', descending=True)
                      .over(['season', 'week', 'position'])
                )
                .otherwise(None)
                .alias('f_position_rank')
        ])
    )
    
    print("  Position ranks calculated")
    return df_with_rank


def generate_opportunity_report(seasons):
    """
    Generate comprehensive player opportunity reports.
    
    Args:
        seasons: int, list of ints, or True for all available seasons
        
    Returns:
        dict: Dictionary mapping season to DataFrame
    """
    print(f"Generating opportunity reports for seasons: {seasons}")
    
    # Load scoring configuration
    scoring_config = load_scoring_config()
    
    # Step 1: Load player mapping
    players_df = load_player_mapping()
    if players_df is None:
        return {}
    
    # Step 2: Load snap counts with WoW changes
    snap_data = process_snap_counts(seasons, players_df)
    if snap_data is None:
        return {}
    
    # Step 3: Aggregate PBP stats
    pbp_stats = aggregate_pbp_stats(seasons, players_df)
    
    # Step 4: Extract opponent mappings (need to reload pbp for this)
    print("\n4. Loading PBP for opponent mapping...")
    pbp_data = nflreadpy.load_pbp(seasons)
    game_opponent_map = extract_game_opponent_mapping(pbp_data)
    
    # Step 5: Load NextGen stats
    passing_ngs, receiving_ngs, rushing_ngs = load_nextgen_stats(seasons, players_df)
    
    # Step 6: Load combine data
    combine_data = load_combine_data(players_df)
    
    # === MASTER JOIN ===
    print("\n9. Performing master join of all data sources...")
    
    # Start with snap data as base
    master = snap_data
    
    # Join player metadata
    master = master.join(
        players_df,
        on='gsis_id',
        how='left'
    )
    
    # Join PBP stats
    if pbp_stats is not None:
        master = master.join(
            pbp_stats,
            on=['gsis_id', 'season', 'week', 'team', 'game_id'],
            how='left'
        )
    
    # Fill in opponent if missing using game mapping
    master = master.join(
        game_opponent_map.select(['game_id', 'team', 'opponent']).rename({'opponent': 'opponent_from_game'}),
        on=['game_id', 'team'],
        how='left'
    ).with_columns([
        pl.when(pl.col('opponent').is_null() | (pl.col('opponent') == ''))
            .then(pl.col('opponent_from_game'))
            .otherwise(pl.col('opponent'))
            .alias('opponent')
    ]).drop('opponent_from_game')
    
    # Join NextGen stats
    if passing_ngs is not None:
        master = master.join(
            passing_ngs,
            on=['gsis_id', 'season', 'week'],
            how='left'
        )
    
    if receiving_ngs is not None:
        master = master.join(
            receiving_ngs,
            on=['gsis_id', 'season', 'week'],
            how='left',
            suffix='_rec'
        )
    
    if rushing_ngs is not None:
        master = master.join(
            rushing_ngs,
            on=['gsis_id', 'season', 'week'],
            how='left'
        )
    
    # Join combine data
    if combine_data is not None:
        master = master.join(
            combine_data,
            on='gsis_id',
            how='left'
        )
    
    print(f"  Master dataset: {len(master)} records")
    
    # Calculate fantasy points
    master = calculate_fantasy_points(master, scoring_config)
    
    # Calculate position ranks
    master = calculate_position_ranks(master)
    
    # === FINAL COLUMN SELECTION ===
    print("\n10. Selecting and ordering final columns...")
    
    # Define column order per spec
    final_cols = [
        'season',
        'week',
        'player_name',
        'gsis_id',
        'position',
        'team',
        'opponent',
        # Player profile
        'draft_year',
        'draft_round',
        'draft_pick',
        'player_height',
        'player_weight',
        'fourty',
        'bench',
        'vertical',
        'broad_jump',
        'cone',
        'shuttle',
        # Participation
        'offense_snaps',
        'offense_snap_share',
        'wow_offense_snap_share_change',
        'defense_snaps',
        'defense_snap_share',
        'wow_defense_snap_share_change',
        'special_teams_snaps',
        'special_teams_snap_share',
        'wow_special_teams_snap_share_change',
        # Passing
        'pass_attempts',
        'pass_complete',
        'pass_touchdowns',
        'passing_yards',
        'interceptions',
        'pass_first_downs',
        'pass_2pt_conversions',
        'avg_time_to_throw',
        'avg_completed_air_yards',
        'avg_intended_air_yards',
        'avg_air_yards_differential',
        'aggressiveness',
        'max_completed_air_distance',
        'avg_air_yards_to_sticks',
        'completion_percentage',
        'expected_completion_percentage',
        'completion_percentage_above_expectation',
        # Receiving
        'receptions',
        'receiving_yards',
        'receiving_air_yards',
        'receiving_yac',
        'receiving_touchdowns',
        'receiving_first_downs',
        'receiving_2pt_conversions',
        'avg_air_distance',
        'max_air_distance',
        'avg_cushion',
        'avg_separation',
        'percent_share_of_intended_air_yards',
        'targets',
        'catch_percentage',
        'avg_yac',
        'avg_expected_yac',
        'avg_yac_above_expectation',
        # Rushing
        'rush_attempts',
        'rushing_yards',
        'rush_touchdowns',
        'rush_first_downs',
        'efficiency',
        'percent_attempts_gte_eight_defenders',
        'avg_time_to_los',
        'expected_rush_yards',
        'rush_yards_over_expected',
        'avg_rush_yards',
        'rush_yards_over_expected_per_att',
        'rush_pct_over_expected',
        # Fantasy
        'fantasy_points',
        'f_position_rank',
    ]
    
    # Select only columns that exist
    available_cols = master.columns
    cols_to_select = [col for col in final_cols if col in available_cols]
    
    master_final = master.select(cols_to_select)
    
    # Split by season
    unique_seasons = master_final['season'].unique().sort()
    print(f"\nSplitting by seasons: {unique_seasons.to_list()}")
    
    results = {}
    for season in unique_seasons:
        season_data = (
            master_final
            .filter(pl.col('season') == season)
            .sort(['week', 'team', 'position', 'player_name'])
        )
        results[season] = season_data
        print(f"  Season {season}: {len(season_data)} records")
    
    return results


def write_reports_to_csv(reports, output_dir='./reports'):
    """
    Write opportunity reports to CSV files.
    
    Args:
        reports: dict mapping season to DataFrame
        output_dir: directory to write CSV files to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for season, df in reports.items():
        filename = output_path / f'opportunity_report_{season}.csv'
        df.write_csv(filename)
        print(f"Wrote {filename} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive player opportunity reports'
    )
    parser.add_argument(
        'seasons',
        nargs='?',
        default='2024',
        help='Season(s) to process. Use "all" for all available, comma-separated list for multiple (e.g., "2023,2024"), or single year. Default: 2024'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='./reports',
        help='Output directory for CSV files. Default: ./reports'
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
        reports = generate_opportunity_report(seasons)
        
        if not reports:
            print("No reports generated.")
            return 1
        
        # Write to CSV
        print("\n" + "="*60)
        print("Writing CSV files...")
        print("="*60)
        write_reports_to_csv(reports, args.output_dir)
        
        print(f"\n✓ Successfully generated {len(reports)} opportunity report(s)")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
