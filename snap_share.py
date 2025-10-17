# yearly week by week snap share report
# shows how many snaps each player played in each week and change week over week

# inputs:
# - snap_counts - https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html

# output:
# one csv per season with per-player per-week snap counts and snap shares
# columns:
# - season 
# - week
# - player
# - pfr_player_id
# - position
# - team
# - opponent
# - offense_snaps
# - offense_snap_share
# - wow_offense_snap_share_change
# - defense_snaps
# - defense_snap_share
# - wow_defense_snap_share_change
# - special_teams_snaps
# - special_teams_snap_share
# - wow_special_teams_snap_share_change
 
import argparse
import sys
from pathlib import Path

import nflreadpy
import polars as pl


def generate_snap_share_report(seasons):
    """
    Generate weekly snap share reports with week-over-week changes.
    
    Args:
        seasons: int, list of ints, or True for all available seasons
        
    Returns:
        dict: Dictionary mapping season to DataFrame
    """
    print(f"Loading snap count data for seasons: {seasons}")
    
    # Load snap count data (includes both regular and postseason)
    snap_data = nflreadpy.load_snap_counts(seasons)
    
    if snap_data.is_empty():
        print("No data loaded. Exiting.")
        return {}
    
    print(f"Loaded {len(snap_data)} snap count records")
    
    # Get unique seasons from the data
    unique_seasons = snap_data['season'].unique().sort()
    print(f"Processing seasons: {unique_seasons.to_list()}")
    
    results = {}
    
    for season in unique_seasons:
        print(f"\nProcessing season {season}...")
        season_data = snap_data.filter(pl.col('season') == season)
        
        # Create complete grid for each player-team combination
        # For each player-team combo, fill all weeks between first and last appearance
        player_team_weeks = (
            season_data
            .group_by(['pfr_player_id', 'team', 'season', 'player', 'position'])
            .agg([
                pl.col('week').min().alias('first_week'),
                pl.col('week').max().alias('last_week')
            ])
        )
        
        # Get all unique weeks in the season
        all_weeks = season_data.select('week').unique().sort('week')
        
        # Create complete grid by joining each player-team with all weeks in their range
        complete_grid = (
            player_team_weeks
            .join(all_weeks, how='cross')
            .filter(
                (pl.col('week') >= pl.col('first_week')) & 
                (pl.col('week') <= pl.col('last_week'))
            )
            .select(['season', 'week', 'pfr_player_id', 'team', 'player', 'position'])
        )
        
        print(f"  Created grid with {len(complete_grid)} player-week combinations")
        
        # Join with actual snap data, filling missing weeks with zeros
        season_filled = (
            complete_grid
            .join(
                season_data,
                on=['season', 'week', 'pfr_player_id', 'team'],
                how='left'
            )
            # Fill missing values with zeros and empty strings
            .with_columns([
                pl.col('player').fill_null(pl.first('player').over(['pfr_player_id', 'team'])),
                pl.col('position').fill_null(pl.first('position').over(['pfr_player_id', 'team'])),
                pl.col('opponent').fill_null(''),
                pl.col('offense_snaps').fill_null(0),
                pl.col('offense_pct').fill_null(0.0),
                pl.col('defense_snaps').fill_null(0),
                pl.col('defense_pct').fill_null(0.0),
                pl.col('st_snaps').fill_null(0),
                pl.col('st_pct').fill_null(0.0),
            ])
        )
        
        # Calculate week-over-week changes
        # Partition by player, team, and season so WoW resets when player changes teams
        season_with_wow = (
            season_filled
            .sort(['pfr_player_id', 'team', 'week'])
            .with_columns([
                # Shift previous week's percentages
                pl.col('offense_pct').shift(1).over(['pfr_player_id', 'team', 'season']).alias('prev_offense_pct'),
                pl.col('defense_pct').shift(1).over(['pfr_player_id', 'team', 'season']).alias('prev_defense_pct'),
                pl.col('st_pct').shift(1).over(['pfr_player_id', 'team', 'season']).alias('prev_st_pct'),
            ])
            .with_columns([
                # Calculate WoW changes (will be NULL for first week automatically)
                (pl.col('offense_pct') - pl.col('prev_offense_pct')).alias('wow_offense_snap_share_change'),
                (pl.col('defense_pct') - pl.col('prev_defense_pct')).alias('wow_defense_snap_share_change'),
                (pl.col('st_pct') - pl.col('prev_st_pct')).alias('wow_special_teams_snap_share_change'),
            ])
        )
        
        # Select and rename columns to match output specification
        final_report = (
            season_with_wow
            .select([
                'season',
                'week',
                'player',
                'pfr_player_id',
                'position',
                'team',
                'opponent',
                pl.col('offense_snaps'),
                pl.col('offense_pct').alias('offense_snap_share'),
                'wow_offense_snap_share_change',
                pl.col('defense_snaps'),
                pl.col('defense_pct').alias('defense_snap_share'),
                'wow_defense_snap_share_change',
                pl.col('st_snaps').alias('special_teams_snaps'),
                pl.col('st_pct').alias('special_teams_snap_share'),
                'wow_special_teams_snap_share_change',
            ])
            .sort(['week', 'team', 'position', 'player'])
        )
        
        results[season] = final_report
        print(f"  Season {season}: {len(final_report)} rows in final report")
    
    return results


def write_reports_to_csv(reports, output_dir='.'):
    """
    Write snap share reports to CSV files.
    
    Args:
        reports: dict mapping season to DataFrame
        output_dir: directory to write CSV files to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for season, df in reports.items():
        filename = output_path / f'snap_share_report_{season}.csv'
        df.write_csv(filename)
        print(f"Wrote {filename} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate weekly snap share reports with week-over-week changes'
    )
    parser.add_argument(
        'seasons',
        nargs='?',
        default='2024',
        help='Season(s) to process. Use "all" for all available, comma-separated list for multiple (e.g., "2023,2024"), or single year. Default: 2024'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory for CSV files. Default: current directory'
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
        reports = generate_snap_share_report(seasons)
        
        if not reports:
            print("No reports generated.")
            return 1
        
        # Write to CSV
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