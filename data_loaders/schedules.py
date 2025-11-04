# NFL Schedule Report
# Loads and processes NFL game schedules with betting lines and rest days

# inputs:
# - schedules - https://nflreadr.nflverse.com/articles/dictionary_schedules.html

# output:
# one csv per season with game schedule information
# columns:
# - game_id
# - season
# - game_type
# - week
# - gameday
# - weekday
# - gametime
# - away_team
# - away_score
# - home_team
# - home_score
# - location
# - result
# - total
# - overtime
# - old_game_id
# - gsis
# - nfl_detail_id
# - pfr
# - pff
# - espn
# - away_rest
# - home_rest
# - away_moneyline
# - home_moneyline
# - spread_line
# - away_spread_odds
# - home_spread_odds
# - total_line
# - under_odds
# - over_odds
# - div_game
# - roof
# - surface
# - temp
# - wind
# - away_qb_id
# - home_qb_id
# - away_qb_name
# - home_qb_name
# - away_coach
# - home_coach
# - referee
# - stadium_id
# - stadium

import argparse
import sys
from pathlib import Path

import nflreadpy
import polars as pl


def generate_schedule_report(seasons):
    """
    Generate NFL schedule reports with game information and betting lines.
    
    Args:
        seasons: int, list of ints, or True for all available seasons
        
    Returns:
        dict: Dictionary mapping season to DataFrame
    """
    print(f"Loading schedule data for seasons: {seasons}")
    
    # Load schedule data
    schedule_data = nflreadpy.load_schedules(seasons)
    
    if schedule_data.is_empty():
        print("No schedule data loaded. Exiting.")
        return {}
    
    print(f"Loaded {len(schedule_data)} schedule records")
    
    # Get unique seasons from the data
    unique_seasons = schedule_data['season'].unique().sort()
    print(f"Processing seasons: {unique_seasons.to_list()}")
    
    results = {}
    
    for season in unique_seasons:
        print(f"\nProcessing season {season}...")
        season_data = schedule_data.filter(pl.col('season') == season)
        
        # Select columns that exist in the data
        # The schedules dataset may have all or some of these columns
        available_cols = season_data.columns
        
        # Define desired column order (only include if they exist)
        desired_cols = [
            'game_id',
            'season',
            'game_type',
            'week',
            'gameday',
            'weekday',
            'gametime',
            'away_team',
            'away_score',
            'home_team',
            'home_score',
            'location',
            'result',
            'total',
            'overtime',
            'old_game_id',
            'gsis',
            'nfl_detail_id',
            'pfr',
            'pff',
            'espn',
            'away_rest',
            'home_rest',
            'away_moneyline',
            'home_moneyline',
            'spread_line',
            'away_spread_odds',
            'home_spread_odds',
            'total_line',
            'under_odds',
            'over_odds',
            'div_game',
            'roof',
            'surface',
            'temp',
            'wind',
            'away_qb_id',
            'home_qb_id',
            'away_qb_name',
            'home_qb_name',
            'away_coach',
            'home_coach',
            'referee',
            'stadium_id',
            'stadium'
        ]
        
        # Select only columns that exist in the data
        cols_to_select = [col for col in desired_cols if col in available_cols]
        
        print(f"  Selected {len(cols_to_select)} columns from {len(available_cols)} available")
        
        final_report = (
            season_data
            .select(cols_to_select)
            .sort(['week', 'gameday', 'gametime'])
        )
        
        results[season] = final_report
        print(f"  Season {season}: {len(final_report)} games in final report")
    
    return results


def write_reports_to_csv(reports, output_dir='./reports'):
    """
    Write schedule reports to CSV files.
    
    Args:
        reports: dict mapping season to DataFrame
        output_dir: directory to write CSV files to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for season, df in reports.items():
        filename = output_path / f'schedule_{season}.csv'
        df.write_csv(filename)
        print(f"Wrote {filename} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate NFL schedule reports with game information and betting lines'
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
        reports = generate_schedule_report(seasons)
        
        if not reports:
            print("No reports generated.")
            return 1
        
        # Write to CSV
        print("\n" + "="*60)
        print("Writing CSV files...")
        print("="*60)
        write_reports_to_csv(reports, args.output_dir)
        
        print(f"\n✓ Successfully generated {len(reports)} schedule report(s)")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
