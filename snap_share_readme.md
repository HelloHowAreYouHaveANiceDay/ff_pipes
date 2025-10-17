# Snap Share Report Generator

Generates weekly snap share reports with week-over-week changes for NFL players.

## Features

- **Week-over-week tracking**: Shows snap share changes from previous week
- **Team-based calculations**: WoW resets when players change teams
- **Missing week handling**: Non-participating weeks are treated as 0 snaps
- **Postseason included**: Regular season and playoff games both included
- **Multiple season support**: Generate reports for single or multiple seasons

## Usage

```bash
# Generate report for 2024 season (default)
uv run snap_share.py

# Generate report for specific season
uv run snap_share.py 2023

# Generate reports for multiple seasons
uv run snap_share.py 2023,2024

# Generate reports for all available seasons (since 2012)
uv run snap_share.py all

# Specify output directory
uv run snap_share.py 2024 -o reports
```

## Output Format

CSV file per season: `snap_share_report_{season}.csv`

### Columns

| Column | Description |
|--------|-------------|
| `season` | NFL season year |
| `week` | Week number (includes postseason) |
| `player` | Player name |
| `position` | Player position |
| `team` | Player's team |
| `opponent` | Opponent team |
| `offense_snaps` | Number of offensive snaps |
| `offense_snap_share` | Percentage of team's offensive snaps (0.0-1.0) |
| `wow_offense_snap_share_change` | Change from previous week (NULL for first week) |
| `defense_snaps` | Number of defensive snaps |
| `defense_snap_share` | Percentage of team's defensive snaps (0.0-1.0) |
| `wow_defense_snap_share_change` | Change from previous week (NULL for first week) |
| `special_teams_snaps` | Number of special teams snaps |
| `special_teams_snap_share` | Percentage of team's special teams snaps (0.0-1.0) |
| `wow_special_teams_snap_share_change` | Change from previous week (NULL for first week) |

## Implementation Details

### Week-over-Week Calculation

- **Partitioned by player AND team**: When a player changes teams, WoW tracking resets
- **First week = NULL**: First week for each player-team combination has NULL WoW values
- **Missing weeks = 0**: If a player doesn't play in a week between their first and last appearance, that week is filled with 0 snaps and 0% snap share
- **Includes postseason**: Both regular season and playoff games are included

### Example

For a player who plays weeks 1, 3, and 4:

| Week | Snaps | Share | WoW Change |
|------|-------|-------|------------|
| 1 | 50 | 0.80 | NULL (first week) |
| 2 | 0 | 0.00 | -0.80 (didn't play) |
| 3 | 45 | 0.75 | +0.75 (returned) |
| 4 | 48 | 0.80 | +0.05 (increased) |

## Data Source

Uses `nflreadpy.load_snap_counts()` which sources data from Pro Football Reference via the nflverse data repository.

- Available since: 2012
- Updated: Weekly during season
- [Data Dictionary](https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html)

## Requirements

- Python 3.12+
- nflreadpy
- polars
- uv (for running)

All dependencies are managed via `uv`.
