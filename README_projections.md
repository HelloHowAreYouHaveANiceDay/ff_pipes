# Fantasy Football Projections Data Loader

R-based data loader for downloading fantasy football projections from multiple sources.

## Overview

This loader scrapes fantasy football projections from various fantasy sports websites (FantasyPros, ESPN, CBS, Yahoo, NFL.com, etc.) and aggregates them into standardized CSV reports. It follows the methodology from [Fantasy Football Analytics Textbook](https://isaactpetersen.github.io/Fantasy-Football-Analytics-Textbook/download-football-data.html#sec-scrapeProjections).

## Prerequisites

### R Installation

1. **Install R** (version 4.0+):
   - Download from: https://cran.r-project.org/
   - Windows: Download and run the `.exe` installer
   - Verify: `R --version`

2. **Install Required R Packages**:
   The script will auto-install missing packages on first run:
   - `ffanalytics` - Fantasy football projections scraper
   - `tidyverse` - Data manipulation
   - `progressr` - Progress bars
   - `lubridate` - Date handling

   Or install manually:
   ```r
   install.packages(c("remotes", "tidyverse", "progressr", "lubridate"))
   remotes::install_github("FantasyFootballAnalytics/ffanalytics")
   ```

## Usage

### Basic Usage

```powershell
# Current season, seasonal projections
Rscript data_loaders/fantasy_projections.R

# Specific season, seasonal projections
Rscript data_loaders/fantasy_projections.R --season 2024

# Current season, weekly projections
Rscript data_loaders/fantasy_projections.R --week 5

# Specific season and week
Rscript data_loaders/fantasy_projections.R --season 2024 --week 5

# Custom output directory
Rscript data_loaders/fantasy_projections.R --output-dir ./custom_reports
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--season YEAR` | Season year to download | Current season |
| `--week NUM` | Week number (0 = seasonal, 1-18 = weekly) | 0 (seasonal) |
| `--output-dir DIR` | Output directory for CSV files | `./reports` |
| `--help, -h` | Show help message | - |

## Output Files

### Seasonal Projections
**File**: `reports/projections_seasonal_{season}.csv`

Contains pre-season projections for all fantasy-relevant players with the following columns:

```
season              - Season year
player_id           - Player ID (various systems: mfl_id, yahoo_id, etc.)
player_name         - Player display name
position            - Player position (QB, RB, WR, TE, K, DST)
team                - Player's team abbreviation
avg_type            - Averaging method (mean, robust, weighted)

[Passing Stats]
pass_attempts       - Projected pass attempts
pass_completions    - Projected completions
pass_yards          - Projected passing yards
pass_tds            - Projected passing TDs
pass_ints           - Projected interceptions

[Rushing Stats]
rush_attempts       - Projected rush attempts
rush_yards          - Projected rushing yards
rush_tds            - Projected rushing TDs

[Receiving Stats]
targets             - Projected targets
receptions          - Projected receptions
rec_yards           - Projected receiving yards
rec_tds             - Projected receiving TDs

[Fantasy Metrics]
fantasy_points      - Projected fantasy points (based on scoring settings)
sd_points           - Projection uncertainty (standard deviation)
ecr                 - Expert Consensus Ranking
adp                 - Average Draft Position
aav                 - Average Auction Value
```

### Weekly Projections
**File**: `reports/projections_weekly_{season}_week{week}.csv`

Contains week-specific projections with similar columns plus:
```
week                - Week number
opponent            - Opponent team abbreviation
ecr_week            - Weekly expert consensus ranking
```

## Scoring Configuration

Projections are calculated using the scoring rules defined in `data_loaders/fantasy_projections_config.R`.

### Current Settings (NFL.com PPR format):
- **Passing**: 0.04 pts/yard, 4 pts/TD, -2 pts/INT
- **Rushing**: 0.1 pts/yard, 6 pts/TD
- **Receiving**: 0.1 pts/yard, 6 pts/TD, **1 pt/reception (PPR)**
- **Fumbles**: -2 pts per fumble lost
- **2-pt conversions**: 2 pts

### Customizing Scoring

Edit `data_loaders/fantasy_projections_config.R` to match your league:

```r
# For Standard (non-PPR) scoring
SCORING_OBJ$rec$rec <- 0

# For Half-PPR scoring
SCORING_OBJ$rec$rec <- 0.5

# Adjust TD values
SCORING_OBJ$pass$pass_tds <- 6    # 6-pt passing TDs
```

## Data Sources

The `ffanalytics` package scrapes projections from multiple sources:
- FantasyPros
- ESPN
- CBS Sports
- Yahoo Sports
- NFL.com
- FantasySharks
- FFToday
- NumberFire
- FantasyFootballNerd
- RTSports

Projections are averaged across all available sources for each player.

## Timing & Updates

- **Seasonal Projections**: Download once pre-season or after major roster changes
- **Weekly Projections**: Download each week, ideally Tuesday-Saturday before games
- **Data Freshness**: Projections update throughout the week as injury reports emerge

## Troubleshooting

### "Could not resolve hostname" or "cannot open URL" errors

**Error**: `cannot open URL 'https://s3.us-east-2.amazonaws.com/ffanalytics/packagedata/player_table.csv'`

This error occurs when the `ffanalytics` package cannot download required data files. This is a **known issue** with the package.

**Solutions**:

1. **Test connectivity**:
   ```powershell
   # Test if you can access the S3 bucket
   curl https://s3.us-east-2.amazonaws.com/ffanalytics/packagedata/player_table.csv
   ```
   Or try opening the URL in your browser.

2. **Check DNS resolution**:
   ```powershell
   nslookup s3.us-east-2.amazonaws.com
   ```
   If DNS fails, try:
   - Use Google DNS (8.8.8.8, 8.8.4.4)
   - Check firewall/proxy settings
   - Try a different network (mobile hotspot, VPN)

3. **Wait and retry**: The S3 bucket or your network may be temporarily unavailable

4. **Reinstall package**:
   ```r
   remove.packages('ffanalytics')
   install.packages('remotes')
   remotes::install_github('FantasyFootballAnalytics/ffanalytics')
   ```

5. **Check package status**: Visit https://github.com/FantasyFootballAnalytics/ffanalytics/issues to see if others are experiencing similar issues

### "Package 'ffanalytics' not found"
```r
install.packages("remotes")
remotes::install_github("FantasyFootballAnalytics/ffanalytics")
```

### Web scraping errors
Projection websites may change their HTML structure. If scraping fails:
1. Check if `ffanalytics` package has updates: `remotes::install_github("FantasyFootballAnalytics/ffanalytics")`
2. Try again later (sites may be temporarily down)
3. Check package issues: https://github.com/FantasyFootballAnalytics/ffanalytics/issues

### Slow downloads
Projections scraping can take 5-15 minutes depending on:
- Number of sources available
- Network speed
- Website response times

Progress bars show the scraping status for each source.

### Missing players
Not all players have projections from all sources. The script:
- Aggregates available projections
- Uses robust averaging to handle outliers
- Returns `NA` for players with no projections

## Integration with Python Pipeline

While this is an R script, you can call it from Python:

```python
import subprocess

def run_projections(season=None, week=0):
    """Run R projections script from Python"""
    cmd = ['Rscript', 'data_loaders/fantasy_projections.R']
    if season:
        cmd.extend(['--season', str(season)])
    if week:
        cmd.extend(['--week', str(week)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Projections failed: {result.stderr}")
    return result.stdout

# Usage
run_projections(season=2024, week=0)
```

## Differences from Other Loaders

Unlike `opportunity_report.py` and `qb_per_game_stats.py` which download **actual historical stats**:
- This loader downloads **forward-looking projections**
- Data comes from **fantasy websites** (not NFL play-by-play data)
- Projections are **predictions** (not facts)
- Multiple **ID systems** (not just `gsis_id`)
- **Updates frequently** during the week

## References

- [Fantasy Football Analytics Textbook](https://isaactpetersen.github.io/Fantasy-Football-Analytics-Textbook/)
- [ffanalytics Package](https://github.com/FantasyFootballAnalytics/ffanalytics)
- [nflverse Data](https://nflverse.nflverse.com/)

## License

This loader is based on methodologies from the Fantasy Football Analytics Textbook and uses the open-source `ffanalytics` package.
