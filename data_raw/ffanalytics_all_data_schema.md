# ffanalytics_all_data Schema

## Data Source
- **Package**: ffanalytics (R)
- **URL**: https://ffanalytics.fantasyfootballanalytics.net/
- **Source Script**: `data_loaders/ffanalytics_all_data.R`
- **Update Frequency**: Pre-season for seasonal projections, weekly during season
- **Data Type**: Fantasy football projections aggregated from multiple sources (FantasyPros, ESPN, CBS, Yahoo, NFL.com, etc.)

## File Naming Convention
- **Seasonal**: `ffanalytics_all_data_{YEAR}.csv`
- **Weekly**: `ffanalytics_all_data_{YEAR}_week{N}.csv`

## Column Specifications

### Identifiers & Player Info
| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Player ID (varies by source: mfl_id, yahoo_id, etc.) |
| `first_name` | string | Player first name |
| `last_name` | string | Player last name |
| `team` | string | Team abbreviation (e.g., BUF, KC) |
| `position.x` | string | Primary position (QB, RB, WR, TE, K, DST) |
| `position.y` | string | Secondary position designation |
| `age` | integer | Player age |
| `exp` | integer | Years of experience in NFL |

### Projection Metadata
| Column | Type | Description |
|--------|------|-------------|
| `avg_type` | string | Averaging method used: "average" (mean), "robust" (median-based), or "weighted" |

### Passing Projections
| Column | Type | Description |
|--------|------|-------------|
| `pass_yds` | float | Projected passing yards |
| `pass_yds_sd` | float | Standard deviation of passing yards projection |
| `pass_tds` | float | Projected passing touchdowns |
| `pass_tds_sd` | float | Standard deviation of passing TDs |
| `pass_int` | float | Projected interceptions thrown |
| `pass_int_sd` | float | Standard deviation of interceptions |

### Rushing Projections
| Column | Type | Description |
|--------|------|-------------|
| `rush_yds` | float | Projected rushing yards |
| `rush_yds_sd` | float | Standard deviation of rushing yards |
| `rush_tds` | float | Projected rushing touchdowns |
| `rush_tds_sd` | float | Standard deviation of rushing TDs |

### Receiving Projections
| Column | Type | Description |
|--------|------|-------------|
| `rec` | float | Projected receptions |
| `rec_sd` | float | Standard deviation of receptions |
| `rec_yds` | float | Projected receiving yards |
| `rec_yds_sd` | float | Standard deviation of receiving yards |
| `rec_tds` | float | Projected receiving touchdowns |
| `rec_tds_sd` | float | Standard deviation of receiving TDs |

### Turnovers
| Column | Type | Description |
|--------|------|-------------|
| `fumbles_lost` | float | Projected fumbles lost |
| `fumbles_lost_sd` | float | Standard deviation of fumbles lost |

### Special Teams / Returns
| Column | Type | Description |
|--------|------|-------------|
| `return_tds` | float | Projected return touchdowns (kick/punt) |
| `return_tds_sd` | float | Standard deviation of return TDs |

### Kicker Projections
| Column | Type | Description |
|--------|------|-------------|
| `fg_0019` | float | Projected field goals made 0-19 yards |
| `fg_0019_sd` | float | Standard deviation of 0-19 yard FGs |
| `fg_2029` | float | Projected field goals made 20-29 yards |
| `fg_2029_sd` | float | Standard deviation of 20-29 yard FGs |
| `fg_3039` | float | Projected field goals made 30-39 yards |
| `fg_3039_sd` | float | Standard deviation of 30-39 yard FGs |
| `fg_4049` | float | Projected field goals made 40-49 yards |
| `fg_4049_sd` | float | Standard deviation of 40-49 yard FGs |
| `fg_50` | float | Projected field goals made 50+ yards |
| `fg_50_sd` | float | Standard deviation of 50+ yard FGs |
| `xp` | float | Projected extra points made |
| `xp_sd` | float | Standard deviation of extra points |

### Defense/Special Teams (DST) Projections
| Column | Type | Description |
|--------|------|-------------|
| `dst_int` | float | Projected team interceptions |
| `dst_int_sd` | float | Standard deviation of team INTs |
| `dst_fum_rec` | float | Projected team fumble recoveries |
| `dst_fum_rec_sd` | float | Standard deviation of fumble recoveries |
| `dst_sacks` | float | Projected team sacks |
| `dst_sacks_sd` | float | Standard deviation of sacks |
| `dst_safety` | float | Projected safeties |
| `dst_safety_sd` | float | Standard deviation of safeties |
| `dst_td` | float | Projected defensive/special teams touchdowns |
| `dst_td_sd` | float | Standard deviation of defensive TDs |

### Individual Defensive Player (IDP) Projections
| Column | Type | Description |
|--------|------|-------------|
| `idp_solo` | float | Projected solo tackles |
| `idp_solo_sd` | float | Standard deviation of solo tackles |
| `idp_asst` | float | Projected assisted tackles |
| `idp_asst_sd` | float | Standard deviation of assisted tackles |
| `idp_pd` | float | Projected passes defended |
| `idp_pd_sd` | float | Standard deviation of passes defended |
| `idp_int` | float | Projected interceptions (IDP) |
| `idp_int_sd` | float | Standard deviation of IDP interceptions |
| `idp_fum_force` | float | Projected forced fumbles |
| `idp_fum_force_sd` | float | Standard deviation of forced fumbles |
| `idp_fum_rec` | float | Projected fumble recoveries (IDP) |
| `idp_fum_rec_sd` | float | Standard deviation of IDP fumble recoveries |
| `idp_td` | float | Projected defensive touchdowns (IDP) |
| `idp_td_sd` | float | Standard deviation of IDP TDs |

## Data Notes

### Standard Deviations
- All `*_sd` columns represent projection uncertainty
- Higher SD = more variance between projection sources or historical volatility
- Used for calculating confidence intervals and risk metrics

### Missing Values
- `NA` indicates data not available or not applicable for that position
- Example: QBs will have `NA` for receiving stats
- DST positions won't have individual player names

### Averaging Methods
Three averaging methods are typically used:
- **average**: Simple mean across all sources
- **robust**: Median-based average, less sensitive to outliers
- **weighted**: Sources weighted by historical accuracy

### Position Codes
- **QB**: Quarterback
- **RB**: Running Back
- **WR**: Wide Receiver
- **TE**: Tight End
- **K**: Kicker
- **DST**: Defense/Special Teams
- **DL**: Defensive Line (IDP)
- **LB**: Linebacker (IDP)
- **DB**: Defensive Back (IDP)

### Seasonal vs Weekly Data
**Seasonal projections** include:
- Full season totals
- All supplemental data (ADP, AAV, ECR, VOR, tiers, risk)

**Weekly projections** include:
- Single week projections
- Limited supplemental data
- Week-specific matchup considerations

## Usage Examples

### Load and explore in R
```r
library(tidyverse)

# Load data
projections <- read_csv("data_raw/ffanalytics_all_data_2025.csv")

# Top QBs by passing yards
projections %>%
  filter(position.x == "QB") %>%
  select(first_name, last_name, team, pass_yds, pass_tds, pass_int) %>%
  arrange(desc(pass_yds)) %>%
  head(10)

# Most uncertain projections (high SD)
projections %>%
  filter(position.x == "RB") %>%
  mutate(total_sd = rush_yds_sd + rec_yds_sd) %>%
  select(first_name, last_name, team, rush_yds, rec_yds, total_sd) %>%
  arrange(desc(total_sd)) %>%
  head(10)
```

### Load and explore in Python
```python
import pandas as pd

# Load data
projections = pd.read_csv("data_raw/ffanalytics_all_data_2025.csv")

# Top WRs by receiving yards
(projections[projections['position.x'] == 'WR']
 [['first_name', 'last_name', 'team', 'rec', 'rec_yds', 'rec_tds']]
 .sort_values('rec_yds', ascending=False)
 .head(10))

# Calculate coefficient of variation (relative uncertainty)
projections['rush_cv'] = projections['rush_yds_sd'] / projections['rush_yds']
```

## Related Files
- **Config**: `data_loaders/fantasy_projections_config.R` - Scoring settings
- **Generator**: `data_loaders/ffanalytics_all_data.R` - Data loader script
- **Documentation**: Script header comments

## Version History
- **2025**: Current season projections
- Schema reflects output from ffanalytics package as of November 2025
