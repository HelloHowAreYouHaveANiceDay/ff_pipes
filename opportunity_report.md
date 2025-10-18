# Player Opportunity Report

A wide table reporting all of a player's relevant dimensions and stats for that week.

Table is unique on season-week-team-gsis_id

## Sources
nflreadr
- [players table](https://nflreadr.nflverse.com/articles/dictionary_players.html)
- [play by play table](https://nflreadr.nflverse.com/articles/dictionary_pbp.html)
- [combine table](https://nflreadr.nflverse.com/articles/dictionary_combine.html)
- [nextgen table](https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html)
local
- [fantasy scoring](./scoring.json)

## Columns
```python
@dataclass
class OpportunityReportObject:
    season: int # 4 digit season year
    week:int # week of the season
    player_name: str # name of the player
    gsis_id: str # player gsis_id 
    position: str # player position
    team: str # player's team in this game
    opponent: str # opposing team in this game
    # player profile
    draft_year: int # player's draft year
    draft_round: int # player's draft round
    draft_pick: int # pick number of the player
    player_height: float # height of the player in inches
    player_weight: float # weight of the player in pounds
    fourty: float # player's 40 yard dash time at combine in seconds
    bench: float # player's bench reps at combine
    vertical: float # player's vertical at combine in inches
    broad_jump: float # player's broad jump at combine in inches
    cone: float # player's cone drill time in seconds
    shuttle: float # player's shuttle time at combine
    # participation
    offense_snaps: int # number of offensive snaps this player played this game
    offense_snap_share: float # pct of offensive snaps this player played this game
    wow_offense_snap_share_change: float # pct change in offensive snaps from previous week. 0 if this week is the first week or previous week is a bye week.
    defense_snaps: int # number of defensive snaps this player played this game
    defense_snap_share: float # pct of defensive snaps this player played this game
    wow_defense_snap_share_change: float # pct change in defensive snaps from previous week. 0 if this week is the first week or previous week is a bye week.
    special_teams_snaps: int # number of special teams snaps this player played this game
    special_teams_snap_share: float # pct of special teams snaps this player played this game 
    wow_special_teams_snap_share_change: float # pct change in special teams snaps from previous week. 0 if this week is the first week or previous week is a bye week
    # passing metrics
    pass_attempts: int # number of passes this player attempted
    pass_complete: int # number of passes this player completed
    pass_touchdowns: int # number of touchdowns this player completed 
    passing_yards: int # number of passing yards this player threw for this game
    interceptions: int # number of interceptions this player threw
    pass_first_downs: int # number of passing first downs
    pass_2pt_conversions: int # number of successful passing 2pt conversions
    avg_time_to_throw: float # nextgen
    avg_completed_air_yards: float # nextgen
    avg_intended_air_yards: float # nextgen
    avg_air_yards_differential: float # nextgen
    aggressiveness: float # nextgen
    max_completed_air_distance: float # nextgen
    avg_air_yards_to_sticks: float # nextgen
    completion_percentage: float 
    expected_completion_percentage: float
    completion_percentage_above_expectation: float
    # receiving metrics
    receptions: int # number of times this player caught a pass
    receiving_yards: int # number of yards this player received for
    receiving_air_yards: int # total air yards for this player's receptions
    receiving_yac: int # total yards after catch
    receiving_touchdowns: int # of receiving TDs
    receiving_first_downs: int # number of receiving first downs
    receiving_2pt_conversions: int # of successful 2pt conversions received
    avg_air_distance: int # adot
    max_air_distance: int # max dot
    avg_cushion: float # nextgen
    avg_separation: float # nextgen
    percent_share_of_intended_air_yards: float # nextgen
    targets: int # number of targets for the receiver
    target_share: float # receiver's targets / all targets from the receiver's team in that game
    catch_percentage: float
    avg_yac: float # average yards after catch
    avg_expected_yac: float
    avg_yac_above_expectation: float
    # rushing metrics
    rush_attempts: int # number of rush attempts
    rushing_yards: int # number of rushing yards
    rush_touchdowns: int # of rushing TDs
    rush_first_downs: int # of first down rushes
    efficiency: float # total yards gained / yards ran. lower = more north/south runner
    percent_attempts_gte_eight_defenders: float
    avg_time_to_los: int 
    expected_rush_yards: int
    rush_yards_over_expected: int
    avg_rush_yards: int
    rush_yards_over_expected_per_att: int
    rush_pct_over_expected: int
    # fantasy
    fantasy_points: float # calculated fantasy points based on scoring.json
    f_position_rank: float # ranking for this player out of all players in his position this week. #1 is highest scorer.

```