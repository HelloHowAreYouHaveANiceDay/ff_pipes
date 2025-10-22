# Fantasy Football Scoring Configuration
# 
# This file defines the scoring settings for calculating fantasy points
# from projections. Modify these values to match your league's scoring rules.
#
# Based on NFL.com default scoring (PPR format)

# Get default scoring object from ffanalytics
SCORING_OBJ <- ffanalytics::scoring

# === PASSING SCORING ===
SCORING_OBJ$pass$pass_att <- 0      # Points per pass attempt
SCORING_OBJ$pass$pass_comp <- 0     # Points per completion
SCORING_OBJ$pass$pass_inc <- 0      # Points per incompletion
SCORING_OBJ$pass$pass_yds <- 0.04   # Points per passing yard
SCORING_OBJ$pass$pass_tds <- 4      # Points per passing TD
SCORING_OBJ$pass$pass_int <- -2     # Points per interception (was -3)
SCORING_OBJ$pass$pass_40_yds <- 0   # Bonus for 40+ yard pass
SCORING_OBJ$pass$pass_300_yds <- 0  # Bonus for 300+ passing yards
SCORING_OBJ$pass$pass_350_yds <- 0  # Bonus for 350+ passing yards
SCORING_OBJ$pass$pass_400_yds <- 0  # Bonus for 400+ passing yards

# === RUSHING SCORING ===
SCORING_OBJ$rush$all_pos <- TRUE     # Apply to all positions
SCORING_OBJ$rush$rush_yds <- 0.1     # Points per rushing yard
SCORING_OBJ$rush$rush_att <- 0       # Points per rush attempt
SCORING_OBJ$rush$rush_40_yds <- 0    # Bonus for 40+ yard rush
SCORING_OBJ$rush$rush_tds <- 6       # Points per rushing TD
SCORING_OBJ$rush$rush_100_yds <- 0   # Bonus for 100+ rushing yards
SCORING_OBJ$rush$rush_150_yds <- 0   # Bonus for 150+ rushing yards
SCORING_OBJ$rush$rush_200_yds <- 0   # Bonus for 200+ rushing yards

# === RECEIVING SCORING ===
SCORING_OBJ$rec$all_pos <- TRUE      # Apply to all positions
SCORING_OBJ$rec$rec <- 1             # Points per reception (PPR - was 0)
SCORING_OBJ$rec$rec_yds <- 0.1       # Points per receiving yard
SCORING_OBJ$rec$rec_tds <- 6         # Points per receiving TD
SCORING_OBJ$rec$rec_40_yds <- 0      # Bonus for 40+ yard reception
SCORING_OBJ$rec$rec_100_yds <- 0     # Bonus for 100+ receiving yards
SCORING_OBJ$rec$rec_150_yds <- 0     # Bonus for 150+ receiving yards
SCORING_OBJ$rec$rec_200_yds <- 0     # Bonus for 200+ receiving yards

# === MISCELLANEOUS SCORING ===
SCORING_OBJ$misc$all_pos <- TRUE          # Apply to all positions
SCORING_OBJ$misc$fumbles_lost <- -2       # Points per fumble lost (was -3)
SCORING_OBJ$misc$fumbles_total <- 0       # Points per fumble (recovered or lost)
SCORING_OBJ$misc$sacks <- 0               # Points per sack (for QBs)
SCORING_OBJ$misc$two_pts <- 2             # Points per 2-point conversion

# === KICKER SCORING ===
SCORING_OBJ$kick$xp <- 1            # Points per extra point
SCORING_OBJ$kick$fg_0019 <- 3       # Points for 0-19 yard FG
SCORING_OBJ$kick$fg_2029 <- 3       # Points for 20-29 yard FG
SCORING_OBJ$kick$fg_3039 <- 3       # Points for 30-39 yard FG
SCORING_OBJ$kick$fg_4049 <- 3       # Points for 40-49 yard FG (was 4)
SCORING_OBJ$kick$fg_50 <- 5         # Points for 50+ yard FG
SCORING_OBJ$kick$fg_miss <- 0       # Points per missed FG

# === RETURN SCORING ===
SCORING_OBJ$ret$all_pos <- TRUE      # Apply to all positions
SCORING_OBJ$ret$return_tds <- 6      # Points per return TD
SCORING_OBJ$ret$return_yds <- 0      # Points per return yard

# === IDP (DEFENSIVE PLAYERS) SCORING ===
SCORING_OBJ$idp$all_pos <- TRUE      # Apply to all positions
SCORING_OBJ$idp$idp_solo <- 1        # Points per solo tackle
SCORING_OBJ$idp$idp_asst <- 0.5      # Points per assisted tackle
SCORING_OBJ$idp$idp_sack <- 2        # Points per sack
SCORING_OBJ$idp$idp_int <- 3         # Points per interception
SCORING_OBJ$idp$idp_fum_force <- 3   # Points per forced fumble
SCORING_OBJ$idp$idp_fum_rec <- 2     # Points per fumble recovery
SCORING_OBJ$idp$idp_pd <- 1          # Points per pass defended
SCORING_OBJ$idp$idp_td <- 6          # Points per defensive TD
SCORING_OBJ$idp$idp_safety <- 2      # Points per safety

# === DEFENSE/SPECIAL TEAMS SCORING ===
SCORING_OBJ$dst$dst_fum_rec <- 2     # Points per fumble recovery
SCORING_OBJ$dst$dst_int <- 2         # Points per interception
SCORING_OBJ$dst$dst_safety <- 2      # Points per safety
SCORING_OBJ$dst$dst_sacks <- 1       # Points per sack
SCORING_OBJ$dst$dst_td <- 6          # Points per TD
SCORING_OBJ$dst$dst_blk <- 1.5       # Points per blocked kick
SCORING_OBJ$dst$dst_ret_yds <- 0     # Points per return yard
SCORING_OBJ$dst$dst_pts_allowed <- 0 # Base points for points allowed (modified by brackets)

# === DEFENSE POINTS ALLOWED BRACKETS ===
# Points awarded based on opponent points allowed
SCORING_OBJ$pts_bracket <- list(
  list(threshold = 0,  points = 10),   # 0 points allowed
  list(threshold = 6,  points = 7),    # 1-6 points allowed
  list(threshold = 13, points = 4),    # 7-13 points allowed
  list(threshold = 20, points = 1),    # 14-20 points allowed
  list(threshold = 27, points = 0),    # 21-27 points allowed
  list(threshold = 34, points = -1),   # 28-34 points allowed
  list(threshold = 99, points = -4)    # 35+ points allowed
)

# === NOTES ===
# This configuration matches NFL.com's default PPR scoring with the following adjustments:
# 1. PPR: 1 point per reception (rec = 1)
# 2. Passing INT: -2 points (was -3 in default)
# 3. Fumbles lost: -2 points (was -3 in default)
# 4. FG 40-49 yards: 3 points (was 4 in default)
#
# To customize for your league:
# - Modify the values above to match your league settings
# - For standard (non-PPR) scoring, set SCORING_OBJ$rec$rec <- 0
# - For half-PPR, set SCORING_OBJ$rec$rec <- 0.5
# - Adjust TD values, yardage points, and bonuses as needed
