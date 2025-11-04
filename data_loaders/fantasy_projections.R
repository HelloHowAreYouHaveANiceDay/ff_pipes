#!/usr/bin/env Rscript
# Fantasy Football Projections Data Loader
# Downloads and processes fantasy football projections from multiple sources
#
# Based on: https://isaactpetersen.github.io/Fantasy-Football-Analytics-Textbook/download-football-data.html#sec-scrapeProjections
#
# Inputs:
# - Multiple fantasy projection websites (via ffanalytics package)
# - League scoring configuration (scoring_config.R)
#
# Outputs:
# - reports/projections_seasonal_{season}.csv - Seasonal projections by player
# - reports/projections_weekly_{season}_week{week}.csv - Weekly projections by player-week
#
# Columns (Seasonal):
# - season, player_id, player_name, position, team
# - avg_type (mean/robust/weighted average across sources)
# - [projection stats]: pass_attempts, pass_completions, pass_yards, pass_tds, pass_ints
#                       rush_attempts, rush_yards, rush_tds
#                       targets, receptions, rec_yards, rec_tds
# - fantasy_points (calculated from scoring settings)
# - sd_points (projection uncertainty)
# - ecr (expert consensus ranking)
# - adp (average draft position)
# - aav (average auction value)
#
# Columns (Weekly):
# - season, week, player_id, player_name, position, team, opponent
# - avg_type, [projection stats], fantasy_points, sd_points, ecr_week

# === LOAD REQUIRED PACKAGES ===
required_packages <- c(
  "tidyverse", # Data manipulation
  "progressr", # Progress bars
  "lubridate" # Date handling
)

# Install missing packages
missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cloud.r-project.org/")
}

# Load packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(progressr)
  library(lubridate)
})

# Try to load ffanalytics with error handling
cat("Loading ffanalytics package...\n")

# First check if ffanalytics is installed
if (!requireNamespace("ffanalytics", quietly = TRUE)) {
  cat("\n")
  cat(rep("!", 70), "\n", sep = "")
  cat("ERROR: ffanalytics package is not installed\n")
  cat(rep("!", 70), "\n", sep = "")
  cat("\nInstalling ffanalytics from GitHub...\n")

  # Install remotes if needed
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org/")
  }

  # Try to install ffanalytics
  tryCatch(
    {
      remotes::install_github("FantasyFootballAnalytics/ffanalytics", quiet = FALSE)
      cat("✓ ffanalytics installed successfully\n")
    },
    error = function(e) {
      cat("\n✗ Failed to install ffanalytics:", e$message, "\n")
      cat("Please install manually:\n")
      cat("  install.packages('remotes')\n")
      cat("  remotes::install_github('FantasyFootballAnalytics/ffanalytics')\n\n")
      quit(status = 1)
    }
  )
}

# Now try to load the package
ffanalytics_available <- tryCatch(
  {
    suppressPackageStartupMessages(library(ffanalytics))
    TRUE
  },
  error = function(e) {
    cat("\n")
    cat(rep("!", 70), "\n", sep = "")
    cat("ERROR: ffanalytics package failed to load\n")
    cat(rep("!", 70), "\n", sep = "")
    cat("\nError details:\n")
    cat(e$message, "\n\n")

    cat("This error typically occurs when:\n")
    cat("1. Network connectivity issues prevent downloading required data\n")
    cat("2. The ffanalytics S3 data source is unavailable\n")
    cat("3. DNS resolution problems\n")
    cat("4. Firewall/proxy blocking the connection\n")

    cat("\nTroubleshooting steps:\n")
    cat("1. Check internet connection and try accessing:\n")
    cat("   https://s3.us-east-2.amazonaws.com/ffanalytics/packagedata/player_table.csv\n")
    cat("   in your browser\n\n")

    cat("2. If the URL doesn't work, the service may be down. Try:\n")
    cat("   - Waiting and retrying later\n")
    cat("   - Checking: https://github.com/FantasyFootballAnalytics/ffanalytics/issues\n\n")

    cat("3. Reinstall the package:\n")
    cat("   remove.packages('ffanalytics')\n")
    cat("   remotes::install_github('FantasyFootballAnalytics/ffanalytics')\n\n")

    cat("4. Check if there's a proxy/firewall blocking S3 access\n\n")

    FALSE
  }
)

if (!ffanalytics_available) {
  cat("\n✗ CANNOT PROCEED: ffanalytics package is required for projections scraping\n\n")
  quit(status = 1)
}

cat("✓ ffanalytics loaded successfully\n")

# === CONFIGURATION ===
# Load scoring configuration from external file
# Simple and robust way to find the script directory
get_script_dir <- function() {
  # Try to get from command args
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)

  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg)
    return(dirname(script_path))
  }

  # Fallback to current directory
  return(getwd())
}

script_dir <- get_script_dir()
config_path <- file.path(script_dir, "fantasy_projections_config.R")

# Try to source the config file
cat("Looking for config at:", config_path, "\n")

if (file.exists(config_path)) {
  source(config_path)
  cat("✓ Loaded scoring configuration\n")
} else {
  # Try alternative paths
  alt_paths <- c(
    "./data_loaders/fantasy_projections_config.R",
    "data_loaders/fantasy_projections_config.R",
    "fantasy_projections_config.R"
  )

  config_loaded <- FALSE
  for (alt_path in alt_paths) {
    if (file.exists(alt_path)) {
      source(alt_path)
      cat("✓ Loaded scoring configuration from:", alt_path, "\n")
      config_loaded <- TRUE
      break
    }
  }

  if (!config_loaded) {
    cat("⚠ Warning: Could not find fantasy_projections_config.R\n")
    cat("  Using default ffanalytics scoring settings\n")
    SCORING_OBJ <- ffanalytics::scoring
  }
}

# Output directory
OUTPUT_DIR <- "./reports"

# === HELPER FUNCTIONS ===

#' Test connection to ffanalytics data source
#' @return TRUE if connection successful, FALSE otherwise
test_connection <- function() {
  cat("\nTesting connection to ffanalytics data source...\n")

  test_url <- "https://s3.us-east-2.amazonaws.com/ffanalytics/packagedata/player_table.csv"

  tryCatch(
    {
      # Try to download a small amount of data
      con <- url(test_url, "r")
      result <- readLines(con, n = 1)
      close(con)

      if (length(result) > 0) {
        cat("✓ Connection successful\n")
        return(TRUE)
      } else {
        cat("✗ Connection failed: No data received\n")
        return(FALSE)
      }
    },
    error = function(e) {
      cat("✗ Connection failed:", e$message, "\n")
      cat("\nPlease check:\n")
      cat("1. Internet connectivity\n")
      cat("2. DNS resolution (try: nslookup s3.us-east-2.amazonaws.com)\n")
      cat("3. Firewall/proxy settings\n")
      cat("4. Try accessing the URL in your browser:\n")
      cat("   ", test_url, "\n\n")
      return(FALSE)
    }
  )
}


#' Parse command line arguments
#' @return list with season, week, and output_dir
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # Default values
  season <- NULL # NULL = current season
  week <- 0 # 0 = seasonal projections
  output_dir <- OUTPUT_DIR

  # Parse arguments
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--season" && i < length(args)) {
      season <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--week" && i < length(args)) {
      week <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--output-dir" && i < length(args)) {
      output_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--help" || args[i] == "-h") {
      cat("Usage: Rscript fantasy_projections.R [options]\n")
      cat("\nOptions:\n")
      cat("  --season YEAR       Season year (default: current season)\n")
      cat("  --week NUM          Week number (0 = seasonal, 1-18 = weekly, default: 0)\n")
      cat("  --output-dir DIR    Output directory (default: ./reports)\n")
      cat("  --help, -h          Show this help message\n")
      cat("\nExamples:\n")
      cat("  Rscript fantasy_projections.R                    # Current season, seasonal projections\n")
      cat("  Rscript fantasy_projections.R --season 2024     # 2024 season, seasonal projections\n")
      cat("  Rscript fantasy_projections.R --week 1          # Current season, week 1 projections\n")
      cat("  Rscript fantasy_projections.R --season 2024 --week 5  # 2024 season, week 5\n")
      quit(status = 0)
    } else {
      i <- i + 1
    }
  }

  list(season = season, week = week, output_dir = output_dir)
}


#' Download projections from multiple sources
#' @param season Season year (NULL = current)
#' @param week Week number (0 = seasonal, 1-18 = weekly)
#' @return Raw projections data
download_projections <- function(season = NULL, week = 0) {
  projection_type <- if (week == 0) "seasonal" else paste0("week ", week)
  cat("\n", rep("=", 60), "\n", sep = "")
  cat(
    "Downloading", projection_type, "projections for season:",
    ifelse(is.null(season), "CURRENT", season), "\n"
  )
  cat(rep("=", 60), "\n", sep = "")

  # Download with progress bar
  cat("\nScraping projection data from multiple sources...\n")
  cat("(This may take several minutes)\n\n")

  tryCatch(
    {
      projections_raw <- progressr::with_progress({
        ffanalytics::scrape_data(
          season = season,
          week = week
        )
      })

      if (length(projections_raw) == 0) {
        stop("No projection data returned from scraping")
      }

      # Count records
      total_records <- projections_raw %>%
        bind_rows() %>%
        nrow()

      cat("\n✓ Successfully downloaded", total_records, "projection records\n")

      return(projections_raw)
    },
    error = function(e) {
      cat("\n✗ Error downloading projections:", e$message, "\n")
      stop(e)
    }
  )
}


#' Calculate projected points using scoring rules
#' @param projections_raw Raw projection data
#' @param scoring_rules Scoring configuration
#' @return Projections with calculated points by source
calculate_projected_points <- function(projections_raw, scoring_rules) {
  cat("\nCalculating projected fantasy points by source...\n")

  tryCatch(
    {
      # Use ffanalytics internal function to impute and score
      projections_scored <- ffanalytics:::impute_and_score_sources(
        data_result = projections_raw,
        scoring_rules = scoring_rules
      )

      # Count records
      total_records <- projections_scored %>%
        bind_rows() %>%
        nrow()

      cat("✓ Calculated points for", total_records, "records\n")

      return(projections_scored)
    },
    error = function(e) {
      cat("✗ Error calculating projected points:", e$message, "\n")
      stop(e)
    }
  )
}


#' Calculate averaged projections across sources
#' @param projections_raw Raw projection data
#' @param scoring_rules Scoring configuration
#' @param return_stats Include raw stats (TRUE) or just points (FALSE)
#' @return Averaged projections
calculate_averaged_projections <- function(projections_raw, scoring_rules, return_stats = TRUE) {
  cat("\nCalculating averaged projections across sources...\n")

  tryCatch(
    {
      projections_avg <- ffanalytics::projections_table(
        projections_raw,
        scoring_rules = scoring_rules,
        return_raw_stats = return_stats
      )

      cat("✓ Averaged projections for", nrow(projections_avg), "players\n")

      return(projections_avg)
    },
    error = function(e) {
      cat("✗ Error calculating averages:", e$message, "\n")
      stop(e)
    }
  )
}


#' Add supplemental player information
#' @param projections Projection data
#' @param week Week number (affects which supplements are added)
#' @return Enriched projections
add_supplemental_info <- function(projections, week = 0) {
  cat("\nAdding supplemental player information...\n")

  tryCatch(
    {
      # Add player info (positions, teams, etc.)
      cat("  - Adding player profile information...\n")
      projections <- projections %>%
        ffanalytics::add_player_info()

      if (week == 0) {
        # Seasonal projections: add draft rankings and values
        cat("  - Adding expert consensus rankings (ECR)...\n")
        projections <- projections %>%
          ffanalytics::add_ecr()

        cat("  - Adding average draft position (ADP)...\n")
        projections <- projections %>%
          ffanalytics::add_adp()

        cat("  - Adding average auction value (AAV)...\n")
        projections <- projections %>%
          ffanalytics::add_aav()

        cat("  - Adding projection uncertainty...\n")
        projections <- projections %>%
          ffanalytics::add_uncertainty()
      } else {
        # Weekly projections: add weekly rankings
        cat("  - Adding weekly expert consensus rankings...\n")
        projections <- projections %>%
          ffanalytics::add_ecr()

        # Note: Weekly projections may not have uncertainty
        tryCatch(
          {
            cat("  - Adding projection uncertainty...\n")
            projections <- projections %>%
              ffanalytics::add_uncertainty()
          },
          error = function(e) {
            cat("    (Uncertainty not available for weekly projections)\n")
          }
        )
      }

      cat("✓ Supplemental information added\n")

      return(projections)
    },
    error = function(e) {
      cat("✗ Error adding supplemental info:", e$message, "\n")
      cat("  (Continuing with available data)\n")
      return(projections)
    }
  )
}


#' Save projections to CSV
#' @param projections Projection data
#' @param season Season year
#' @param week Week number
#' @param output_dir Output directory
save_projections_csv <- function(projections, season, week, output_dir) {
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Determine season value
  if (is.null(season)) {
    season <- year(today())
  }

  # Generate filename
  if (week == 0) {
    filename <- file.path(output_dir, paste0("projections_seasonal_", season, ".csv"))
  } else {
    filename <- file.path(output_dir, paste0("projections_weekly_", season, "_week", week, ".csv"))
  }

  # Write CSV
  cat("\nWriting projections to:", filename, "\n")

  tryCatch(
    {
      write_csv(projections, filename)
      cat("✓ Successfully wrote", nrow(projections), "rows to", filename, "\n")
    },
    error = function(e) {
      cat("✗ Error writing CSV:", e$message, "\n")
      stop(e)
    }
  )
}


#' Main pipeline function
#' @param season Season year (NULL = current)
#' @param week Week number (0 = seasonal)
#' @param output_dir Output directory
main <- function(season = NULL, week = 0, output_dir = OUTPUT_DIR) {
  cat("\n")
  cat(rep("=", 70), "\n", sep = "")
  cat("  FANTASY FOOTBALL PROJECTIONS DATA LOADER\n")
  cat(rep("=", 70), "\n", sep = "")

  # Test connection to data source
  if (!test_connection()) {
    cat("\n✗ Cannot proceed without data source connectivity\n\n")
    quit(status = 1)
  }

  start_time <- Sys.time()

  # 1. Download raw projections
  projections_raw <- download_projections(season, week)

  # 2. Calculate projected points by source
  projections_scored <- calculate_projected_points(projections_raw, SCORING_OBJ)

  # 3. Calculate averaged projections
  projections_avg <- calculate_averaged_projections(projections_raw, SCORING_OBJ, return_stats = TRUE)

  # 4. Add supplemental information
  projections_enriched <- add_supplemental_info(projections_avg, week)

  # 5. Save to CSV
  save_projections_csv(projections_enriched, season, week, output_dir)

  # Report completion
  elapsed <- difftime(Sys.time(), start_time, units = "secs")
  cat("\n")
  cat(rep("=", 70), "\n", sep = "")
  cat("✓ PIPELINE COMPLETED SUCCESSFULLY\n")
  cat("  Time elapsed:", round(elapsed, 1), "seconds\n")
  cat(rep("=", 70), "\n", sep = "")
  cat("\n")
}


# === RUN PIPELINE ===
if (!interactive()) {
  # Parse command line arguments
  args <- parse_args()

  # Run main pipeline
  tryCatch(
    {
      main(
        season = args$season,
        week = args$week,
        output_dir = args$output_dir
      )
      quit(status = 0)
    },
    error = function(e) {
      cat("\n✗ PIPELINE FAILED\n")
      cat("Error:", e$message, "\n\n")
      quit(status = 1)
    }
  )
}
