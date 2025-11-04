#!/usr/bin/env Rscript
# ffanalytics All Data Loader
# Downloads comprehensive wide table from ffanalytics: projections, ECR, ADP (CBS/ESPN/FFC/NFL/RTS/Yahoo),
# AAV (ESPN/Yahoo/NFL), risk metrics, VOR, tiers, and confidence intervals.
#
# Usage: Rscript ffanalytics_all_data.R [--season YEAR] [--week NUM] [--output-dir DIR]
# Output: data_raw/ffanalytics_all_data_{season}.csv (or _week{N}.csv)
# Config: Uses fantasy_projections_config.R for scoring rules
# Reference: https://ffanalytics.fantasyfootballanalytics.net/reference/index.html

# === LOAD REQUIRED PACKAGES ===
required_packages <- c(
  "tidyverse",   # Data manipulation
  "progressr",   # Progress bars
  "lubridate"    # Date handling
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

if (!requireNamespace("ffanalytics", quietly = TRUE)) {
  cat("\n")
  cat(rep("!", 70), "\n", sep = "")
  cat("ERROR: ffanalytics package is not installed\n")
  cat(rep("!", 70), "\n", sep = "")
  cat("\nInstalling ffanalytics from GitHub...\n")

  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org/")
  }

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

# Load the package
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
    FALSE
  }
)

if (!ffanalytics_available) {
  cat("\n✗ CANNOT PROCEED: ffanalytics package is required\n\n")
  quit(status = 1)
}

cat("✓ ffanalytics loaded successfully\n")

# === CONFIGURATION ===
# Load scoring configuration
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg)
    return(dirname(script_path))
  }
  
  return(getwd())
}

script_dir <- get_script_dir()
config_path <- file.path(script_dir, "fantasy_projections_config.R")

cat("Looking for config at:", config_path, "\n")

if (file.exists(config_path)) {
  source(config_path)
  cat("✓ Loaded scoring configuration\n")
} else {
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
OUTPUT_DIR <- "./data_raw"

# === HELPER FUNCTIONS ===

# Test connection to ffanalytics S3 data source
test_connection <- function() {
  cat("\nTesting connection to ffanalytics data source...\n")
  
  test_url <- "https://s3.us-east-2.amazonaws.com/ffanalytics/packagedata/player_table.csv"
  
  tryCatch(
    {
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
      return(FALSE)
    }
  )
}

# Parse command line arguments
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  
  # Default values
  season <- NULL # NULL = current season
  week <- 0      # 0 = seasonal projections
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
      cat("Usage: Rscript ffanalytics_all_data.R [options]\n")
      cat("\nOptions:\n")
      cat("  --season YEAR       Season year (default: current season)\n")
      cat("  --week NUM          Week number (0 = seasonal, 1-18 = weekly, default: 0)\n")
      cat("  --output-dir DIR    Output directory (default: ./data_raw)\n")
      cat("  --help, -h          Show this help message\n")
      cat("\nExamples:\n")
      cat("  Rscript ffanalytics_all_data.R                    # Current season\n")
      cat("  Rscript ffanalytics_all_data.R --season 2025     # 2025 season\n")
      cat("  Rscript ffanalytics_all_data.R --week 1          # Current season, week 1\n")
      quit(status = 0)
    } else {
      i <- i + 1
    }
  }
  
  list(season = season, week = week, output_dir = output_dir)
}

# Download projections from multiple sources (season: NULL=current, week: 0=seasonal)
download_projections <- function(season = NULL, week = 0) {
  projection_type <- if (week == 0) "seasonal" else paste0("week ", week)
  cat("\n", rep("=", 60), "\n", sep = "")
  cat(
    "Downloading", projection_type, "projections for season:",
    ifelse(is.null(season), "CURRENT", season), "\n"
  )
  cat(rep("=", 60), "\n", sep = "")
  
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

# Calculate projected points using scoring rules
calculate_projected_points <- function(projections_raw, scoring_rules) {
  cat("\nCalculating projected fantasy points...\n")
  
  tryCatch(
    {
      projections_scored <- ffanalytics:::impute_and_score_sources(
        data_result = projections_raw,
        scoring_rules = scoring_rules
      )
      
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

# Create projections table with averaged stats (mean/robust/weighted)
create_projections_table <- function(projections_raw, scoring_rules) {
  cat("\nCreating projections table with averaged stats...\n")
  
  tryCatch(
    {
      projections_table <- ffanalytics::projections_table(
        projections_raw,
        scoring_rules = scoring_rules,
        return_raw_stats = TRUE
      )
      
      cat("✓ Created projections table with", nrow(projections_table), "players\n")
      
      return(projections_table)
    },
    error = function(e) {
      cat("✗ Error creating projections table:", e$message, "\n")
      stop(e)
    }
  )
}

# Add player information (names, positions, teams)
add_player_information <- function(data) {
  cat("\nAdding player information...\n")
  
  tryCatch(
    {
      data_with_info <- data %>%
        ffanalytics::add_player_info()
      
      cat("✓ Player information added\n")
      
      return(data_with_info)
    },
    error = function(e) {
      cat("⚠ Warning: Could not add player info:", e$message, "\n")
      return(data)
    }
  )
}

# Add Expert Consensus Rankings (ECR) from FantasyPros
add_expert_rankings <- function(data) {
  cat("\nAdding Expert Consensus Rankings (ECR)...\n")
  
  tryCatch(
    {
      data_with_ecr <- data %>%
        ffanalytics::add_ecr()
      
      cat("✓ ECR added\n")
      
      return(data_with_ecr)
    },
    error = function(e) {
      cat("⚠ Warning: Could not add ECR:", e$message, "\n")
      return(data)
    }
  )
}

# Add ADP from CBS, ESPN, FFC, NFL, RTS, Yahoo
add_draft_position <- function(data) {
  cat("\nAdding Average Draft Position (ADP) from multiple sources...\n")
  
  tryCatch(
    {
      data_with_adp <- data %>%
        ffanalytics::add_adp()
      
      cat("✓ ADP added\n")
      
      return(data_with_adp)
    },
    error = function(e) {
      cat("⚠ Warning: Could not add ADP:", e$message, "\n")
      return(data)
    }
  )
}

# Add AAV from ESPN, Yahoo, NFL
add_auction_values <- function(data) {
  cat("\nAdding Average Auction Values (AAV)...\n")
  
  tryCatch(
    {
      data_with_aav <- data %>%
        ffanalytics::add_aav()
      
      cat("✓ AAV added\n")
      
      return(data_with_aav)
    },
    error = function(e) {
      cat("⚠ Warning: Could not add AAV:", e$message, "\n")
      return(data)
    }
  )
}

# Add projection uncertainty and confidence intervals
add_projection_uncertainty <- function(data) {
  cat("\nAdding projection uncertainty...\n")
  
  tryCatch(
    {
      data_with_uncertainty <- data %>%
        ffanalytics::add_uncertainty()
      
      cat("✓ Projection uncertainty added\n")
      
      return(data_with_uncertainty)
    },
    error = function(e) {
      cat("⚠ Warning: Could not add uncertainty:", e$message, "\n")
      return(data)
    }
  )
}

# Add risk calculations based on projection variance
add_risk_metrics <- function(data) {
  cat("\nAdding risk calculations...\n")
  
  tryCatch(
    {
      data_with_risk <- data %>%
        ffanalytics::add_risk()
      
      cat("✓ Risk metrics added\n")
      
      return(data_with_risk)
    },
    error = function(e) {
      cat("⚠ Warning: Could not add risk:", e$message, "\n")
      return(data)
    }
  )
}

# Add Value Over Replacement (VOR) calculations
add_value_over_replacement <- function(data) {
  cat("\nAdding Value Over Replacement (VOR)...\n")
  
  tryCatch(
    {
      data_with_vor <- data %>%
        ffanalytics::add_vor()
      
      cat("✓ VOR added\n")
      
      return(data_with_vor)
    },
    error = function(e) {
      cat("⚠ Warning: Could not add VOR:", e$message, "\n")
      return(data)
    }
  )
}

# Add tier assignments by position
add_player_tiers <- function(data) {
  cat("\nAdding player tiers...\n")
  
  tryCatch(
    {
      data_with_tiers <- data %>%
        ffanalytics::set_tiers()
      
      cat("✓ Player tiers added\n")
      
      return(data_with_tiers)
    },
    error = function(e) {
      cat("⚠ Warning: Could not add tiers:", e$message, "\n")
      return(data)
    }
  )
}

# Save data to CSV with column summary
save_to_csv <- function(data, season, week, output_dir) {
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
    filename <- file.path(output_dir, paste0("ffanalytics_all_data_", season, ".csv"))
  } else {
    filename <- file.path(output_dir, paste0("ffanalytics_all_data_", season, "_week", week, ".csv"))
  }
  
  # Write CSV
  cat("\nWriting data to:", filename, "\n")
  
  tryCatch(
    {
      write_csv(data, filename)
      cat("✓ Successfully wrote", nrow(data), "rows to", filename, "\n")
      
      # Print column summary
      cat("\nColumns included (", ncol(data), " total):\n", sep = "")
      col_names <- names(data)
      
      # Group columns by category for better readability
      id_cols <- col_names[grepl("^(id|player_name|player|pos|team)", col_names, ignore.case = TRUE)]
      proj_cols <- col_names[grepl("^(pass|rush|rec|points|avg_)", col_names, ignore.case = TRUE)]
      rank_cols <- col_names[grepl("^(ecr|rank|tier)", col_names, ignore.case = TRUE)]
      draft_cols <- col_names[grepl("^(adp|aav)", col_names, ignore.case = TRUE)]
      risk_cols <- col_names[grepl("^(risk|sd|lower|upper|ceiling|floor)", col_names, ignore.case = TRUE)]
      vor_cols <- col_names[grepl("^vor", col_names, ignore.case = TRUE)]
      other_cols <- setdiff(col_names, c(id_cols, proj_cols, rank_cols, draft_cols, risk_cols, vor_cols))
      
      if (length(id_cols) > 0) {
        cat("  Player IDs & Info (", length(id_cols), "): ", paste(head(id_cols, 10), collapse = ", "), 
            ifelse(length(id_cols) > 10, "...", ""), "\n", sep = "")
      }
      if (length(proj_cols) > 0) {
        cat("  Projections (", length(proj_cols), "): ", paste(head(proj_cols, 10), collapse = ", "), 
            ifelse(length(proj_cols) > 10, "...", ""), "\n", sep = "")
      }
      if (length(rank_cols) > 0) {
        cat("  Rankings & Tiers (", length(rank_cols), "): ", paste(rank_cols, collapse = ", "), "\n", sep = "")
      }
      if (length(draft_cols) > 0) {
        cat("  Draft Data (", length(draft_cols), "): ", paste(draft_cols, collapse = ", "), "\n", sep = "")
      }
      if (length(risk_cols) > 0) {
        cat("  Risk & Uncertainty (", length(risk_cols), "): ", paste(head(risk_cols, 10), collapse = ", "), 
            ifelse(length(risk_cols) > 10, "...", ""), "\n", sep = "")
      }
      if (length(vor_cols) > 0) {
        cat("  VOR (", length(vor_cols), "): ", paste(vor_cols, collapse = ", "), "\n", sep = "")
      }
      if (length(other_cols) > 0) {
        cat("  Other (", length(other_cols), "): ", paste(head(other_cols, 10), collapse = ", "), 
            ifelse(length(other_cols) > 10, "...", ""), "\n", sep = "")
      }
    },
    error = function(e) {
      cat("✗ Error writing CSV:", e$message, "\n")
      stop(e)
    }
  )
}

# Main pipeline: download, enrich, and save comprehensive ffanalytics data
main <- function(season = NULL, week = 0, output_dir = OUTPUT_DIR) {
  cat("\n")
  cat(rep("=", 70), "\n", sep = "")
  cat("  FFANALYTICS ALL DATA LOADER\n")
  cat(rep("=", 70), "\n", sep = "")
  
  # Test connection
  if (!test_connection()) {
    cat("\n✗ Cannot proceed without data source connectivity\n\n")
    quit(status = 1)
  }
  
  start_time <- Sys.time()
  
  # 1. Download raw projections
  projections_raw <- download_projections(season, week)
  
  # 2. Calculate projected points
  projections_scored <- calculate_projected_points(projections_raw, SCORING_OBJ)
  
  # 3. Create projections table with averages
  data <- create_projections_table(projections_raw, SCORING_OBJ)
  
  # 4. Add all supplemental data
  data <- add_player_information(data)
  data <- add_expert_rankings(data)
  
  # Only add seasonal data for week 0
  if (week == 0) {
    data <- add_draft_position(data)
    data <- add_auction_values(data)
    data <- add_projection_uncertainty(data)
    data <- add_risk_metrics(data)
    data <- add_value_over_replacement(data)
    data <- add_player_tiers(data)
  } else {
    # For weekly, only add uncertainty if available
    data <- add_projection_uncertainty(data)
  }
  
  # 5. Save to CSV
  save_to_csv(data, season, week, output_dir)
  
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
