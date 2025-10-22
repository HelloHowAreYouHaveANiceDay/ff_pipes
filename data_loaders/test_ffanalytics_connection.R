#!/usr/bin/env Rscript
# Test connectivity to ffanalytics data sources
# Run this to diagnose why ffanalytics package is failing to load

cat("\n")
cat(rep("=", 70), "\n", sep = "")
cat("  FFANALYTICS CONNECTION DIAGNOSTIC TOOL\n")
cat(rep("=", 70), "\n", sep = "")
cat("\n")

# Test 1: Basic internet connectivity
cat("Test 1: Internet connectivity\n")
cat("------------------------------\n")
test_url <- "https://www.google.com"
internet_works <- tryCatch({
  con <- url(test_url, "r")
  close(con)
  cat("✓ Internet connection: WORKING\n\n")
  TRUE
}, error = function(e) {
  cat("✗ Internet connection: FAILED\n")
  cat("  Error:", e$message, "\n\n")
  FALSE
})

# Test 2: S3 bucket accessibility
cat("Test 2: S3 bucket access\n")
cat("------------------------\n")
s3_url <- "https://s3.us-east-2.amazonaws.com/ffanalytics/packagedata/player_table.csv"
cat("Attempting to access:", s3_url, "\n")

s3_works <- tryCatch({
  con <- url(s3_url, "r")
  result <- readLines(con, n = 1, warn = FALSE)
  close(con)
  
  if (length(result) > 0) {
    cat("✓ S3 bucket access: WORKING\n")
    cat("  First line:", substr(result, 1, 60), "...\n\n")
    TRUE
  } else {
    cat("✗ S3 bucket access: NO DATA\n\n")
    FALSE
  }
}, error = function(e) {
  cat("✗ S3 bucket access: FAILED\n")
  cat("  Error:", e$message, "\n")
  cat("\n  This is the same error preventing ffanalytics from loading\n\n")
  FALSE
})

# Test 3: DNS resolution
cat("Test 3: DNS resolution\n")
cat("----------------------\n")
cat("Testing DNS for: s3.us-east-2.amazonaws.com\n")

# Try to resolve DNS
dns_works <- tryCatch({
  # This uses nsl.tools which requires internet
  result <- nsl::nsl("s3.us-east-2.amazonaws.com")
  if (length(result) > 0) {
    cat("✓ DNS resolution: WORKING\n")
    cat("  IP addresses:", paste(result, collapse = ", "), "\n\n")
    TRUE
  } else {
    cat("✗ DNS resolution: NO RESULTS\n\n")
    FALSE
  }
}, error = function(e) {
  cat("⚠ DNS test: Could not test (nsl package not available)\n")
  cat("  Run manually: nslookup s3.us-east-2.amazonaws.com\n\n")
  NA
})

# Test 4: Check if ffanalytics is installed
cat("Test 4: ffanalytics package status\n")
cat("-----------------------------------\n")
if (requireNamespace("ffanalytics", quietly = TRUE)) {
  cat("✓ ffanalytics package: INSTALLED\n")
  
  # Try to load it
  loaded <- tryCatch({
    suppressPackageStartupMessages(library(ffanalytics))
    cat("✓ ffanalytics loading: SUCCESS\n\n")
    TRUE
  }, error = function(e) {
    cat("✗ ffanalytics loading: FAILED\n")
    cat("  Error:", e$message, "\n\n")
    FALSE
  })
} else {
  cat("✗ ffanalytics package: NOT INSTALLED\n")
  cat("  Install with: remotes::install_github('FantasyFootballAnalytics/ffanalytics')\n\n")
}

# Summary
cat("\n")
cat(rep("=", 70), "\n", sep = "")
cat("  DIAGNOSTIC SUMMARY\n")
cat(rep("=", 70), "\n", sep = "")
cat("\n")

if (internet_works && s3_works) {
  cat("✓ All connectivity tests passed!\n")
  cat("  The ffanalytics package should be able to load.\n")
  cat("  If it still fails, try reinstalling:\n")
  cat("    remove.packages('ffanalytics')\n")
  cat("    remotes::install_github('FantasyFootballAnalytics/ffanalytics')\n")
} else if (!internet_works) {
  cat("✗ Internet connectivity is not working\n")
  cat("  Please check your network connection\n")
} else if (!s3_works) {
  cat("✗ Cannot access S3 bucket (this is the core issue)\n")
  cat("\nPossible causes:\n")
  cat("  1. DNS resolution failure\n")
  cat("  2. Firewall blocking AWS S3\n")
  cat("  3. Corporate proxy restrictions\n")
  cat("  4. S3 bucket is temporarily unavailable\n")
  cat("\nSuggested actions:\n")
  cat("  1. Check firewall/antivirus settings\n")
  cat("  2. Try from a different network (mobile hotspot, VPN)\n")
  cat("  3. Configure DNS servers (try 8.8.8.8, 8.8.4.4)\n")
  cat("  4. Contact IT department if on corporate network\n")
  cat("  5. Wait and try again later (service may be down)\n")
}

cat("\n")
cat("For more help, visit:\n")
cat("  https://github.com/FantasyFootballAnalytics/ffanalytics/issues\n")
cat("\n")
