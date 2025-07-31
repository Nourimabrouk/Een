# Test script for the R love letter
# Run this to test if the love letter works

# Check if required packages are installed
required_packages <- c("tidyverse", "scales", "glue", "lubridate")

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) > 0) {
  cat("Missing packages:", paste(missing_packages, collapse = ", "), "\n")
  cat("Install them with: install.packages(c(", paste(paste0("'", missing_packages, "'"), collapse = ", "), "))\n")
} else {
  cat("✅ All required packages are available!\n")
  cat("🌟 You can now run the love letter with:\n")
  cat("source('love_letter_tidyverse_2025.R')\n")
  
  # Quick test of core functionality
  tryCatch({
    library(tidyverse, quietly = TRUE, warn.conflicts = FALSE)
    library(glue, quietly = TRUE, warn.conflicts = FALSE)
    
    phi <- 1.618033988749895
    test_moment <- 1
    heartbeat <- sin(test_moment * phi) * exp(-test_moment/21)
    
    cat("📊 Quick test successful!\n")
    cat("Golden ratio φ =", phi, "\n")
    cat("Test heartbeat intensity =", round(heartbeat, 4), "\n")
    cat("💖 Love letter is ready to run! 💖\n")
    
  }, error = function(e) {
    cat("❌ Error in love letter:", e$message, "\n")
  })
}