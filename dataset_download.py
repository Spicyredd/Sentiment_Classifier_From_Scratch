import os
"""
This script downloads the latest version of the "movie-review-dataset" from Kaggle using the kagglehub library.

Steps performed:
1. Sets the KAGGLEHUB_CACHE environment variable to the current directory, ensuring downloaded files are cached locally.
2. Downloads the "vipulgandhi/movie-review-dataset" dataset from Kaggle.
3. Prints the local path to the downloaded dataset files.

Requirements:
- The 'kagglehub' library must be installed.
- Appropriate Kaggle API credentials must be configured for access.

Usage:
    python dataset_download.py
"""
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = "./"

# Download latest version
path = kagglehub.dataset_download("vipulgandhi/movie-review-dataset")

print("Path to dataset files:", path)
