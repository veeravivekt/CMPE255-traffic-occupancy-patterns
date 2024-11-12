#!/bin/bash

# Define the URL and target directory
URL="https://archive.ics.uci.edu/static/public/204/pems+sf.zip"
TARGET_DIR="data"
ZIP_FILE="$TARGET_DIR/pems_sf.zip"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download the ZIP file
echo "-------------Downloading the dataset-------------"
curl -L -o "$ZIP_FILE" "$URL"

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download the file."
    exit 1
fi

# Extract the ZIP file
echo "Extracting the dataset..."
unzip -o "$ZIP_FILE" -d "$TARGET_DIR"

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract the ZIP file."
    exit 1
fi

# Clean up: remove the ZIP file after extraction
echo "Cleaning up..."
rm "$ZIP_FILE"

echo "Dataset downloaded and extracted successfully into the '$TARGET_DIR' folder."

echo "-------------Setting up environment-------------"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

source .venv/bin/activate

echo "Setup complete"
