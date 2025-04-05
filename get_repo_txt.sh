#!/usr/bin/env bash

# Name of the output file
OUTPUT="repo.txt"

# Remove the output file if it exists, so we start fresh
rm -f "$OUTPUT"

# Get the basename of this script (so we can exclude it)
SCRIPT_NAME="$(basename "$0")"

# A dashed separator line (adjust length to your preference)
SEPARATOR="--------------------------------------------------------------------------------"

# Recursively find all files from the current directory,
# then loop over them
find . -type f | while read -r file; do

    # Exclude this script itself
    if [ "$(basename "$file")" = "$SCRIPT_NAME" ]; then
        continue
    fi

    # Exclude the output file (repo.txt) if it exists
    if [ "$(basename "$file")" = "$OUTPUT" ]; then
        continue
    fi

    # Keep the path as returned by 'find', which includes the leading './'
    RELATIVE_PATH="$file"

    # Mark the beginning of this file's section
    echo "$SEPARATOR" >> "$OUTPUT"
    echo "BEGIN FILE: $RELATIVE_PATH" >> "$OUTPUT"
    echo "$SEPARATOR" >> "$OUTPUT"

    # Append the file's content
    cat "$file" >> "$OUTPUT"

    # Mark the end of this file's section
    echo "$SEPARATOR" >> "$OUTPUT"
    echo "END FILE: $RELATIVE_PATH" >> "$OUTPUT"
    echo "$SEPARATOR" >> "$OUTPUT"

    # (Optional) Add a blank line separator
    echo "" >> "$OUTPUT"

done

echo "All files have been concatenated into $OUTPUT."