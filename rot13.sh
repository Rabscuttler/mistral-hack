#!/bin/bash

# Check if input is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <text to encrypt>"
    exit 1
fi

# Combine all arguments into a single string
input="$*"

# Perform ROT13 encryption using tr
encrypted=$(echo "$input" | tr 'A-Za-z' 'N-ZA-Mn-za-m')

# Output the encrypted text
echo "$encrypted"
