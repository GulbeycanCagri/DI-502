#!/bin/bash

# Number of prompts defined in evaluate_rag_v2.py (currently 12)
NUM_PROMPTS=12

echo "Starting sequential evaluation of $NUM_PROMPTS prompts..."

for ((i=1; i<NUM_PROMPTS; i++)); do
    echo "----------------------------------------------------------------"
    echo "Running evaluation for prompt index $i"
    echo "----------------------------------------------------------------"
    
    python src/evaluate_rag_v2.py --prompt_idx $i
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "Error running prompt index $i. Stopping."
        exit 1
    fi
    
    echo "Finished prompt index $i"
    echo "Sleeping for 5 seconds to ensure cleanup..."
    sleep 5
done

echo "All evaluations completed."