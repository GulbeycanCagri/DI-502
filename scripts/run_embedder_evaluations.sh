#!/bin/bash

# Number of embedders defined in evaluate_rag_embedder.py (currently 5)
NUM_EMBEDDERS=5

echo "Starting sequential evaluation of $NUM_EMBEDDERS embedders..."

for ((i=0; i<NUM_EMBEDDERS; i++)); do
    echo "----------------------------------------------------------------"
    echo "Running evaluation for embedder index $i"
    echo "----------------------------------------------------------------"
    
    python src/evaluate_rag_embedder.py --embedder_idx $i
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "Error running embedder index $i. Stopping."
        exit 1
    fi
    
    echo "Finished embedder index $i"
    echo "Sleeping for 5 seconds to ensure cleanup..."
    sleep 5
done

echo "All embedder evaluations completed."
