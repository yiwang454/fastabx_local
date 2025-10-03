cd /home/s2522559/workspace/fastabx/scripts
base_directory="/home/s2522559/datastore/CMU+L2_Arctic/CMU_Arctic_features"
for SPEAKER_DIR in "$base_directory"/*; do
    if [ -d "$SPEAKER_DIR" ]; then
        SPEAKER_NAME=$(basename "$SPEAKER_DIR")
        
        # Skip the 'tsv' directory as requested
        if [ "$SPEAKER_NAME" == "hubert_large_l18" ]; then
            echo "Skipping continous directory: $SPEAKER_NAME"
            continue
        fi

        echo "processing "$SPEAKER_DIR
        python reorg_features.py $SPEAKER_DIR $SPEAKER_DIR
    fi
done