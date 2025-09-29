# for num in 100 200 300 400 600 900
# do 
#     python scripts/reorg_accenterror_df.py /home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100_released/VCTK_tokens_reorg_abx_errors_nolevel.csv number $num;
# done

# for ratio in 0.025 0.05 0.1 0.2 
# do
#     python scripts/reorg_accenterror_df.py /home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100_released/VCTK_tokens_reorg_abx_errors_nolevel.csv ratio $ratio;
# done
for num in 100 200 300 400 600 900
do 
    python scripts/reorg_accenterror_df.py /home/s2522559/workspace/fastabx/scripts/results/abx_errors_nolevel.csv "number" $num;
done

for ratio in 0.025 0.05 0.1 0.2 
do
    python scripts/reorg_accenterror_df.py /home/s2522559/workspace/fastabx/scripts/results/abx_errors_nolevel.csv "ratio" $ratio;
done