for num in 100 200 300 400 600 900
do 
    python scripts/reorg_accenterror_df.py /home/s2522559/workspace/fastabx/scripts/results/abx_errors_nolevel.csv number $num;
done

for ratio in 0.025 0.05 0.1 0.2 
do
    python scripts/reorg_accenterror_df.py /home/s2522559/workspace/fastabx/scripts/results/abx_errors_nolevel.csv ratio $ratio;
done

for thres in 0.1 0.2 0.3
do
    python scripts/reorg_accenterror_df.py /home/s2522559/workspace/fastabx/scripts/results/abx_errors_nolevel.csv threshold $thres;
done