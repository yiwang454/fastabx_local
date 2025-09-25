for num in 100, 200, 300, 400, 600, 900
do 
    python scripts/reorg_accenterror_df.py /home/s2522559/workspace/fastabx/scripts/results/abx_errors_nolevel.csv $num --select_mode number
done

for ratio in 0.025 0.05 0.1 0.2 
do
    python scripts/reorg_accenterror_df.py /home/s2522559/workspace/fastabx/scripts/results/abx_errors_nolevel.csv $ratio --select_mode ratio
done
for thres in 0.1 0.2 0.3
    python scripts/reorg_accenterror_df.py /home/s2522559/workspace/fastabx/scripts/results/abx_errors_nolevel.csv $thres --select_mode threshold
done