# !/bin/bash
echo "Init requirements..."


source /mnt/ceph_rbd/applications/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/ceph_rbd/applications/anaconda3/envs/hf_torch_abx

# echo "Running Python script..."
cd /mnt/ceph_rbd/workspace/fastabx/scripts
python vctk_abx_accent.py /mnt/ceph_rbd/workspace/fastabx/scripts/result_vctk_hubert_accent.txt || echo "accent experiment failed" >> "/mnt/ceph_rbd/workspace/fastabx/scripts/result_vctk_hubert_accent.txt"; sleep infinity

sleep infinity
sleeper_pid=$!

# # Create a function that gets executed when this main process receives signal TERM
shutdown_pod(){
  1>&2 echo Exiting main process!
  kill $sleeper_pid
#   kill $timer_pid
  exit 0
}
trap shutdown_pod TERM

# Wait for the sleeper process to finish. This is important!
wait
