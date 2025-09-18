cd /home/s2522559/workspace/fastabx/scripts
for dir in '' '_merge0.2vctk' '_merge0.5vctk' '_merge1.0vctk' '_released'
do
 input='/home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100'$dir'/VCTK_tokens'
 output='/home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100'$dir'/VCTK_tokens_reorg'
 python reorg_features.py $input $output
done