for dir in '_merge0.2vctk' '_merge0.5vctk' '_merge1.0vctk' '_released'
do
  python scripts/vctk_codec_abx.py /home/s2522559/datastore /home/s2522559/datastore/items_vctk/vctk_word_test.item 18 'RepCodec_codecs/repcodec_hubert_large_l18_ls100'$dir'/VCTK_tokens_reorg' \
	--codebook_path 'RepCodec_codecs/repcodec_hubert_large_l18_ls100'$dir'/repcodec_hubert_large_l18_ls100'$dir'_codebook.pt' --abx_mode accent_word;
  # echo "scripts/results/repcodec_hubert_large_l18_ls100"$dir".txt"
done
