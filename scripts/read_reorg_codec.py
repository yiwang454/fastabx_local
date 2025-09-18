import torch
import os

# path = "/mnt/ceph_rbd/data/vctk/features/content_codebook.pt"

# content_codebook = torch.load(path, weights_only=True, map_location=torch.device('cpu'))
# print("content_codebook size", content_codebook.size())

def reorg_codecs(path_codec, root_path):

    content_codecs_vctk = torch.load(path_codec, weights_only=True, map_location=torch.device('cpu'))

    print("content_codecs_vctk keys", len(list(content_codecs_vctk.keys())))
    for key in content_codecs_vctk.keys():
        spk_folder = key.split("_")[0]
        # print(content_codecs_vctk[key].size())
        if not os.path.isdir(os.path.join(root_path, spk_folder)):
            os.makedirs(os.path.join(root_path, spk_folder))
        feat_path = os.path.join(root_path, spk_folder, key + ".pt")
        torch.save(content_codecs_vctk[key], feat_path)
        
    print("finished successfully")


path2 = "/mnt/ceph_rbd/data/vctk/features/semantic_codecs_vctk.pt"
feature_path = "/mnt/ceph_rbd/data/vctk/features/semantic_codecs_vctk"

reorg_codecs(path2, feature_path)

def check_dims(spk_dir):
    for file in os.listdir(spk_dir):
        realpath = os.path.join(spk_dir, file)
        codes = torch.load(realpath, weights_only=True, map_location=torch.device('cpu'))
        print(codes.size())
    
# check_dims("/mnt/ceph_rbd/data/vctk/features/content_codecs_vctk/p299")
# "/mnt/ceph_rbd/data/vctk/hubert_feature/large_l18_mic1/p299"
