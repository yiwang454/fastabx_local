import torch
# from fastabx import zerospeech_abx
from fastabx import Dataset, Subsampler, Task
from fastabx import Score
from fastabx import zerospeech_abx
import sys, os

layer = 18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_percent = 0.1

def maker(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True)

def maker_codec_external(path: str, codec_codebook: torch.Tensor):
    indices = torch.load(path, weights_only=True).squeeze(0) # , map_location=torch.device('cpu')
    feature = codec_codebook[indices]
    return feature


def main(output_path, rank=0):
    with open(output_path, "a+") as output_w:
        output_w.write("start running")
    frequency = 50
    if rank == 0:
        item = "/mnt/ceph_rbd/muavic/scripts/items_vctk/vctk_largeclass_debug.item"
    elif rank == 0.5:
        item = "/mnt/ceph_rbd/muavic/scripts/items_vctk/vctk_largeclass_debug_half.item"
    else:
        item = f"/mnt/ceph_rbd/muavic/scripts/items_vctk/vctk_largeclass_debug_fourth{rank}.item"

    features = f"/mnt/ceph_rbd/data/vctk/hubert_feature/large_l{layer}_mic1"
    # codebook_path = f"/mnt/ceph_rbd/data/vctk/features/semantic_codebook.pt"
    # codec_codebook = torch.load(codebook_path, weights_only=True) # , map_location=torch.device('cpu')
    dataset = Dataset.from_item(item, features, frequency, feature_maker=maker) # feature_maker

    subsampler = Subsampler(max_size_group=10, max_x_across=2)
    # task = Task(dataset, on="#phone", across=["speaker"], subsampler=subsampler,) # by=["next-phone", "prev-phone", "accent"], 
    task = Task(dataset, on="accent", by=["#phone", "next-phone", "prev-phone"], across=["speaker"], subsampler=subsampler,) #  "accent"
    with open(output_path, "a+") as output_w:
        output_w.write("rank {}, task len {}\n".format(rank, len(task)))
        if len(task) > 0:
            output_w.write("task 0 {}\n".format(str(task)))

    score = Score(task, "angular")
    abx_error_rate = score.collapse(levels=[("#phone", "prev-phone", "next-phone"), "speaker"]) # "prev-phone", "next-phone", "accent"
    with open(output_path, "a+") as output_w:
        output_w.write("rank {}, abx_error_rate {} \n".format(rank, abx_error_rate))

    # abx = zerospeech_abx(
    #     item,
    #     features,
    #     max_size_group=10
    #     max_x_across=5,
    #     feature_maker=maker,
    #     extension=".pt",
    # )
    # print(abx)

if __name__ == "__main__":
    # for r in range(1, 3):
    #     main(r)
    o_path = sys.argv[1]
    # for r in range(1, 5):
    main(o_path, rank=0.5)



def maker(path: str) -> torch.Tensor:
    features = torch.load(path, weights_only=True)
    # assert sr == bundle.sample_rate
    # features, _ = model.extract_features(x.to(device))
    return features 

def zerospeech_abx_trial():
    abx = zerospeech_abx(
        "/mnt/ceph_rbd/muavic/scripts/vctk_temp_item.item",
        f"/mnt/ceph_rbd/data/vctk/hubert_feature/large_l{layer}_mic1",
        max_size_group=10,
        max_x_across=None,
        feature_maker=maker,
        extension=".pt",
    )
    print(abx)
    """
    0.0 # something wrong, maybe because there's only one speaker above
    """


# def post_delete_item():
#     [170498, 264684, 397790, 505072]
