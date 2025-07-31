import torch
# from fastabx import zerospeech_abx
from fastabx import Dataset, Subsampler, Task
from fastabx import Score
from fastabx import zerospeech_abx

layer = 18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_percent = 0.1

def maker(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True)

def main():
    item, frequency = "/mnt/ceph_rbd/muavic/scripts/vctk_largeclass_accent_tenth.item", 50
    features = f"/mnt/ceph_rbd/data/vctk/hubert_feature/large_l{layer}_mic1"
    dataset = Dataset.from_item(item, features, frequency) # feature_maker

    subsampler = Subsampler(max_size_group=10, max_x_across=2)
    task = Task(dataset, on="#phone", by=["next-phone", "prev-phone", "accent"], across=["speaker"], subsampler=subsampler,)
    print(len(task))
    print(task[0])
    score = Score(task, "angular")
    abx_error_rate = score.collapse(levels=[("prev-phone", "next-phone", "accent"), "speaker"])
    print(abx_error_rate)

    # abx = zerospeech_abx(
    #     item,
    #     features,
    #     max_size_group=10,
    #     max_x_across=5,
    #     feature_maker=maker,
    #     extension=".pt",
    # )
    # print(abx)
if __name__ == "__main__":
    main()




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
