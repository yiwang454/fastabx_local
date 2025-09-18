import torch
# from fastabx import zerospeech_abx
from fastabx import Dataset, Subsampler, Task
from fastabx import Score
from fastabx import zerospeech_abx
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_percent = 0.1

def maker(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True)

def main():
    item = sys.argv[1] # /home/s2522559/datastore/items_vctk/vctk_largeclass_debug_fourth1.item
    features_root = sys.argv[2] # /home/s2522559/datastore
    layer = sys.argv[3]
    frequency = 50
    features = f"{features_root}/vctk/hubert_feature/large_l{layer}_mic1"
    # f"/mnt/ceph_rbd/data/vctk/hubert_feature/large_l{layer}_mic1"
    # dataset = Dataset.from_item(item, features, frequency) # feature_maker

    abx = zerospeech_abx(
        item,
        features,
        max_size_group=10,
        max_x_across=5,
        feature_maker=maker,
        extension=".pt",
    )
    print(abx)

    # subsampler = Subsampler(max_size_group=10, max_x_across=5)
    # task = Task(dataset, on="#phone", by=["next-phone", "prev-phone"], across=["speaker"], subsampler=subsampler,)
    # print(len(task))
    # print(task[0])
    # score = Score(task, "angular")
    # abx_error_rate = score.collapse(levels=[("prev-phone", "next-phone"), "speaker"])
    # print(abx_error_rate)

if __name__ == "__main__":
    main()




def maker(path: str) -> torch.Tensor:
    features = torch.load(path, weights_only=True)
    # assert sr == bundle.sample_rate
    # features, _ = model.extract_features(x.to(device))
    return features 


