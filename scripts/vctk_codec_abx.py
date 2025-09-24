import torch
# from fastabx import zerospeech_abx
from fastabx import Dataset, Subsampler, Task
from fastabx import Score
from fastabx import zerospeech_abx
import sys, os
import argparse
from pathlib import Path
fastabx_root = str(Path(__file__).resolve().parent.parent)
mod_flag = False
if fastabx_root.split("/")[-1] == "fastabx":
    sys.path.append(fastabx_root)
    from utils.dataset_mod import DatasetMod
    mod_flag = True


def parse_args():
    """
    Parses command-line arguments for the file reorganization script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('features_root',type=str)
    parser.add_argument('item_file',type=str)
    parser.add_argument('layer',type=int)
    parser.add_argument('features_path',type=str)
    parser.add_argument('codebook_path',type=str)
    parser.add_argument('--abx_mode',type=str, default="phone_abx")
    parser.add_argument('--frame_mean', action='store_true')

    args = parser.parse_args()
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_percent = 0.1

def maker(path: str, codebook: torch.Tensor) -> torch.Tensor:
    data = torch.load(path, weights_only=True)
    return codebook[data]

if __name__ == "__main__":
    args = parse_args()
    features_root = args.features_root # /home/s2522559/datastore
    item = os.path.join(features_root, args.item_file)
    layer = args.layer
    frequency = 50
    features = os.path.join(features_root, args.features_path) 
    # /home/s2522559/datastore/RepCodec_codecs/repcodec_hubert_large_l18_ls100/VCTK_tokens_reorg 
    # print("features path", features, "frame_mean", args.frame_mean)
    codebook = torch.load(os.path.join(args.features_root, args.codebook_path), weights_only=True) # RepCodec_codecs/repcodec_hubert_large_l18_ls100/repcodec_hubert_large_l18_ls100_codebook.pt"

    if args.abx_mode == "phone_abx":
        dataset = Dataset.from_item(item, features, frequency, feature_maker=lambda x: maker(x, codebook)) # feature_maker
        subsampler = Subsampler(max_size_group=10, max_x_across=5)
        task = Task(dataset, on="#phone", by=["next-phone", "prev-phone"], across=["speaker"], subsampler=subsampler,)
        # print(len(task))
        # print(task[0])
        score = Score(task, "angular", frame_mean=args.frame_mean)
        abx_error_rate = score.collapse(levels=[("prev-phone", "next-phone"), "speaker"])
        print("phone_abx score", abx_error_rate)
    
    elif args.abx_mode == "zerospeech_abx":
        abx = zerospeech_abx(
            item,
            features,
            max_size_group=10,
            max_x_across=5,
            feature_maker=lambda x: maker(x, codebook),
            extension=".pt",
        )
        print("zerospeech_abx score", abx)

    elif args.abx_mode == "accent_abx":
        dataset = Dataset.from_item(item, features, frequency, feature_maker=lambda x: maker(x, codebook)) # feature_maker
        subsampler = Subsampler(max_size_group=10, max_x_across=5)
        task = Task(dataset, on="accent", by=["next-phone", "#phone", "prev-phone"], subsampler=subsampler,) # across=["speaker"]
        # print(len(task))
        # print(task[0])
        score = Score(task, "angular", frame_mean=args.frame_mean)
        abx_error_rate = score.collapse(levels=[("next-phone", "#phone", "prev-phone"),])
        print("accent_abx score", abx_error_rate)

    elif args.abx_mode == "accent_word":
        if mod_flag:
            dataset = DatasetMod.from_item_limitlen(item, features, frequency, feature_maker=lambda x: maker(x, codebook), len_lim=100) # feature_maker
        else:
            dataset = Dataset.from_item(item, features, frequency, feature_maker=lambda x: maker(x, codebook))
            print("warning: wrong dataset implementation with schema")
        task = Task(dataset, on="accent", by=["#phone"]) # across=["speaker"]
        print(len(task))
        print([task[i] for i in range(len(task) - 10, len(task))])
        score = Score(task, "angular", frame_mean=args.frame_mean)
        print(type(score))
        abx_error_rate = score.collapse(levels=[("#phone"),])
        print("accent_word score", abx_error_rate)

    else:
        raise NotImplemented


def maker(path: str) -> torch.Tensor:
    features = torch.load(path, weights_only=True)
    # assert sr == bundle.sample_rate
    # features, _ = model.extract_features(x.to(device))
    return features 


