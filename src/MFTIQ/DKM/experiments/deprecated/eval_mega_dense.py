from argparse import ArgumentParser
from MFTIQ.DKM.dkm import (
    DKM,
    local_corr,
    corr_channels,
    linear,
    baseline,
)
from MFTIQ.DKM.dkm.benchmarks import (
    MegadepthDenseBenchmark,
)
import json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ablation_target",
        type=str,
        choices=["DKM", "local_corr", "no_regression", "naive_regression", "baseline"],
        default="DKM",
    )
    args, _ = parser.parse_known_args()
    model_name = args.ablation_target
    if model_name == "DKM":
        model = DKM(pretrained=True, version="mega_synthetic")
    elif model_name == "local_corr":
        model = local_corr(pretrained=True, version="mega_synthetic")
    elif model_name == "no_regression":
        model = corr_channels(pretrained=True, version="mega_synthetic")
    elif model_name == "baseline":
        model = baseline(pretrained=True, version="mega_synthetic")
    elif model_name == "naive_regression":
        model = linear(pretrained=True, version="mega_synthetic")

    mega_dense_benchmark = MegadepthDenseBenchmark("data/megadepth")

    mega_dense_results = []
    for s in range(5):
        mega_dense_results.append(mega_dense_benchmark.benchmark(model))
        json.dump(
            mega_dense_results, open(f"results/mega_dense_{model_name}.json", "w")
        )
