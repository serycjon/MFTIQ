
# MFTIQ: Multi-Flow Tracker with Independent Matching Quality Estimation

[Project Page](https://cmp.felk.cvut.cz/~serycjon/MFTIQ/)
Official implementation of the MFTIQ tracker from the paper:

[Jonáš Šerých](https://cmp.felk.cvut.cz/~serycjon/), [Michal Neoral](https://scholar.google.com/citations?user=fK9nkmQAAAAJ&hl=en&oi=ao) [Jiří Matas](https://cmp.felk.cvut.cz/~matas/): "**MFTIQ: Multi-Flow Tracker with Independent Matching Quality Estimation**", accepted to WACV 2025

Please cite our paper, if you use any of this.

    @inproceedings{serych2025mftiq,
                   title={{MFTIQ}: Multi-Flow Tracker with Independent Matching Quality Estimation},
                   author={Serych, Jonas and Neoral, Michal and Matas, Jiri},
                   journal={arXiv preprint arXiv:TBD},
                   year={2024},
    }


## Install

Create and activate a new virtualenv:

    # we have tested with python 3.11.3
    python -m venv venv
    source venv/bin/activate

Then install the package and all its dependencies.
It must be done in two steps due to spatial-correlation-sampler requiring torch during installation:

    pip install .
	pip install .[full]
	# depending on your shell, it may be something like
	# pip install '.[full]'

We did this with the following versions:
	
	module load cuDNN/8.4.1.50-CUDA-11.7.0
    module load CUDA/11.7.0
    module load Python/3.11.3-GCCcore-12.3.0
    module load GCCcore/11.3.0 # for compilation of the spatial-correlation-sampler

## Run the demo
Download the trained model:

	bash download_model.sh

Then simply running:

    python demo.py

should produce a `demo_out` directory with two visualizations.

See available options like this:

	python demo.py --help
	
and feel free to run it on your own videos. If you don't want to create your own video edit template, run:

	python demo.py --video demo_in/camel/ --edit checkerboard --gpu 0

You can replace the `demo_in/camel/` with a path to your video file, or a directory with video frames.


## Run eval report
To run evaluation on the TAP-Vid dataset install few more dependencies with:

	pip install .[full,extra-eval]
	
Symlink / copy the evaluation datasets into the `datasets/` directory.
The `tapvid_davis.pkl` can be downloaded from [here](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid),
The `tapvid_kinetics` directory should contain a set of `.pkl` files, also downloaded [here](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid).
The `robotap` directory should contain robotap split `.pkl`s downloaded from [here](https://github.com/google-deepmind/tapnet?tab=readme-ov-file#roboTAP).

Then, run following script (potentially changing the first argument for different dataset_configs. Consider running with `--mode first`.):

    python run_eval_report.py dataset_configs/pkl-tapvid-davis-256x256_512x512.py --gpu 1 --export /path/to/results/ --cache /path/to/cache/ configs/MFTIQ4_ROMA_200k_cfg.py
	
## Training
See [here](src/MFTIQ/UOM/readme.md).

## License

The camel demo video is a preview of the ["camel" DAVIS16 sequence](https://davischallenge.org/davis2016/one_result.html?seq_id=camel).
The lioness demo video in `demo_in` was extracted from [youtube](https://www.youtube.com/watch?v=ugsJtsO9w1A).

This work is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

The `src/MFTIQ` directory contains subdirectories with copies (with tiny modifications to plug them into our codebase) of various optical flow and wide-baseline matching methods: [DKM - MIT license](https://github.com/Parskatt/DKM), [FlowFormer++ - Apache license](https://github.com/XiaoyuShi97/FlowFormerPlusPlus), [MemFlow, Apache license](https://github.com/DQiaole/MemFlow), [NeuFlow - Apache license](https://github.com/neufieldrobotics/NeuFlow), [NeuFlow v2 - Apache license](https://github.com/neufieldrobotics/NeuFlow_v2), [RoMa - MIT license](https://github.com/Parskatt/RoMa) - check the LICENSE files in the appropriate directories.
The `src/MFTIQ/RAFT` directory contains a modified version of [RAFT](https://github.com/princeton-vl/RAFT), which is licensed under BSD-3-Clause license.
The modifications from the [MFT tracker](https://cmp.felk.cvut.cz/~serycjon/MFT) (`OcclusionAndUncertaintyBlock` and its integration in `raft.py`) are licensed again under the [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

This work was supported by Toyota Motor Europe,
by the Grant Agency of the Czech Technical University in Prague, grant No. `SGS23/173/OHK3/3T/13`, and
by the Research Center for Informatics project `CZ.02.1.01/0.0/0.0/16_019/0000765` funded by OP VVV.

