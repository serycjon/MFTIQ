# Training of Uncertainty Occlusion Module

The data for training MFTIQ's Uncertainty Occlusion Module (UOM) is generated using the Kubric dataset generation tool with `custom_worker.py` configured for a custom scenario. The final dataset is large (several terabytes). Therefore, we provide only scripts for dataset generation and a sample to illustrate the directory structure.


## Example of Training Data

Download [sample data here](https://drive.google.com/file/d/1n1H5fBcZXcWudi40D04UUX-46ijwtzJi/view?usp=sharing).

The sample data includes two directories:
- **sample_rendered_data**: contains Kubric-generated data (first 50 frames only for size constraints).
- **sample_training_data**: includes paired images with ground truth for training.

The full dataset comprises 200 rendered sequences, each 240 frames long, creating 20,000 training frame pairs with time differences ($\Delta_t$) in the range $\Delta_t \in \{2, 3, ..., 30 \}$ and 10,000 pairs with $\Delta_t \in \{30, 31, ..., 150 \}$.
 

## Set Up Environment 

To install the required packages, run:

    pip install pypng imageio tensorflow tensorflow-datasets rich

### FlowFormer

Optical flow estimates from FlowFormer are precomputed for part of the training data.

    pip install yacs loguru timm

### Download Kubric Docker/Singularity Image

Kubric’s standard rendering pipeline uses Docker. However, our scripts are configured for **Singularity 3.7.0**. To use Docker instead, you will need to modify the `custom_data_for_longtermflow.sh` script.

    singularity pull docker://kubricdockerhub/kubruntu:latest
    singularity build kubruntu-latest.sif kubruntu-latest.simg  # maybe not necessary

### Optional - install GNU parallel

For easier parallelization of dataset generation, we recommend to use [GNU parallel](https://www.gnu.org/software/parallel/).


## Dataset Generation

### KUBRIC rendering - 200 sequences with Kubric tool
```bash
cd src/MFTIQ/UOM/datasets/
export KUBRIC_RENDER='/ABSOLUTE/PATH/RENDER_ROOT_DIR' # Your directory for rendered files
export KUBRIC_TRAIN='/ABSOLUTE/PATH/TRAIN_ROOT_DIR' # Your directory for training files
~/bin/parallel -j 2 './custom_data_for_longtermflow.sh -t 12 -s {} -r ${KUBRIC_RENDER}; pwd;' ::: $(seq 1 200)
```

### Generate training pairs from rendered sequences 

```bash
export KUBRIC_SUBDIR='FPS_120__NFRAMES_240__CAM_linear_movement_linear_lookat__GE_forward_backward_cycle__camshake__TYPE_012' # do not change
python -m MFTIQ.UOM.create_kubric_data --gpuid 0 --samples_start 0 --samples_end 20000 --sequence_length_min 2 --sequence_length_max 30 --step_size 1 --dataroot ${KUBRIC_RENDER}/RES_1024x1024 --subdir ${KUBRIC_SUBDIR} --saveroot ${KUBRIC_TRAIN}/002_030/;
python -m MFTIQ.UOM.create_kubric_data --gpuid 0 --samples_start 0 --samples_end 10000 --sequence_length_min 30 --sequence_length_max 150 --step_size 10 --dataroot ${KUBRIC_RENDER}/RES_1024x1024 --subdir ${KUBRIC_SUBDIR} --saveroot ${KUBRIC_TRAIN}/030_150/;

~/bin/parallel -j 10 'nice python -m MFTIQ.UOM.create_kubric_data --second_stage --saveroot ${KUBRIC_TRAIN}/002_030/ --seed {}; pwd' ::: $(seq 10 20)
~/bin/parallel -j 10 'nice python -m MFTIQ.UOM.create_kubric_data --second_stage --saveroot ${KUBRIC_TRAIN}/030_150/ --seed {}; pwd' ::: $(seq 10 20)
```

## Training

Place or symlink the generated dataset into **datasets/kubric_datasets/**.

```bash
python -m MFTIQ.UOM.train --gpus 0 --config mftiq --batch_size 8
```
Note: The first run will generate cache for training files. It may took several minutes.  
