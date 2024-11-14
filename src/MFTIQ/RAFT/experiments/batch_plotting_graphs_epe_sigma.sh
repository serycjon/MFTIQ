#!/usr/bin/env bash

GPU=$1
BATCH_RUN=$2

function create_graphs_sintel_validation_subsplit ()
{
    GPU=$1
    MODEL=$2
    SAVE_PATH=$3
    OCCLUSION_MODULE=$4

    python3 experiments/plotting_epe_sigma_occl_statistics.py --gpus $GPU --save_path $SAVE_PATH --model $MODEL --subsplit validation --occlusion_module $OCCLUSION_MODULE &
}

function create_graphs_sintel_kubric_validation_subsplit_all ()
{
    GPU=$1
    MODEL=$2
    SAVE_PATH=$3
    OCCLUSION_MODULE=$4

    python3 experiments/plotting_epe_sigma_occl_statistics_all.py --gpus $GPU --save_path $SAVE_PATH --model $MODEL --subsplit validation --occlusion_module $OCCLUSION_MODULE
}

export MPLBACKEND=agg

if [[ $BATCH_RUN -eq 1 ]]
then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs'

#     MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/50000_raft-things-sintel-splitted-occlusion-uncertainty-base-sintel.pth'
#     create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-l2loss-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

elif [[ $BATCH_RUN -eq 2 ]]
then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-reweighting-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-reweighting-l2-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-base-sintel_bad_loss.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'
elif [[ $BATCH_RUN -eq 3 ]]
then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-l2loss-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-reweighting-l2-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'
elif [[ $BATCH_RUN -eq 4 ]]
then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-huber-epe-direct-non-occluded-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-huber-epe-direct-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'
elif [[ $BATCH_RUN -eq 49 ]]
then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-l2loss-non-occluded-base-sintel.pth'
    create_graphs_sintel_validation_subsplit $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'


elif [[ $BATCH_RUN -eq 5 ]]
then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs_all'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth'
    create_graphs_sintel_kubric_validation_subsplit_all $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-base-sintel.pth'
    create_graphs_sintel_kubric_validation_subsplit_all $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

# elif [[ $BATCH_RUN -eq 6 ]]
# then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs_all'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-l2loss-base-sintel.pth'
    create_graphs_sintel_kubric_validation_subsplit_all $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-l2loss-non-occluded-base-sintel.pth'
    create_graphs_sintel_kubric_validation_subsplit_all $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

# elif [[ $BATCH_RUN -eq 7 ]]
# then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs_all'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-kubric-splitted-fullflowou-base-sintel.pth'
    create_graphs_sintel_kubric_validation_subsplit_all $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-kubric-splitted-occlusion-uncertainty-base-sintel.pth'
    create_graphs_sintel_kubric_validation_subsplit_all $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'
#
# elif [[ $BATCH_RUN -eq 8 ]]
# then
    SAVE_PATH='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/base-sintel/statistics_graphs_all'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth'
    create_graphs_sintel_kubric_validation_subsplit_all $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'

    MODEL='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-splitted-occlusion-uncertainty-base-sintel_bad_loss.pth'
    create_graphs_sintel_kubric_validation_subsplit_all $GPU $MODEL $SAVE_PATH 'separate_with_uncertainty'
fi
