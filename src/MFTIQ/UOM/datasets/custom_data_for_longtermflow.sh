#!/bin/bash

# ~/bin/parallel -j 10 'custom_data_for_longtermflow.sh {}' ::: $(seq 1000)

START_SEED_LOOP=1
END_SEED_LOOP=-1
TYPE=1
root_savedir="~/kubric/"

while getopts s:e:t:r: flag;
do
    case "${flag}" in
        s) START_SEED_LOOP=${OPTARG};;
        e) END_SEED_LOOP=${OPTARG};;
        t) TYPE=${OPTARG};;
        r) root_savedir=${OPTARG};;
    esac
done

if test "$END_SEED_LOOP" -eq "-1"; then
    END_SEED_LOOP=$START_SEED_LOOP
fi

echo $root_savedir
echo "START_SEED_LOOP: $START_SEED_LOOP";
echo "STOP: $END_SEED_LOOP";
echo "TYPE: $TYPE";

STATIC_OBJ_MIN=10
STATIC_OBJ_MAX=20

DYNAMIC_OBJ_MIN=3
DYNAMIC_OBJ_MAX=10


MAX_CAMERA_MOVEMENT=12.0
MIN_CAMERA_MOVEMENT=8.0
FRAME_RATE=12
N_FRAMES=24
RESOLUTION="256x256"
CAMERA_TYPE="linear_movement"
GENERAL_SCENARIO="forward"
CYCLE_FRAMES_OBJECTS=24
CAMERA_STEPS=2

STATIC_SPAWN_MIN=( -7 -7 0 )
STATIC_SPAWN_MAX=( 7 7 10 )

if [[ $TYPE -eq 1 ]]
then
    RESOLUTION="256x256"
    FRAME_RATE=12
    N_FRAMES=24
elif [[ $TYPE -eq 2 ]]
then
    RESOLUTION="1024x1024"
    FRAME_RATE=120
    N_FRAMES=240
elif [[ $TYPE -eq 3 ]]
then
    RESOLUTION="256x256"
    FRAME_RATE=24
    N_FRAMES=240
    CYCLE_FRAMES_OBJECTS=48
    CAMERA_STEPS=5
    GENERAL_SCENARIO="forward_backward_cycle"
    CAMERA_TYPE="linear_movement"
elif [[ $TYPE -eq 4 ]]
then
    RESOLUTION="1024x1024"
    FRAME_RATE=24
    N_FRAMES=240
    CYCLE_FRAMES_OBJECTS=48
    CAMERA_STEPS=5
    GENERAL_SCENARIO="forward_backward_cycle"
    CAMERA_TYPE="linear_movement"
elif [[ $TYPE -eq 5 ]]
then
    RESOLUTION="1024x1024"
    FRAME_RATE=24
    N_FRAMES=240
    GENERAL_SCENARIO="forward"
    CAMERA_TYPE="linear_movement_linear_lookat"
elif [[ $TYPE -eq 6 ]]
then
    RESOLUTION="256x256"
    FRAME_RATE=24
    N_FRAMES=240
    CYCLE_FRAMES_OBJECTS=48
    CAMERA_STEPS=5
    GENERAL_SCENARIO="forward_backward_cycle"
    CAMERA_TYPE="linear_movement_linear_lookat"
elif [[ $TYPE -eq 7 ]]
then
    RESOLUTION="256x256"
    FRAME_RATE=24
    N_FRAMES=48
#     CYCLE_FRAMES_OBJECTS=48
#     CAMERA_STEPS=5
    GENERAL_SCENARIO="forward"
    CAMERA_TYPE="carlike_frontback_movement"
elif [[ $TYPE -eq 8 ]]
then
    RESOLUTION="256x256"
    FRAME_RATE=120
    N_FRAMES=240
    CYCLE_FRAMES_OBJECTS=48
    CAMERA_STEPS=5
    GENERAL_SCENARIO="forward_backward_cycle"
    CAMERA_TYPE="linear_movement_linear_lookat"
elif [[ $TYPE -eq 9 ]]
then
    RESOLUTION="256x256"
    FRAME_RATE=24
    N_FRAMES=48
#     CYCLE_FRAMES_OBJECTS=48
#     CAMERA_STEPS=5
    GENERAL_SCENARIO="forward_test1"
    CAMERA_TYPE="carlike_frontback_movement"
    STATIC_SPAWN_MIN=( -14 -14 0 )
    STATIC_SPAWN_MAX=( 14 14 10 )
elif [[ $TYPE -eq 10 ]]
then
    RESOLUTION="256x256"
    FRAME_RATE=120
    N_FRAMES=240
    CYCLE_FRAMES_OBJECTS=48
    CAMERA_STEPS=5
    GENERAL_SCENARIO="forward_backward_cycle__camshake"
    CAMERA_TYPE="linear_movement_linear_lookat"
elif [[ $TYPE -eq 11 ]]
then
    RESOLUTION="128x128"
    FRAME_RATE=24
    N_FRAMES=48
    CYCLE_FRAMES_OBJECTS=12
    CAMERA_STEPS=3
    GENERAL_SCENARIO="forward_backward_cycle__camshake"
    CAMERA_TYPE="linear_movement_linear_lookat"
elif [[ $TYPE -eq 12 ]]
then
    RESOLUTION="1024x1024"
    FRAME_RATE=120
    N_FRAMES=240
    CYCLE_FRAMES_OBJECTS=48
    CAMERA_STEPS=3
    GENERAL_SCENARIO="forward_backward_cycle__camshake"
    CAMERA_TYPE="linear_movement_linear_lookat"
fi


for (( c=$START_SEED_LOOP; c<=$END_SEED_LOOP; c++ ))
do
    foo=$(printf "%05d" $c)
    TYPENUM=$(printf "%03d" $TYPE)
    savedir=/data/RES_${RESOLUTION}/${foo}/FPS_${FRAME_RATE}__NFRAMES_${N_FRAMES}__CAM_${CAMERA_TYPE}__GE_${GENERAL_SCENARIO}__TYPE_${TYPENUM}/

    echo ROOTSAVEDIR: ${root_savedir}
    echo SAVEDIR? ${savedir}

    # original container docker://kubricdockerhub/kubruntu
    nice -n 30 singularity run --no-home --cleanenv --bind "$PWD:/kubric" --bind "${root_savedir}:/data" kubruntu_latest.sif python3 /kubric/custom_worker.py \
    --camera=${CAMERA_TYPE} \
    --min_motion_blur=0.25 \
    --max_motion_blur=1.0 \
    --gravity_z=-9.81 \
    --resolution=${RESOLUTION} \
    --focal_length=35.0 \
    --object_list_name=gso  \
    --general_scenario=${GENERAL_SCENARIO} \
    --camera_steps=${CAMERA_STEPS} \
    --cycle_frames_for_objects=${CYCLE_FRAMES_OBJECTS} \
    --frame_end=${N_FRAMES} --frame_rate=${FRAME_RATE} \
    --max_camera_movement ${MAX_CAMERA_MOVEMENT} \
    --min_camera_movement ${MIN_CAMERA_MOVEMENT} \
    --min_num_static_objects=${STATIC_OBJ_MIN} \
    --max_num_static_objects=${STATIC_OBJ_MAX} \
    --min_num_dynamic_objects=${DYNAMIC_OBJ_MIN} \
    --max_num_dynamic_objects=${DYNAMIC_OBJ_MAX} \
    --static_spawn_min=${STATIC_SPAWN_MIN} \
    --static_spawn_max=${STATIC_SPAWN_MAX} \
    --seed=${c} \
    --job-dir=${savedir} || true
done
