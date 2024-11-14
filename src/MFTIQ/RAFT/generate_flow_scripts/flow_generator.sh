#!/usr/bin/env bash

source /datagrid/personal/neoral/app/pytorch131python37cuda10/bin/activate;

cd /datagrid/personal/neoral/repos/raft_debug;

#python3 generate_flow_scripts/kitti_generator.py \
#--model=models/kitti.pth;
#
#python3 generate_flow_scripts/sintel_generator.py \
#--model=models/sintel.pth;

#python3 generate_flow_scripts/tomi_generator.py;
python3 generate_flow_scripts/flow_for_stepan.py;
