mkdir -p checkpoints
cd checkpoints
wget https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/UOM_bs4_200k.pth
wget https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/flowformerpp-sintel.pth
wget https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth

cd ../src/MFTIQ/NeuFlow/
wget https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/neuflow_sintel.pth

cd ../NeuFlow_v2
wget https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/neuflow_mixed.pth
