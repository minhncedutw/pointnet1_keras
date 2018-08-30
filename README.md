# pointnet1_keras

## Run steps of Normal version:
1. Download and unzip the [Shapenet dataset](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip) to `./pointnet1_keras/DATA` directory
1. Run `./pointnet1_keras/DATA/Seg_dataprep.py` to convert data to `*.h5` type.
1. Run `./pointnet1_keras/train_seg.py` to run training PointNet
 ### Notice:
 - this PointNet version is able to save model after training
 ### This normal version is referenced from repository: 
 `https://github.com/garyli1019/pointnet-keras`
 
 ## Run steps of Dynamic data loading version: 
 1. Download and unzip the [Shapenet dataset](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip) to `./pointnet1_keras/DATA` directory
 1. Run `./pointnet1_keras/train_seg_2.py` to run training PointNet
 ### Notice: 
 - this version, data is loaded when training so the train speed is very slow compared to Normal version. But this version can flexiblely change size of data and size of network.
