# SN-Graph Generator Usage

## Installation
Before installation, make sure [edt](https://github.com/seung-lab/euclidean-distance-transform-3d) and [tinyply](https://github.com/ddiakopoulos/tinyply) has been download in this folder. `tinyply` is used for parse voxel file in `./test_data`. 

The test code has been tested in windows10 + MSVC(C++17)

## Usage
`main.cpp` include a sample usage of the generator. You can read your own file by replace the function `read_ply` in SNG.cpp.

## Test data
All models are selected from ModelNet and have been voxelized. 