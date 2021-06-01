# SN-Graph: a Minimalist 3D Object Representation for Classification

## Introduction
Using deep learning techniques to process 3D objects has achieved many successes. However, few methods focus on the representation of 3D objects, which could be more effective for specific tasks than traditional representations, such as point clouds, voxels, and multi-view images. In this paper, we propose a Sphere Node Graph (SN-Graph) to represent 3D objects. Specifically, we extract a certain number of internal spheres (as nodes) from the signed distance field (SDF), and then establish connections (as edges) among the sphere nodes to construct a graph, which is seamlessly suitable for 3D analysis using graph neural network (GNN). Experiments conducted on the ModelNet40 dataset show that when there are fewer nodes in the graph or the tested objects are rotated arbitrarily, the classification accuracy of SN-Graph is significantly higher than the state-of-the-art methods.

## Usage
### SN-Graph Generator
The usage of this part is to generate SN-Graph from voxel files.

### SN-Graph Network
The code has been tested on pytorch 1.7.1 + PyG 1.6.3 + CUDA 11.0

## License

This project is licensed under the terms of the MIT license (see `LICENSE` for details).