# SN-Graph: a Minimalist 3D Object Representation for Classification

This repository contains the implementation of the paper:

Siyu Zhang, Hui Cao, Yuqi Liu, Shen Cai∗, Yanting Zhang, Yuanzhan Li and Xiaoyu Chi. SN-Graph: a Minimalist 3D Object Representation for Classification. IEEE International Conference on Multimedia and Expo (ICME) 2021, oral presentation.

## Abstract
Using deep learning techniques to process 3D objects has achieved many successes. However, few methods focus on the representation of 3D objects, which could be more effective for specific tasks than traditional representations, such as point clouds, voxels, and multi-view images. In this paper, we propose a Sphere Node Graph (SN-Graph) to represent 3D objects. Specifically, we extract a certain number of internal spheres (as nodes) from the signed distance field (SDF), and then establish connections (as edges) among the sphere nodes to construct a graph, which is seamlessly suitable for 3D analysis using graph neural network (GNN). Experiments conducted on the ModelNet40 dataset show that when there are fewer nodes in the graph or the tested objects are rotated arbitrarily, the classification accuracy of SN-Graph is significantly higher than the state-of-the-art methods.

## Sphere Generation And Connection
In order to represent 3D objects by skeleton-like graphs, each node of the graph that also refers to an internal sphere should be selected from voxels.
We use furthest sphere sampling with special distance difinetion to select nodes.
The distance between two nodes satisfies the following formula:

![Distance Formula](/images/distance.jpg)

The following image display the comparison of different concise representations with the resolution of 32.

![Different Sampling Method](/images/32resolution.jpg)

Inspired by the idea of human joint connection, we propose to construct SN-Graph according to four node connection rules.
1. The edge between two nodes should be close enough to the object.
2. The edge between two sphere nodes should not intersect another selected sphere.
3. The maximum number of connections from each node to other nodes is limited to $q$
4. For an isolated sphere node, it needs to be connected to another nearest node.


This image shows the process of generate a SN-Graph

![process](/images/generate-process.gif)
![process](/images/connect-process.gif)

This image shows the SNG of airplane with different nodes.

![Airplane SNG with 8, 16, 32, 64, 128, 256 Nodes](/images/sng.jpg)


And the following images shows the SNG of different objects.

![SNG 1](/images/sngs-part1.jpg)

![SNG 2](/images/sngs-part2.jpg)

![SNG 3](/images/sngs-part3.jpg)

## Networks
Similar to most graph classification networks, our network follows the design of `graph convolution => readout => classifier`. 

We first use an MLP/EdgeConv layer to increase the input feature dimension $k$ to $64$, and then aggregate the features through a 4-layer graph convolution or attention operation.
Then, the concatenation of the global max feature (with dimension of 256) and global mean features (with dimension of 256) of each GraphConv layer is fed to a 3-layer FC network to obtain the classification score.

![Network Architecture](/images/networks.jpg)

## Experiment
![](/images/table1.jpg)

![](/images/fig6.jpg)

![](/images/table2.jpg)

## Usage
### SN-Graph Generator
The usage of this part is to generate SN-Graph from voxel files.

### SN-Graph Network
The code has been tested on pytorch 1.7.1 + PyG 1.6.3 + CUDA 11.0

### Paper file
[Paper PDF](https://arxiv.org/pdf/2105.14784.pdf)

## License

This project is licensed under the terms of the MIT license (see `LICENSE` for details).
