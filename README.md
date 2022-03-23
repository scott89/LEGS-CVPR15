## Deep Networks for Saliency Detection via Local Estimation and Global Search

### Introduction

LEGS is a state-of-the-art salient object detection system which utilizes deep networks to combine local estimation and global search. This pachage contains the source code to reproduce the experimental results of LEGS reported in the [CVPR 2015 paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2A_104.pdf)

### Citing LEGS

If you find LEGS useful in your research, please consider to cite our paper:

     @inproceedings{ wang2015deep,
        title={Deep Networks for Saliency Detection via Local Estimation and Global Search},
        author={Wang, Lijun and Lu, Huchuan and Ruan, Xiang and Yang, Ming-Hsuan},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        pages={3183--3192},
        year={2015}
     }

### Dependencies

* Deep learning framework Caffe [R1] and its dependencies. We have modified Caffe slightly based on [R2] for efficiency.
* Geodesic Object Proposal [R3] and its dependencies.
* Cuda enabled GPUs.

### How to Run

* First make sure that all the software dependencies have been installed following their guidelines. 
* Then run test in Matlab to generate saliency maps in the "maps" folder.
* The current implementation has been tested on MATLAB R2013b on a Linux 64-bit machine with a TITAN Black GPU. It runs at about 2 s/f.


### Note

* If your MATLAB version is higher than R2012b, running linear algebra functions could returen "cannot load any more object with static TLS" error. Please follow the following [instructions](http://www.mathworks.com/support/bugreports/961964) to fix this MATLAB bug. 

### Contact

wlj@mail.dlut.edu.cn

### Reference

[R1] Jia Y, Shelhamer E, Donahue J, et al. Caffe: Convolutional architecture for fast feature embedding, ACM International Conference on Multimedia, 2014.

[R2] Li H, Zhao R, Wang X. Highly efficient forward and backward propagation of convolutional neural networks for pixelwise classification. arXiv preprint, 2014.

[R3] Krähenbühl P, Koltun V. Geodesic object proposa, ECCV 2014.



