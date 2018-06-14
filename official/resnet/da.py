from __future__ import print_function

"""

	Data Augmentation via Spatial Transformers Networks 
	(https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf) 
	use warping module to apply hand coded affine transportations
	
	reference kernels https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg

	hflip: flip tensor horizontally 

"""

import tensorflow as tf 
import transformer

def augment(tensor,theta):
	## requires theta: affine transform tensor of shape (B, 6)
	idx = 0
	theta = tf.constant([theta for _ in range(tensor.get_shape()[idx])],tf.float32)
	return transformer.spatial_transformer_network(tensor, theta)

def hflip(tensor):
	return augment(tensor,[[-1,0,0],[0,1,0]])

def vflip(tensor):
	return augment(tensor,[[1,0,0],[0,-1,0]])