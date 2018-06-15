from __future__ import print_function

"""

    Data Augmentation via Spatial Transformers Networks 
    (https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf) 
    use warping module to apply hand coded affine transportations
    
    reference kernels flipping https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg
    reference rotating https://stackoverflow.com/questions/2259678/easiest-way-to-rotate-by-90-degrees-an-image-using-opencv#5912847

    hflip: flip tensor horizontally 

"""

import tensorflow as tf 
import transformer

ALL_AUGMENTATIONS = ['none','hflip','vflip','hvflip','rot90ccw','rot90cw']
#ALL_AUGMENTATIONS = ['none','hflip','vflip','hvflip']

def data_augmented(input,op,direction='forward',data_format='channels_last'):
    ## flips
    if op == 'hflip':
        return hflip(input,data_format)
    if op == 'vflip':
        return vflip(input,data_format)     
    if op == 'hvflip':
        return vflip(hflip(input,data_format),data_format)

    ## rotations
    if op == 'rot90ccw':
        if direction == 'forward':
            return rot90ccw(input,data_format)
        else:
            return rot90cw(input,data_format)
    if op == 'rot90cw':
        if direction == 'forward':
            return rot90cw(input,data_format)
        else:
            return rot90ccw(input,data_format)
    return input   

def rot90ccw(input,data_format):
    # rotate counter-clockwise
    x = tf.transpose(input,perm=[0,1,3,2])
    x = hflip(x,data_format)
    return x

def rot90cw(input,data_format):
    # rotate clockwise
    if data_format == 'channels_last':
        x = tf.transpose(input,perm=[0,2,1,3])
    else:
        x = tf.transpose(input,perm=[0,1,3,2])
    x = vflip(x,data_format)
    return x

def augment(tensor,theta,n=None,data_format='channels_last'):
    ## requires theta: affine transform tensor of shape (B, 6)
    if data_format == 'channels_last':
        idx = 0
    else:
        idx = 0
    bs = 32 * 4
    #print(n,'shape',tensor.get_shape())
    #if n is None:
    #    n = tensor.get_shape()[idx]
    n = bs 
    theta = tf.constant([theta for _ in range(n)],tf.float32)
    #if n is None:
    #    n = int(tensor.get_shape()[idx])
    #theta = tf.constant([theta for _ in range(n)],tf.float32)
    #print(theta)
    return transformer.spatial_transformer_network(tensor, theta)

def hflip(tensor,n=None,data_format='channels_last'):
    return augment(tensor,[-1,0,0,0,1,0],n,data_format)

def vflip(tensor,n=None,data_format='channels_last'):
    return augment(tensor,[1,0,0,0,-1,0],n,data_format)




def test():
    import numpy as np 
    import cv2

    B,H,W,F = 8,10,20,10
    fn = '/home/dolokov/mask.png'
    x = cv2.imread(fn)
    x = x.reshape( [1]+list(x.shape) )
    B,H,W,F = x.shape
    #x = np.random.randint(low=0, high=100, size=(B,H,W,F),dtype=np.int32).astype(np.float32)
    
    x = x.astype(np.float32)/255.

    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32,shape=[B,H,W,F])

        print(x.shape,x.dtype,x.min(),x.max(),x.sum())

        hflipped = hflip(inputs,B)
        backhflipped = hflip(hflipped,B)

        vflipped = vflip(inputs,B) 
        backvflipped = vflip(vflipped,B)

        sess.run(tf.global_variables_initializer())

        val_hflipped, val_vflipped = sess.run([hflipped,vflipped],feed_dict={inputs:x})

        val_bhflipped, val_bvflipped = sess.run([backhflipped,backvflipped],feed_dict={inputs:x})

    im_hflipped = np.around(255. * (val_hflipped-val_hflipped.min())/(val_hflipped.max()-val_hflipped.min())).astype(np.uint8).reshape([H,W,F])
    im_vflipped = np.around(255. * (val_vflipped-val_vflipped.min())/(val_vflipped.max()-val_vflipped.min())).astype(np.uint8).reshape([H,W,F])
    im_bhflipped = np.around(255. * (val_bhflipped-val_bhflipped.min())/(val_bhflipped.max()-val_bhflipped.min())).astype(np.uint8).reshape([H,W,F])
    im_bvflipped = np.around(255. * (val_bvflipped-val_bvflipped.min())/(val_bvflipped.max()-val_bvflipped.min())).astype(np.uint8).reshape([H,W,F])
        

    print('hfl',im_hflipped.shape,im_hflipped.dtype,im_hflipped.min(),im_hflipped.max())

    cv2.imwrite('/tmp/hflipped.jpg',im_hflipped)
    cv2.imwrite('/tmp/vflipped.jpg',im_vflipped)
    cv2.imwrite('/tmp/bhflipped.jpg',im_bhflipped)
    cv2.imwrite('/tmp/bvflipped.jpg',im_bvflipped)

    print('hflip',np.linalg.norm(x - val_bhflipped))
    print('vflip',np.linalg.norm(x - val_bvflipped))

    assert np.linalg.norm(x - val_bhflipped) < 1e-4
    assert np.linalg.norm(x - val_bvflipped) < 1e-4
        

if __name__ == '__main__':
    test()
