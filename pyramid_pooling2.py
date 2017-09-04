from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

import tensorflow as tf
slim = tf.contrib.slim


def pyramid_pooling(inputs, pool_size, depth, original_HW):
    dims = inputs.get_shape().dims
    # out_height, out_width = dims[1].value, dims[2].value
    pool1 = slim.avg_pool2d(inputs, pool_size, stride=pool_size)
    conv1 = slim.conv2d(pool1, depth, [1, 1], stride=1)
    # output = tf.image.resize_bilinear(conv1, [original_HW[0], original_HW[1]])
    output = tf.image.resize_bilinear(conv1, original_HW)

    return output


def pyramid_pooling_module(inputs, original_HW):    
    pyramid1 = pyramid_pooling(inputs, (2, 2), 512, original_HW)
    pyramid2 = pyramid_pooling(inputs, (4, 4), 512, original_HW)
    pyramid3 = pyramid_pooling(inputs, (8, 8), 512, original_HW)
    pyramid4 = pyramid_pooling(inputs, (16, 16), 512, original_HW)
    output = tf.concat(axis=3, values=[inputs, pyramid1, pyramid2, pyramid3, pyramid4])

    return output


def fully_connected(inputs, num_classes, is_training):
    net = slim.conv2d(inputs, 512, [3, 3], stride=1)
    # net = slim.dropout(net, keep_prob=0.9, is_training=is_training)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                      normalizer_fn=None)

    dims = inputs.get_shape().dims
    out_height, out_width = dims[1].value, dims[2].value
    output = tf.image.resize_bilinear(net, [8*out_height, 8*out_width], name="logits")

    return output


# def pspnet_v1(inputs,
#               blocks,
#               levels,
#               num_classes=None,
#               is_training=True,
#               reuse=None,
#               scope=None):

#   with tf.variable_scope(scope, 'pspnet_v1', [inputs], reuse=reuse) as sc:
#     end_points_collection = sc.name + '_end_points'
#     with slim.arg_scope([slim.conv2d, bottleneck, pyramid_pooling,
#                          pspnet_utils.stack_blocks_dense,
#                          pspnet_utils.pyramid_pooling_module],
#                         outputs_collections=end_points_collection):
#       with slim.arg_scope([slim.batch_norm], is_training=is_training):
#         net = inputs

#         net = root_block(net)
#         net = pspnet_utils.stack_blocks_dense(net, blocks, None)

#         net = pspnet_utils.pyramid_pooling_module(net, levels)
#         net = slim.conv2d(net, 512, [3, 3], stride=1, scope='fc1')
#         net = slim.dropout(net, keep_prob=0.9, is_training=is_training)

#         net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
#                           normalizer_fn=None, scope='logits')

#         dims = inputs.get_shape().dims
#         out_height, out_width = dims[1].value, dims[2].value
#         net = tf.image.resize_bilinear(net, [out_height, out_width])

#         # TODO
#         end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#         end_points['predictions'] = slim.softmax(net, scope='predictions')

#         return net, end_points

