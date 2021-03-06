"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import collections
import glob
import random
import math

import tensorflow as tf
import numpy as np

from resnet import Dilated_ResNet_101, Dilated_ResNet_101_SSPS
import pyramid_pooling as pp

IMG_RGB_MEAN = np.array([104.42819731, 111.76448516, 125.9762497], dtype=np.float32)
BATCH_SIZE = 1
DATA_DIRECTORY = '/Users/i859032/images/Red_Bull/512X512/data/train'
# DATA_DIRECTORY = '../../../data/redbull/train'
DATA_LIST_PATH = './dataset/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 3
NUM_STEPS = 100000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './deeplab_resnet_init.ckpt'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
# SNAPSHOT_DIR = '../../../data/redbull_ckpt/20170628'
WEIGHT_DECAY = 0.0005
CROP_SIZE = 512
SCALE_SIZE = 544


Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def load_examples(args):
    if args.data_dir is None or not os.path.exists(args.data_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(args.data_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(args.data_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=args.is_training)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        
        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        # raw_input[:,:width//2,:] -= IMG_MEAN
        img = tf.cast(raw_input[:,:width//2,:], tf.float32) - IMG_RGB_MEAN
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        a_images = tf.concat(axis=2, values=[img_b, img_g, img_r])

        # a_images = preprocess(tf.image.convert_image_dtype(raw_input[:,:width//2,:], dtype=tf.float32))
        b_images = raw_input[:,width//2:,:]
        inputs, targets = [a_images, b_images]


    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if False:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [SCALE_SIZE, SCALE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if SCALE_SIZE > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif SCALE_SIZE < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)


    with tf.name_scope("target_images"):
        targets1 = tf.squeeze(tf.to_int32(targets[:,:,2:]), squeeze_dims=[2])
        targets_one_hot = tf.one_hot(targets1, depth=args.num_classes, axis=-1)
        targets2 = transform(targets_one_hot)
        target_images = tf.argmax(targets2, axis=-1)


    paths_batch, inputs_batch, targets_batch = tf.train.shuffle_batch( 
        [paths, input_images, target_images],
        batch_size=args.batch_size,
        capacity=3000,
        num_threads=2,
        min_after_dequeue=1000)
    
    steps_per_epoch = int(math.ceil(len(input_paths) / args.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    # h, w = map(int, args.input_size.split(','))
    # input_size = (h, w)
    
    tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):

        examples = load_examples(args)
        print("examples count = %d" % examples.count)

        final_input = tf.placeholder_with_default(examples.inputs, shape=(None, CROP_SIZE, CROP_SIZE, 3), name="final_input")
        image_batch, label_batch = [final_input, examples.targets]
    
    # Create network.
    # print('image_batch:', image_batch.shape)
    net = Dilated_ResNet_101({'data': image_batch}, is_training=args.is_training)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    old_vars = tf.global_variables()

    # print("old_vars:", len(old_vars))

    init_op = tf.variables_initializer(old_vars)
    # init = tf.global_variables_initializer()
    sess.run(init_op)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
    
    # Load variables if the checkpoint is provided.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]

    # print("restore_var:", len(restore_var))

    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)


    # print('net.layers[fc1_voc12]:', net.layers['fc1_voc12'].shape)

    with tf.name_scope("psp"):

        # net.layers['res5c_relu'].shape == [none, CROP_SIZE/8, CROP_SIZE/8, 2048]
        # pooled_output.shape == (none, CROP_SIZE/8, CROP_SIZE/8, 4096)    
        pooled_output = pp.pyramid_pooling_module(net.layers['res5c_relu'], original_HW=(CROP_SIZE/8, CROP_SIZE/8))
        # print('pooled_output:', pooled_output.shape)

        # raw_output.shape == (none, CROP_SIZE, CROP_SIZE, 3)
        raw_output = pp.fully_connected(pooled_output, num_classes=args.num_classes , is_training=args.is_training)
        print('raw_output:', raw_output.shape)


    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.


    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    
    resnet_trainable = [v for v in all_trainable if 'fc' not in v.name and 'Conv' not in v.name] # lr * 1.0
    psp_trainable = [v for v in all_trainable if 'fc' in v.name or 'Conv' in v.name]
    # print("=========== psp_trainable =========== \n", psp_trainable)
    psp_w_trainable = [v for v in psp_trainable if 'weights' in v.name] # lr * 10.0
    psp_b_trainable = [v for v in psp_trainable if 'biases' in v.name] # lr * 20.0
    
    # print("=========== resnet_trainable =========== \n", resnet_trainable)
    # exit()
    assert(len(all_trainable) == len(psp_trainable) + len(resnet_trainable))
    assert(len(psp_trainable) == len(psp_w_trainable) + len(psp_b_trainable))

    # conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    # fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    # fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    # fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    # assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    # assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    # raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_batch = tf.expand_dims(label_batch, axis=-1)


    def prepare_label(input_batch, new_size, num_classes):
        """Resize masks and perform one-hot encoding.

        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.
          num_classes: number of classes to predict (including background).
          one_hot: whether perform one-hot encoding.

        Returns:
          Outputs a tensor of shape [batch_size h w 21]
          with last dimension comprised of 0's and 1's only.
        """
        with tf.name_scope('label_encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
            
        return input_batch

    print("label_batch.shape:", label_batch.shape)
    # label_proc = prepare_label(label_batch, (CROP_SIZE/8, CROP_SIZE/8), num_classes=args.num_classes) # [batch_size, h, w]
    label_proc = tf.squeeze(label_batch, squeeze_dims=[3])
    print("label_proc.shape:", label_proc.shape)
    # raw_gt = tf.reshape(label_proc, [-1,])
    # indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    # gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    # prediction = tf.gather(raw_prediction, indices)
    # tf.squeeze(label_proc, squeeze_dims=[2])
    gt = label_proc
    prediction = raw_output

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3, name="final_output")

    def depreprocess(image_BGR, num_images, img_mean):
        _, h, w, c = image_BGR.shape
        if num_images > args.batch_size:
            num_images = args.batch_size
        outputs = []        
        for i in range(num_images):
            input_RGB = image_BGR[i][:,:,::-1]
            input_plus_mean = input_RGB + img_mean
            input_casted = tf.cast(input_plus_mean, tf.uint8)
            outputs.append(input_casted)
        return tf.stack(outputs)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = depreprocess(image_batch, args.save_num_images, IMG_RGB_MEAN)        

    with tf.name_scope("convert_targets"):
        # targets = tf.expand_dims(label_batch, axis=-1)
        converted_targets = (label_batch * (np.round(256 / (args.num_classes-1))-1))
        converted_targets = tf.cast(converted_targets, tf.uint8)

    with tf.name_scope("convert_outputs"):
        converted_outputs = (pred * (np.round(256 / (args.num_classes-1))-1))
        converted_outputs = tf.cast(converted_outputs, tf.uint8)

    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    

    # optimiser = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    # grads = tf.gradients(reduced_loss, all_trainable)
    # train_op = optimiser.apply_gradients(zip(grads, all_trainable))



    opt_resnet = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_psp_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_psp_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    grads = tf.gradients(reduced_loss, resnet_trainable + psp_w_trainable + psp_b_trainable)
    grads_resnet = grads[:len(resnet_trainable)]
    grads_psp_w = grads[len(resnet_trainable) : (len(resnet_trainable) + len(psp_w_trainable))]
    grads_psp_b = grads[(len(resnet_trainable) + len(psp_w_trainable)):]

    train_op_resnet = opt_resnet.apply_gradients(zip(grads_resnet, resnet_trainable))
    train_op_psp_w = opt_psp_w.apply_gradients(zip(grads_psp_w, psp_w_trainable))
    train_op_psp_b = opt_psp_b.apply_gradients(zip(grads_psp_b, psp_b_trainable))
    train_op = tf.group(train_op_resnet, train_op_psp_w, train_op_psp_b)
    
    # summaries
    input_img_summary = tf.summary.image("inputs", converted_inputs)
    target_img_summary = tf.summary.image("targets", converted_targets)
    output_img_summary = tf.summary.image("outputs", converted_outputs)
    lr_summary = tf.summary.scalar("learning_rate", learning_rate)
    loss_summary = tf.summary.scalar("loss", reduced_loss)
    summary_op = tf.summary.merge(
        [lr_summary, loss_summary, input_img_summary, target_img_summary, output_img_summary])
    

    # # Set up tf session and initialize variables. 
    # config = tf.ConfigProto()
    ## config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    
    # # Saver for storing checkpoints of the model.
    # saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
    
    # # Load variables if the checkpoint is provided.
    # if args.restore_from is not None:
    #     loader = tf.train.Saver(var_list=restore_var)
    #     load(loader, sess, args.restore_from)

    # print(restore_var)
    # print(new_var)
    # print(tf.global_variables())

    new_var = [v for v in tf.global_variables() if v not in old_vars]
    # print("new_var:", len(new_var))

    assert(len(tf.global_variables()) == len(old_vars) + len(new_var))
    # init_new_vars_op = tf.initialize_variables(new_var)
    init_new_vars_op = tf.variables_initializer(new_var)
    
    sess.run(init_new_vars_op)


    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        
        if step % args.save_pred_every == 0:
            loss_value, summary, _ = sess.run([reduced_loss, summary_op, train_op], feed_dict=feed_dict)
            # loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, image_batch, label_batch, pred, summary_op, train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % 20 == 0:
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
