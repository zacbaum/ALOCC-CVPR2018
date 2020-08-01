import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import pp, visualize, to_json, show_all_variables
from models import ALOCC_Model
import matplotlib.pyplot as plt
from kh_tools import *
import numpy as np
import scipy.misc
from utils import *
import time
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer(
    "attention_label",
    1,
    "Conditioned label that growth attention of training label [1]",
)
flags.DEFINE_float("r_alpha", 0.4, "Refinement parameter [0.4]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 45, "The size of image to use. [45]")
flags.DEFINE_integer(
    "input_width",
    None,
    "The size of image to use. If None, same value as input_height [None]",
)
flags.DEFINE_integer(
    "output_height", 45, "The size of the output images to produce [45]"
)
flags.DEFINE_integer(
    "output_width",
    None,
    "The size of the output images to produce. If None, same value as output_height [None]",
)
flags.DEFINE_string("dataset", "UCSD", "The name of dataset [UCSD, mnist]")
flags.DEFINE_string(
    "dataset_address",
    "./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test",
    "The path of dataset",
)
flags.DEFINE_string(
    "input_fname_pattern", "*", "Glob pattern of filename of input images [*]"
)
flags.DEFINE_string(
    "checkpoint_dir",
    "./checkpoint/UCSD_128_45_45/",
    "Directory name to save the checkpoints [checkpoint]",
)
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string(
    "sample_dir", "samples", "Directory name to save the image samples [samples]"
)
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("use_ckpt", -1, "CHeckpoint number to use, -1 uses most recent [-1]")
FLAGS = flags.FLAGS


def check_some_assertions():
    """
    to check some assertions in inputs and also check sth else.
    """
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)


def main(_):
    print("Program is started at", time.clock())

    n_per_itr_print_results = 100
    kb_work_on_patch = False
    nd_input_frame_size = (180, 270)
    nd_slice_size = (180, 270)
    n_stride = 1

    FLAGS.input_width = nd_slice_size[0]
    FLAGS.input_height = nd_slice_size[1]
    FLAGS.output_width = nd_slice_size[0]
    FLAGS.output_height = nd_slice_size[1]

    FLAGS.dataset = "data-alocc"
    FLAGS.dataset_address = "./dataset/data-alocc/test/in"
    FLAGS.checkpoint_dir = "./checkpoint/" + "{}_{}_{}_{}_{}".format(
        FLAGS.dataset,
        FLAGS.batch_size,
        FLAGS.output_height,
        FLAGS.output_width,
        FLAGS.r_alpha
    )
    FLAGS.sample_dir = os.path.join("./samples/in", (str(FLAGS.use_ckpt) + "_" + str(FLAGS.r_alpha)))

    check_some_assertions()

    nd_patch_size = (FLAGS.input_width, FLAGS.input_height)
    nd_patch_step = (n_stride, n_stride)

    FLAGS.nStride = n_stride
    # FLAGS.input_fname_pattern = '*'
    FLAGS.train = False
    FLAGS.epoch = 1

    pp.pprint(flags.FLAGS.__flags)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        tmp_ALOCC_model = ALOCC_Model(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            attention_label=FLAGS.attention_label,
            r_alpha=FLAGS.r_alpha,
            is_training=FLAGS.train,
            dataset_name=FLAGS.dataset,
            dataset_address=FLAGS.dataset_address,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            nd_patch_size=nd_patch_size,
            n_stride=n_stride,
            n_per_itr_print_results=n_per_itr_print_results,
            kb_work_on_patch=kb_work_on_patch,
            nd_input_frame_size=nd_input_frame_size,
        )

        # show_all_variables()

        print("--------------------------------------------------")
        print("Load Pretrained Model...")
        tmp_ALOCC_model.f_check_checkpoint(checkpoint_number=FLAGS.use_ckpt)

        if FLAGS.dataset == "data-alocc":
            lst_image_paths = [
                x
                for x in glob(
                    os.path.join(FLAGS.dataset_address, FLAGS.input_fname_pattern)
                )
            ]
        t = time.time()
        images = read_lst_images(lst_image_paths, None, None, b_work_on_patch=False)
        t = time.time() - t
        print(" [*] Loaded Data in {:3f}s".format(t))

        tmp_ALOCC_model.f_test_frozen_model(images, lst_image_paths)


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    tf.app.run()
