from __future__ import division
import re
from ops import *
from utils import *
from kh_tools import *
import logging
import matplotlib.pyplot as plt
import time


class ALOCC_Model(object):
    def __init__(
        self,
        sess,
        input_height=45,
        input_width=45,
        output_height=64,
        output_width=64,
        batch_size=128,
        sample_num=128,
        attention_label=1,
        is_training=True,
        z_dim=100,
        gf_dim=16,
        df_dim=16,
        gfc_dim=512,
        dfc_dim=512,
        c_dim=3,
        dataset_name=None,
        dataset_address=None,
        input_fname_pattern=None,
        checkpoint_dir=None,
        log_dir=None,
        sample_dir=None,
        r_alpha=0.2,
        kb_work_on_patch=True,
        nd_input_frame_size=(240, 360),
        nd_patch_size=(10, 10),
        n_stride=1,
        n_fetch_data=10,
        n_per_itr_print_results=500,
    ):
        """
    This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection
    :param sess: TensorFlow session      
    :param batch_size: The size of batch. Should be specified before training. [128]
    :param attention_label: Conditioned label that growth attention of training label [1]
    :param r_alpha: Refinement parameter [0.2]        
    :param z_dim:  (optional) Dimension of dim for Z. [100] 
    :param gf_dim: (optional) Dimension of gen filters in first conv layer. [64] 
    :param df_dim: (optional) Dimension of discrim filters in first conv layer. [64] 
    :param gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024] 
    :param dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024] 
    :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]  
    :param sample_dir: Directory address which save some samples [.]
    :param kb_work_on_patch: Boolean value for working on PatchBased System or not [True]
    :param nd_input_frame_size:  Input frame size 
    :param nd_patch_size:  Input patch size
    :param n_stride: PatchBased data preprocessing stride
    :param n_fetch_data: Fetch size of Data 
    :param n_per_itr_print_results: # of printed iteration   
    """

        self.n_per_itr_print_results = n_per_itr_print_results
        self.nd_input_frame_size = nd_input_frame_size
        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir

        self.sess = sess
        self.is_training = is_training

        self.r_alpha = r_alpha

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name="d_bn1")
        self.d_bn2 = batch_norm(name="d_bn2")
        self.d_bn3 = batch_norm(name="d_bn3")
        self.d_bn4 = batch_norm(name="d_bn4")
        self.g_bn0 = batch_norm(name="g_bn0")
        self.g_bn1 = batch_norm(name="g_bn1")
        self.g_bn2 = batch_norm(name="g_bn2")
        self.g_bn3 = batch_norm(name="g_bn3")
        self.g_bn4 = batch_norm(name="g_bn4")
        self.g_bn5 = batch_norm(name="g_bn5")
        self.g_bn6 = batch_norm(name="g_bn6")

        self.dataset_name = dataset_name
        self.inlier_dataset_address = os.path.join(dataset_address, 'in')
        self.outlier_dataset_address = os.path.join(dataset_address, 'out')
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.attention_label = attention_label

        if self.is_training:
            logging.basicConfig(filename="ALOCC_loss.log", level=logging.INFO)

        if self.dataset_name == "data-alocc":
            inlier_image_paths = []
            for sImageDirFiles in glob(
                os.path.join(self.inlier_dataset_address, self.input_fname_pattern)
            ):
                inlier_image_paths.append(sImageDirFiles)
            self.inlier_data = inlier_image_paths

            outlier_image_paths = []
            for sImageDirFiles in glob(
                os.path.join(self.outlier_dataset_address, self.input_fname_pattern)
            ):
                outlier_image_paths.append(sImageDirFiles)
            self.outlier_data = outlier_image_paths

            self.c_dim = 1

        else:
            assert "Error in loading dataset"

        self.grayscale = self.c_dim == 1
        self.build_model()

    # =========================================================================================================
    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name="real_images"
        )
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name="sample_inputs"
        )

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name="z")

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # Simple GAN's losses
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)
            )
        )
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)
            )
        )
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)
            )
        )

        # Refinement loss
        self.g_r_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G, labels=self.z)
        )
        self.g_loss = self.g_loss + self.g_r_loss * self.r_alpha
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if "d_" in var.name]
        self.g_vars = [var for var in t_vars if "g_" in var.name]

    # =========================================================================================================
    def train(self, config):
        d_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(
            self.d_loss, var_list=self.d_vars
        )
        g_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(
            self.g_loss, var_list=self.g_vars
        )

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.saver = tf.train.Saver(max_to_keep=100)

        self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])

        sample_files = self.inlier_data[0 : self.sample_num]
        sample = read_lst_images(sample_files, None, None, self.b_work_on_patch)
        sample = sample[0 : self.sample_num]
        sample_in = np.array(sample).astype(np.float32)
        imageio.imwrite(
            "./{}/train_in_samples.jpg".format(config.sample_dir),
            montage(sample_in[:, :, :, 0]),
        )

        sample_files = self.inlier_data[0 : self.sample_num]
        sample = read_lst_images(sample_files, None, None, self.b_work_on_patch)
        sample = get_noisy_data(sample)
        sample = sample[0 : self.sample_num]
        sample_noise = np.array(sample).astype(np.float32)
        imageio.imwrite(
            "./{}/train_noise_samples.jpg".format(config.sample_dir),
            montage(sample_noise[:, :, :, 0]),
        )

        sample_files = self.outlier_data[0 : self.sample_num]
        sample = read_lst_images(sample_files, None, None, self.b_work_on_patch)
        sample = sample[0 : self.sample_num]
        sample_out = np.array(sample).astype(np.float32)
        imageio.imwrite(
            "./{}/train_out_samples.jpg".format(config.sample_dir),
            montage(sample_out[:, :, :, 0]),
        )

        # load previous checkpoint
        counter = 1
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # load traning data
        if config.dataset == "data-alocc":
            t = time.time()
            in_sample = read_lst_images(self.inlier_data, None, None, self.b_work_on_patch)
            t = time.time() - t
            print(" [*] Loaded Inlier Data in {:3f}s".format(t))

            t = time.time()
            noise_sample = get_noisy_data(in_sample)
            t = time.time() - t
            print(" [*] Loaded Noisy Data in {:3f}s".format(t))

            t = time.time()
            out_sample = read_lst_images(self.outlier_data, None, None, self.b_work_on_patch)
            t = time.time() - t
            print(" [*] Loaded Outlier Data in {:3f}s".format(t))

        for epoch in xrange(config.epoch):
            print(
                "Epoch ({}/{})-------------------------------------------------".format(
                    epoch, config.epoch
                )
            )
            if config.dataset == "data-alocc":
                batch_idxs = min(len(in_sample), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if config.dataset == "data-alocc":
                    in_batch = in_sample[
                        idx * config.batch_size : (idx + 1) * config.batch_size
                    ]
                    noise_batch = noise_sample[
                        idx * config.batch_size:(idx + 1) * config.batch_size
                    ]
                    '''
                    out_idxs = np.random.randint(len(out_sample), size=config.batch_size)
                    out_batch = out_sample[out_idxs]
                    '''
                in_batch_images = np.array(in_batch).astype(np.float32)
                noise_batch_images = np.array(noise_batch).astype(np.float32)

                # update discriminator
                _, summary_str = self.sess.run(
                    [d_optim, self.d_sum],
                    feed_dict={
                        self.inputs: in_batch_images,
                        self.z: noise_batch_images,
                    },
                )

                # update refinement(generator)
                _, summary_str = self.sess.run(
                    [g_optim, self.g_sum], feed_dict={self.z: noise_batch_images}
                )

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run(
                    [g_optim, self.g_sum], feed_dict={self.z: noise_batch_images}
                )

                errD_fake = self.d_loss_fake.eval({self.z: noise_batch_images})
                errD_real = self.d_loss_real.eval({self.inputs: in_batch_images})
                errG = self.g_loss.eval({self.z: noise_batch_images})

                counter += 1

                msg = "Epoch:[%2d][%4d/%4d]--> d_loss: %.8f, g_loss: %.8f" % (
                    epoch,
                    idx,
                    batch_idxs,
                    errD_fake + errD_real,
                    errG,
                )
                print(msg)
                logging.info(msg)

                if np.mod(counter, self.n_per_itr_print_results) == 0:
                    # INLIERS
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_in,
                            self.inputs: sample_in,
                        },
                    )
                    imageio.imwrite(
                        "./{}/train_in_{:02d}_{:04d}.png".format(
                            config.sample_dir, epoch, idx
                        ),
                        montage(samples[:, :, :, 0]),
                    )
                    msg = "[Sample In] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)
                    print(msg)
                    logging.info(msg)

                    # NOISY
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_noise,
                            self.inputs: sample_noise,
                        },
                    )
                    imageio.imwrite(
                        "./{}/train_noise_{:02d}_{:04d}.png".format(
                            config.sample_dir, epoch, idx
                        ),
                        montage(samples[:, :, :, 0]),
                    )
                    msg = "[Sample Noise] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)
                    print(msg)
                    logging.info(msg)

                    # OUTLIERS
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_out,
                            self.inputs: sample_out,
                        },
                    )
                    imageio.imwrite(
                        "./{}/train_out_{:02d}_{:04d}.png".format(
                            config.sample_dir, epoch, idx
                        ),
                        montage(samples[:, :, :, 0]),
                    )
                    msg = "[Sample Out] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)
                    print(msg)
                    logging.info(msg)

            self.save(config.checkpoint_dir, epoch)

    # =========================================================================================================
    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name="d_h0_conv"))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name="d_h1_conv")))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name="d_h2_conv")))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name="d_h3_conv")))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, "d_h3_lin")
            h5 = tf.nn.sigmoid(h4, name="d_output")
            return h5, h4

    # =========================================================================================================
    def generator(self, z):
        with tf.variable_scope("generator") as scope:

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            hae0 = lrelu(
                self.g_bn4(conv2d(z, self.df_dim * 2, name="g_encoder_h0_conv"))
            )
            hae1 = lrelu(
                self.g_bn5(conv2d(hae0, self.df_dim * 4, name="g_encoder_h1_conv"))
            )
            hae2 = lrelu(
                self.g_bn6(conv2d(hae1, self.df_dim * 8, name="g_encoder_h2_conv"))
            )

            h2, self.h2_w, self.h2_b = deconv2d(
                hae2,
                [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                name="g_decoder_h1",
                with_w=True,
            )
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim * 1],
                name="g_decoder_h0",
                with_w=True,
            )
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3,
                [self.batch_size, s_h, s_w, self.c_dim],
                name="g_decoder_h00",
                with_w=True,
            )

            return tf.nn.tanh(h4, name="g_output")

    # =========================================================================================================
    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            hae0 = lrelu(
                self.g_bn4(conv2d(z, self.df_dim * 2, name="g_encoder_h0_conv"))
            )
            hae1 = lrelu(
                self.g_bn5(conv2d(hae0, self.df_dim * 4, name="g_encoder_h1_conv"))
            )
            hae2 = lrelu(
                self.g_bn6(conv2d(hae1, self.df_dim * 8, name="g_encoder_h2_conv"))
            )

            h2, self.h2_w, self.h2_b = deconv2d(
                hae2,
                [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                name="g_decoder_h1",
                with_w=True,
            )
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2,
                [self.batch_size, s_h2, s_w2, self.gf_dim * 1],
                name="g_decoder_h0",
                with_w=True,
            )
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3,
                [self.batch_size, s_h, s_w, self.c_dim],
                name="g_decoder_h00",
                with_w=True,
            )

            return tf.nn.tanh(h4, name="g_output")

    # =========================================================================================================
    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size, self.output_height, self.output_width, self.r_alpha
        )

    # =========================================================================================================
    def save(self, checkpoint_dir, step):
        model_name = "ALOCC_Model.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(
            self.sess, os.path.join(checkpoint_dir, model_name), global_step=step
        )

    # =========================================================================================================
    def load(self, checkpoint_dir):
        import re

        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
          counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
          print(" [*] Success to read {}".format(ckpt_name))
          return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    # =========================================================================================================

    def f_check_checkpoint(self, checkpoint_number=-1):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        print(" [*] Reading checkpoints...")
        self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.all_model_checkpoint_paths[checkpoint_number])
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            could_load = True
            checkpoint_counter = counter
        else:
            print(" [*] Failed to find a checkpoint")
            could_load = False
            checkpoint_counter = 0

        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            return counter
        else:
            print(" [!] Load failed...")
            return -1

    # =========================================================================================================
    def f_test_frozen_model(self, lst_image_slices):
        lst_generated_img = []
        lst_discriminator_v = []
        lst_discriminator_of_g_v = []
        tmp_shape = lst_image_slices.shape
        tmp_lst_slices = lst_image_slices

        batch_idxs = len(tmp_lst_slices) // self.batch_size

        print("start new process ...")
        for i in xrange(0, batch_idxs):
            batch_data = tmp_lst_slices[i * self.batch_size : (i + 1) * self.batch_size]

            results_g = self.sess.run(self.G, feed_dict={self.z: batch_data})
            
            results_d = self.sess.run(
                self.D_logits, feed_dict={self.inputs: batch_data}
            )
            results_d_of_g = self.sess.run(
                self.D_logits, feed_dict={self.inputs: results_g}
            )

            lst_discriminator_v.extend(results_d)
            lst_discriminator_of_g_v.extend(results_d_of_g)
            lst_generated_img.extend(results_g)
            print("finish pp ... {}/{}".format(i, batch_idxs))

        f = plt.figure()
        x = np.sort(np.squeeze(np.array(lst_discriminator_v)))
        plt.plot(x)
        f.savefig("./" + self.sample_dir + "/d_values.jpg")

        f = plt.figure()
        x = np.sort(np.squeeze(np.array(lst_discriminator_of_g_v)))
        plt.plot(x)
        f.savefig("./" + self.sample_dir + "/d_of_g_values.jpg")

        np.save("./" + self.sample_dir + "/d_values.npy", lst_discriminator_v)
        np.save("./" + self.sample_dir + "/d_of_g_values.npy", lst_discriminator_of_g_v)

        imageio.imwrite(
            "./" + self.sample_dir + "/ALOCC_generated.jpg",
            montage(np.array(lst_generated_img)[:100, :, :, 0]),
        )
        imageio.imwrite(
            "./" + self.sample_dir + "/ALOCC_input.jpg",
            montage(np.array(tmp_lst_slices)[:100, :, :, 0]),
        )
