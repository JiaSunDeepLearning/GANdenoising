#! /usr/bin/python
# -*- coding: utf8 -*-

# import os, time, pickle, random, time
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from datetime import datetime
# import numpy as np
# from time import localtime, strftime
# import logging, scipy
# from PIL import Image
# import matplotlib.pyplot as plt

# import tensorflow as tf
# import tensorlayer as tl

from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config

# ##====================== HYPER-PARAMETERS ===========================###
# # Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
# # initialize G
n_epoch_init = config.TRAIN.n_epoch_init
# # adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():
    print("creating folders to save result images and trained model...")
    # # create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    # ##====================== PRE-LOAD DATA ===========================###
    print("PRE-LOADING DATA...")
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    # # If your machine have enough memory, please pre-load the whole train set.
    print("pre-loading the whole train set...")
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    # ##========================== DEFINE MODEL ============================###
    # # train inference
    # ============================ srgan =====================================#
    # t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    # ============================ srgan =====================================#

    t_image = tf.placeholder('float32', [batch_size, 224, 224, 3], name='t_image_input_to_SRGAN_generator')
    print("Creating placeholder for training set ,size is : [", batch_size, "224 224 3]")
    # ============================ srgan =====================================#
    # t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')
    # ============================ srgan =====================================#
    t_target_image = tf.placeholder('float32', [batch_size, 224, 224, 3], name='t_target_image')
    print("Creating placeholder for target set ,size is : [", batch_size, "224 224 3]")
    print("Generator : Forward propagation is starting ...")
    net_g = SRGAN_g(t_image, sizeof_input=None, is_train=True, reuse=False)
    print("Generator : Forward propagation ends ")
    print("shape of net_g.outputs[0]: ", net_g.outputs[0].shape)
    print("net_g.outputs[0]: ", net_g.outputs[0])

    '''
    plt.figure("valid_gen.png")
    plt.imshow(net_g.outputs[0])
    plt.show()
    '''

    # tl.vis.save_image(net_g.outputs[0], 'training_temps' + '/valid_gen.png')
    print("Discriminator : Forward propagation of target image is starting ...")
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    print("Discriminator : Forward propagation of target image ends ...")

    print("Discriminator : Forward propagation of generated image is starting ...")
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)
    print("Discriminator : Forward propagation of generated image ends ...")

    net_g.print_params(False)
    net_g.print_layers()
    net_d.print_params(False)
    net_d.print_layers()

    # # vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    print("resizing target images for vgg")
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg
    #  http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    print("resizing generated images for vgg")
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)
    # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    # # test inference
    print("test inference")
    net_g_test = SRGAN_g(t_image, sizeof_input=None, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    print("define D Loss")
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    print("define G Loss")
    g_gan_loss = 0.25 * 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 0.14 * 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    print("adam ")
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    # # Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    # # SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    # ##========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    print("RESTORE MODEL")
    tl.layers.initialize_global_variables(sess)
    # tf.global_variables_initializer(sess)
    # print("seems stops here")
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_2k.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_2k.npz'.format(tl.global_flag['mode']), network=net_g)

    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}2k.npz'.format(tl.global_flag['mode']), network=net_d)

    # ##============================= LOAD VGG ===============================###
    import os
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    # ##============================= TRAINING ===============================###
    # # use first `batch_size` of train set to have a quick test during training
    print("use first batch_size of train set to have a quick test during training")
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32)
    # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=imnoise_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())

    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan + '/_train_sample_384.png')

    # ##========================= initialize G ====================###

    # # fixed learning rate

    import time
    '''
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        
        #if epoch != 0 and (epoch % config.TRAIN.decay_every_init) == 0:
        #    new_lr_decay_init = config.TRAIN.lr_decay_init ** (epoch // config.TRAIN.decay_every_init)
         #   sess.run(tf.assign(lr_v, lr_init * new_lr_decay_init))
          #  log = " ** new learning rate: %f (for generator init)" % (lr_init * new_lr_decay_init)
           # print(log)
        

        # # If your machine cannot load all images into memory, you should use
        # # this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        # # If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=imnoise_fn)
            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        # save mse every epoch to txt file
        with open('logfiles/every_epoch_mse_init.txt', 'a') as file_object:
            file_object.write(str(total_mse_loss / n_iter))
            file_object.write('\n')

        # # quick evaluation on train set
        
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)
        
        # # save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

        if (epoch != 0) and (epoch % 100 == 0):
            tl.files.save_npz(net_g.all_params,
                              name=checkpoint_dir + '/' + str(epoch+200) + 'g_{}_init.npz'.format(tl.global_flag['mode']),
                              sess=sess)
    '''
    # ##========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch + 1):
        # # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter, total_mse_loss, total_vgg_loss, total_adv_loss = 0, 0, 0, 0, 0, 0

        # # If your machine cannot load all images into memory, you should use
        # # this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        # # If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=imnoise_fn)

            # # update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            # # update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})

            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))

            # errD:d_loss, errG:g_loss, errM:mse, errV:vgg, errA:adv;

            total_d_loss += errD
            total_g_loss += errG
            total_mse_loss += errM
            total_vgg_loss += errV
            total_adv_loss += errA
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        # save logs
        # g_loss = mse_loss + vgg_loss + g_gan_loss

        # g_loss: joint loss function
        with open('logfiles/every_epoch_g_loss.txt', 'a') as file_object:
            file_object.write(str(total_g_loss / n_iter))
            file_object.write('\n')

        # d_loss: average possibility
        with open('logfiles/every_epoch_d_loss.txt', 'a') as file_object:
            file_object.write(str(total_d_loss / n_iter))
            file_object.write('\n')

        # mse_loss
        with open('logfiles/every_epoch_mse_loss.txt', 'a') as file_object:
            file_object.write(str(total_mse_loss / n_iter))
            file_object.write('\n')

        # vgg_loss: mse of vgg feature map (2e-6)
        with open('logfiles/every_epoch_vgg_loss.txt', 'a') as file_object:
            file_object.write(str(total_vgg_loss / n_iter))
            file_object.write('\n')

        # g_gan_loss(adv): (1e-3)
        with open('logfiles/every_epoch_adv_loss.txt', 'a') as file_object:
            file_object.write(str(total_adv_loss / n_iter))
            file_object.write('\n')

        # # quick evaluation on train set
        '''
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
            # ; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)
        '''
        # # save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
        # every 100 epoch
        if (epoch != 0) and (epoch % 100 == 0):
            tl.files.save_npz(net_g.all_params,
                              name=checkpoint_dir + '/' + str(epoch) + 'g_{}.npz'.format(tl.global_flag['mode']),
                              sess=sess)


def evaluate():
    # # create folders to save result images
    print("create or check folders to save result images...")
    # save_dir = "samples/{}".format(tl.global_flag['mode'])
    save_dir = "./results/Denoised/"
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    #save_npy_dir = "./train2valid_npy/"
    #tl.files.exists_or_mkdir(save_npy_dir)

    # ##====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=True))

    print("Reading images from valid data set...")

    # # If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32, printable=False)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    # print("use gan to generate sr image")
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    # ##========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    # tf.global_variables_initializer(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/600g_srgan_init.npz', network=net_g)

    # ##========================== DEFINE MODEL ============================###
    # imid = 0  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    for imid in range(0, len(valid_hr_imgs)):
        print("time : ", imid, " / ", len(valid_hr_imgs)-1)
        # valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
        # valid_lr_img = get_imgs_fn('test_lr.png', 'data2017/')  # if you want to test your own image
        # valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
        # valid_hr_img = get_imgs_fn('head_GT.bmp', './samples/evaluate/')
        # if you want to test your own image , 'data2017/DIV2K_valid_HR/'
        valid_hr_img = (valid_hr_img / (255. / 2.)) - 1  # if you want to test your own image
        # print(valid_hr_img.min(), valid_hr_img.max())
        # valid_hr_img = crop_sub_imgs_fn(valid_hr_img, is_random=True)
        size = valid_hr_img.shape
        valid_lr_img = imnoise_valid_fn(valid_hr_img, size_of_image=size)

        print("size and range of valid_lr_image is : ", size, valid_lr_img.min(), valid_lr_img.max())
        # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
        # the old version of TL need to specify the image size


        # ##======================= EVALUATION =============================###
        import time
        # print("Generating sr image is starting... ")
        start_time = time.time()
        # time.asctime( time.localtime(time.time()) )
        print(time.asctime(time.localtime(start_time)))
        # print(start_time)
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        # out is not the SR image
        # print(out[0])

        # out_downsample = out[0]
        # print(out_downsample.shape)
        # out = scipy.misc.imresize(out_downsample, [size[0], size[1]], interp='bicubic', mode=None)
        print("took: %4.4fs" % (time.time() - start_time))
        print('generated image:', out.shape, out.min(), ' - ', out.max())
        # print(" mission success...")

        # print("LR size: %s /  generated HR size: %s" % (size, out.shape))
        # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] saving images")

        image = out[0]
        image = (image + 1) * (255. / 2.)
        image[image > 255] = 255
        image[image < 0] = 0

        #image_float = image

        image = np.rint(image)
        image = np.uint8(image)

        tl.vis.save_image(image, save_dir + '/0' + str(imid+101) + 'gen.png')
        # tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
        # tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')
        # print(save_dir + '/0' + str(imid+101) + 'gen.png')
        # out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        # tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')
        # np.save(save_npy_dir + '/0' + str(imid+101) + 'gen.npy', image_float)

        '''
        psnr_imid = psnr_me(image, valid_hr_img)
        with open(save_dir + 'psnr_log.txt', 'a') as file_object:
            file_object.write(str(psnr_imid))
            file_object.write('\n')

        ssim_imid = ssim_me(image, valid_hr_img)
        with open(save_dir + 'ssim_log.txt', 'a') as file_object:
            file_object.write(str(ssim_imid))
            file_object.write('\n')
        '''


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='evaluate', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
