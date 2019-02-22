from __future__ import division
import numpy as np
import tensorflow as tf
import rawpy
import math
import tensorflow.contrib.slim as slim
from Cchen import Cchen

class ModelBuilder:
    #Tensorflow wrapper fuctions
    #Assumes NHWC
    def sliding_window(x,filter_size=1,strides=1,padding='VALID'):
        if(type(filter_size) is int):
            size_y = filter_size
            size_x = filter_size
        else:
            #It must be tuple
            (size_y,size_x) = filter_size

        if(type(strides) is int):
            stride_y = strides
            stride_x = strides
        else:
            #It must be tuple
            (stride_y,stride_x) = strides
        #shape_fix = [x.shape[i].value if x.shape[i].value is not None else tf.shape(x)[i] for i in range(len(x.shape))]
        #x = tf.reshape(x,shape_fix)
        #window =tf.image.extract_image_patches(x,[1,size_y,size_x,1],[1,stride_y,stride_x,1],[1,1,1,1],padding)
        #use convolution with filter_num = filter_size*in_channels
        in_chan = x.shape[-1]
        filt = tf.ones([size_y*size_x*in_chan,size_y*size_x])
        #Identity filter; only diagonal is 1 other elements 0
        filt = tf.matrix_band_part(filt,0,0)
        filt = tf.reshape(filt,[size_y,size_x,in_chan,size_y*size_x])
        window = tf.nn.conv2d(x,filt,[1,stride_y,stride_x,1],padding)
        return window
    #inner 2 dimensions are equal
    def get_triangular(x):
        h = x.shape[-2]
        w = x.shape[-1]
        axis = len(x.shape)
        reshaped = tf.reshape(x,[-1,h,w])
        triangular_mat = tf.matrix_band_part(tf.ones([h,w]),0,-1)
        triangular_vect = tf.boolean_mask(x,tf.not_equal(triangular_mat,0),axis=axis-2)
        reshape = [x.shape[i] if x.shape[i].value is not None else tf.shape(x)[i] for i in range(axis-2)]
        shape = [x.shape[i] for i in range(axis-2)]
        shape.append(x.shape[-2]*(x.shape[-1]+1)//2)
        reshape.append(x.shape[-2]*(x.shape[-1]+1)//2)
        triangular_vect.set_shape(shape)
        return tf.reshape(triangular_vect,reshape)

    def create_polynomial(x):
        original_dims = x.shape
        original_shape = tf.shape(x)
        x = tf.reshape(x,[-1,x.shape[-1]])
        ones = tf.ones_like(x[:,0:1])
        x = tf.concat([x,ones],-1)
        newshape = [int(original_dims[i]) if original_dims[i].value is not None else original_shape[i] for i in range(len(original_dims))]
        newshape[-1] = newshape[-1]+1
        newshape.append(int(original_dims[-1]+1))
        base = tf.expand_dims(x,-1)
        base_T = tf.expand_dims(x,-2)
        quadratic = tf.matmul(base,base_T)
        quadratic = tf.reshape(quadratic,newshape)
        return ModelBuilder.get_triangular(quadratic)

    #concatenate along last dimension
    def concat(x1,x2,name='concat'):
        return tf.concat([x1,x2],axis=-1,name=name)

    def max_pool(x,pool_size=2,strides=2,padding='valid',name=None):
        return tf.layers.max_pooling2d(x,pool_size,strides,padding=padding,name=name)

    def global_avg_pool(x,name='global_avg_pool',keepdims=False):
        return tf.reduce_mean(x,axis=(1,2),name=name,keepdims=keepdims)

    def global_max_pool(x,name='global_max_pool'):
        return tf.reduce_max(x,axis=(1,2),name=name)

    def global_min_pool(x,name='global_avg_pool'):
        return tf.reduce_min(x,axis=(1,2),name=name)
    #Consequent padding strategy for B,H,W,C
    def pad_for_filter_sym(x,filter_size):
        filter_removes = filter_size-1
        pad_before = math.ceil(filter_removes/2)
        pad_after = filter_removes-pad_before
        paddings = [[0,0],[pad_before,pad_after],[pad_before,pad_after],[0,0]]
        return tf.pad(x,paddings,'SYMMETRIC')

    #Consequent padding strategy for B,H,W,C
    def pad_for_filter_ref(x,filter_size):
        filter_removes = filter_size-1
        pad_before = math.ceil(filter_removes/2)
        pad_after = filter_removes-pad_before
        paddings = [[0,0],[pad_before,pad_after],[pad_before,pad_after],[0,0]]
        return tf.pad(x,paddings,'REFLECT')

    #Custom layer definitions




    def deep_isp_lowlevel(feature,residual,feature_filt_num=61,feature_act=tf.nn.relu,residual_act=tf.nn.tanh,name='low_level',no_feature=False):
        residual_filt_num = residual.shape[3]
        concat = None
        #If the feature map and residual is the same do not concat
        if(feature == residual):
            concat = residual
        else:
            concat = tf.concat([feature,residual],3,name=name+'_concat')
        #prepad to keep dims with reflect
        concat = ModelBuilder.pad_for_filter_ref(concat,3)
        if(not no_feature):
            feature_conv = tf.layers.conv2d(concat,feature_filt_num,3,activation=feature_act,name=name+'_feature',padding='valid')
            residual_conv = tf.layers.conv2d(concat,residual_filt_num,3,activation=residual_act,name=name+'_residual',padding='valid')
            residual_im = residual+residual_conv
            return feature_conv, residual_im
        else:
            residual_conv = tf.layers.conv2d(concat,residual_filt_num,3,activation=residual_act,name=name+'_residual',padding='valid')
            residual_im = residual+residual_conv
            return residual_im

    def deep_isp_highlevel(feature,filt_num=64,feature_act=tf.nn.relu):
        conv = tf.layers.conv2d(feature,filt_num,3,activation=feature_act,strides=2)
        return ModelBuilder.max_pool(conv)


    #Keep all information and reduce dimensions
    def downsample_std(x,target_filter_num,down_scale=2,name='downsample'):
        d1 = tf.space_to_depth(x,down_scale)
        return tf.layers.conv2d(d1,target_filter_num,1,name=name)

    def downsample_std_size(x,target_filter_num,filter_size=3,down_scale=2,name='downsample'):
        d1 = tf.space_to_depth(x,down_scale)
        d1 = ModelBuilder.pad_for_filter_ref(d1,filter_size)
        return tf.layers.conv2d(d1,target_filter_num,filter_size,name=name)

    #Upsample by learning to merge and deconstruct - sub pixel level sampling
    def upsample_dts(x,target_filter_num,upsample_scale,name='upsample'):
        intermediate_filter_number = target_filter_num*upsample_scale
        c1 = tf.layers.conv2d(x,intermediate_filter_number,1, name=name)
        return tf.depth_to_space(c1,upsample_scale)

    def upsample_dts_size(x,target_filter_num,filter_size=3,upsample_scale=2,name='upsample'):
        intermediate_filter_number = target_filter_num*upsample_scale
        #use reflect padding
        x = ModelBuilder.pad_for_filter_ref(x,filter_size)
        c1 = tf.layers.conv2d(x,intermediate_filter_number,filter_size, name=name)
        return tf.depth_to_space(c1,upsample_scale)

    def upsample_transpose(x,filter_num,filter_size,padding='valid',activation=None,kernel_init=None,name='upconv'):
        return tf.layers.conv2d_transpose(x,filter_num,filter_size,strides=filter_size,padding=padding,activation=activation,kernel_initializer=kernel_init,name=name)

    def upsample_resizeconv(x,target_filter_num,target_multiplier,kernel_init=None,name='upsample',activation=None):
        in_shape = tf.shape(x);
        resize = tf.image.resize_nearest_neighbor(x,(target_multiplier*in_shape[1],target_multiplier*in_shape[2]), name=name+'_upsample')
        return tf.layers.conv2d(resize,target_filter_num,3,padding='SAME',kernel_initializer=kernel_init,name=name+'_upconv',activation=activation)

    def upsample_resizeconv_ref(x,target_filter_num,target_multiplier,activation=None,kernel_init=None,name='upsample'):
        in_shape = tf.shape(x);
        resize = tf.image.resize_nearest_neighbor(x,(target_multiplier*in_shape[1],target_multiplier*in_shape[2]), name=name+'_upsample')
        resize = ModelBuilder.pad_for_filter_ref(resize,3)
        return tf.layers.conv2d(resize,target_filter_num,3,padding='valid',activation=activation,kernel_initializer=kernel_init,name=name+'_upconv')


    #Performance and loss functions
    def rgb_to_luma(rgb):
        coeffs = tf.constant([0.2126, 0.7152, 0.0722])
        return tf.reduce_sum(coeffs*rgb,-1,keepdims=True)

    def packed_to_luma(packed):
        r = packed[:,0::2,0::2,0:1] * 0.2126
        g1 = packed[:,0::2,1::2,0:1] * 0.7152
        g2 = packed[:,1::2,0::2,0:1] * 0.7152
        b = packed[:,1::2,1::2,0:1] * 0.0722
        return tf.concat([r,g1,g2,b],-1)
    def l1_loss_optim(result,gt,regularizer=None,optimizer=tf.train.AdamOptimizer):
        l1 = tf.abs(result-gt)
        loss = tf.reduce_mean(l1)
        if(regularizer is not None):
            loss = loss + regularizer()
        lr = tf.placeholder(tf.float32)
        optim = optimizer(learning_rate=lr).minimize(loss)
        return [loss, optim, lr]

    def l1_loss_optim_plus_scaled(result,gt,regularizer=None,optimizer=tf.train.AdamOptimizer):
        res_mean = tf.reduce_mean(result,axis=(1,2,3))
        gt_mean = tf.reduce_mean(gt,axis=(1,2,3))
        res_mean = tf.to_float(tf.equal(res_mean,0))*1e-9+res_mean #If result is all black dont divide by zero
        scaled_res = gt_mean/res_mean*result
        l1 = tf.abs(result-gt)
        loss = tf.reduce_mean(l1)
        l1_scaled = tf.abs(scaled_res-gt)
        loss = loss + tf.reduce_mean(l1_scaled)
        if(regularizer is not None):
            loss = loss + regularizer()
        lr = tf.placeholder(tf.float32)
        optim = optimizer(learning_rate=lr).minimize(loss)
        return [loss, optim, lr]

    def l1_loss(result,gt):
        l1 = tf.abs(result-gt)
        loss = tf.reduce_mean(l1)
        return [loss, None, None]

    def l1_reg(reg_scale=0.0001):
        regularizer = tf.contrib.layers.l1_regularizer(reg_scale)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        reg_term = sum([regularizer(param) for param in params])
        return reg_term

    def l2_reg(reg_scale=0.0001):
        regularizer = tf.contrib.layers.l2_regularizer(reg_scale)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        reg_term = sum([regularizer(param) for param in params])
        return reg_term

    def l1_loss_optim_l1reg(result,gt):
        return ModelBuilder.l1_loss_optim(result,gt,regularizer=ModelBuilder.l1_reg)

    def logcosh_loss_optim_l1reg(result,gt,a=3):
        return ModelBuilder.logcosh_loss_optim(result,gt,a=a,regularizer=ModelBuilder.l1_reg)

    def logcosh_loss_optim_l2reg(result,gt,a=3):
        return ModelBuilder.logcosh_loss_optim(result,gt,a=a,regularizer=ModelBuilder.l2_reg)

    def logcosh_loss_optim(result, gt,a=3,regularizer=None,optimizer=tf.contrib.opt.NadamOptimizer):
        error = (result-gt)*a
        cosh = (tf.exp(error)+tf.exp(-1.0*error))/2.0
        log_cosh = tf.log(cosh)/a
        #sum of logcosh in each channel
        loss = tf.reduce_sum(log_cosh,-1)
        #average over samples
        loss = tf.reduce_mean(loss)
        if(regularizer is not None):
            loss = loss + regularizer()
        lr = tf.placeholder(tf.float32)
        optim = optimizer(learning_rate=lr).minimize(loss)
        return [loss, optim, lr]

    def performance(result,gt):
        psnr = tf.image.psnr(result,gt,max_val=1.0)
        luma = ModelBuilder.rgb_to_luma(result)
        luma_gt = ModelBuilder.rgb_to_luma(gt)
        ssim = tf.image.ssim(luma,luma_gt,max_val=1.0)
        ms_ssim = tf.image.ssim_multiscale(luma,luma_gt,max_val=1.0)
        return [psnr, ssim, ms_ssim]

    def summaries(loss,ssim,ms_ssim,psnr):
        loss_sum = tf.summary.scalar('loss',loss)
        psnr_sum = tf.summary.scalar('psnr',tf.reduce_mean(psnr))
        ssim_sum = tf.summary.scalar('ssim',tf.reduce_mean(ssim))
        ms_ssim_sum = tf.summary.scalar('msssim',tf.reduce_mean(ms_ssim))
        merged = tf.summary.merge([loss_sum,psnr_sum,ssim_sum,ms_ssim_sum])
        return merged

    def summaries_epoch(prefix='train_'):
        epoch_loss_pl = tf.placeholder(tf.float32,shape=())
        epoch_loss = tf.summary.scalar(prefix+'loss_epoch',epoch_loss_pl)
        epoch_psnr_pl = tf.placeholder(tf.float32,shape=())
        epoch_ssim_pl = tf.placeholder(tf.float32,shape=())
        epoch_mssim_pl = tf.placeholder(tf.float32,shape=())
        epoch_psnr = tf.summary.scalar(prefix+'psnr_epoch',epoch_psnr_pl)
        epoch_ssim = tf.summary.scalar(prefix+'ssim_epoch',epoch_ssim_pl)
        epoch_mssim = tf.summary.scalar(prefix+'msssim_epoch',epoch_mssim_pl)
        merged_epoch = tf.summary.merge([epoch_loss,epoch_psnr,epoch_ssim,epoch_mssim])
        return (merged_epoch, epoch_loss_pl, epoch_psnr_pl, epoch_ssim_pl, epoch_mssim_pl)

    def preprocess_packed_sony_clip(inp,y,ratios,black_level,preproc_on_cpu=True,multiply_ratio=True):
        if(preproc_on_cpu):
            with tf.device("/cpu:0"):
                with tf.variable_scope('preprocess'):
                    ratios = tf.reshape(ratios,[-1,1,1,1])
                    inp = tf.cast(inp, tf.float32)
                    inp = tf.maximum(inp-black_level,0.0)/(16383-black_level)
                    if(multiply_ratio):
                        inp=inp*ratios
                    inp = tf.minimum(inp,1.0)
                    y = tf.cast(y,tf.float32)/65535.0
            return inp, y
        else:
            with tf.variable_scope('preprocess'):
                    ratios = tf.reshape(ratios,[-1,1,1,1])
                    inp = tf.cast(inp, tf.float32)
                    inp = tf.maximum(inp-black_level,0.0)/(16383-black_level)
                    if(multiply_ratio):
                        inp=inp*ratios
                    inp = tf.minimum(inp,1.0)
                    y = tf.cast(y,tf.float32)/65535.0
            return inp, y

    def preprocess_sony_concat_bl_and_ratios(inp,y,ratios,black_level,preproc_on_cpu=True):
        if(preproc_on_cpu):
            with tf.device("/cpu:0"):
                with tf.variable_scope('preprocess'):
                    inp = tf.cast(inp, tf.float32)
                    ones = tf.ones_like(inp[:,:,:,0:1])
                    ratios = tf.reshape(ratios,[-1,1,1,1])*ones/300.0 #Max amp rate 300 so normalize to it
                    bl = black_level*ones/16383.0
                    inp = inp/16383.0
                    inp = tf.concat([inp,bl,ratios],-1)
                    y = tf.cast(y,tf.float32)/65535.0
            return inp, y
        else:
            with tf.variable_scope('preprocess'):
                    inp = tf.cast(inp, tf.float32)
                    ones = tf.ones_like(inp[:,:,:,0:1])
                    ratios = tf.reshape(ratios,[-1,1,1,1])*ones/300.0
                    bl = black_level*ones/16383.0
                    inp = inp/16383.0
                    inp = tf.concat([inp,bl,ratios],-1)
                    y = tf.cast(y,tf.float32)/65535.0
            return inp, y

    def preprocess_packed_sony_exp(inp,y,ratios,black_level,preproc_on_cpu=True,shutter_speed=False):
        if(preproc_on_cpu):
            with tf.device("/cpu:0"):
                with tf.variable_scope('preprocess'):
                    inp = tf.cast(inp, tf.float32)
                    inp = tf.maximum(inp-black_level,0.0)/(16383-black_level)
                    inp = tf.minimum(inp,1.0)
                    y = tf.cast(y,tf.float32)/65535.0
                    if(not shutter_speed):
                        inp = tf.concat([inp,tf.ones_like(inp)*ratios],-1)
                    else:
                        inp = tf.concat([inp,tf.ones_like(inp)/ratios],-1)
            return inp, y
        else:
            with tf.variable_scope('preprocess'):
                    inp = tf.cast(inp, tf.float32)
                    inp = tf.maximum(inp-black_level,0.0)/(16383-black_level)
                    inp = tf.minimum(inp,1.0)
                    y = tf.cast(y,tf.float32)/65535.0
                    if(not shutter_speed):
                        inp = tf.concat([inp,tf.ones_like(inp)*ratios],-1)
                    else:
                        inp = tf.concat([inp,tf.ones_like(inp)/ratios],-1)
            return inp, y


    #Model building functions
    def build_unet_transpose(x,depth=4,init_filter_num=32):
        pre_pool = []
        currentOut = x

        #Down part
        for i in range(0,depth):
            currentFilterNum = init_filter_num*(2**i)
            layerNum = str(i+1)
            conv1 = tf.layers.conv2d(currentOut,currentFilterNum,3,activation=tf.nn.leaky_relu,name=('conv'+layerNum+'_1'),padding='same')
            conv2 = tf.layers.conv2d(conv1,currentFilterNum,3,activation=tf.nn.leaky_relu,name=('conv'+layerNum+'_2'),padding='same')
            pre_pool.append(conv2)
            currentOut = ModelBuilder.max_pool(pre_pool[i],name=('max_pool'+layerNum))

        #Bottom part
        currentFilterNum = (2**depth)*init_filter_num
        currentOut = tf.layers.conv2d(currentOut,currentFilterNum,3,activation=tf.nn.leaky_relu, name=('conv'+str(depth+1)+'_1'),padding='same')
        currentOut = tf.layers.conv2d(currentOut,currentFilterNum,3,activation=tf.nn.leaky_relu, name=('conv'+str(depth+1)+'_2'),padding='same')
        
        #Up part
        for i in range(0,depth):
            index = depth-1-i
            layerNum = str(depth+2+i)
            currentFilterNum = (2**index)*init_filter_num
            upsample = ModelBuilder.upsample_transpose(currentOut, currentFilterNum, 2, kernel_init=tf.initializers.truncated_normal(stddev=0.02), name='upconv'+layerNum)
            concat = ModelBuilder.concat(upsample,pre_pool[index],name='upconcat'+layerNum)
            conv1 =  tf.layers.conv2d(concat,currentFilterNum,3,activation=tf.nn.leaky_relu, name=('conv'+layerNum+'_1'),padding='same')
            currentOut = tf.layers.conv2d(conv1,currentFilterNum,3,activation=tf.nn.leaky_relu, name=('conv'+layerNum+'_2'),padding='same')
        return currentOut

    def build_unet_resize(x,depth=4,init_filter_num=32):
        pre_pool = []
        currentOut = x

        #Down part
        for i in range(0,depth):
            currentFilterNum = init_filter_num*(2**i)
            layerNum = str(i+1)
            conv1 = tf.layers.conv2d(currentOut,currentFilterNum,3,activation=tf.nn.leaky_relu,name=('conv'+layerNum+'_1'),padding='same')
            conv2 = tf.layers.conv2d(conv1,currentFilterNum,3,activation=tf.nn.leaky_relu,name=('conv'+layerNum+'_2'),padding='same')
            pre_pool.append(conv2)
            currentOut = ModelBuilder.max_pool(pre_pool[i],name=('max_pool'+layerNum))

        #Bottom part
        currentFilterNum = (2**depth)*init_filter_num
        currentOut = tf.layers.conv2d(currentOut,currentFilterNum,3,activation=tf.nn.leaky_relu, name=('conv'+str(depth+1)+'_1'),padding='same')
        currentOut = tf.layers.conv2d(currentOut,currentFilterNum,3,activation=tf.nn.leaky_relu, name=('conv'+str(depth+1)+'_2'),padding='same')
        
        #Up part
        for i in range(0,depth):
            index = depth-1-i
            layerNum = str(depth+2+i)
            currentFilterNum = (2**index)*init_filter_num
            upsample = ModelBuilder.upsample_resizeconv(currentOut, currentFilterNum, 2, kernel_init=tf.initializers.truncated_normal(stddev=0.02), name='upconv'+layerNum,activation=tf.nn.leaky_relu)
            concat = ModelBuilder.concat(upsample,pre_pool[index],name='upconcat'+layerNum)
            conv1 =  tf.layers.conv2d(concat,currentFilterNum,3,activation=tf.nn.leaky_relu, name=('conv'+layerNum+'_1'),padding='same')
            currentOut = tf.layers.conv2d(conv1,currentFilterNum,3,activation=tf.nn.leaky_relu, name=('conv'+layerNum+'_2'),padding='same')
        return currentOut



    #Input transformations
    def pack_sony_tf(inp,name='pack_sony'):
        return tf.space_to_depth(inp,2,name=name)

    def fullres_sony_tf(packed,name='fullres_sony'):
        r = packed[:,:,:,0:1]
        zeros = tf.zeros_like(r)
        r = tf.concat([r,zeros,zeros],3)
        g1 = packed[:,:,:,1:2]
        g1 = tf.concat([zeros,g1,zeros],3)
        g2 = packed[:,:,:,2:3]
        g2 = tf.concat([zeros,g2,zeros],3)
        b = packed[:,:,:,3:4]
        b = tf.concat([zeros,zeros,b],3)
        mosaic = tf.concat([r,g1,g2,b],3)
        mosaic = tf.depth_to_space(mosaic,2)
        return mosaic


    #Assumes packed input
    def build_cchen_sony(x):
        unet = ModelBuilder.build_unet_transpose(x)
        subPixelSampler = tf.layers.conv2d(unet,12,1,name='final_upsample')
        return tf.depth_to_space(subPixelSampler,2)

    def build_cchen_sony_resize(x):
        unet = ModelBuilder.build_unet_resize(x)
        subPixelSampler = tf.layers.conv2d(unet,12,1,name='final_upsample')
        return tf.depth_to_space(subPixelSampler,2)

    #Assumes flat input
    def build_unet_extra_skip(x,ratio=None):
        packed = ModelBuilder.pack_sony_tf(x)
        mosaic = ModelBuilder.fullres_sony_tf(packed)
        if(ratio is not None):
            #concatenate it to both packed and mosaic
            ratio1 = tf.ones_like(packed[:,:,:,0:1])*ratio
            ratio2 = tf.ones_like(mosaic[:,:,:,0:1])*ratio
            packed = tf.concat([packed,ratio1],-1)
            mosaic = tf.concat([mosaic,ratio2],-1)
        unet = ModelBuilder.build_unet_transpose(packed)
        #Extra upsample and concat
        subPixelSampler = ModelBuilder.upsample_transpose(unet,32,2, kernel_init=tf.initializers.truncated_normal(stddev=0.02), name='upconv_last')
        skip_con = tf.concat([subPixelSampler,mosaic],3)
        final_conv = tf.layers.conv2d(skip_con,32,3,activation=tf.nn.leaky_relu,name='final_conv1',padding='SAME')
        final_conv = tf.layers.conv2d(final_conv,32,3,activation=tf.nn.leaky_relu,name='final_conv2',padding='SAME')
        out_im = tf.layers.conv2d(final_conv,3,1,name='output')
        return out_im

    def build_unet_extra_skip_resize(x):
        packed = ModelBuilder.pack_sony_tf(x)
        mosaic = ModelBuilder.fullres_sony_tf(packed)
        unet = ModelBuilder.build_unet_resize(packed)
        #Extra upsample and concat
        subPixelSampler = ModelBuilder.upsample_resizeconv(unet,32,2, kernel_init=tf.initializers.truncated_normal(stddev=0.02), name='upconv_last',activation=tf.nn.leaky_relu)
        skip_con = tf.concat([subPixelSampler,mosaic],3)
        final_conv = tf.layers.conv2d(skip_con,32,3,activation=tf.nn.leaky_relu,name='final_conv1',padding='SAME')
        final_conv = tf.layers.conv2d(final_conv,32,3,activation=tf.nn.leaky_relu,name='final_conv2',padding='SAME')
        out_im = tf.layers.conv2d(final_conv,3,1,name='output')
        return out_im


    #Experiment model building functions, standardized for flat inputs
    def build_exp_pl():
        x_pl = tf.placeholder(tf.uint16,shape=(None,None,None,1))
        y_pl = tf.placeholder(tf.uint16,shape=(None,None,None,3))
        ratios_pl = tf.placeholder(tf.float32,shape=(None,))
        pre_x_pl = tf.placeholder(tf.float32,shape=(None,None,None,1))
        pre_y_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        out_im_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        result_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        loss_pl = tf.placeholder(tf.float32,shape=())
        return {'x':x_pl,'y':y_pl,'ratio':ratios_pl,'pre_x_pl':pre_x_pl,'pre_y_pl':pre_y_pl,'out_im_pl':out_im_pl,'loss_pl':loss_pl}

    def build_exp_pl_concat_bl_and_rat():
        x_pl = tf.placeholder(tf.uint16,shape=(None,None,None,1))
        y_pl = tf.placeholder(tf.uint16,shape=(None,None,None,3))
        ratios_pl = tf.placeholder(tf.float32,shape=(None,))
        pre_x_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        pre_y_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        out_im_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        result_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        loss_pl = tf.placeholder(tf.float32,shape=())
        return {'x':x_pl,'y':y_pl,'ratio':ratios_pl,'pre_x_pl':pre_x_pl,'pre_y_pl':pre_y_pl,'out_im_pl':out_im_pl,'loss_pl':loss_pl}

    def build_exp_pl_concat_bl():
        x_pl = tf.placeholder(tf.uint16,shape=(None,None,None,1))
        y_pl = tf.placeholder(tf.uint16,shape=(None,None,None,3))
        ratios_pl = tf.placeholder(tf.float32,shape=(None,))
        pre_x_pl = tf.placeholder(tf.float32,shape=(None,None,None,2))
        pre_y_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        out_im_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        result_pl = tf.placeholder(tf.float32,shape=(None,None,None,3))
        loss_pl = tf.placeholder(tf.float32,shape=())
        return {'x':x_pl,'y':y_pl,'ratio':ratios_pl,'pre_x_pl':pre_x_pl,'pre_y_pl':pre_y_pl,'out_im_pl':out_im_pl,'loss_pl':loss_pl}



    def add_preproc_to_model(model,pre_x,pre_y):
        model['pre_x'] = pre_x
        model['pre_y'] = pre_y
        return model

    def add_epoch_perf_to_model(model,step_sum,perf_t,perf_v):
        model['step_sum'] = step_sum
        t_epoch_sum = perf_t[0]
        t_epoch_loss_pl = perf_t[1]
        t_epoch_psnr_pl = perf_t[2]
        t_epoch_ssim_pl = perf_t[3]
        t_epoch_mssim_pl= perf_t[4]
        v_epoch_sum = perf_v[0]
        v_epoch_loss_pl = perf_v[1]
        v_epoch_psnr_pl = perf_v[2]
        v_epoch_ssim_pl = perf_v[3]
        v_epoch_mssim_pl= perf_v[4]
        model['t_epoch_sum'] = t_epoch_sum
        model['t_epoch_loss_pl'] = t_epoch_loss_pl
        model['t_epoch_ssim_pl'] = t_epoch_ssim_pl
        model['t_epoch_mssim_pl'] = t_epoch_mssim_pl
        model['t_epoch_psnr_pl'] = t_epoch_psnr_pl
        model['v_epoch_sum'] = v_epoch_sum
        model['v_epoch_loss_pl'] = v_epoch_loss_pl
        model['v_epoch_ssim_pl'] = v_epoch_ssim_pl
        model['v_epoch_mssim_pl'] = v_epoch_mssim_pl
        model['v_epoch_psnr_pl'] = v_epoch_psnr_pl
        return model

    def add_inference_to_model(model,out_im):
        model['image'] = out_im
        return model

    def add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr):
        model['step'] = step_up
        model['gs'] = gs_var
        model['loss'] = loss
        model['psnr'] = psnr
        model['ssim'] = ssim
        model['ms_ssim'] = ms_ssim
        model['optim'] = optim
        model['lr'] = lr
        return model

    def build_cchen_sony_exp(loss_fn=l1_loss_optim):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        packed = ModelBuilder.pack_sony_tf(model['pre_x_pl'])
        result = ModelBuilder.build_cchen_sony(packed)
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        [loss, optim, lr] = loss_fn(result,model['pre_y_pl'])
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],model['pre_y_pl'])
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = tf.get_variable("global_step",shape=[], initializer = tf.zeros_initializer, dtype=tf.int32)
        step_up = gs_var.assign(gs_var+1)
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model


    def build_unet_s_sony_exp(loss_fn=l1_loss_optim):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        result = ModelBuilder.build_unet_extra_skip(model['pre_x_pl'])
        [loss, optim, lr] = loss_fn(result,model['pre_y_pl'])
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],model['pre_y_pl'])
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = tf.get_variable("global_step",shape=[], initializer = tf.zeros_initializer, dtype=tf.int32)
        step_up = gs_var.assign(gs_var+1)
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model

    def build_cchen_sony_exp_resize(loss_fn=l1_loss_optim):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        packed = ModelBuilder.pack_sony_tf(model['pre_x_pl'])
        result = ModelBuilder.build_cchen_sony_resize(packed)
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        [loss, optim, lr] = loss_fn(result,model['pre_y_pl'])
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],model['pre_y_pl'])
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = tf.get_variable("global_step",shape=[], initializer = tf.zeros_initializer, dtype=tf.int32)
        step_up = gs_var.assign(gs_var+1)
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model

    def build_unet_s_sony_exp_resize(loss_fn=l1_loss_optim):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        result = ModelBuilder.build_unet_extra_skip_resize(model['pre_x_pl'])
        [loss, optim, lr] = loss_fn(result,model['pre_y_pl'])
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],model['pre_y_pl'])
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = tf.get_variable("global_step",shape=[], initializer = tf.zeros_initializer, dtype=tf.int32)
        step_up = gs_var.assign(gs_var+1)
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model

    def pack_for_cchen(x):
        y = tf.space_to_depth(x,2)
        a = y[:,:,:,0:1] 
        b = y[:,:,:,1:2] 
        c = y[:,:,:,2:3] 
        d = y[:,:,:,3:4] 
        return tf.concat([a,b,d,c],3)

    def build_loadable_cchen(loss_fn=l1_loss):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        pre_x = ModelBuilder.pack_for_cchen(model['pre_x_pl'])
        pre_y = model['pre_y_pl']
        result = Cchen.network(pre_x)
        [loss, optim, lr] = loss_fn(result,pre_y)
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],pre_y)
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = None
        step_up = None
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model


    def build_deep_isp_exp(loss_fn=l1_loss_optim):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        packed = ModelBuilder.pack_sony_tf(model['pre_x_pl'])
        currentOut = packed
        currentOutR = packed
        for i in range(0,10):
            currentOut, currentOutR = ModelBuilder.deep_isp_lowlevel(currentOut,currentOutR,feature_filt_num=61,feature_act=tf.nn.leaky_relu,residual_act=tf.nn.tanh,name='low_level'+str(i))
        for i in range(0,3):
            currentOut = ModelBuilder.deep_isp_highlevel(currentOut,filt_num=64,feature_act=tf.nn.leaky_relu)
        currentOut = ModelBuilder.global_avg_pool(currentOut)
        currentOut = tf.layers.dense(currentOut,45)
        #get the quadratic transformation of the residual
        ones = tf.ones_like(currentOutR[:,:,:,0:1])
        currentOutR = tf.concat([currentOutR,ones],-1)
        currentOutR_transposed = tf.expand_dims(currentOutR,-1)
        currentOutR = tf.expand_dims(currentOutR,-2)
        quadratic = tf.matmul(currentOutR_transposed,currentOutR)
        #extract upper triangular (15 elements),since we are working on packed
        #quadratic = tf.matrix_band_part(quadratic,0,-1)
        unique_elements = ModelBuilder.get_triangular(quadratic)
        #use first 10 elements for r channel
        r = unique_elements*currentOut[:,0:15]
        g = unique_elements*currentOut[:,15:30]
        b = unique_elements*currentOut[:,30:45]
        result = tf.concat([r,g,b],-1)
        result = tf.layers.conv2d(result,12,1,name='final_upsample')
        result = tf.depth_to_space(result,2)
        [loss, optim, lr] = loss_fn(result,model['pre_y_pl'])
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],model['pre_y_pl'])
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = tf.get_variable("global_step",shape=[], initializer = tf.zeros_initializer, dtype=tf.int32)
        step_up = gs_var.assign(gs_var+1)
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model

    def build_deep_isp_exp2(loss_fn=l1_loss_optim):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        packed = ModelBuilder.pack_sony_tf(model['pre_x_pl'])
        currentOut = packed
        currentOutR = packed
        for i in range(0,5):
            currentOut, currentOutR = ModelBuilder.deep_isp_lowlevel(currentOut,currentOutR,feature_filt_num=61,feature_act=tf.nn.leaky_relu,residual_act=tf.nn.tanh,name='low_level'+str(i))
        currentOutR = ModelBuilder.upsample_resizeconv_ref(currentOutR,3,2,name='upsample_res')
        currentOut = ModelBuilder.upsample_resizeconv_ref(currentOut,61,2,name='upsample_feat')
        for i in range(5,10):
            currentOut, currentOutR = ModelBuilder.deep_isp_lowlevel(currentOut,currentOutR,feature_filt_num=61,feature_act=tf.nn.leaky_relu,residual_act=tf.nn.tanh,name='low_level'+str(i))
        for i in range(0,3):
            currentOut = ModelBuilder.deep_isp_highlevel(currentOut,filt_num=64,feature_act=tf.nn.leaky_relu)
        currentOut = ModelBuilder.global_avg_pool(currentOut)
        currentOut = tf.layers.dense(currentOut,30)
        #get the quadratic transformation of the residual
        ones = tf.ones_like(currentOutR[:,:,:,0:1])
        currentOutR = tf.concat([currentOutR,ones],-1)
        currentOutR_transposed = tf.expand_dims(currentOutR,-1)
        currentOutR = tf.expand_dims(currentOutR,-2)
        quadratic = tf.matmul(currentOutR_transposed,currentOutR)
        #extract upper triangular (10 elements),since we are working on packed
        #quadratic = tf.matrix_band_part(quadratic,0,-1)
        unique_elements = ModelBuilder.get_triangular(quadratic)
        #use first 10 elements for r channel
        r = tf.reduce_sum(unique_elements*currentOut[:,0:10],axis=-1,keepdims=True)
        g = tf.reduce_sum(unique_elements*currentOut[:,10:20],axis=-1,keepdims=True)
        b = tf.reduce_sum(unique_elements*currentOut[:,20:30],axis=-1,keepdims=True)
        result = tf.concat([r,g,b],-1)
        [loss, optim, lr] = loss_fn(result,model['pre_y_pl'])
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],model['pre_y_pl'])
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = tf.get_variable("global_step",shape=[], initializer = tf.zeros_initializer, dtype=tf.int32)
        step_up = gs_var.assign(gs_var+1)
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model

    def upper_rect(x,max_val=1.0):
        return tf.to_float(x <= max_val)*max_val-(tf.nn.relu(max_val-x))


    def build_unet_amp_infer(loss_fn=l1_loss_optim_plus_scaled):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False,multiply_ratio=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        packed = ModelBuilder.pack_sony_tf(model['pre_x_pl'])
        #Calculate relative brightness
        bright = ModelBuilder.packed_to_luma(packed)
        conv1 = tf.layers.conv2d(bright,16,3,strides=2,activation=tf.nn.relu)
        conv1 = ModelBuilder.max_pool(conv1)
        conv1 = tf.layers.conv2d(conv1,32,2,strides=2,activation=tf.nn.relu)
        conv1 = ModelBuilder.max_pool(conv1)
        conv1 = tf.reduce_mean(conv1,axis=(1,2))
        dense = tf.layers.dense(conv1,8,activation=tf.nn.relu)
        dense = tf.layers.dense(dense,1,activation=tf.nn.relu)
        #Upper rect scaling ratio to learned max value and assure at least 1 times scale
        learned_max_scale = tf.get_variable("learnable_scale_max",shape=[],dtype=tf.float32,initializer = tf.constant_initializer(300.),trainable=True)
        dense = ModelBuilder.upper_rect(dense+1,max_val=learned_max_scale)
        packed = ModelBuilder.upper_rect(dense*packed)
        #convert back to flat because unet_extra_skip assumes flat
        flat = tf.depth_to_space(packed,2)
        result = ModelBuilder.build_unet_extra_skip(flat)
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        [loss, optim, lr] = loss_fn(result,model['pre_y_pl'])
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],model['pre_y_pl'])
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = tf.get_variable("global_step",shape=[], initializer = tf.zeros_initializer, dtype=tf.int32)
        step_up = gs_var.assign(gs_var+1)
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model

    def build_unet_self_scale(loss_fn=l1_loss_optim_plus_scaled):
        model = ModelBuilder.build_exp_pl()
        pre_x,pre_y = ModelBuilder.preprocess_packed_sony_clip(model['x'],model['y'],model['ratio'],512,preproc_on_cpu=False,multiply_ratio=False)
        model = ModelBuilder.add_preproc_to_model(model,pre_x,pre_y)
        result = ModelBuilder.build_unet_extra_skip(model['pre_x_pl'])
        out_im = tf.minimum(tf.maximum(result,0.0),1.0)
        [loss, optim, lr] = loss_fn(result,model['pre_y_pl'])
        model = ModelBuilder.add_inference_to_model(model,out_im)
        [psnr, ssim, ms_ssim] = ModelBuilder.performance(model['out_im_pl'],model['pre_y_pl'])
        #Tensorboard summaries
        step_sum = ModelBuilder.summaries(model['loss_pl'],ssim,ms_ssim,psnr)
        perf_t = ModelBuilder.summaries_epoch()
        perf_v = ModelBuilder.summaries_epoch(prefix='valid_')
        model = ModelBuilder.add_epoch_perf_to_model(model,step_sum,perf_t,perf_v)
        #Global step
        gs_var = tf.get_variable("global_step",shape=[], initializer = tf.zeros_initializer, dtype=tf.int32)
        step_up = gs_var.assign(gs_var+1)
        model = ModelBuilder.add_training_vars_to_model(model,step_up,gs_var,loss,psnr,ssim,ms_ssim,optim,lr)
        return model
