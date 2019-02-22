from __future__ import division
import numpy as np
import tensorflow as tf
from SIDLoader import SIDLoader
from ModelBuilder import ModelBuilder
from Experiment import Experiment
import time,datetime,os,glob

path_prefix = '.'
checkpoint_dir = path_prefix+'/chk'
dataset_dir = path_prefix+'/dataset'
black_level = 512
seed = 1337
tensorboard_dir = path_prefix+'/tensorboard/'
#Set initial seed
np.random.seed(seed)
#Load flat matrix
dataset = SIDLoader(dataset_dir, patch_fn=None,keep_raw=False,keep_gt=True, set_id='test')
#Set up experiments
expList = []
expList.append(Experiment(name='Sony',model_fn={'fn':ModelBuilder.build_loadable_cchen},device="/device:GPU:0",tensorboard_dir=tensorboard_dir,checkpoint_dir='../checkpoint',dataset=dataset))
#expList.append(Experiment(name='cchen_sony_noflip',model_fn={'fn':ModelBuilder.build_cchen_sony_exp},device="/device:GPU:0",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='unet_s_sony_noflip',model_fn={'fn':ModelBuilder.build_unet_s_sony_exp},device="/device:GPU:1",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='deep_isp_noflip',model_fn={'fn':ModelBuilder.build_deep_isp_exp},device="/device:GPU:2",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='cchen_resize_sony_noflip',model_fn={'fn':ModelBuilder.build_cchen_sony_exp_resize},device="/device:GPU:3",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='unet_s_resize_sony_noflip',model_fn={'fn':ModelBuilder.build_unet_s_sony_exp_resize},device="/device:GPU:4",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='cchen_sony_flip',model_fn={'fn':ModelBuilder.build_cchen_sony_exp},device="/device:GPU:0",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='unet_s_sony_flip',model_fn={'fn':ModelBuilder.build_unet_s_sony_exp},device="/device:GPU:1",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='deep_isp_flip',model_fn={'fn':ModelBuilder.build_deep_isp_exp},device="/device:GPU:2",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='cchen_resize_sony_flip',model_fn={'fn':ModelBuilder.build_cchen_sony_exp_resize},device="/device:GPU:3",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='unet_s_resize_sony_flip',model_fn={'fn':ModelBuilder.build_unet_s_sony_exp_resize},device="/device:GPU:4",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='unet_self_amp2',model_fn={'fn':ModelBuilder.build_unet_self_scale},device="/device:GPU:0",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
#expList.append(Experiment(name='unet_amp_infer2',model_fn={'fn':ModelBuilder.build_unet_amp_infer},device="/device:GPU:1",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir,dataset=dataset))
epoch = 0
dataset.start()
try:
    #test loop
    for exp in expList:
        exp.create_test_writer()
    while(epoch < 1):
        #Get batch from batchloader
        (x,y,r) = dataset.get_batch()
        #start running training step on each GPU
        for exp in expList:
            exp.test_action(x,y,r)
        #Wait for all to finish
        for exp in expList:
            exp.finish_test_action()
        epoch = dataset.readEpoch

        if(dataset.readC == 0): #It is the end of the epoch
            for exp in expList:
                exp.end_of_epoch_test()
                        
except KeyboardInterrupt:
    print('Keyboard interrupt accepted')
finally:
    print("Stopping dataset")
    dataset.stop()
    for exp in expList:
        exp.model['sess'].close()
