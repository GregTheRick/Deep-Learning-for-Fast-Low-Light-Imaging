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
valid_freq = 20
seed = 1337
tensorboard_dir = path_prefix+'/tensorboard'
#Set initial seed
np.random.seed(seed)
#Set up experiments
expList = []
expList.append(Experiment(name='cchen_sony_noflip',model_fn={'fn':ModelBuilder.build_cchen_sony_exp},device="/device:GPU:0",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir))
expList.append(Experiment(name='unet_s_sony_noflip',model_fn={'fn':ModelBuilder.build_unet_s_sony_exp},device="/device:GPU:1",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir))
expList.append(Experiment(name='deep_isp_noflip',model_fn={'fn':ModelBuilder.build_deep_isp_exp},device="/device:GPU:2",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir))
expList.append(Experiment(name='cchen_resize_sony_noflip',model_fn={'fn':ModelBuilder.build_cchen_sony_exp_resize},device="/device:GPU:3",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir))
expList.append(Experiment(name='unet_s_resize_sony_noflip',model_fn={'fn':ModelBuilder.build_unet_s_sony_exp_resize},device="/device:GPU:4",tensorboard_dir=tensorboard_dir,checkpoint_dir=checkpoint_dir))


#Load flat matrix
dataset = SIDLoader(dataset_dir, patch_fn=SIDLoader.patch_unprocessed_sony,keep_raw=True,keep_gt=True)
validSet = None
#Get epoch from first experiment
epoch = expList[0].epoch
dataset.writeEpoch = epoch
dataset.readEpoch = epoch
dataset.start()
learning_rate = 1e-4
try:
    #train loop
    for exp in expList:
        exp.begin_train_cycle()
    while(epoch < 300):
        if(epoch >= 149):
            learning_rate = 1e-5
        #Get batch from batchloader
        (x,y,r) = dataset.get_batch()
        #start running training step on each GPU
        for exp in expList:
            exp.train_action(x,y,r,learning_rate)
        #Wait for all to finish
        for exp in expList:
            exp.finish_train_action()
        epoch = dataset.readEpoch

        if(dataset.readC == 0): #It is the end of the epoch
            for exp in expList:
                exp.end_of_epoch_train(epoch)
            #Validate
            if(epoch%valid_freq == 0):
                if(validSet == None):
                    validSet = SIDLoader(dataset_dir, patch_fn=None,keep_raw=True,keep_gt=True, set_id='valid')
                    validSet.start()
                validSet.readEpoch = dataset.readEpoch-1
                validSet.writeEpoch = dataset.readEpoch-1
                vepoch = validSet.readEpoch
                for exp in expList:
                    exp.begin_valid_cycle(epoch)
                while(vepoch < epoch):
                    (x,y,r) = validSet.get_batch()
                    for exp in expList:
                        exp.valid_action(x,y,r)
                    for exp in expList:
                        exp.finish_valid_action()
                    vepoch = validSet.readEpoch
                    if(validSet.readC == 0):
                        #get validation epoch summaries
                        for exp in expList:
                            exp.end_of_epoch_valid(epoch)
                        
except KeyboardInterrupt:
    print('Keyboard interrupt accepted. Shutting down')
finally:
    dataset.stop()
    if(validSet is not None):
        validSet.stop()
    for exp in expList:
        exp.finish_train_cycle()
        exp.model['sess'].close()