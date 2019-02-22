from __future__ import division
import rawpy
import math
import numpy as np
import glob
import os
import threading
import tensorflow as tf
import time

class SIDLoader(threading.Thread):
    #Static methods:
    def get_imageID(path):
        return int(os.path.basename(path)[0:5])

    def get_exposure(path):
        return float(os.path.basename(path)[9:-5])

    def load_unprocessed_raw(path):
        toReturn = None
        with rawpy.imread(path) as raw:
            toReturn = np.expand_dims(raw.raw_image_visible.astype(np.uint16),axis=2)
        return toReturn
    def load_processed_raw(path):
        toReturn = None
        with rawpy.imread(path) as raw:
            toReturn = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            toReturn = np.uint16(toReturn) #Store as uint16 to save memory
        return toReturn
    def patch_unprocessed_sony(inp,gt,ps):
        #Only start patches on even indicies
        (H,W,_) = inp.shape
        ps = 2*ps
        px = np.random.randint(0,(W-ps)/2)*2
        py = np.random.randint(0,(H-ps)/2)*2
        #burn three other randints to match seed
        np.random.randint(2, size=1)
        np.random.randint(2, size=1)
        np.random.randint(2, size=1)
        return (inp[py:py+ps,px:px+ps,:],gt[py:py+ps,px:px+ps,:])

    def patch_flip_unprocessed_sony(inp,gt,ps):
        (H,W,_) = inp.shape
        ps = 2*ps
        px = np.random.randint(0,(W-ps)/2)*2
        py = np.random.randint(0,(H-ps)/2)*2
        patch = inp[py:py+ps,px:px+ps,:]
        gt_patch = gt[py:py+ps,px:px+ps,:]
        patch_packed = SIDLoader.pack_raw_sony(patch)
        if np.random.randint(2, size=1)[0] == 1:  # random flip
            patch_packed = np.flip(patch_packed, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1:
            patch_packed = np.flip(patch_packed, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            patch_packed = np.transpose(patch_packed, ( 1, 0, 2))
            gt_patch = np.transpose(gt_patch, ( 1, 0, 2))
        patch = SIDLoader.unpack_packed_sony(patch_packed)
        return (patch, gt_patch)

    def patch_packed_sony(inp,gt,ps):
        (H,W,_) = inp.shape
        px = np.random.randint(0,W-ps)
        py = np.random.randint(0,H-ps)
        return (inp[py:py+ps,px:px+ps,:],gt[2*py:2*(py+ps),2*px:2*(px+ps),:])

    def patch_flip_packed_sony(inp,gt,ps):
        (H,W,_) = inp.shape
        px = np.random.randint(0,W-ps)
        py = np.random.randint(0,H-ps)
        patch = inp[py:py+ps,px:px+ps,:]
        gt_patch = gt[2*py:2*(py+ps),2*px:2*(px+ps),:]
        if np.random.randint(2, size=1)[0] == 1:  # random flip
            patch = np.flip(patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1:
            patch = np.flip(patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            patch = np.transpose(patch, ( 1, 0, 2))
            gt_patch = np.transpose(gt_patch, ( 1, 0, 2))
        return (patch,gt_patch)


    def pack_raw_sony(inp):
        (H, W, _) = inp.shape
        r = inp[0:H:2,0:W:2,:]
        g1 = inp[0:H:2,1:W:2,:]
        g2 = inp[1:H:2,0:W:2,:]
        b = inp[1:H:2,1:W:2,:]
        return np.concatenate((r,g1,g2,b),axis=2)

    def unpack_packed_sony(inp):
        (H, W, _) = inp.shape
        unpacked = np.zeros((2*H,2*W,1),np.int16)
        unpacked[0:2*H:2,0:2*W:2,0] = inp[0:H,0:W,0]
        unpacked[0:2*H:2,1:2*W:2,0] = inp[0:H,0:W,1]
        unpacked[1:2*H:2,0:2*W:2,0] = inp[0:H,0:W,2]
        unpacked[1:2*H:2,1:2*W:2,0] = inp[0:H,0:W,3]
        return unpacked


    #Instance methods ratios_or_exposure: 0 -> ratios otherwise exposure
    def __init__(self, data_path, data_type='Sony', preload_buff=2, patch_size=512, patch_per_image=1,image_per_batch=1, set_id='train', seed=29586, process_in=None, patch_fn=None, keep_raw=True, keep_gt=True, ratios_or_exposure=0, group=None, target=None, name=None, args=(), kwargs=None, daemon=True):
        super().__init__(group=group, target=target, name=name, args=args,kwargs=kwargs,daemon=daemon)
        self.data_type = data_type
        self.data_path = data_path+'/'+data_type+'/short/'
        self.gt_path = data_path+'/'+data_type+'/long/'
        self.extension = '.RAF'
        if(data_type.lower() == 'sony'):
            self.extension = '.ARW'
        self.file_prefix = '0'
        if(set_id.lower() == 'valid'):
            self.file_prefix = '2'
        elif(set_id.lower() == 'test'):
            self.file_prefix = '1'
        self.fileNames = glob.glob(self.data_path+self.file_prefix+'*'+self.extension) #Contains file names for location in image buffer
        self.gtFileNames = glob.glob(self.gt_path+self.file_prefix+'*'+self.extension) #Contain unique GT file names for location in gt buffer
        print('Number of inputs in "',set_id.lower(),'" set: \t',len(self.fileNames))
        print('Number of ground truths in "',set_id.lower(),'" set: \t',len(self.gtFileNames))
        self.mappings = [self.gtFileNames.index(self.get_gt_file(f)) for f in self.fileNames] #Contains corresponding GTs at same position as fileNames
        if(keep_raw):
            self.data = [None]*len(self.fileNames)
        if(keep_gt):
            self.gtData = [None]*len(self.gtFileNames)
        #Batchloading parameters
        self.keep_raw = keep_raw
        self.keep_gt = keep_gt
        self.interrupt = False
        self.seed = seed
        self.writeC = 0
        self.readC = 0
        self.writeEpoch = 0
        self.readEpoch = 0
        self.ppi = patch_per_image
        self.ipb = image_per_batch
        self.ps = patch_size
        self.locks = [None] * preload_buff
        for i in range(0,preload_buff):
            self.locks[i] = threading.Condition(lock=threading.Lock())
        self.buffer = [[None, None, None]] * preload_buff #0->in,1->gt,2->ratio
        self.bufferReadable = [False] * preload_buff
        self.nextWriteBuf = 0
        self.nextReadBuf = 0
        self.imagesInDataset = len(self.fileNames)
        self.process_in = process_in
        self.patch_fn = patch_fn
        self.ratios_or_exposure = ratios_or_exposure
        #Init random seed
        self.jump_to_epoch(0)
        self.iteration = None

    def run(self):
        self.jump_to_epoch(self.writeEpoch)
        self.update_iteration()
        try:
            while(not self.interrupt):
                lock = self.locks[self.nextWriteBuf]
                with lock:
                    #Wait until buffer is writable
                    lock.wait_for(self.buffer_is_writable)
                    if(self.interrupt):
                        return None
                    remaining_images = self.imagesInDataset - self.writeC
                    images_in_batch = self.ipb if self.ipb <= remaining_images else remaining_images
                    ins = []
                    gts = []
                    ratios = []
                    for image in range(0,images_in_batch):
                        curr_im = self.get_in(self.writeC+image)
                        curr_gt = self.get_gt(self.writeC+image)
                        if(self.ratios_or_exposure == 0):
                            curr_rat = self.get_ratio(self.writeC+image)
                        else:
                            curr_rat = self.get_inExp(self.writeC+image)
                        for patch in range(0,self.ppi):
                            im_patch = None
                            gt_patch = None
                            if(self.patch_fn is None):
                                im_patch = curr_im
                                gt_patch = curr_gt
                            else:
                                (im_patch,gt_patch) = self.patch_fn(curr_im,curr_gt,self.ps)
                            ins.append([im_patch])
                            gts.append([gt_patch])
                            ratios.append([curr_rat])
                    self.buffer[self.nextWriteBuf][0] = np.concatenate(ins,axis=0)
                    self.buffer[self.nextWriteBuf][1] = np.concatenate(gts,axis=0)
                    self.buffer[self.nextWriteBuf][2] = np.concatenate(ratios,axis=0)
                    self.writeC = self.writeC+images_in_batch
                    if(self.writeC == self.imagesInDataset):
                        self.writeEpoch = self.writeEpoch+1
                        self.writeC = 0
                        self.jump_to_epoch(self.writeEpoch)
                        self.update_iteration()
                    self.buffer_is_written()
                    lock.notify()
        except KeyboardInterrupt:
            print('Keyboard interrupt accepted - Best Regards SID Loader')
        finally:
            self.buffer_is_written()
            if(lock is not None):
                lock.notify()


    def load_and_proc_in(self,path):
        toReturn = None
        if (self.process_in is not None):
            toReturn = self.process_in(SIDLoader.load_unprocessed_raw(path))
        else:
            toReturn = SIDLoader.load_unprocessed_raw(path)
        return toReturn


    #Verify if input is loaded, if not then load
    def get_in(self,inID):
        inID = self.iteration[inID]
        path = self.fileNames[inID]
        if(self.keep_raw):
            if(self.data[inID] is None):
                self.data[inID] = self.load_and_proc_in(path)
            return self.data[inID]
        else:
            temp = self.load_and_proc_in(path)
            #temp = np.array(temp,np.float32)
            #temp = np.maximum(0,temp-512)
            #print(np.mean(temp)/(2**14-1-512),'\t\t',np.amin(temp)/(2**14-1-512),'\t\t',np.amax(temp)/(2**14-1-512),'\t\t\t\t',self.get_ratio(inID))
            return temp
    #Verify if GT is loaded, if not then load
    def get_gt(self,inID):
        inID = self.iteration[inID]
        mapID = self.mappings[inID]
        path = self.gtFileNames[mapID]
        if(self.keep_gt):
            if(self.gtData[mapID] is None):
                self.gtData[mapID] = SIDLoader.load_processed_raw(path)
            return self.gtData[mapID]
        else:
            return SIDLoader.load_processed_raw(path)

    def get_ratio(self,inID):
        inID = self.iteration[inID]
        in_exp = SIDLoader.get_exposure(self.fileNames[inID])
        gt_exp = SIDLoader.get_exposure(self.gtFileNames[self.mappings[inID]])
        return min(gt_exp/in_exp,300.0)

    def get_inExp(self,inID):
        inID = self.iteration[inID]
        in_exp = SIDLoader.get_exposure(self.fileNames[inID])
        return in_exp

    def get_gt_file(self, infile):
        return glob.glob(self.gt_path + ('%05d_00*'+self.extension) % SIDLoader.get_imageID(infile))[0]

    def jump_to_epoch(self, epoch):
        if(self.file_prefix == '0'):
            np.random.seed(self.seed+epoch)
            tf.set_random_seed(self.seed+epoch)

    def update_iteration(self):
        if(self.file_prefix == '0'):
            self.iteration = np.random.permutation(len(self.fileNames))
        else:
            self.iteration = range(0,len(self.fileNames))

    def buffer_is_read(self):
        curr_buff = self.nextReadBuf
        self.buffer[curr_buff] = [None,None,None]
        self.nextReadBuf = (self.nextReadBuf+1)%len(self.bufferReadable)
        self.bufferReadable[curr_buff] = False

    def buffer_is_written(self):
        curr_buff = self.nextWriteBuf
        self.nextWriteBuf = (self.nextWriteBuf+1)%len(self.bufferReadable)
        self.bufferReadable[curr_buff] = True

    def buffer_is_readable(self):
        return (self.bufferReadable[self.nextReadBuf] or self.interrupt)

    def buffer_is_writable(self):
        return (not self.bufferReadable[self.nextWriteBuf] or self.interrupt)

    def get_batch(self):
        toReturn = [None,None,None]
        lock = self.locks[self.nextReadBuf]
        with lock:
            lock.wait_for(self.buffer_is_readable)
            if(self.interrupt):
                return None
            toReturn[0] = np.copy(self.buffer[self.nextReadBuf][0])
            toReturn[1] = np.copy(self.buffer[self.nextReadBuf][1])
            toReturn[2] = np.copy(self.buffer[self.nextReadBuf][2])
            self.readC = self.readC + toReturn[0].shape[0]/self.ppi
            if(self.readC == self.imagesInDataset):
                self.readC = 0
                self.readEpoch = self.readEpoch+1
            self.buffer_is_read()
            lock.notify()
        return toReturn

    def stop(self):
        self.interrupt = True
