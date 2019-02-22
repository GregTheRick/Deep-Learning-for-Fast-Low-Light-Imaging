from __future__ import division
import numpy as np
import tensorflow as tf
import threading
import os, glob
from PIL import Image

class Experiment:
    def preprocess_step(model,x,y,ratio):
        sess = model['sess']
        pre_x = model['pre_x']
        pre_y = model['pre_y']
        feed_dict = {model['x']:x,model['y']:y,model['ratio']:ratio}
        return sess.run([pre_x, pre_y],feed_dict=feed_dict)

    def infer_step(model,pre_x,pre_y=None,train=False,get_loss=False,lr=None):
        sess = model['sess']
        if(not train and not get_loss):
            feed_dict = {model['pre_x_pl']:pre_x}
            return sess.run([model['image']],feed_dict=feed_dict)
        elif(not train and get_loss):
            loss = model['loss']
            feed_dict = {model['pre_x_pl']:pre_x,model['pre_y_pl']:pre_y}
            return sess.run([loss,model['image']],feed_dict=feed_dict)
        else:
            if(pre_y is None):
                raise ValueError('During training we need to know the targets... However they were not provided.')
            if(lr is None):
                raise ValueError('No Learning rate supplied! How can we train like this?')
            #When training always get loss
            loss = model['loss']
            optim = model['optim']
            step = model['step']
            feed_dict = {model['pre_x_pl']:pre_x,model['pre_y_pl']:pre_y,model['lr']:lr}
            [l,out_im,_,_] = sess.run([loss,model['image'],step,optim],feed_dict=feed_dict)
            return [l,out_im]

    def get_perf_step(model,out_im,pre_y,loss=None):
        sess = model['sess']
        psnr = model['psnr']
        ssim = model['ssim']
        ms_ssim = model['ms_ssim']
        if(loss is None):
            feed_dict = {model['out_im_pl']:out_im,model['pre_y_pl']:pre_y}
            return sess.run([psnr,ssim,ms_ssim],feed_dict=feed_dict)
        else:
            feed_dict = {model['out_im_pl']:out_im,model['pre_y_pl']:pre_y,model['loss_pl']:loss}
            return sess.run([psnr,ssim,ms_ssim,model['step_sum']],feed_dict=feed_dict)



    def __init__(self, name='experiment',model_fn={'fn':None,'loss_fn':None},device="/device:GPU:0",tensorboard_dir='/scratch/s162994/tensorboard',checkpoint_dir='/scratch/s162994/chk',seed=1337,dataset=None,save_images=True):
        checkpoint_dir = checkpoint_dir+'/%s/' % (name)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        self.seed =seed
        self.model_fn = model_fn
        self.perf = {'ssim':[],'ms_ssim':[],'psnr':[]}
        self.train_loss = []
        self.vperf = {'ssim':[],'ms_ssim':[],'psnr':[]}
        self.valid_loss = []
        self.saver = None
        tensorboard_dir = tensorboard_dir + '/%s' % (self.name)
        self.tensorboard_dir = tensorboard_dir
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        self.count = -1
        self.epoch = 0
        self.train_thread = None
        self.valid_thread = None
        self.model = None
        self.create_graph_and_sess(device)
        self.trainWriter = None
        self.validWriter = None
        self.dataset = dataset
        self.save_images = save_images

    def init_and_restore_if_possible(self):
        self.model['sess'].run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt:
            print('Checkpoint: ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.model['sess'], ckpt.model_checkpoint_path)
            #Check number of latest epoch
            epochfolders = glob.glob(self.checkpoint_dir+'e_*')
            epoch = 0
            for fold in epochfolders:
                epoch = np.maximum(epoch,int(fold[-4:]))
            print('Experiment '+self.name+' has been restored to epoch '+str(epoch))
            self.epoch = epoch

    def create_writer(self,path,sess):
        return tf.summary.FileWriter(path,sess.graph)

    def create_test_writer(self):
        self.validWriter = self.create_writer(self.tensorboard_dir+'/test',self.model['sess'])
        self.count = 1
        self.vperf = {'ssim':[],'ms_ssim':[],'psnr':[]}
        self.valid_loss = []

    def create_graph_and_sess(self,device):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(self.seed)
            model = {}
            if(self.model_fn['fn'] is not None):
                with tf.device(device):
                    #Switchable loss function and defaulting
                    if('loss_fn' in self.model_fn and self.model_fn['loss_fn'] is not None):
                        model = self.model_fn['fn'](loss_fn=self.model_fn['loss_fn'])
                    else:
                        model = self.model_fn['fn']()
            model['sess'] = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver = tf.train.Saver(save_relative_paths=True)
            self.model = model
            self.init_and_restore_if_possible()

    def train_action(self,x,y,ratio,lr):
        train_thread = Experiment.ParalellTrainer(self.model,x,y,ratio,lr)
        train_thread.start()
        self.train_thread = train_thread

    def finish_train_action(self):
        train_thread = self.train_thread
        train_thread.join()
        self.train_loss.append(train_thread.res['loss'])
        self.perf['psnr'].append(train_thread.res['psnr'])
        self.perf['ssim'].append(train_thread.res['ssim'])
        self.perf['ms_ssim'].append(train_thread.res['ms_ssim'])

    def begin_train_cycle(self):
        self.trainWriter = self.create_writer(self.tensorboard_dir+'/train',self.model['sess'])

    def finish_train_cycle(self):
        if(self.trainWriter is not None):
            self.trainWriter.close()
            self.trainWriter = None

    def begin_valid_cycle(self,epoch):
        self.validWriter = self.create_writer(self.tensorboard_dir+'/'+str(epoch)+'/valid',self.model['sess'])
        self.count = 1


    def valid_action(self,x,y,ratio):
        valid_thread = Experiment.ParalellValidator(self.model,x,y,ratio)
        valid_thread.start()
        self.valid_thread = valid_thread

    def finish_valid_action(self):
        valid_thread = self.valid_thread
        valid_thread.join()
        self.valid_loss.append(valid_thread.res['loss'])
        self.vperf['psnr'].append(valid_thread.res['psnr'])
        self.vperf['ssim'].append(valid_thread.res['ssim'])
        self.vperf['ms_ssim'].append(valid_thread.res['ms_ssim'])
        self.validWriter.add_summary(valid_thread.res['summ'],self.count)
        self.count = self.count+1

    def test_action(self,x,y,ratio):
        valid_thread = Experiment.ParalellTester(self.model,x,y,ratio)
        valid_thread.start()
        self.valid_thread = valid_thread

    def finish_test_action(self):
        valid_thread = self.valid_thread
        valid_thread.join()
        self.valid_loss.append(valid_thread.res['loss'])
        self.vperf['psnr'].append(valid_thread.res['psnr'])
        self.vperf['ssim'].append(valid_thread.res['ssim'])
        self.vperf['ms_ssim'].append(valid_thread.res['ms_ssim'])
        self.validWriter.add_summary(valid_thread.res['summ'],self.count)
        self.count = self.count+1
        if(self.save_images):
            #Save image
            if not os.path.isdir(self.checkpoint_dir+'test/'):
                os.makedirs(self.checkpoint_dir+'test/')
            #Save results to checkpoints
            #Aslo save GT
            im_name = ('%04d_res_'+self.name+'.png') % (self.count-1)
            if(self.dataset is not None):
                im_name = os.path.basename(self.dataset.fileNames[self.count-2])+'_result_'+self.name+'.png'
            im_pairs = np.concatenate((valid_thread.res['out_im'][0],valid_thread.y[0]/(2**16)),axis=0)
            Image.fromarray((im_pairs*255).astype('uint8')).save(self.checkpoint_dir+'test/'+im_name)

    def end_of_epoch_train(self,epoch):
        loss = np.mean(self.train_loss)
        psnr = np.mean(self.perf['psnr'])
        ssim = np.mean(self.perf['ssim'])
        mssim = np.mean(self.perf['ms_ssim'])
        [e_sum] = self.model['sess'].run([self.model['t_epoch_sum']],feed_dict={self.model['t_epoch_loss_pl']:loss, self.model['t_epoch_psnr_pl']:psnr, self.model['t_epoch_ssim_pl']:ssim, self.model['t_epoch_mssim_pl']:mssim})
        self.trainWriter.add_summary(e_sum,epoch)
        self.train_loss = []
        self.perf = {'ssim':[],'ms_ssim':[],'psnr':[]}
        self.saver.save(self.model['sess'],self.checkpoint_dir + self.name + '.ckpt',global_step=self.model['gs'].eval(session=self.model['sess']))
        #Save epoch number with directories
        os.makedirs(self.checkpoint_dir + 'e_%04d' % epoch)
        #remove previous dir if present
        if(os.path.isdir(self.checkpoint_dir + 'e_%04d' % (epoch-1))):
            os.rmdir((self.checkpoint_dir + 'e_%04d' % (epoch-1)))
        print('Train loss exp '+self.name+' for epoch', str(epoch),' is: ',loss)
        print('Train perf exp '+self.name+' for epoch', str(epoch),' ssim: ',ssim,' ms_ssim: ',mssim,' psnr: ',psnr)

    def end_of_epoch_valid(self,epoch):
        loss = np.mean(self.valid_loss)
        psnr = np.mean(self.vperf['psnr'])
        ssim = np.mean(self.vperf['ssim'])
        mssim = np.mean(self.vperf['ms_ssim'])
        [e_sum] = self.model['sess'].run([self.model['v_epoch_sum']],feed_dict={self.model['v_epoch_loss_pl']:loss, self.model['v_epoch_psnr_pl']:psnr, self.model['v_epoch_ssim_pl']:ssim, self.model['v_epoch_mssim_pl']:mssim})
        self.trainWriter.add_summary(e_sum,epoch)
        self.vperf = {'ssim':[],'ms_ssim':[],'psnr':[]}
        self.valid_loss = []
        self.validWriter.close()
        print('Valid loss exp '+self.name+' for epoch', str(epoch),' is: ',loss)
        print('Valid perf exp '+self.name+' for epoch', str(epoch),' ssim: ',ssim,' ms_ssim: ',mssim,' psnr: ',psnr)

    def end_of_epoch_test(self):
        loss = np.mean(self.valid_loss)
        psnr = np.mean(self.vperf['psnr'])
        ssim = np.mean(self.vperf['ssim'])
        mssim = np.mean(self.vperf['ms_ssim'])
        [e_sum] = self.model['sess'].run([self.model['v_epoch_sum']],feed_dict={self.model['v_epoch_loss_pl']:loss, self.model['v_epoch_psnr_pl']:psnr, self.model['v_epoch_ssim_pl']:ssim, self.model['v_epoch_mssim_pl']:mssim})
        self.validWriter.add_summary(e_sum,0)
        self.vperf = {'ssim':[],'ms_ssim':[],'psnr':[]}
        self.valid_loss = []
        self.validWriter.close()
        print('Test loss exp '+self.name+' is: ',loss)
        print('Test perf for ', self.name, ' ssim: ',ssim,' ms_ssim: ',mssim,' psnr: ',psnr)


    class ParalellTrainer(threading.Thread):
        def __init__(self,model,x,y,ratio,lr,group=None, target=None, name=None, args=(), kwargs=None, daemon=None):
            super().__init__(group=group, target=target, name=name, args=args,kwargs=kwargs,daemon=daemon)
            self.model = model
            self.res = {}
            self.lr = lr
            self.x = x
            self.y = y
            self.ratio = ratio
        
        def run(self):
            #preprocess in separate run to save mem
            [pre_x,pre_y] = Experiment.preprocess_step(self.model,self.x,self.y,self.ratio)
            #Get inference and training
            [loss,out_im] = Experiment.infer_step(self.model,pre_x,pre_y,train=True,lr=self.lr)
            #Get performance in separate run to save mem on GPU
            [psnr,ssim,msssim] = Experiment.get_perf_step(self.model,out_im,pre_y)
            self.res['loss'] = loss
            self.res['psnr'] = psnr
            self.res['ssim'] = ssim
            self.res['ms_ssim'] = msssim

    class ParalellValidator(threading.Thread):
        def __init__(self,model,x,y,ratio,group=None, target=None, name=None, args=(), kwargs=None, daemon=None):
            super().__init__(group=group, target=target, name=name, args=args,kwargs=kwargs,daemon=daemon)
            self.model = model
            self.res = {}
            self.x = x
            self.y = y
            self.ratio = ratio
        
        def run(self):
            #preprocess in separate run to save mem
            [pre_x,pre_y] = Experiment.preprocess_step(self.model,self.x,self.y,self.ratio)
            [loss,out_im] = Experiment.infer_step(self.model,pre_x,pre_y,train=False,get_loss=True)
            [p,s,ms,summ] = Experiment.get_perf_step(self.model,out_im,pre_y,loss=loss)
            self.res['loss'] = loss
            self.res['psnr'] = p
            self.res['ssim'] = s
            self.res['ms_ssim'] = ms
            self.res['summ'] = summ

    class ParalellTester(threading.Thread):
        def __init__(self,model,x,y,ratio,group=None, target=None, name=None, args=(), kwargs=None, daemon=None):
            super().__init__(group=group, target=target, name=name, args=args,kwargs=kwargs,daemon=daemon)
            self.model = model
            self.res = {}
            self.x = x
            self.y = y
            self.ratio = ratio
        
        def run(self):
            #preprocess in separate run to save mem
            [pre_x,pre_y] = Experiment.preprocess_step(self.model,self.x,self.y,self.ratio)
            [loss,out_im] = Experiment.infer_step(self.model,pre_x,pre_y,train=False,get_loss=True)
            [p,s,ms,summ] = Experiment.get_perf_step(self.model,out_im,pre_y,loss=loss)
            self.res['loss'] = loss
            self.res['psnr'] = p
            self.res['ssim'] = s
            self.res['ms_ssim'] = ms
            self.res['summ'] = summ
            self.res['out_im'] = out_im



