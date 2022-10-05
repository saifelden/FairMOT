import cv2
import torch
from torch import nn
import numpy as np
from model import DetHead,BackBone
from coco_dataloader import CocoDataLoader
from torch.optim import Adam as adam
import numpy as np 
from tqdm import tqdm
import sys
from glob import glob
from einops import rearrange 
class CenterNet():
    def __init__(self,classes=['person','car','bicycle'],batch_size = 32,learning_rate =0.001,save_epochs=2,mode='train'):

        self.detection_model = BackBone().to('cuda')

        self.classes = classes
        self.save_epochs=save_epochs
        self.loss = nn.L1Loss()
        self.ht_weight_loss = torch.tensor(0.5)
        self.offset_weight_loss = torch.tensor(0.3)
        self.bbsize_weight_loss = torch.tensor(0.2)
        self.params = list(self.detection_model.parameters())
        self.params.extend([self.ht_weight_loss,self.offset_weight_loss,self.bbsize_weight_loss])
        self.optimizer = adam(list(self.detection_model.parameters()),lr=learning_rate)
        if mode=='train':
            self.train_dl = CocoDataLoader(classes=classes,split='train',number_of_samples = 5000,batch_size = batch_size)
        elif mode =='test':
            self.test_dl = CocoDataLoader(classes=classes,split='validation',number_of_samples = 1000,batch_size = batch_size)
        self.model_name = ''
        self.iteration_num=0


    def set_model_name(self,model_name):
        self.model_name = model_name


    def train(self,number_of_epochs):
        
        
        for i in tqdm(range(number_of_epochs)):
            avg_loss = 0
            for j in range(self.train_dl.number_of_batches):
                image_batch,heatmap_gt,offset_gt,bbox_size_gt = self.train_dl.next_batch()
                self.detection_model.zero_grad()
                image_batch = torch.tensor(image_batch).to('cuda')
                heatmap_predicted,offset_predicted,bbox_size_predicted = self.detection_model(image_batch)
                ht_loss = self.loss(heatmap_predicted,torch.tensor(heatmap_gt).to('cuda'))
                offset_loss = self.loss(offset_predicted,torch.tensor(offset_gt).to('cuda'))
                bbsize_loss = self.loss(bbox_size_predicted,torch.tensor(bbox_size_gt).to('cuda'))
                all_loss = self.ht_weight_loss*ht_loss+self.offset_weight_loss*offset_loss+self.bbsize_weight_loss*bbsize_loss
                all_loss.backward()
                avg_loss+=all_loss
                self.optimizer.step()
                if j%25==0:
                    print('current moving avg_Loss  at',j,'is:',avg_loss/(j+1))

            if i%self.save_epochs==0:

                print('current avg_Loss is:',avg_loss/self.train_dl.number_of_batches)
                torch.save(self.detection_model.state_dict(),'model_checkpoints/model_'+self.model_name+'_'+str(self.iteration_num+i))

            self.train_dl.on_epoch_ends()
        self.iteration_num+=number_of_epochs


    def test(self):

        with torch.no_grad():
            self.detection_model.eval()
            for j in range(self.test_dl.number_of_batches):
                image_batch,heatmap_gt,offset_gt,bbox_size_gt = self.test_dl.next_batch()
                image_batch = torch.tensor(image_batch).to('cuda')
                heatmap_predicted,offset_predicted,bbox_size_predicted = self.detection_model(image_batch)
                ht_loss = self.loss(heatmap_predicted,torch.tensor(heatmap_gt).to('cuda'))
                offset_loss = self.loss(offset_predicted,torch.tensor(offset_gt).to('cuda'))
                bbsize_loss = self.loss(bbox_size_predicted,torch.tensor(bbox_size_gt).to('cuda'))
                all_loss = self.ht_weight_loss*ht_loss+self.offset_weight_loss*offset_loss+self.bbsize_weight_loss*bbsize_loss
                print('current batch loss is',all_loss)

        self.test_dl.on_epoch_ends()


    def load_saved_model(self,checkpoint):

        self.iteration_num=int(checkpoint.split('_')[-1])
        if self.model_name=='':
            checkpoint_path = 'model_checkpoints/model_'+checkpoint
        else:
            checkpoint_path = 'model_checkpoints/model_'+self.model_name+'_'+checkpoint

        try:
            self.detection_model.load_state_dict(torch.load(checkpoint_path))
        except:
            print('Invalid chekpoint at path',checkpoint_path)
            print('loading Random Initialized weights.....')


    def predict_bbox(self,folder_imgs,confthresh=0.9):

        imgs_files = glob(folder_imgs+'/*')
        for img_path in imgs_files:
            img = cv2.imread(img_path)
            img = cv2.resize(img,(512,512)).astype(np.float32)/255
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).reshape(1,img.shape[0],img.shape[1],img.shape[2])
            img = rearrange(img,'b h w c -> b c h w')
            with torch.no_grad():
                self.detection_model.eval()
                img = torch.tensor(img).to('cuda')
                heatmap_predicted,offset_predicted,bbox_size_predicted = self.detection_model(img)
                heatmap_indexes = heatmap_predicted>confthresh
                heatmap_predicted = heatmap_indexes*heatmap_predicted
                heatmap_predicted*=255
                heatmap_predicted = heatmap_predicted.to('cpu').numpy().reshape(3,128,128)
                heatmap_predicted = rearrange(heatmap_predicted,'c h w -> h w c')
                
                for i in range(len(self.classes)):
                    img_name = img_path.split('/')[-1].split('.jpg')[0]
                    # import ipdb;ipdb.set_trace()
                    cv2.imwrite(img_name+'_hm_'+str(i)+'.jpg',heatmap_predicted[:,:,i])


if __name__ =='__main__':

    action = sys.argv[1]
    model = CenterNet(mode = action)   
    checkpoint = sys.argv[2]
    if checkpoint!= '-1':
        model.load_saved_model(checkpoint)
        model.set_model_name(sys.argv[2].split('_')[0])
    else:
        model.set_model_name(sys.argv[4])
    if action =='train':
        number_of_epochs = int(sys.argv[3])
        model.train(number_of_epochs)
    elif action=='test':
        model.test()

    elif action=='predict':
        image_files = sys.argv[3]
        model.predict_bbox(image_files,confthresh=0.9)





