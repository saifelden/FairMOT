import numpy as np
import cv2
from einops import rearrange
import torch
from torch import nn
import fiftyone as fo
import random
from PIL import Image
from PIL import ImageFilter


class CocoDataLoader():

    def __init__(self,classes=['person','car'],split='train',number_of_samples = 5000,batch_size = 32,shuffle = True,input_dim = (512,512),stride_scale = 4):

        self.dataset = fo.zoo.load_zoo_dataset("coco-2017",classes = classes,split=split)
        self.dataset.persistent = True
        self.used_dataset = self.dataset.take(number_of_samples)
        self.image_paths = [sample.filepath for sample in self.used_dataset.iter_samples()]
        self.annotations = []
        self.classes = classes
        for annotation in self.used_dataset.iter_samples():
            curr_annotation = []
            for det in annotation.ground_truth.detections:
                curr_det = {}
                if det['label'] not in classes:
                    continue
                curr_det['label'] = det['label']
                curr_det['bbox'] = det['bounding_box']
                curr_annotation.append(curr_det)
            self.annotations.append(curr_annotation)
        self.batch_size=batch_size
        self.number_of_samples= number_of_samples
        self.shuffle = shuffle
        self.batch_index = 0
        self.number_of_batches = int(self.number_of_samples/self.batch_size)
        self.input_dim = input_dim
        self.stride_scale = stride_scale
        self.output_height = int(self.input_dim[0]/self.stride_scale)
        self.output_width = int(self.input_dim[1]/self.stride_scale)
        self.label_map = {}
        for i in range(len(self.classes)):
            self.label_map[self.classes[i]]=i




    def next_batch(self):

        if self.batch_index+self.batch_size < self.number_of_samples:
            impath_batch = self.image_paths[self.batch_index:self.batch_index+self.batch_size]
            annotation_batch = self.annotations[self.batch_index:self.batch_index+self.batch_size]
        else:
            impath_batch = self.image_paths[self.batch_index:]
            annotation_batch = self.annotations[self.batch_index:]
        
        self.batch_index+=self.batch_size

        images_batch = []
        org_size = []
        for impath in impath_batch:
            img = cv2.imread(impath)
            height,width,_ = img.shape
            org_size.append((height,width))
            img = cv2.resize(img,(self.input_dim[1],self.input_dim[0]))
            images_batch.append(img)
        
        images_batch = np.array(images_batch)
        self.channel_size = images_batch.shape[-1]


        self.generate_gt_heatmap(annotation_batch,org_size)
        
        self.generate_gt_offset(annotation_batch,org_size)
        self.generate_gt_bbox_size(annotation_batch,org_size)
        self.apply_gaussian_filter_v2()

        self.generate_gt_bbox(images_batch)
        self.heatmap = rearrange(self.heatmap,'b h w c -> b c h w').astype(np.float32)
        self.offset = rearrange(self.offset,'b h w c -> b c h w').astype(np.float32)
        self.bbox_size/=np.array([self.output_height,self.output_width])
        self.bbox_size = rearrange(self.bbox_size,'b h w c  -> b c h w').astype(np.float32)
        return rearrange(images_batch,'b h w c -> b c h w').astype(np.float32)/255,self.heatmap,self.offset,self.bbox_size


    def generate_gt_heatmap(self,annotation_batch,org_size):

        self.heatmap = np.zeros([self.batch_size,self.output_width,self.output_height,len(self.classes)],dtype=np.float32)
        for i in range(len(annotation_batch)):
            for j in range(len(annotation_batch[i])):
                bbox = annotation_batch[i][j]['bbox']
                label = annotation_batch[i][j]['label']
                org_bbox = [bbox[0]*self.input_dim[1],bbox[1]*self.input_dim[0],bbox[2]*self.input_dim[1],bbox[3]*self.input_dim[0]]
                output_bbox = (np.array(org_bbox,dtype=np.float32)/self.stride_scale).astype(np.int32)
                label_id = self.label_map[label]
                center_x = int(output_bbox[0]+(output_bbox[2]/2))
                center_y = int(output_bbox[1]+(output_bbox[3]/2))

                self.heatmap[i][center_y][center_x][label_id]=255

    def apply_gaussian_filter_v1(self):


        for i in range(self.heatmap.shape[0]):
            for j in range(len(self.classes)):
                # image = Image.fromarray(self.heatmap[i,:,:,j].reshape([self.output_height,self.output_width,1]))
                img = cv2.blur(self.heatmap[i,:,:,j].reshape([self.output_height,self.output_width,1]),(10,10))
                self.heatmap[i,:,:,j] = img.reshape([self.output_height,self.output_width])

    def apply_gaussian_filter_v2(self,filter=(3,3)):

        for i in range(self.heatmap.shape[0]):
            for j in range(len(self.classes)):
                indices = np.where(self.heatmap[i,:,:,j]==255)
                if type(indices)==tuple:

                    size = indices[0].shape[0]
                    for k in range(size):
                        point = (indices[0][k],indices[1][k])
                        bb_size = self.bbox_size[i,point[0],point[1]]
                        if bb_size[0]==0. or bb_size[1]==0.:
                            continue
                        for m in range(max(point[0]-int(bb_size[0]/filter[0]),0),min(point[0]+int(bb_size[0]/filter[0])+1,self.heatmap.shape[1])):
                            for n in range(max(point[1]-int(bb_size[1]/filter[1]),0),min(point[1]+int(bb_size[1]/filter[1])+1,self.heatmap.shape[2])):
                                sub = int((((m - point[0])*(255/(bb_size[0]/filter[0])))**2+((n - point[1])*(255/(bb_size[1]/filter[1])))**2)**0.5)

                                self.heatmap[i,m,n,j]=max(255-sub,self.heatmap[i,m,n,j])
                                





    def generate_gt_offset(self,annotation_batch,org_size):

        self.offset = np.zeros([self.batch_size,self.output_width,self.output_height,2],dtype=np.int8)

        for i in range(len(annotation_batch)):
            for j in range(len(annotation_batch[i])):
                bbox = annotation_batch[i][j]['bbox']
                org_bbox = [bbox[0]*self.input_dim[1],bbox[1]*self.input_dim[0],bbox[2]*self.input_dim[1],bbox[3]*self.input_dim[0]]
                output_bbox =  (np.array(org_bbox,dtype=np.float32)/self.stride_scale).astype(np.int32)
                output_center_x = int(output_bbox[0]+(output_bbox[2]/2))
                output_center_y = int(output_bbox[1]+(output_bbox[3]/2))
                org_center_x = int(org_bbox[0]+(org_bbox[2]/2))
                org_center_y = int(org_bbox[1]+(org_bbox[3]/2))
                offset_x = org_center_x - (output_center_x*self.stride_scale)
                offset_y = org_center_x - (output_center_y*self.stride_scale)        
                self.offset[i][output_center_y][output_center_x][0]=offset_x/self.stride_scale
                self.offset[i][output_center_y][output_center_x][1]=offset_y/self.stride_scale

    
    def generate_gt_bbox_size(self,annotation_batch,org_size):

        self.bbox_size = np.zeros([self.batch_size,self.output_width,self.output_height,2],dtype=np.float32)

        for i in range(len(annotation_batch)):
            for j in range(len(annotation_batch[i])):
                bbox = annotation_batch[i][j]['bbox']
                label = annotation_batch[i][j]['label']
                org_bbox = [bbox[0]*self.input_dim[1],bbox[1]*self.input_dim[0],bbox[2]*self.input_dim[1],bbox[3]*self.input_dim[0]]
                output_bbox =  (np.array(org_bbox,dtype=np.float32)/self.stride_scale).astype(np.int32)
                output_center_x = int(output_bbox[0]+(output_bbox[2]/2))
                output_center_y = int(output_bbox[1]+(output_bbox[3]/2))
                label_id = self.label_map[label]
                if output_bbox[3]<=0. or output_bbox[2]<=0.:
                    continue
                self.bbox_size[i][output_center_y][output_center_x][0]=output_bbox[3]
                self.bbox_size[i][output_center_y][output_center_x][1]=output_bbox[2]

    def generate_gt_bbox(self,image_batch):

        for i in range(image_batch.shape[0]):
            cv2.imwrite('gt_img_'+str(i)+'.jpg',image_batch[i])
            for j in range(len(self.classes)):
                cv2.imwrite('hm_'+str(i)+'_'+str(j)+'.jpg',self.heatmap[i,:,:,j])
            

    def on_epoch_ends(self):

        self.batch_index=0
        if self.shuffle:
            ind_list = list(range(self.number_of_samples))
            random.shuffle(ind_list)
            self.image_paths = list(np.array(self.image_paths)[ind_list])
            self.annotations = list(np.array(self.annotations)[ind_list])


if __name__=='__main__':
    train_dl = CocoDataLoader(split='validation',number_of_samples = 5000)
    batch_data = train_dl.next_batch()
    train_dl.on_epoch_ends()
    import ipdb;ipdb.set_trace()
    


        




