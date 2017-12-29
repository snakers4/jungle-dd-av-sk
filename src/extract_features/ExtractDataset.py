# import torch
import torch.utils.data as data

import moviepy.editor as mpe
import numpy as np 
from random import randint
from numpy import random
import os
import torch
from itertools import repeat
    
class ExtractDataset(data.Dataset):
    def __init__(self,
                 root,
                 video_dict,
                 transform=None,
                 dataset_name='micro_chimps',
                 fps = 2,
                 target_size = (64,64)):
        self.root = root
        self.video_dict = video_dict
        self.transform = transform
        self.name = dataset_name
        self.video_ids = list(self.video_dict.keys())
        self.fps = fps
        self.target_size = target_size
        

        unique_values = sorted(list(set([x for x in self.video_dict.values()])))
        print(unique_values)
        self.idx_2_value = {i:value for i,value in enumerate(unique_values) }
        self.value_2_idx = {value:i for i,value in enumerate(unique_values) }
        del unique_values

    def __getitem__(self, index):
        video_id = self.video_ids[index]
    
        try:
            clip = mpe.VideoFileClip(filename = os.path.join(self.root, video_id),
                                    audio = False,
                                    target_resolution = (self.target_size[0],self.target_size[1]))

            video_matrix = None
            
            frame_counter = 0
            # loop over the extracted images 
            for im in clip.iter_frames(fps=self.fps): 
                
                # Videos mostly give 31 frames, but sometimes 30
                if frame_counter < self.fps * 15:
                    im = im[np.newaxis, ...]
                    if video_matrix is None:

                        # write the size of the first frame to the file
                        # will not work because the video is already resized
                        # with open("video_sizes.csv", "a") as myfile:
                        #    myfile.write('{},{},{}'.format(video_id,im.shape[0],im.shape[1] ))

                        video_matrix = im
                    else:
                        video_matrix = np.append(video_matrix,im, axis=0)
                else:
                    # break out of the loop if there are more than 15 * fps frames
                    break
                    
                frame_counter += 1
            
            if frame_counter != self.fps * 15:
                print ('Short video detected - {} - {} frames'.format(video_id,str(frame_counter)) )
                
                # write the size of the first frame to the file
                with open("short_videos.csv", "a") as myfile:
                    myfile.write('{},{}\n'.format(video_id,frame_counter ))
                    
                for append_frames in repeat(None, self.fps*15 - frame_counter):
                    # append the last image number of times we need to get 30 frames
                    video_matrix = np.append(video_matrix,im, axis=0)

            video_matrix = video_matrix.astype(np.float32) 
            
            if self.transform is not None:
                for i,im in enumerate(video_matrix):
                    video_matrix[i,...] = self.transform(im)
     
        except Exception as e:
            print ('Video open exception triggered with video {}'.format(video_id))
            print(str(e))

            # write the size of the first frame to the file
            with open("broken_videos.csv", "a") as myfile:
                myfile.write('{},{}\n'.format(video_id,str(e) ))
                        
            video_matrix = random.random((self.fps * 15,self.target_size[0],self.target_size[1],3))
            video_matrix = video_matrix.astype(np.float32)            
            
            if self.transform is not None:
                for i,im in enumerate(video_matrix):
                    video_matrix[i,...] = self.transform(im)          

        output = torch.stack ( [( torch.from_numpy(img.transpose((2, 0, 1))) ) for img in video_matrix] )
        
        # for extracting numpy datasets
        gt = video_id
        
        # for e2e bce image pipeline
        # gt = self.value_2_idx[self.video_dict[video_id]]
        # gt_one_hot = np.eye(len(self.value_2_idx))[ np.array(gt).reshape(-1) ][0]
        # gt_one_hot = torch.from_numpy(gt_one_hot)            
    
        return output,gt

    def __len__(self):
        return len(self.video_ids)