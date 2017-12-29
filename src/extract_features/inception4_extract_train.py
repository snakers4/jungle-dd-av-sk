# standard pytorch imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# standard ds and os tools
import pandas as pd
import numpy as np
import tqdm
import os

# custom written classes and imports
from ExtractDataset import ExtractDataset
from ExtractAugs import Normalize
from torchvision.transforms import Compose

from InceptionResnetv2 import inceptionresnetv2
from Inception4Cadene import inceptionv4

# garbage collector
import gc

# stupid hack - add params instead of doing a proper CLI parsing
# also works with jupyter notebooks
args = type('test', (), {})()
args.finetune = False
args.resume = False
args.batch_size = 4
args.workers = 4
args.lognumber = 'inception4_train'
args.save_path = '../../data/interim/inception4_train'
args.numclasses = 23
args.arch = 'inception4'
args.evaluate = False
args.data = '../../data/raw/'

def get_train_videos():
    from sklearn.model_selection import train_test_split     

    df_train = pd.read_csv('../../data/train.csv')
    df_submit = pd.read_csv('../../data/submission.csv')
    
    # remove lion - because it cannot be split properly
    # bring lion back - for savva
    ANIMALS = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant',
           'forest buffalo', 'gorilla', 'hippopotamus', 'human', 'hyena', 'lion',
           'large ungulate', 'leopard', 'other (non-primate)',
           'other (primate)', 'pangolin', 'porcupine', 'reptile', 'rodent',
           'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']
    
    df_train['animal'] = ''
    df_train.loc[df_train.sum(axis=1)>1,'animal'] = '2 animals'
    df_train.loc[df_train.sum(axis=1)==1, 'animal'] = df_train[df_train.sum(axis=1)==1][ANIMALS].idxmax(axis=1)

    df_sample = pd.DataFrame()

    # just take the dataset as is w/o any limitations
    # do not take 2 animals because of ambiguity
    for animal in ANIMALS:
        df_sample = df_sample.append(df_train[df_train.animal == animal])

    for animal in ['2 animals']:
        df_sample = df_sample.append(df_train[df_train.animal == animal])        
        
    # df_sample = df_sample[0:128]
    video_2_animal_train = dict(zip(df_sample.filename.values,['all_animals'] * len(list( df_sample.filename.values )) ))     
    # video_2_animal_train = dict(zip(df_sample.filename.values,['all_animals'] * 128 ))     
    return video_2_animal_train
class InceptionExtractor(nn.Module):
    def __init__(self,
                 inception):
        super(InceptionExtractor, self).__init__()
        
        self.stem = nn.Sequential(
            inception.features[0],
            inception.features[1],
            inception.features[2],
            inception.features[3],
            inception.features[4],
            inception.features[5],
        )
        
        self.block1 = nn.Sequential(
            inception.features[6],
            inception.features[7],
            inception.features[8],
            inception.features[9],
        )        
        
        self.tr1 = inception.features[10]
        
        self.block2 = nn.Sequential(
            inception.features[11],
            inception.features[12],
            inception.features[13],
            inception.features[14],
            inception.features[15],
            inception.features[16],
            inception.features[17],            
        )
        
        self.tr2 = inception.features[18]
        
        self.block3 = nn.Sequential(
            inception.features[19],
            inception.features[20],
            inception.features[21]        
        )        
        
        self.avgpool1 = nn.AvgPool2d(48, stride=1)
        self.avgpool2 = nn.AvgPool2d(23, stride=1)
        self.avgpool3 = nn.AvgPool2d(11, stride=1)
  

    def forward(self, x):
        
        out = []
        for frame in x:
            x1 = self.stem(frame)
            x1 = self.block1(x1)
            
            x1_resume = self.tr1(x1)
            x2 = self.block2(x1_resume)

            x2_resume = self.tr2(x2)
            x3 = self.block3(x2_resume)

            feature_vector = torch.cat((self.avgpool1(x1)
                                        .view(x1.size(0),x1.size(1)),
                                        self.avgpool2(x2)
                                        .view(x2.size(0),x2.size(1)),
                                        self.avgpool3(x3)
                                        .view(x3.size(0),x3.size(1)),
                                        ), dim=1)
            out.append(feature_vector)
            
        out = torch.stack ([features for features in out])             
        
        return out    

inception = inceptionv4(num_classes=1000, pretrained='imagenet')
model = InceptionExtractor(inception)

predict_dict = get_train_videos()
df_predict = pd.DataFrame(list(predict_dict.items()))
df_predict.to_csv('../../data/interim/{}.csv'.format(args.lognumber))

if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()

cudnn.benchmark = True    
print('Total dictionary length is {}'.format( len(predict_dict) ) )

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

extract_augs = Compose([
    Normalize(mean,std),       
])

predict_folder = ExtractDataset(root = args.data,
                           video_dict = predict_dict,
                           transform = extract_augs,
                           dataset_name = 'predict_chimps',
                           fps = 3,
                           target_size = (404, 404)
                       )
   
predict_loader = torch.utils.data.DataLoader(
    dataset = predict_folder,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True)

print('Total predict folder length is {}'.format( len(predict_folder) ) )

def predict(predict_loader,
            model,
            iter_number):
    # switch to evaluate mode
    model.eval()

    with tqdm.tqdm(total=iter_number) as pbar:
        for i, (input, target) in enumerate(predict_loader):
            # target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input.float(), volatile=True)
            # target_var = torch.autograd.Variable(target, volatile=True)
            # compute output
            output = model(input_var)
            
            for idx,video_id in enumerate(target):
                # print (idx,video_id)
                video_array = output[idx].data.cpu().numpy()
                # print( video_array.shape )
                # print( os.path.join(args.save_path,video_id) )
                np.save( os.path.join(args.save_path,video_id),video_array)
                del video_array

            pbar.update(1)
            gc.collect()


iter_number = len(predict_folder)

predict(predict_loader,
        model,
        len(predict_loader)
       )

