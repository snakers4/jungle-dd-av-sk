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

from nasnet import nasnetalarge

# garbage collector
import gc

# stupid hack - add params instead of doing a proper CLI parsing
# also works with jupyter notebooks
args = type('test', (), {})()
args.finetune = False
args.resume = False
args.batch_size = 4
args.workers = 4
args.lognumber = 'nasnet_train'
args.save_path = '../../data/interim/nasnet_train'
args.numclasses = 23
args.arch = 'nasnet'
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
class NasnetExtractor(nn.Module):
    def __init__(self,
                 nasnet):
        super(NasnetExtractor, self).__init__()
        
        self.conv0 = nasnet.conv0 
        self.cell_stem_0 = nasnet.cell_stem_0
        self.cell_stem_1 = nasnet.cell_stem_1

        self.cell_0 = nasnet.cell_0
        self.cell_1 = nasnet.cell_1
        self.cell_2 = nasnet.cell_2
        self.cell_3 = nasnet.cell_3
        self.cell_4 = nasnet.cell_4
        self.cell_5 = nasnet.cell_5

        self.reduction_cell_0 = nasnet.reduction_cell_0

        self.cell_6 = nasnet.cell_6
        self.cell_7 = nasnet.cell_7
        self.cell_8 = nasnet.cell_8
        self.cell_9 = nasnet.cell_9
        self.cell_10 = nasnet.cell_10
        self.cell_11 = nasnet.cell_11

        self.reduction_cell_1 = nasnet.reduction_cell_1

        self.cell_12 = nasnet.cell_12
        self.cell_13 = nasnet.cell_13
        self.cell_14 = nasnet.cell_14
        self.cell_15 = nasnet.cell_15
        self.cell_16 = nasnet.cell_16
        self.cell_17 = nasnet.cell_17
        
        self.avgpool1 = nn.AvgPool2d(21, stride=1)
        self.avgpool2 = nn.AvgPool2d(11, stride=1)
        self.avgpool3 = nn.AvgPool2d(11, stride=1)

    def forward(self, x):
        
        out = []
        for frame in x:
           
            x_conv0 = self.conv0(frame)
            x_stem_0 = self.cell_stem_0(x_conv0)
            x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

            x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
            x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
            x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
            x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
            x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
            x_cell_5 = self.cell_5(x_cell_4, x_cell_3)

            x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)

            x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
            x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
            x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
            x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
            x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
            x_cell_11 = self.cell_11(x_cell_10, x_cell_9)

            x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)

            x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
            x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
            x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
            x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
            x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
            x_cell_17 = self.cell_17(x_cell_16, x_cell_15)            

            
            feature_vector = torch.cat((self.avgpool1(x_reduction_cell_0)
                                        .view(x_reduction_cell_0.size(0),x_reduction_cell_0.size(1)),
                                        self.avgpool2(x_reduction_cell_1)
                                        .view(x_reduction_cell_1.size(0),x_reduction_cell_1.size(1)),
                                        self.avgpool3(x_cell_17)
                                        .view(x_cell_17.size(0),x_cell_17.size(1)),
                                        ), dim=1)
            out.append(feature_vector)
            
        out = torch.stack ([features for features in out])             
        
        return out

nasnet = nasnetalarge(num_classes=1000, pretrained='imagenet')
model = NasnetExtractor(nasnet)

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
                           fps = 2,
                           target_size = (331, 331)
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

