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

# garbage collector
import gc

# stupid hack - add params instead of doing a proper CLI parsing
# also works with jupyter notebooks
args = type('test', (), {})()
args.finetune = False
args.resume = False
args.batch_size = 4
args.workers = 6
args.lognumber = 'resnet_test'
args.save_path = '../../data/interim/resnet_test'
args.numclasses = 23
args.arch = 'resnet152'
args.evaluate = False
args.data = '../../data/raw/'

def get_predict_videos():
    df_submit = pd.read_csv('../../data/submission.csv')
    predict_dict = dict(zip(list(df_submit.filename.values),['all_animals'] * len(list(df_submit.filename.values)) ))
    return predict_dict
class ResnetVideoExtractor(nn.Module):
    def __init__(self,
                 resnet):
        super(ResnetVideoExtractor, self).__init__()
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )   
        
        self.avgpool1 = nn.AvgPool2d(101, stride=1)
        self.avgpool2 = nn.AvgPool2d(51, stride=1)
        self.avgpool3 = nn.AvgPool2d(26, stride=1)
        self.avgpool4 = nn.AvgPool2d(13, stride=1)
        
        self.skip1 = resnet.layer1
        self.skip2 = resnet.layer2
        self.skip3 = resnet.layer3
        self.skip4 = resnet.layer4

    def forward(self, x):
        
        out = []
        for frame in x:
            x1 = self.stem(frame)
            x1 = self.skip1(x1)
            x2 = self.skip2(x1)
            x3 = self.skip3(x2)
            x4 = self.skip4(x3)

            feature_vector = torch.cat((self.avgpool1(x1)
                                        .view(x1.size(0),x1.size(1)),
                                        self.avgpool2(x2)
                                        .view(x2.size(0),x2.size(1)),
                                        self.avgpool3(x3)
                                        .view(x3.size(0),x3.size(1)),
                                        self.avgpool4(x4)
                                        .view(x4.size(0),x4.size(1)),
                                        ), dim=1)
            out.append(feature_vector)

        out = torch.stack ([features for features in out])             
            
        return out

    

resnet = models.resnet152(pretrained=True)
model = ResnetVideoExtractor(resnet) 

predict_dict = get_predict_videos()
df_predict = pd.DataFrame(list(predict_dict.items()))
df_predict.to_csv('../../data/interim/{}.csv'.format(args.lognumber))

if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()

cudnn.benchmark = True    
print('Total dictionary length is {}'.format( len(predict_dict) ) )

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

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

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

extract_augs = Compose([
    Normalize(mean,std),       
])

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

