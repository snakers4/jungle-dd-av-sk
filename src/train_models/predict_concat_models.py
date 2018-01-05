import argparse
from local_utils import *
from train_kfold_models_utils import *

parser = argparse.ArgumentParser(description='Predicting models based on extracted features')
parser.add_argument('-m', '--model_name', metavar='model_name')
parser.add_argument('-resnet152', '--resnet152_folder', metavar='resnet152_folder')
parser.add_argument('-resnet152_test', '--resnet152_test_folder', metavar='resnet152_test_folder')
parser.add_argument('-inception_resnet', '--inception_resnet_folder', metavar='inception_resnet_folder')
parser.add_argument('-inception_resnet_test', '--inception_resnet_test_folder', metavar='inception_resnet_test_folder')
parser.add_argument('-inception4', '--inception4_folder', metavar='inception4_folder')
parser.add_argument('-inception4_test', '--inception4_test_folder', metavar='inception4_test_folder')

parser.add_argument('-blank', '--blank_only', metavar='blank_only', type=int, default=0)
parser.add_argument('-k', '--num_folds', metavar='num_folds', type=int, default=5)
parser.add_argument('-bs', '--batch_size', metavar='batch_size', type=int, default=48)

def main():
    global args
    args = parser.parse_args()
    
    folders = [args.resnet152_folder, args.inception_resnet_folder, args.inception4_folder]
    test_folders = [args.resnet152_test_folder, args.inception_resnet_test_folder, args.inception4_test_folder]
    shapes = [(45, 3840), (45, 3488), (45, 2944)]
    
    predict_kfold_models(args.model_name, folders, test_folders, shapes, 
                         args.num_folds, args.batch_size, 
                         True if args.blank_only==1 else False)
            
if __name__ == '__main__':
    main() 