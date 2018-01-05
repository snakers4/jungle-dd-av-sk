import argparse
from local_utils import *
from train_kfold_models_utils import *

parser = argparse.ArgumentParser(description='Predicting models based on extracted features')
parser.add_argument('-m', '--model_name', metavar='model_name')
parser.add_argument('-s', '--shape', metavar='shape', type=int, nargs='+')
parser.add_argument('-f', '--folder', metavar='folder')
parser.add_argument('-tf', '--test_folder', metavar='test_folder')

parser.add_argument('-k', '--num_folds', metavar='num_folds', type=int, default=5)
parser.add_argument('-bs', '--batch_size', metavar='batch_size', type=int, default=64)

def main():
    global args
    args = parser.parse_args()
    
    predict_kfold_models(args.model_name, [args.folder], [args.test_folder], [tuple(args.shape)], 
                         args.num_folds, args.batch_size)
            
if __name__ == '__main__':
    main() 