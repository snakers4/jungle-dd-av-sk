import argparse
from local_utils import *
from train_kfold_models_utils import *

parser = argparse.ArgumentParser(description='Training KFolds based on extracted concatenated features')
parser.add_argument('-m', '--model_name', metavar='model_name')
parser.add_argument('-resnet152', '--resnet152_folder', metavar='resnet152_folder')
parser.add_argument('-resnet152_test', '--resnet152_test_folder', metavar='resnet152_test_folder')
parser.add_argument('-inception_resnet', '--inception_resnet_folder', metavar='inception_resnet_folder')
parser.add_argument('-inception_resnet_test', '--inception_resnet_test_folder', metavar='inception_resnet_test_folder')
parser.add_argument('-inception4', '--inception4_folder', metavar='inception4_folder')
parser.add_argument('-inception4_test', '--inception4_test_folder', metavar='inception4_test_folder')

parser.add_argument('-blank', '--blank_only', metavar='blank_only', type=int, default=0)
parser.add_argument('-init_k', '--init_fold', metavar='init_fold', type=int, default=0)
parser.add_argument('-k', '--num_folds', metavar='num_folds', type=int, default=5)
parser.add_argument('-e', '--epochs', metavar='epochs', type=int, default=15)
parser.add_argument('-pe', '--ps_epochs', metavar='ps_epochs', type=int, default=5)
parser.add_argument('-bs', '--batch_size', metavar='batch_size', type=int, default=48)
parser.add_argument('-ps_bs', '--ps_batch_size', metavar='ps_batch_size', type=int, default=32)
parser.add_argument('-ps_tbs', '--ps_test_batch_size', metavar='ps_test_batch_size', type=int, default=16)

def getConcatModel(resnet152_input_shape, inception_resnet_input_shape, inception4_input_shape,
                 meta_input_shape, output_shape):
    x_resnet152_input = Input(shape=resnet152_input_shape)
    x_inception_resnet_input = Input(shape=inception_resnet_input_shape)
    x_inception4_input = Input(shape=inception4_input_shape)
    
    """Concatenating extracted features from resnet152, inception_resnet and inception4
    
    It's reasonable to add more features from different pre-trained networks like nasnet.
    """
    x_input = concatenate([x_resnet152_input, x_inception_resnet_input, x_inception4_input])
    
    # Attention
    x_att = BatchNormalization()(x_input)
    x_att = Attention(name='attlayer')(x_att)
    
    # minmax
    x_minmax = Lambda(lambda x: K.max(x, axis=1) - K.min(x, axis=1))(x_input)
    
    # Meta
    x_meta_input = Input(shape=meta_input_shape)
    x_meta = Dense(128)(x_meta_input)
    x_meta = BatchNormalization()(x_meta)
    x_meta = Activation('relu')(x_meta)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(128)(x_meta)
    x_meta = BatchNormalization()(x_meta)
    x_meta = Activation('relu')(x_meta)
    
    # Concatenate
    x = concatenate([x_att, x_minmax, x_meta])
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    
    # Output
    x_output = Dense(output_shape, activation='sigmoid')(x)
    return Model(inputs=[x_resnet152_input, x_inception_resnet_input, x_inception4_input, x_meta_input], outputs=x_output)

def main():
    global args
    args = parser.parse_args()
    
    folders = [args.resnet152_folder, args.inception_resnet_folder, args.inception4_folder]
    test_folders = [args.resnet152_test_folder, args.inception_resnet_test_folder, args.inception4_test_folder]
    shapes = [(45, 3840), (45, 3488), (45, 2944)]
    
    train_kfold_models(getConcatModel, args.model_name, folders, test_folders, shapes, 
                       args.init_fold, args.num_folds, args.epochs, args.ps_epochs, 
                       args.batch_size, args.ps_batch_size, args.ps_test_batch_size,
                       True if args.blank_only==1 else False)

if __name__ == '__main__':
    main()   