import argparse
from local_utils import *
from train_kfold_models_utils import *

parser = argparse.ArgumentParser(description='Training KFolds based on extracted features')
parser.add_argument('-m', '--model_name', metavar='model_name')
parser.add_argument('-s', '--shape', metavar='shape', type=int, nargs='+')
parser.add_argument('-f', '--folder', metavar='folder')
parser.add_argument('-tf', '--test_folder', metavar='test_folder')

parser.add_argument('-rnn', '--rnn', metavar='rnn', type=int, default=0)
parser.add_argument('-init_k', '--init_fold', metavar='init_fold', type=int, default=0)
parser.add_argument('-k', '--num_folds', metavar='num_folds', type=int, default=5)
parser.add_argument('-e', '--epochs', metavar='epochs', type=int, default=15)
parser.add_argument('-pe', '--ps_epochs', metavar='ps_epochs', type=int, default=5)
parser.add_argument('-bs', '--batch_size', metavar='batch_size', type=int, default=64)
parser.add_argument('-ps_bs', '--ps_batch_size', metavar='ps_batch_size', type=int, default=44)
parser.add_argument('-ps_tbs', '--ps_test_batch_size', metavar='ps_test_batch_size', type=int, default=20)

def getModel(input_shape, meta_input_shape, output_shape):
    x_input = Input(shape=input_shape)
    
    """Applying Attention layer (kind of learnable pooling)""" 
    x_att = BatchNormalization()(x_input)
    x_att = Attention(name='attlayer')(x_att)
    
    """Max-Min (kind of pooling) layer
    
    The idea behind this layer is to get representation how features change across timeline 
    and get better predictions between blank/not_blank classes. It gives reasonable boost in score.
    """
    x_minmax = Lambda(lambda x: K.max(x, axis=1) - K.min(x, axis=1))(x_input)
    
    """2x128 Dense layers for meta-data
    
    Meta-data provides slightly faster convergence as well as slight boost in blank/not_blank 
    predictions (blank videos are usually lighter).
    """
    x_meta_input = Input(shape=meta_input_shape)
    x_meta = Dense(128)(x_meta_input)
    x_meta = BatchNormalization()(x_meta)
    x_meta = Activation('relu')(x_meta)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(128)(x_meta)
    x_meta = BatchNormalization()(x_meta)
    x_meta = Activation('relu')(x_meta)
    
    """Concatenating extracted features: attention, max-min and meta-data features"""
    x = concatenate([x_att, x_minmax, x_meta])
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    """2x1024 Dense layers"""
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    
    """Last Dense layer with output_shape equal to number of classes
    
    Sigmoid activation function with binary crossentropy loss are used for multi-label purposes.
    """ 
    x_output = Dense(output_shape, activation='sigmoid')(x)
    return Model(inputs=[x_input, x_meta_input], outputs=x_output)

def getRNNModel(input_shape, meta_input_shape, output_shape):
    x_input = Input(shape=input_shape)
    
    x_feat = BatchNormalization()(x_input)
    x_feat = Dropout(0.15)(x_feat)
    
    
    """Bidirectional RNN could be used as alternative but provides slightly worse score"""
#     rnn_output = Bidirectional(CuDNNGRU(256, return_sequences=True, name="rnn"))(x_feat)
#     rnn_output = Dropout(0.15)(rnn_output)
#     x_feat = concatenate([rnn_output, x_feat])

    """2x512 RNN layers with return sequences and go backwards (to capture more info from the first frames)"""
    rnn_output1 = CuDNNGRU(512, return_sequences=True, name="rnn1", go_backwards=True)(x_feat)
    rnn_output1 = Dropout(0.15)(rnn_output1)
    rnn_output2 = CuDNNGRU(512, return_sequences=True, name="rnn2", go_backwards=True)(rnn_output1)
    rnn_output2 = Dropout(0.15)(rnn_output2)
    
    """Concatenating output from 1 and 2 RNN layers with features"""
    x_feat = concatenate([rnn_output1, rnn_output2, x_feat])

    """Applying AttentionWeightedAverage layer (kind of learnable pooling)

    NOTE: Attention layer from local_utils.py could be used there instead of AttentionWeightedAverage 
    and might provide better score.
    """ 
    x_att = AttentionWeightedAverage(name='attlayer', return_attention=False)(x_feat)
    x_att = Dropout(0.25)(x_att)
    
    """Max-Min (kind of pooling) layer
    
    The idea behind this layer is to get representation how features change across timeline 
    and get better predictions between blank/not_blank classes. It gives reasonable boost in score.
    """
    x_minmax = Lambda(lambda x: K.max(x, axis=1) - K.min(x, axis=1))(x_input)
    x_minmax = Dropout(0.25)(x_minmax)
    
    """2x128 Dense layers for meta-data
    
    Meta-data provides slightly faster convergence as well as slight boost in blank/not_blank 
    predictions (blank videos are usually lighter).
    """
    x_meta_input = Input(shape=meta_input_shape)
    x_meta = Dense(128)(x_meta_input)
    x_meta = BatchNormalization()(x_meta)
    x_meta = Activation('relu')(x_meta)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(128)(x_meta)
    x_meta = BatchNormalization()(x_meta)
    x_meta = Activation('relu')(x_meta)
    x_meta = Dropout(0.25)(x_meta)
    
    """Concatenating extracted features: rnn+attention, max-min and meta-data features"""
    x = concatenate([x_att, x_minmax, x_meta])
    x = BatchNormalization()(x)

    """2x1024 Dense layers"""
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    """Last Dense layer with output_shape equal to number of classes
    
    Sigmoid activation function with binary crossentropy loss are used for multi-label purposes.
    """ 
    x_output = Dense(output_shape, activation='sigmoid')(x)
    return Model(inputs=[x_input, x_meta_input], outputs=x_output)

def main():
    global args
    args = parser.parse_args()
     
    get_model_fun = getRNNModel if (args.rnn==1) else getModel
    
    train_kfold_models(get_model_fun, args.model_name, [args.folder], [args.test_folder], [tuple(args.shape)], 
                       args.init_fold, args.num_folds, args.epochs, args.ps_epochs, 
                       args.batch_size, args.ps_batch_size, args.ps_test_batch_size)
            
if __name__ == '__main__':
    main()   