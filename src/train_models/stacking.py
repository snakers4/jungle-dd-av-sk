import argparse
from local_utils import *

def getSimpleNNModel(input_shape, output_shape):
    x_input = Input(shape=input_shape)
    x_feat = BatchNormalization()(x_input)
    x_feat = Dropout(0.15)(x_feat) # Providing higher dropout should increase score 

    x_feat = Dense(512)(x_feat)
    x_feat = BatchNormalization()(x_feat)
    x_feat = Activation('relu')(x_feat)
    x_feat = Dropout(0.25)(x_feat)

    x_feat = Dense(512)(x_feat)
    x_feat = BatchNormalization()(x_feat)
    x_feat = Activation('relu')(x_feat)
    x_feat = Dropout(0.25)(x_feat)

    x_output = Dense(output_shape, activation='sigmoid')(x_feat)
    return Model(inputs=x_input, outputs=x_output)

def stack_models():
    video_names, X_meta, Y, test_video_names, test_X_meta, inx2label, label2inx = load_data()
    
    resnet152_skip_f5_pred = np.load(results_dir+'resnet152_skip_CuDNNGRU512x2_dense1024_dense1024_bs64_pred.npy')
    resnet152_skip_f5_test_pred = np.load(results_dir+'resnet152_skip_CuDNNGRU512x2_dense1024_dense1024_bs64_test_pred.npy')
    
    resnet152_skip1_f5_pred = np.load(results_dir+'resnet152_skip_dense1024x2_bs64_pred.npy')
    resnet152_skip1_f5_test_pred = np.load(results_dir+'resnet152_skip_dense1024x2_bs64_test_pred.npy')

    inception_resnet_skip_f5_pred = np.load(results_dir+'inception_resnet_skip_CuDNNGRU512x2_dense1024_dense1024_bs64_pred.npy')
    inception_resnet_skip_f5_test_pred = np.load(results_dir+'inception_resnet_skip_CuDNNGRU512x2_dense1024_dense1024_bs64_test_pred.npy')

    inception_resnet_skip1_f5_pred = np.load(results_dir+'inception_resnet_skip_dense1024x2_bs64_pred.npy')
    inception_resnet_skip1_f5_test_pred = np.load(results_dir+'inception_resnet_skip_dense1024x2_bs64_test_pred.npy')

    inception4_skip_f5_pred = np.load(results_dir+'inception4_skip_CuDNNGRU512x2_dense1024_dense1024_bs64_pred.npy')
    inception4_skip_f5_test_pred = np.load(results_dir+'inception4_skip_CuDNNGRU512x2_dense1024_dense1024_bs64_test_pred.npy')

    inception4_skip1_f5_pred = np.load(results_dir+'inception4_skip_dense1024x2_bs64_pred.npy')
    inception4_skip1_f5_test_pred = np.load(results_dir+'inception4_skip_dense1024x2_bs64_test_pred.npy')

    nasnet_skip1_f5_pred = np.load(results_dir+'nasnet_skip_dense1024x2_bs48_pred.npy')
    nasnet_skip1_f5_test_pred = np.load(results_dir+'nasnet_skip_dense1024x2_bs48_test_pred.npy')

    concat3_skip_f5_pred = np.load(results_dir+'concat3_skip_dense1024x2_bs48_pred.npy')
    concat3_skip_f5_test_pred = np.load(results_dir+'concat3_skip_dense1024x2_bs48_test_pred.npy')

    concat3_blank_skip_f5_pred = np.load(results_dir+'concat3_blank_skip_dense1024x2_bs48_pred.npy')
    concat3_blank_skip_f5_test_pred = np.load(results_dir+'concat3_blank_skip_dense1024x2_bs48_test_pred.npy')

    pred = np.concatenate([
        resnet152_skip_f5_pred, 
        inception_resnet_skip_f5_pred, 
        inception4_skip_f5_pred,
        resnet152_skip1_f5_pred, 
        inception_resnet_skip1_f5_pred, 
        inception4_skip1_f5_pred,
        nasnet_skip1_f5_pred, 
        concat3_skip_f5_pred, 
        concat3_blank_skip_f5_pred], -1)
    
    pred_test = np.concatenate([
        resnet152_skip_f5_test_pred.mean(0), 
        inception_resnet_skip_f5_test_pred.mean(0), 
        inception4_skip_f5_test_pred.mean(0), 
        resnet152_skip1_f5_test_pred.mean(0), 
        inception_resnet_skip1_f5_test_pred.mean(0), 
        inception4_skip1_f5_test_pred.mean(0),
        nasnet_skip1_f5_test_pred.mean(0), 
        concat3_skip_f5_test_pred.mean(0), 
        concat3_blank_skip_f5_test_pred.mean(0)], -1)
    
    trn_folds, val_folds = stratified_kfold_sampling(Y, 10, seed)

    blend_pred = np.zeros((pred.shape[0], Y.shape[1]))
    blend_pred_test = np.zeros((len(trn_folds), pred_test.shape[0], Y.shape[1]))

    for f_inx in range(0, len(trn_folds)):
        print("Training fold {}".format(f_inx))
        blend_model = getSimpleNNModel((pred.shape[1],), Y.shape[1])

        model_name = "blend__dense512x2__10f__concat3__concat3_blank__3skip__3skip1__nasnet"
        model_file_name = model_name+"_f"+str(f_inx)
        model_file = blend_dir+model_file_name+'.h5'

        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', mode='min', verbose=0, 
                                           save_best_only=True, save_weights_only=False, period=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.7, patience=2, min_lr=0.0001, verbose=0)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')

        opt=optimizers.Adam(lr=1e-3);
        blend_model.compile(optimizer=opt, loss='binary_crossentropy')

        blend_model.fit(
            pred[trn_folds[f_inx]], 
            Y[trn_folds[f_inx]],
            validation_data=(pred[val_folds[f_inx]], Y[val_folds[f_inx]]),
            batch_size=512,
            epochs=50,
            shuffle=True,
            verbose=0,
            callbacks=[reduce_lr, early_stop, model_checkpoint])

        del blend_model
        blend_model = load_model(model_file, compile=True)
        blend_pred[val_folds[f_inx]] = blend_model.predict(pred[val_folds[f_inx]], 10240)
        blend_pred_test[f_inx] = blend_model.predict(pred_test, 10240)

        losses = compute_losses(Y[val_folds[f_inx]], blend_pred[val_folds[f_inx]], eps=1e-5)
        print("fold: {}, loss: {}".format(f_inx, sum(losses)/len(losses)))

    np.save(results_dir+model_name+'_pred.npy', blend_pred)
    np.save(results_dir+model_name+'_test_pred.npy', blend_pred_test)

    losses = compute_losses(Y, blend_pred, eps=1e-5)
    print("full loss: {}".format(sum(losses)/len(losses)))
    
    
    target = Y
    blend_pred = blend_pred
    blend_test_pred = blend_pred_test.mean(0)
    
    clips = find_opt_clip_map(target, blend_pred)
    for cl_inx in range(0, blend_pred.shape[1]):
        np.clip(blend_pred[:,cl_inx], clips[cl_inx][0], clips[cl_inx][1], blend_pred[:,cl_inx])
        
    for cl_inx in range(0, blend_test_pred.shape[1]):
        np.clip(blend_test_pred[:,cl_inx], clips[cl_inx][0], clips[cl_inx][1], blend_test_pred[:,cl_inx])        

    losses = compute_losses(target, blend_pred)
    print()
    print("clipped loss: {}".format(sum(losses)/len(losses)))
    
    
    submission_df = pd.DataFrame(data=blend_test_pred, index=test_video_names, columns=inx2label, dtype=np.float32)
    submission_df.index.name = 'filename'
    submission_file_name = 'blend__dense512x2__10f__concat3__concat3_blank__3skip__3skip1__nasnet__submission.csv'
    submission_df.to_csv(results_dir+submission_file_name, index=True)
    
def main():
    stack_models()
            
if __name__ == '__main__':
    main()  