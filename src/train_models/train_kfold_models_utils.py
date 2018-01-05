from local_utils import *

def train_kfold_models(get_model_fun, model_name, folders, test_folders, shapes, init_fold, num_folds,
                       epochs, ps_epochs, batch_size, ps_batch_size, ps_test_batch_size, blank_only=False, seed=seed):
    video_names, X_meta, Y, test_video_names, test_X_meta, inx2label, label2inx = load_data()
    
    if (blank_only):
        Y = np.expand_dims(Y[:,1], 1)
    
    trn_folds, val_folds = stratified_kfold_sampling(Y, num_folds, seed)
    
    pred = np.zeros((video_names.shape[0], Y.shape[1]))
    test_pred = np.zeros((num_folds, test_video_names.shape[0], Y.shape[1]))
    
    test_seq = FeatureSequence(test_folders, test_video_names, shapes,
                               test_X_meta, np.zeros((test_video_names.shape[0], 1)), 
                               batch_size, shuffle=False, parallel=False)
    
    
    for f_inx in range(init_fold, num_folds):
        print("Training fold {}".format(f_inx))
        
        model = get_model_fun(*shapes, (X_meta.shape[1],), Y.shape[1])
        model_file_name = model_name+"f"+str(f_inx)
        model_file = models_dir+model_file_name+'.h5'

        # Train and valid seq
        trn_seq = FeatureSequence(folders, video_names[trn_folds[f_inx]], shapes,
                                  X_meta[trn_folds[f_inx]], Y[trn_folds[f_inx]], 
                                  batch_size, shuffle=True, parallel=False)
        val_seq = FeatureSequence(folders, video_names[val_folds[f_inx]], shapes,
                                  X_meta[val_folds[f_inx]], Y[val_folds[f_inx]], 
                                  batch_size, shuffle=False, parallel=False)
        
        # Test pseudo seq
        num_ps_test_samples = (math.ceil(Y[trn_folds[f_inx]].shape[0]/ps_batch_size) + 1)*ps_test_batch_size
        test_ps_inx = np.arange(0,test_video_names.shape[0])
        np.random.shuffle(test_ps_inx)
        test_ps_samples = test_ps_inx[:num_ps_test_samples]

        test_ps_seq = FeatureSequence(test_folders, test_video_names[test_ps_samples], shapes,
                                      test_X_meta[test_ps_samples], np.zeros((num_ps_test_samples, 1)), 
                                      batch_size, shuffle=False, parallel=False)

        # Callbacks
        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, 
                                           save_best_only=True, save_weights_only=False, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=0.0001, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
        
        print("Training")
        opt=optimizers.Adam(lr=1e-3);
        model.compile(optimizer=opt, loss='binary_crossentropy')
        model.fit_generator(
            generator=trn_seq, steps_per_epoch=len(trn_seq),
            validation_data=val_seq, validation_steps=len(val_seq),
            initial_epoch=0, epochs=epochs, shuffle=False, verbose=1,
            callbacks=[model_checkpoint, reduce_lr, early_stop],
            use_multiprocessing=False, workers=cpu_cores, max_queue_size=cpu_cores+2)
        
        print("Pseudo-label training")
        del model
        model = load_model(model_file, compile=True, 
                           custom_objects={'Attention':Attention,'AttentionWeightedAverage':AttentionWeightedAverage})
        for ps_inx in range(0, ps_epochs):         
            test_Y = model.predict_generator(test_ps_seq, len(test_ps_seq), verbose=1,
                                             use_multiprocessing=False, workers=cpu_cores, max_queue_size=cpu_cores+2)
            trn_ps_seq = PseudoFeatureSequence(folders, video_names[trn_folds[f_inx]], shapes,
                                               X_meta[trn_folds[f_inx]], Y[trn_folds[f_inx]], ps_batch_size,
                                               test_folders, test_video_names[test_ps_samples], 
                                               test_X_meta[test_ps_samples], test_Y, ps_test_batch_size, 
                                               shuffle=True, parallel=False)
            model.fit_generator(
                generator=trn_ps_seq, steps_per_epoch=len(trn_ps_seq),  
                validation_data=val_seq, validation_steps=len(val_seq),
                initial_epoch=epochs+ps_inx, 
                epochs=epochs+ps_inx+1, 
                shuffle=False, verbose=1,
                callbacks=[model_checkpoint, reduce_lr],
                use_multiprocessing=False, workers=cpu_cores, max_queue_size=cpu_cores+2)

            del test_Y
            del trn_ps_seq
            gc.collect()

            
        print("Predicting")
        del model  
        model = load_model(model_file, compile=True, 
                           custom_objects={'Attention':Attention,'AttentionWeightedAverage':AttentionWeightedAverage})
        pred[val_folds[f_inx]] = model.predict_generator(val_seq, len(val_seq), verbose=1, use_multiprocessing=False, 
                                                         workers=cpu_cores, max_queue_size=cpu_cores+2)
        test_pred[f_inx] = model.predict_generator(test_seq, len(test_seq), verbose=1, use_multiprocessing=False, 
                                                   workers=cpu_cores, max_queue_size=cpu_cores+2)

        losses = compute_losses(Y[val_folds[f_inx]], pred[val_folds[f_inx]], eps=1e-5)
        print("fold: {}, loss: {}".format(f_inx, sum(losses)/len(losses)))
        print()

    np.save(results_dir+model_name+'_pred.npy', pred)
    np.save(results_dir+model_name+'_test_pred.npy', test_pred)

    losses = compute_losses(Y, pred, eps=1e-5)
    print("full loss: {}".format(sum(losses)/len(losses)))
    print()
    
def predict_kfold_models(model_name, folders, test_folders, shapes, num_folds,
                         batch_size, blank_only=False, seed=seed):
    video_names, X_meta, Y, test_video_names, test_X_meta, inx2label, label2inx = load_data()
    
    if (blank_only):
        Y = np.expand_dims(Y[:,1], 1)
    
    trn_folds, val_folds = stratified_kfold_sampling(Y, num_folds, seed)
    
    pred = np.zeros((video_names.shape[0], Y.shape[1]))
    test_pred = np.zeros((num_folds, test_video_names.shape[0], Y.shape[1]))
    
    test_seq = FeatureSequence(test_folders, test_video_names, shapes,
                               test_X_meta, np.zeros((test_video_names.shape[0], 1)), 
                               batch_size, shuffle=False, parallel=False)
    
    
    for f_inx in range(0, num_folds):
        print("Predicting fold {}".format(f_inx))
        
        model_file_name = model_name+"f"+str(f_inx)
        model_file = models_dir+model_file_name+'.h5'
        model = load_model(model_file, compile=True, 
                           custom_objects={'Attention':Attention,'AttentionWeightedAverage':AttentionWeightedAverage})
        
        val_seq = FeatureSequence(folders, video_names[val_folds[f_inx]], shapes,
                                  X_meta[val_folds[f_inx]], Y[val_folds[f_inx]], 
                                  batch_size, shuffle=False, parallel=False)
        
        pred[val_folds[f_inx]] = model.predict_generator(val_seq, len(val_seq), verbose=1, use_multiprocessing=False, 
                                                         workers=cpu_cores, max_queue_size=cpu_cores+2)
        test_pred[f_inx] = model.predict_generator(test_seq, len(test_seq), verbose=1, use_multiprocessing=False, 
                                                   workers=cpu_cores, max_queue_size=cpu_cores+2)

        losses = compute_losses(Y[val_folds[f_inx]], pred[val_folds[f_inx]], eps=1e-5)
        print("fold: {}, loss: {}".format(f_inx, sum(losses)/len(losses)))
        print()

    np.save(results_dir+model_name+'_pred.npy', pred)
    np.save(results_dir+model_name+'_test_pred.npy', test_pred)

    losses = compute_losses(Y, pred, eps=1e-5)
    print("full loss: {}".format(sum(losses)/len(losses)))
    print()
    