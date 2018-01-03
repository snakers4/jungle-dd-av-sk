import os, sys, pickle, multiprocessing
from joblib import Parallel, delayed

import skvideo.io as skv
import numpy as np
import pandas as pd

cpu_cores = multiprocessing.cpu_count()

def get_video_metadata(folder, file_name):
    return [int(skv.ffprobe(folder+file_name)['video']['@bit_rate']), os.path.getsize(folder+file_name)]

def load_videos(folder, video_names, load_fun, cpu_cores=cpu_cores):
    return np.array(Parallel(n_jobs=cpu_cores)(delayed(load_fun)(folder, video_name) for video_name in video_names))

def extract_meta_data(data_folder='../../data/', micro_folder='../../data/micro/'):
    df = pd.read_csv(data_folder + 'train_labels.csv', index_col='filename')
    test_df = pd.read_csv(data_folder + 'submission_format.csv', index_col='filename')
    
    meta_data = load_videos(micro_folder, df.index, get_video_metadata)
    pickle.dump(meta_data, open(data_folder + 'meta_data.pkl', 'wb'))

    test_meta_data = load_videos(micro_folder, test_df.index, get_video_metadata)
    pickle.dump(test_meta_data, open(data_folder + 'test_meta_data.pkl', 'wb'))

def main():
    extract_meta_data()
            
if __name__ == '__main__':
    main() 