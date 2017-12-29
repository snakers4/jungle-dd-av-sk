import glob as glob
import pandas as pd
import tqdm
import skvideo.io as skv

g = glob.glob('../slow_max_space/raw/*.mp4')
# g = glob.glob('micro_chimps/micro/*.mp4')
video_ids = [(video.split('/')[-1]) for video in g]

widths = []
heights = []
nb_frames = []
avg_frame_rates = []

df = pd.DataFrame(columns=['video_id','height','width','nb_frames','avg_frame_rate'])

chunk = 0

with tqdm.tqdm(total=len(g)) as pbar:
    for i,video_id,video_path in zip(range(0,len(g)),video_ids,g):
        try:
            probe = skv.ffprobe(g[i])
            width = probe['video']['@width']
            height = probe['video']['@height']
            nb_frame = probe['video']['@nb_frames']       
            avg_frame_rate = probe['video']['@avg_frame_rate']        
        except Exception as e:
            print(str(e))
            probe = 0
            width = 0
            height = 0
            nb_frame = 0
            avg_frame_rate = 0
        

        df.loc[i] = video_id,height,width,nb_frame,avg_frame_rate
        
        if (i+1)%50000 == 0:
            print('Save initiated, iteration {}'.format(i) )
            df.to_csv('video_metadata_chunk_{}.csv'.format(chunk) )
            chunk+=1
            df = pd.DataFrame(columns=['video_id','height','width','nb_frames','avg_frame_rate'])

        pbar.update(1)

df.to_csv('video_metadata_chunk_{}.csv'.format(chunk) )