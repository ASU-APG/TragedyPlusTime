import glob
import os

openpose_path = '/mnt1/arnav/pose-estimation/openpose/'
video_dir = '/mnt1/arnav/pose-estimation/oops-videos/'

video_paths = glob.glob(f'{video_dir}*.mp4')
for example in video_paths:
    base_name = os.path.splitext(os.path.basename(example))[0]
    cmd = f'cd {openpose_path} && ./build/examples/openpose/openpose.bin --video {example} --write_json ../results3/{base_name} --display 0  --write_video ../results3/{base_name}/processed.avi --model_pose COCO'
    os.system(cmd)