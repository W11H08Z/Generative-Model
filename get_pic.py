import numpy as np
import imageio
import os

def get_gif(folder_path):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(('.png'))]
    image_files.sort()

    frames = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = imageio.v2.imread(image_path)
        frames.append(frame)
    
    imageio.mimsave(f'gif/{os.path.basename(folder_path)}.gif', frames, duration=0.5)

if __name__ == '__main__':
    get_gif("/home/wangchai/zhw/Generative-Model/GAN_images")