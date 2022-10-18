# SPDX-FileCopyrightText: 2021 ejcgt
# SPDX-License-Identifier: MIT

from matplotlib import pyplot as plt
from matplotlib import animation
import os
import cv2
import numpy as np

from utils import evaluation


class Visualize():

    def __init__(self, frames_path, df, heatmaps):
        self.frames_path = frames_path
        self.frames = os.listdir(frames_path)
        self.frames.sort()
        self.frames = [f for f in self.frames if f.endswith('.jpg')]
        
        self.df = df
        self.heatmaps = heatmaps
        
        self.img = cv2.imread(os.path.join(frames_path, self.frames[0]))
        self.fig = plt.figure()
        self.disp = plt.imshow(self.img)
        plt.close()
    
    
    def ani_init(self):
        self.disp.set_data(self.img)


    def animate(self, i):
        img = cv2.imread(os.path.join(self.frames_path, self.frames[i]))

        # Draw head bbox 
        color = (0, 255, 0)   
        thickness = 4  
        start = (int(self.df.loc[self.frames[i],'left']), int(self.df.loc[self.frames[i],'top']))
        end = (int(self.df.loc[self.frames[i],'right']), int(self.df.loc[self.frames[i],'bottom']))
        img = cv2.rectangle(img, start, end, color, thickness)
        
        # Mark gaze
        alpha = 0.2
        beta = 1 - alpha
        hm_stacked = np.repeat(np.expand_dims(self.heatmaps[i], 2), 3, 2)
        hm_stacked = hm_stacked.astype(np.uint8)
        img = cv2.addWeighted(img, alpha, hm_stacked, beta, 0.0)
        
        self.disp.set_data(img)
        return self.disp
    
    def save_video(self, out):
        ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.ani_init, frames=len(self.frames),
                               interval=50)
        ani.save(out)