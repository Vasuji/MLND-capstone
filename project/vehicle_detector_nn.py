import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
import os
from keras.models import load_model
import h5py
from collections import deque

from project.feature_extraction import FeatureExtraction







class DetectVehicle(object):


    def __init__(self,param_dict,model):
        self.param_dict = param_dict
        self.nn = model
        self.X_scalar = self.param_dict['X_scalar']
        self.orient = self.param_dict['orient']
        self.pix_per_cell = self.param_dict['pix_per_cell'] 
        self.cell_per_block = self.param_dict['cell_per_block']
        self.hist_bins = self.param_dict['hist_bins'] 
        self.spatial_size = self.param_dict['spatial_size'] 
    
    
        self.FE = FeatureExtraction(hist_bins = self.hist_bins,\
                            hist_bins_range =  (0,256),\
                            spatial_size = self.spatial_size,\
                            orient = self.orient,\
                            pix_per_cell =  self.pix_per_cell,\
                            cell_per_block =  self.cell_per_block)
        
        
        self.threshold = 1.0
        self.heatmap = None
        
        # Heat Image for the Last Three Frames
        self.heat_images = deque(maxlen=3)
        
        # Current Frame Count
        self.frame_count = 0
        self.full_frame_processing_interval = 4
        
        # Xstart
        self.xstart = 600
        
        
        # Various Scales
        self.ystart_ystop_scale = [(360, 560, 1.5), (400, 600, 1.8), (440, 700, 2.5)]
        

        # Kernal For Dilation
        self.kernel = np.ones((50, 50))
        
        self.image = None
        
        
    
    def find_cars(self,image,vid=True,vis=False):
        self.image = image
        box_list = []
        draw_img = np.copy(self.image)
        img = self.image.astype(np.float32)/255
        
        if vid: # video
            
            if self.frame_count % self.full_frame_processing_interval == 0:
                mask = np.ones_like(img[:, :, 0])
            else:
                mask = np.sum(np.array(self.heat_images), axis=0)
                mask[(mask > 0)] = 1
                mask = cv2.dilate(mask, self.kernel, iterations=1)

            self.frame_count += 1

            for (self.ystart, self.ystop, self.scale) in self.ystart_ystop_scale:

                nonzero = mask.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])

                if len(nonzeroy) != 0:
                    self.ystart = max(np.min(nonzeroy), self.ystart)
                    self.ystop = min(np.max(nonzeroy), self.ystop)
                if len(nonzeroy) != 0:
                    x_start = max(np.min(nonzerox), self.xstart)
                    x_stop = np.max(nonzerox)
                else:
                    continue

                if x_stop <= x_start or self.ystop <= self.ystart:
                    continue
                    
                ibox_list = self.window_search(img)  
                for k in range(len(ibox_list)):
                    box_list.append(ibox_list[k])
                    
            # Add heat to each box in box list
            self.add_heat_and_threshold(draw_img, box_list)
            # Find final boxes from heatmap using label function
            labels = label(self.heatmap)
            draw_img = self.draw_labeled_bboxes(draw_img, labels)  
            
            return draw_img
         
                    
        else: # picture
            
            for (self.ystart, self.ystop, self.scale) in self.ystart_ystop_scale:
                ibox_list = self.window_search(img)  
                
                for k in range(len(ibox_list)):
                    box_list.append(ibox_list[k])
            
            if vis: # visualize picture with all boaxes found
                return box_list
                
            else: # visualize only threshold boxes  
                draw_image = self.window_search(img)
                # Add heat to each box in box list
                self.add_heat_and_threshold(draw_img, box_list)
                # Find final boxes from heatmap using label function
                labels = label(self.heatmap)
                draw_img = self.draw_labeled_bboxes(draw_img, labels)  
            
                return [draw_img,self.heatmap] 
            
       
        
        
        
    def window_search(self,img):
        
            box_list = []
            img_tosearch = img[self.ystart:self.ystop,:,:]
            ctrans_tosearch = self.FE.convert_color(img_tosearch, cspace='YCrCb')
    
            if self.scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1]/self.scale),
                                      np.int(imshape[0]/self.scale)))
        
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]

    
            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
            nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
            nfeat_per_block = self.orient*self.cell_per_block**2
    
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
            # Compute individual channel HOG features for the entire image

            hog1 = self.FE.get_hog_features(ch1,vis=False,feature_vec = False)
            hog2 = self.FE.get_hog_features(ch2,vis=False,feature_vec = False)
            hog3 = self.FE.get_hog_features(ch3,vis=False,feature_vec = False)

    
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel() 
            
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos*self.pix_per_cell
                    ytop = ypos*self.pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window,\
                                                    xleft:xleft+window], (64,64))
        
                    # Get color features
                    spatial_features = self.FE.bin_spatial(subimg)
                    hist_features = self.FE.color_hist(subimg)
            
                    color_features = np.hstack((spatial_features, hist_features))
                
            
                    X = np.hstack((color_features,hog_features)).reshape(1, -1)
            
                    test_features = self.X_scalar.transform(X) 
               
                    test_prediction = self.nn.predict_classes(test_features,verbose=0)
            
            
                    if test_prediction == 1:
                    
                        xbox_left = np.int(xleft*self.scale)
                        ytop_draw = np.int(ytop*self.scale)
                        win_draw = np.int(window*self.scale)
                    
                        box_list.append(((xbox_left, ytop_draw + self.ystart),\
                                     (xbox_left + win_draw, ytop_draw + \
                                      win_draw + self.ystart)))
                        
                        
           
            return box_list
            
        
        
       
    
    def add_heat_and_threshold(self,draw_img, bbox_list):
        # Iterate through list of bboxes
        h_map = np.zeros_like(draw_img[:,:,0]).astype(np.float)
        
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            h_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
            
        self.heat_images.append(h_map)
        self.heatmap = np.sum(np.array(self.heat_images),axis=0)
        # Return thresholded map
        self.heatmap[self.heatmap <= self.threshold] = 0
        # Return updated heatmap
        return 



    def draw_labeled_bboxes(self,img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),\
                    (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
    

def findCars(img, ystart, ystop, scale, nn,\
              X_scaler, color_space, orient,\
              pix_per_cell, cell_per_block,\
              hog_channel,spatial_size, hist_bins):
    
    
   

    FE = FeatureExtraction(hist_bins = hist_bins,\
                            hist_bins_range = (0,256),\
                            spatial_size = spatial_size,\
                            orient = orient,\
                            pix_per_cell =  pix_per_cell,\
                            cell_per_block =  cell_per_block)

    
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = FE.convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,\
                        (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = FE.get_hog_features(ch1,vis = False, feature_vec=False)
        hog2 = FE.get_hog_features(ch2,vis = False, feature_vec=False)
        hog3 = FE.get_hog_features(ch3,vis = False, feature_vec=False)
    else:
        hog1 = FE.get_hog_features(hog_channel,vis =False, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch

            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,\
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,\
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,\
                                 xpos:xpos + nblocks_per_window].ravel()
                
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog1[ypos:ypos + nblocks_per_window,\
                                    xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop +\
                                window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = FE.bin_spatial(subimg)
            hist_features = FE.color_hist(subimg)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = nn.predict_classes(test_features,verbose=0)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img
    