import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog


class FeatureExtraction(object):
    
    
    def __init__(self,hist_bins,hist_bins_range,spatial_size,\
                 orient,pix_per_cell,cell_per_block):
        
        self.hist_bins = hist_bins # hist_bins=32
        self.hist_bins_range= hist_bins_range # hist_bins_range = (0, 256)
        self.spatial_size=spatial_size # spatial_size=(32, 32)
        self.orient = orient # orient=9
        self.pix_per_cell = pix_per_cell # pix_per_cell=8
        self.cell_per_block = cell_per_block # cell_per_block=2
        
     
    def bin_spatial(self,img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img,(16,16)).ravel() 
        # Return the feature vector
        return features
    
    
    def convert_color(self,img, cspace):
        
        if cspace == 'BGR':
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cspace == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif cspace == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    
 
    
    def color_hist(self,img,plot_info=False):
        
        ch1_hist = np.histogram(img[:,:,0],\
                                     bins= self.hist_bins,\
                                     range=self.hist_bins_range)
        
        ch2_hist = np.histogram(img[:,:,1],\
                                     bins=self.hist_bins,\
                                     range=self.hist_bins_range)
        
        ch3_hist = np.histogram(img[:,:,2],\
                                     bins=self.hist_bins,\
                                     range=self.hist_bins_range)
        bin_edges = ch1_hist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
        
        # Return the individual histograms, bin_centers and feature vector
        if plot_info:
            return  ch1_hist,ch2_hist,ch3_hist, bin_centers, hist_features
        else:
            return hist_features
    
    
        
        
    ''' Define a function to extract features from a list of images
        Have this function call bin_spatial() and color_hist()'''
    
    def extract_color_features(self,imgs, cspace='RGB'):
                      
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                feature_image = self.convert_color(image,cspace)
            else: feature_image = np.copy(image)   
            
            
            # Apply bin_spatial() to get spatial color features
            spatial_features = self.bin_spatial(feature_image)
            
            # Apply color_hist() also with a color space option now
            hist_features = self.color_hist(feature_image)
            
            # Append the new feature vector to the features list
            features.append(np.hstack((spatial_features, hist_features)))
            
        
        
        # Return list of feature vectors
        return features
        
        
    
    
    def get_hog_features(self,img,vis,feature_vec):
        if vis == True:
            features, hog_image = hog(img,
                                  orientations=self.orient,
                                  pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                  cells_per_block=(self.cell_per_block, self.cell_per_block),
                                  transform_sqrt=False,
                                  visualise=True,
                                  feature_vector=feature_vec)
        
            return features, hog_image
        else:      
            features = hog(img,
                       orientations=self.orient,
                       pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block, self.cell_per_block),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=feature_vec)
        
            return features 
        
        
        
    ''' Define a function to extract features from a list of images
        Have this function call bin_spatial() and color_hist()'''
    
    def extract_hog_features(self,imgs, cspace='RGB', hog_channel=0):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                feature_image = self.convert_color(image,cspace)
            else: feature_image = np.copy(image)      

            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.get_hog_features(feature_image[:,:,channel],
                                                              vis=False, 
                                                              feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = self.get_hog_features(feature_image[:,:,hog_channel],
                                                     vis=False,
                                                     feature_vec=True)
                
            # Append the new feature vector to the features list
            features.append(hog_features)
        # Return list of feature vectors
        return features
    
    
    
    
    
    