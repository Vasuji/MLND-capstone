
# Capstone Project Proposal
## Machine Learning Engineer Nano Degree


### Domain Background

Object detection[[1]](https://en.wikipedia.org/wiki/Object_detection) is one of the exciting part of the study in Computer vision and Machine learning. For example human detection[[2]](http://www.sciencedirect.com/science/article/pii/S0031320315003179) in servelence cameras, face and facial expression detection[[3]](http://ii.tudelft.nl/pub/dragos/euromedia.pdf) in speech videos etc. This project is greatly inspired by Vehicle detection and Tracking project of the Self Driving Car Nanodegree which is offered at Udacity. The main focus on this project is to build a model for image classification and using it to detect and track the identified image in videos. More specifically it detect and track vehicle in the video.


### Problem Statement

The statement of the problem is ***Vehicle detection and tracking in a video using classifier trained over vehicle and non vehicle image data set*** . The solution to the problem is that, it should be able to select proper pixcels in the image frame in videos as vehicle and draw a rectangular boundary. More technically, the accuracy of the classifier will be calculated with respect to train, test data as well as the performance of classifier on videos. The results thus obtained should be replicable using same data set.

### Data Set and Inputs
 This project will build a classifier using two sets of data called vehicle and nonvehicle. One a model is ready, it will be used to test it's performance on test video. Following are the data sources:
 
 1. Vehicle Images: [Link](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
 2. Non-vehicle Images: [Link](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
 3. Test Video: [Link](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/project_video.mp4)


### Solution Statement:
 The problem of detecting vehicle in the test video will be solved by using techniques of Computer vision for feature extraction and  and Machine Learning for search of best classifier model. Different scheme and hyperparameters for ```SVM``` classifiers for eample ```linear, rbf, 'C' ``` will be tried. For parameter tuning, scikit-learn's ```GridSearchCV, RandomizedSearchCV``` will be used. Best classifier model will be extracted using labeled data set and it will be used to the test video. Once the part of the image representing vehicle is identified in each frame of video, a rectangular box will be drawn around it for vehicle tracking purpose. The correctly counted vehicle in the duration of the video will count the success of the algorithm.


### Benchmark model






### Evaluation Metric
Two different metrics will be used in this project. First one includes the accuracy of the trained support vector machine on training and test data. The second includes the count of correctly detected vehicle and false positive. The actual no of the cars will be counted manually and it will be compaired to the vehicle detected by the algorithhm.



### Project Design

* Part 1: ***Getting Familiar to data and exploratory data analysis*** 

* Part 2: ***Feature Extraction***

* Part 3. ***Train a linear SVM***

* Part 4. ***Technique Slidding Window Search***

* Part 5. ***Search and Classifiy an image***

* Part 6. ***Search and Classifiy a Video***




### References:
1. [Object Detection Wikipedia](https://en.wikipedia.org/wiki/Object_detection)
2. [Human detection from images and videos: A survey](http://www.sciencedirect.com/science/article/pii/S0031320315003179)
3. [Machine Learinig Technique for Face Analaysis](http://ii.tudelft.nl/pub/dragos/euromedia.pdf)











