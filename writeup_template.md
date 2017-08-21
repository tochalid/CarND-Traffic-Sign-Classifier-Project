#**Traffic Sign Recognition** 
---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/before.jpg "Before Grayscaling"
[image4]: ./ts_downloads/id4_speed_limit_70.jpg "Traffic Sign 1"
[image5]: ./ts_downloads/id14_stop.jpg "Traffic Sign 2"
[image6]: ./ts_downloads/id3_speed_limit_60.jpg "Traffic Sign 3"
[image7]: ./ts_downloads/id25_road_work.jpg "Traffic Sign 4"
[image8]: ./ts_downloads/id12_priority_road.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set. In the code, the analysis is done using python and numpy methods.

I used python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed over each class. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. Additionally the data can be augmented, eg rotated. Good [samples code](https://github.com/aleju/imgaug). Implementations can be done eg with [sklearn.preprocessing package](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler) just to name one framework. A good reference for scaling attempts and there effects can be found here: [preprocessing](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)

As a first step, I decided to convert the images to grayscale because it improves accuracy and reduces computing time. I used the standard function of tensorflow [rgb_to_grayscale](https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_grayscale).  Second, I normalize the data for better convergence and backpropagation. Further reading you'll find [here](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html). 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3] ![alt text][image2]

Further augmentation was not necessary to accomplish required 93% validation accuracy.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| Input = 32x32x1							| 
| L1: Convolution 5x5     	| 1x1 stride, valid padding, output = 28x28x6, relu activation 	|
| Max pooling	      	| 2x2 stride,  valid padding, Output = 14x14x6 				|
| L2: Convolution 5x5	    | 1x1 stride, valid padding, output = 10x10x16. relu activation   	|
| Max pooling	      	| 2x2 stride,  valid padding, Output = 5x5x16 				|
|	Flatten					|					Output = 400  |
| L3: Fully connected		| Output = 120, relu activation				|
| L4 Fully connected		| Output = 84, relu activation, dropout |
| Outupt: Fully connected				| Output = 43								|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an an iterative approach for the hyperparameters. First checked underfitting with batchsize 2 which produced constant accuracy. Than switched to 128 which is overfitting. Accomplished some 0.89 validation accuracy. Decided to "early-terminate" in order to prevent overfitting, limited epochs to 10. Decreased sigma from 1 to 0.05 to reduce "oszillation" of accuracy. Learning kept to 0.001 due to overfitting less sensible.

####4. Reused LeNet architecture which is suitable due to its application for character classification, thus well-known for its accuracy. In order to get the validation set accuracy to be at least 0.93 it has been sufficient to apply grayscale and normalization, than regularization and dropout to minimize impact of overfitting.

My final model results were:
* training set accuracy of 0.99290?
* validation set accuracy of 0.93832? 
* test set accuracy of 0.9220?

If an iterative approach was chosen:
* The major problem with the initial architecture is overfitting, which can be addressed in multiple ways, but 
* A design choise would be to squeeze further the spacial dimension using deeper convolutions.

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? Well know architecture that has been fine-tuned (even if i didn't change any structure here) for many applications by the community. Thus, a good starting point, that need to be further modified to additionally increase accuracy and improve prediction.  
 
###Test a Model on New Images

####1. Five German traffic signs found on the web and provide them in the report. 

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. The prediction seems not sufficient, thus the model statistics report high accuracy. I find it obviouse that probabilities have very small values around 0 ranging from e-1 to e-2. However, the small values could explain that the prediction is vage. Comparing probabilities from new signs and from training and validation sets, indicates similar scale or worth (mean very small), which would points me for double-checking the preprocessing mechanisms. Right now I cannot preclude a programming error, but couldnt find any yet. It could be a problem with numerical stability.

Here are the results of the prediction: (instable, eg Predicted Labels:  [31 29  7 20 33])

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| id4_speed_limit_70  		| other   									| 
| id14_stop     			| other 										|
| id3_speed_limit_60					| other											|
| id25_road_work      		| other					 				|
| id12_priority_road		| other      							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Meaningful certainty cannot be determined at this stage

The code for making predictions on my final model is located in the section with header "Analyze Performance" of the Ipython notebook.

The model is relatively unsure. See top 5.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| [ 0.08945462,  0.06818956,  0.06765534,  0.05835355,  0.05278334]     | [38, 34, 36,  3, 14]  				| 
| [ 0.06100992,  0.04847522,  0.04079581,  0.03980098,  0.03709331]     | [38,  3, 17, 36, 24] 					|
| [ 0.07438038,  0.05912182,  0.04951916,  0.04601262,  0.03545175]					| [38, 34, 36,  3, 22]						|
| [ 0.08282059,  0.06962395,  0.05189766,  0.04466608,  0.04315159]	   	| [38, 34,  3, 14, 22]						|
| [ 0.0701469 ,  0.06590696,  0.05478808,  0.03873353,  0.03673509]			  | [38,  3, 34, 22, 14]						|



Expected Lables [ 4 14  3 25 12]

['ClassId', 'SignName']
['0', 'Speed limit (20km/h)']
['1', 'Speed limit (30km/h)']
['2', 'Speed limit (50km/h)']
['3', 'Speed limit (60km/h)']
['4', 'Speed limit (70km/h)']
['5', 'Speed limit (80km/h)']
['6', 'End of speed limit (80km/h)']
['7', 'Speed limit (100km/h)']
['8', 'Speed limit (120km/h)']
['9', 'No passing']
['10', 'No passing for vehicles over 3.5 metric tons']
['11', 'Right-of-way at the next intersection']
['12', 'Priority road']
['13', 'Yield']
['14', 'Stop']
['15', 'No vehicles']
['16', 'Vehicles over 3.5 metric tons prohibited']
['17', 'No entry']
['18', 'General caution']
['19', 'Dangerous curve to the left']
['20', 'Dangerous curve to the right']
['21', 'Double curve']
['22', 'Bumpy road']
['23', 'Slippery road']
['24', 'Road narrows on the right']
['25', 'Road work']
['26', 'Traffic signals']
['27', 'Pedestrians']
['28', 'Children crossing']
['29', 'Bicycles crossing']
['30', 'Beware of ice/snow']
['31', 'Wild animals crossing']
['32', 'End of all speed and passing limits']
['33', 'Turn right ahead']
['34', 'Turn left ahead']
['35', 'Ahead only']
['36', 'Go straight or right']
['37', 'Go straight or left']
['38', 'Keep right']
['39', 'Keep left']
['40', 'Roundabout mandatory']
['41', 'End of no passing']
['42', 'End of no passing by vehicles over 3.5 metric tons']


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

not yet available
