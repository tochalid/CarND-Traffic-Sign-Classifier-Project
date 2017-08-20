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
[image3]: ./examples/before.png "Befor Grayscaling"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

![alt text][image2]
![alt text][image3]

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

To train the model, I used an an iterative approach. First checked underfitting with batchsize 2 which produced constant accuracy. Than switched to 128 which is overfitting. Accomplished some 0.89 validation accuracy. Decided to "early-terminate" in order to prevent overfitting, limited epochs to 10. Decreased sigma from 1 to 0.05 to reduce "oszillation" of accuracy. Learning kept to 0.001 due to overfitting less sensible.

####4. Reused LeNet architecture which is suitable due to its application for character classification, thus well-known for its accuracy. In order to get the validation set accuracy to be at least 0.93 it has been sufficient to apply grayscale and normalization, than regularization and dropout to minimize impact of overfitting.

My final model results were:
* training set accuracy of 0.99290?
* validation set accuracy of 0.93832? 
* test set accuracy of 0.9220?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


