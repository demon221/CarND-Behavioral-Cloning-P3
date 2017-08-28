#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NVIDIA_model.png "Model Visualization of NVIDIA Model"
[image2]: ./examples/left_image.jpg "Left Image"
[image3]: ./examples/center_image.jpg "Center Image"
[image4]: ./examples/right_image.jpg "Right Image"
[image5]: ./examples/processed.jpg "Processed Image"
[image6]: ./examples/center_image.jpg "Normal Image"
[image7]: ./examples/flip.jpg "Flipped Image"
[image8]: ./examples/translate.jpg "Translated Image"
[image9]: ./examples/shadow.jpg "Shadowed Image"
[image10]: ./examples/brightness.jpg "Brighted Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

| Files       |  Description  |
|:-----------:|:-------------:|
| model.py    | the script to create and train the model|
| load_data.py| the script to import raw image, image preprocessing and augmentation|
| drive.py    | contain simulator callbacks, model prediction results and simple vehicle control|
| video.py    | create video from automation mode in simulator|
| model.json  | contain the model architecture of convolution neural network|
| model.h5    | contain the trained model weights for model.json|

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around both the track1 and track2 by executing
```sh
python drive.py model.json
```


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. Model architecture design

The design of the network is based on the [NVIDIA model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (model.py lines 52-63), which has been used by NVIDIA for the end-to-end self driving test. During attempts, it is well suited for the project.

It is a deep convolution network which works well with supervised image classification / regression problems. As the NVIDIA model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.

##### - Model

This model consists of three convolution neural network layers with 5x5 filter sizes, two convolution neural network layers with 3x3 filter sizes and four fully connected layers.
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.
Because the test result has less overfitting, dropout trick is not used.

The final model architecture seems like following:

* Image normalization
* Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
* Fully connected: neurons: 100, activation: RELU
* Fully connected: neurons: 50, activation: RELU
* Fully connected: neurons: 10, activation: RELU
* Fully connected: neurons: 1 (output)

Below is an model structure output from the Keras summary, which gives more details on the shapes and the number of parameters.

|Layer (type)         |        Output Shape        |      Param |
|:-------------------:|:--------------------------:|:----------:|
|lambda_1 (Lambda)    |        (None, 66, 200, 3)  |      0     |
|conv2d_1 (Conv2D)    |        (None, 31, 98, 24)  |      1824  |
|conv2d_2 (Conv2D)    |        (None, 14, 47, 36)  |      21636 |
|conv2d_3 (Conv2D)    |        (None, 5, 22, 48)   |      43248 |
|conv2d_4 (Conv2D)    |        (None, 3, 20, 64)   |      27712 |
|conv2d_5 (Conv2D)    |        (None, 1, 18, 64)   |      36928 |
|flatten_1 (Flatten)  |        (None, 1152)        |      0     |
|dense_1 (Dense)      |        (None, 100)         |      115300|
|dense_2 (Dense)      |        (None, 50)          |      5050  |
|dense_3 (Dense)      |        (None, 10)          |      510   |
|dense_4 (Dense)      |        (None, 1)           |      11    |

Here is a visualization of the NVIDIA's model architecture.

![alt text][image1]

##### - Training

The model was trained using Adam optimiser with a learning rate = 1e-04 and mean squared error as a loss function.
I split the data into 80% for the training data and 20% for validation.
With batch size of 32, training epoch of 5, the model seems to perform quite well on both tracks.

#### 2.Training data collection

When using the simulator to collect training data, the training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving for 3 circles, counter-clock driving for 1 circle, and some places recovering from the left and right sides of the road.
To make the model more general, data form the second track is also collected. It contains much more steering angles values than the first track.
So the training data is a combination of driving data from track 1 and track2 at last.

The training data collected includes two tracks data, with a CSV file that contains the file path to the current center, right and left camera images, and the current throttle, brake, speed and steering information, also with an image folder includes all center, right and left camera images.

The trainer, instead of reading the entire dataset into memory, is used with model.fit_generator function in Keras to bring in just one image at a time to feed the GPU I use for the training and validation. This `batch_generator()` function (load_data.py lines 127-148) is like this.

```python
# Generator for the training images and associated steering angles
def batch_generator(image_paths, angles, batch_size, is_training=True):

    images = np.empty([batch_size, img_rows, img_cols, ch])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            angle = angles[index]
            # augmentation
            if is_training and np.random.rand() < 0.6:
                image, angle = augment(center, left, right, angle)
            else:
                image = load_image(center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
```

#### 3. Data augmentation and preprocessing

After the data collection from manual mode driving on the two tracks, I uses three steps to augment and preprocess the data.

##### - Use left and right camera images
(load_data.py lines 47-55)

Along with each frame we receive images from 3 camera positions: left, center and right. Although we are only going to use central camera while driving, we can still use left and right cameras data during training after applying steering angle correction, increasing number of examples by a factor of 3.

The `correction` steering angle I choose for the left and right camera image is **0.25**.

| Left camera | Center camera | Right camera |
|:-----------:|:-------------:|:------------:|
|![alt text][image2]|![alt text][image3]|![alt text][image4]|


##### - Data preprocessing
(load_data.py lines 19-44)

* Crop the images to remove the sky and the car front parts (`crop()`)
* Resize the images to 66x200 for input of NVIDIA model (`resize()`)
* Convert the images from RGB to YUV (`rgb2yuv()`)

This is an example of the preprocessed image.

![alt text][image5]

##### - Data augmentation
(load_data.py lines 58-124)

For training, the following augmentation techniques along with Python generator are used to generate augmented images:

* Randomly flip image left/right with steering angle (multiply -1 when flip) (`random_flip()`)
* Randomly translate image vertically and horizontally with steering angle adjustment (0.002 per pixel shift) (`random_translate()`)
* Randomly add shadows to the image (`random_shadow()`)
* Randomly brightness image (`random_brightness()`)

```python
# Image augmentation and steering angle adjust
def augment(center, left, right, angle, range_x=100, range_y=10):

    image, angle = choose_image(center, left, right, angle)
    image, angle = random_flip(image, angle)
    image, angle = random_translate(image, angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, angle
```

Original image:

![alt text][image6]

Random flip image:

![alt text][image7]

Random translate image:

![alt text][image8]

Random shadow image:

![alt text][image9]

Random brightness image:

![alt text][image10]

With data collection from two tracks and data augmentation, training data has a totally amount of **14190**.

### Simulation result

At the end of the process, the vehicle is able to drive autonomously around both the first and second track without leaving the road.

Here are the autonomous driving video on the two tracks.

* The Lake Track

(https://youtu.be/DCQp6IOHqak)

* The Jungle Track

(https://youtu.be/8BDNEGsyuu0)

We even performed some destructive tests. During autonomous driving, we tried to give manually interference and steer the car off course by acceleration or random steering.
In most cases, the car reacted by steering back to the center. Only in extreme cases, the car lost control and wonder off into the woods or into the lake.
