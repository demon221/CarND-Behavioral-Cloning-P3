import cv2
import os
import numpy as np
import matplotlib.image as mpimg

fileDataPath = './data'
fileDataCSV = 'driving_log.csv'

img_rows, img_cols, ch = 66, 200, 3
input_shape = (img_rows, img_cols, ch)
correction = 0.25


def load_image(image_file):

    return mpimg.imread(os.path.join(fileDataPath, image_file.strip()))


# Crop the image, remove the sky at the top and the car front at the bottom
def crop(image, top_offset=.375, bottom_offset=.125):

    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    return image[top:-bottom, :, :] # remove the sky and the car front


# Resize the image to the input shape used by Nvidia's network model
def resize(image):

    return cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_AREA)


# Convert the image to YUV color space used by Nvidia's network model
def rgb2yuv(image):

    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):

    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


# Randomly choose an image from the center, left or right, and adjust the steering angle
def choose_image(center, left, right, angle):

    choice = np.random.choice(3)
    if choice == 0:
        return load_image(left), angle + correction
    elif choice == 1:
        return load_image(right), angle - correction
    return load_image(center), angle


# Random flip the image left <-> right
def random_flip(image, angle):

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle


# Random translate the image vertically and horizontally
def random_translate(image, angle, range_x, range_y):

    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, angle


# Random addition of shadows
def random_shadow(image):

    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = image.shape[1] * np.random.rand(), 0
    x2, y2 = image.shape[1] * np.random.rand(), image.shape[0]
    xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


# Random brightness adjust
def random_brightness(image):

    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] = hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# Image augmentation and steering angle adjust
def augment(center, left, right, angle, range_x=100, range_y=10):

    image, angle = choose_image(center, left, right, angle)
    image, angle = random_flip(image, angle)
    image, angle = random_translate(image, angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, angle


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

