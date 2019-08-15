## Advanced Lane-Lines by demigecko 

### This writeup is still in a working progress. 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/undistort_test1.png "Undistorted"
[image3-1]: /output_images/combo_all_test_images.jpg "combo"
[image3-2]: ./output_images/line-plot.png "Line-Plot"
[image4]: ./output_images/binary_combo_final.png "Binary Example"
[image5]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image6]: ./examples/color_fit_lines.jpg "Fit Visual"
[image7]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the Jypyter notebook located in "./CarND-Advanced-Lane-Lines/Advanced-Lane-Lines.ipynb"  

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the calibration images using the `cv2.undistort()` function and obtained this result from the first 4 images in the folder of  `camera_cal/` :  

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I setup a new function called `cal_undistort`  to undistort the test image and obtained this outcome from  `test_images/straight_lines2.jpg`:

![alt text][image3]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color thresholds and binray opertaion to generate a binary image, and the code is in the 5 session. 

1. Import  images in BGR space by  `cv2.imread()`, it is important to distinguish this from `cv2.imread`, which is in RGB space.  
2. Use `cv2.split()` to separeate three channels into (b, g, r) accordingly. Alternatively `b=img[:,:,0],g=img[:,:,1],r=img[:,:,2]`. 
> In [OpenCV-Python Tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html?highlight=split) :
> cv2.split() is a costly operation (in terms of time), so only use it if necessary. Numpy indexing is much more efficient and should be used if possible. 
3. The trick is to detect the yellow lanes and white lanes without involving the impact of shaow or other features.
    Yelow is the combination of the similar amount of *Red* and *Green*, but no *Blue*, so c1 is a true-and-false image array based on  the following threshold  `(b < 120) & (r > 140) & (g > 140)`,  define `binary_1[c1]=1`
    White is the combination of the same aomnut of *Red*, *Green*, and *Blue*, so c2 is a true-and-false image array based on the following threhold  `c2 = (b > 180) & (r > 180) & (g > 180)` define `binary_2[c2]=1`
4. then I combined binray_1 and binary_2 by `bitwise_or()` that compute the bit-wise OR of two arrays element-wise.
5. In addition, to ensure the good quliaty of image process, I aslo calculate the mean value of the image by `np.mean()`. If the mean value is higher than 100, then the input image will be substarted by a uniform background image. This step imporves the robustness of image process. 

Comments: in the OpenCV website they split colors due to heavy image proces, however, if we can implmenting such simple color pixel detetion in the CCD in the first place, then this can greatly speed up the image process. 

```python
def advance_procces(image):
    (b, g, r) = cv2.split(image)
    c1 = (b < 120) & (r > 140) & (g > 140) # pick the yellow lane 
    c2 = (b > 180) & (r > 180) & (g > 180) # pick the white lane
    binary_1 = np.zeros_like(b)
    binary_2 = np.zeros_like(b)
    binary_1[c1]=1
    binary_2[c2]=1
    bitwise_or = cv2.bitwise_or(binary_1, binary_2)
    return bitwise_or
```
```python
if np.mean(warped) > 100:
    background = np.ones(warped.shape, dtype="uint8") * 20
else: 
    background = np.ones(warped.shape, dtype="uint8") * 0    
    subtracted_image = cv2.subtract(warped, background)
```
Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3-1]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, and the code is in the 3 session. 
1. Convert the test image from RGB to HLS (hue, lightness, saturation) and extarct the S-channel. The parameter of sautation can improve the lane detetion in differnent color (i.e. White or Yellow) even if under shadow of trees. 
2. Convert the same test image to gray, and use Sobel operator to detect edges. I use only x-Sobel for lane detetion to remove any horizontal features. 
3. Combine the outcome from  (step1 **or** step 2) to provide the good lane detetion. 


```python
# Threshold x gradient
thresh_min = 20 
thresh_max = 200 

# Threshold color channel
s_thresh_min = 120 
s_thresh_max = 255 
```
Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 3rd code cell of the Jupyter notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose  the source and destination points by the line plots carefully.

![alt text][image3-1]

```python
src = np.float32(
    [[274,680],[1046,680],[546,485],[743,485]])
dst = np.float32(
    [[274,680],[1046,680],[274,485],[1046,485])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 274, 680     | 274, 680        | 
| 1046, 680   | 1046, 680      |
| 546, 485     | 274, 468      |
| 743, 485     | 1046, 468        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most of time I spend is to find the roubust lane detetion by cv2 and  


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Here is the point I can imporve.

1. to save the `objpoints` and  `imgpoints` by pickel, and recall them when I need. It would save some processing time.
2. to write the pipeplie in `class`, I think this is the technique I should develop sooner or later 
3. I recoreded some videos by HERO, I would like to try the roubustness of my 
