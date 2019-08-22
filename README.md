## Advanced Lane-Lines by demigecko 

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
[image5]: ./output_images/warped_straight_lines.png "Warp Example"
[image6]: ./output_images/lane_pixels_fit.jpg "Poly Fit"
[image7]: ./output_images/image_unwarp_highlight.jpg "Unwarped and highlighted"
[video1]: ./output_video/project_video_annotated_full.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Please pay attention that I have two jupyter notebooks, named `Advanced Lane Finding_Steps` and `Advanced Lane Finding_Video`. The former file is to generate the step-by-step outcomes for Rubric Points, and the later one is mainly focused on the video output and debugging.  

### Camera Calibration 

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code of this step is contained in the first code cell of the Jypyter notebook located in `"./CarND-Advanced-Lane-Lines/Advanced Lane Finding_Steps.ipynb`  

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the transformation matrix of camera calibration and distortion coefficients using the `cv2.calibrateCamera()` .  I correct the distortion to those test images by using the `cv2.undistort()` and the first 4 images in the folder of  `camera_cal/` is shown below:  

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like the following:
![alt text][image2]

I setup a new function called `cal_undistort`  to undistort the test image and obtained this outcome from  `test_images/straight_lines2.jpg`:

![alt text][image3]

Note: my color space is in BGR

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `cal_undistort()`, which appears in the 3rd code cell of the Jupyter notebook).  The `cal_undistort()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the source and destination points by the line plots carefully. Note: The trick is: don't try to get the prefect straight lines after perspective transform, it will be hard for later polyfitting due to small curvature. Therefore, some degree of nonideal transform is good for later polyfitting.

This resulted in the following source and destination points:

```python
src = np.float32(
    [[280,  700],  # Bottom left
    [595,  460],   # Top left
    [725,  460],   # Top right
    [1125, 700]])  # Bottom right

dst = np.float32(
    [[250,  720],  # Bottom left
    [250,    0],   # Top left
    [1065,   0],   # Top right
    [1065, 720]])  # Bottom right

M = cv2.getPerspectiveTransform(src, dst)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][image5]


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color thresholds and binray opertaion to generate a binary image, and the code is in the 5 session. 

1. Import  images in BGR space by  `cv2.imread()`, it is important to distinguish this from `cv2.imread`, which is in RGB space.  
2. Use `cv2.split()` to separate three channels into (b, g, r) accordingly. Alternatively `b=img[:,:,0],g=img[:,:,1],r=img[:,:,2]`. 
> In [OpenCV-Python Tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html?highlight=split) :
> cv2.split() is a costly operation (in terms of time), so only use it if necessary. Numpy indexing is much more efficient and should be used if possible. 
3. The trick is to detect the yellow lanes and white lanes without involving the impact of shadow or other features.
    Yelow is the combination of the similar amount of *Red* and *Green*, but no *Blue*, so c1 is a true-and-false image array based on the following threshold  `(b < 120) & (r > 140) & (g > 140)`,  define `binary_1[c1]=1`
    White is the combination of the same aomnut of *Red*, *Green*, and *Blue*, so c2 is a true-and-false image array based on the following threhold  `c2 = (b > 180) & (r > 180) & (g > 180)` define `binary_2[c2]=1`
4. then I combined binray_1 and binary_2 by `bitwise_or()` that compute the bit-wise OR of two arrays element-wise.
5. In addition, to ensure the good quality of image process, I aslo calculate the mean value of the image by `np.mean()`. If the mean value is higher than 100, then the input image will be substarted by a uniform background image. This step improves the robustness of image process. 

Comments: in the OpenCV website they split colors due to heavy image proces, however, if we can implement such simple color pixel detection in the CCD in the first place, then this can greatly speed up the image process. 

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
The following step is optional.

```python
if np.mean(warped) > 100:
    background = np.ones(warped.shape, dtype="uint8") * 20
else: 
    background = np.ones(warped.shape, dtype="uint8") * 0    
    subtracted_image = cv2.subtract(warped, background)
```
Here's the all test images after my image process steps. As you can see, the signal-to-noise ratio is good.  

![alt text][image3-1]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the four functions `find_lane_pixels` , `fit_polynomial`,  `search_around_poly`  that Udacity provided in our leanring process. 

It is import to trace all the input the output in each function, and thier processing flow 

1. function of `find_lane_pixels(binary_warped)`
```
def find_lane_pixels(binary_warped):
    return leftx, lefty, rightx, righty, curve_fit, left_fitx, right_fitx, ploty
```
2. function of  `fit_poly(img_shape, leftx, lefty, rightx, righty)`

```
def fit_polynomial(binary_warped, leftx, lefty, rightx, righty, out_img, delta):
    return out_img, ploty, left_fit, right_fit
```
> `input` are `leftx`, `lefty`, `rightx`, `righty`   
> `output` are the fitted indices of  left lands and right lands in x and ploty is the shared y indices


3. function of `search_around_poly(binary_warped, left_fit, right_fit)` 
```
def search_around_poly(binary_warped, left_fit, right_fit)
    return result, left_fitx, right_fitx, ploty 
```

![alt text][image6]  
This image might not be shown properly in githut, please see it in Jupyter notebook

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Please switch to `Advanced Lane Finding_Video` 

I did this in the function of  `measure_curvature_pixels` , I basically followed the instruction to evaluate the curvature. In my code, I made a fair assumption that both left and right lanes have same curvature.  Therefore, I introduce a parameter `delta`, it is defined as the diffence of the max values of left and right of line graph of histogram.  In case of not being able to captaure the right lane accurately due to the nature of dashed lanes, I defined a global variable of `prev_delta` to keep the pipeline vaild while running. I set the condition that if the differnce of the peak values is large than 300, or the points of right lane is less than 100, the pipeline will use the previous `delta` that saved in `prev_delta` globally. 
```
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
leftx_value = max(histogram[:midpoint])
rightx_value = max(histogram[midpoint:])
if abs(leftx_value - rightx_value) > 300 or rightx_value < 100:
    delta = prev_delta
else: 
    delta = rightx_base - leftx_base
```
I assumed the curvature of the right and left lanes are the same. Therefore, I shifted all points of right lane by `delta`, and combine all data points of left and right lanes for the whole fitting. 
```
    x = np.append(leftx, rightx - delta, axis=0)
    y = np.append(lefty, righty, axis=0)
    left_fit = np.polyfit(y, x, 2)
    right_fit = np.polyfit(y, x, 2)
```
However, in the case of not being able to find the `curve_fit`, I introduce four global variables:  `prev_curve_fit` , `prev_leftx` , and  `prev_rightx`. These allows the program to keep running when there is any null array generated. I set the condition as below. 

```
if  leftx.size > 10 and rightx.size > 10:
    x = np.append(leftx, rightx-delta, axis=0)
    y = np.append(lefty, righty, axis=0)
    curve_fit = np.polyfit(y, x, 2)
    prev_curve_fit = curve_fit
    prev_leftx =leftx
    prev_rightx = leftx

else: 
    curve_fit = prev_curve_fit
    leftx = prev_leftx
    leftx = prev_rightx
```
After all these tricks, the proper `curve_fit` can be generated to the calculate the curvature. The key information is to know the pixel / m in both x and y directions.  
```
def measure_curvature_pixels(ploty,curve_fit):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    curvature = ((1 + (2*curve_fit[0]*y_eval + curve_fit[1])**2)**1.5) / np.absolute(2*curve_fit[0])
    return curvature
```

Once I have the curvature, then I defined the car center in the following: 
```
def car_offset(leftx, rightx, img_shape, xm_per_pix=3.7/800):
    mid_imgx = img_shape[1]//2      
    car_pos = (leftx[-1] + rightx[-1])/2
    offsetx = (mid_imgx - car_pos) * xm_per_pix
    return offsetx
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the following function: 
```
def unwarp_highlight(img, warp, left_points, right_points, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_fitx = left_points[0]
    right_fitx = right_points[0]
    ploty = left_points[1]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    reverse_warp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    return cv2.addWeighted(img, 1, reverse_warp, 0.3, 0)
```
Here is an example of my result on a test image with annotation: 

![alt text][image7]

---

### Pipeline (video)

#### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video is made based on `Advanced Lane Finding_Video.ipynb` that is a slim version of my codes to produce the video and video debugging. I successfully annotated the video of `project_video.mp4` in full length (50 secends) with some wobbly lines. No catastrophic failures that would cause the car to drive off the road! 

Here's a [link to my video result](./output_video/project_video_annotated_full.mp4)

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most of time I spent is to find the roubust lane detetion by cv2, and I didn't use any Sobel or gradient detetion. Solely I did it based on colors of White and Yellow and properly subtract the background when it is needed (not used in the video). This part of calculations can be implemented in the camera, a specialized camera for this purpose to speed up the image process. The pipeline esialy fails when it cannot capture the lanes. It is critcial to save some variables as globals variables. When the pipeline cannot detect the lane, it can take the previous values to continue running.

Here are the points I can imporve.

1. to save the `objpoints` and  `imgpoints` by pickle, and recall them when I need. 
2. to write the pipeplie in `class`, it is the technique I should develop sooner or later.
3. I have some recoreded videos by HERO, I would like to try the roubustness of my pipeline and futhur improve my pipeline
