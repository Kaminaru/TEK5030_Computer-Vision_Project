#https://medium.com/@alexppppp/adding-objects-to-image-in-python-133f165b9a01
import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
import imutils
import time

"""CONFIG"""
# HSV values to pick colour for the ball to track
ColourLower = (0, 100, 20) # 29, 86, 6
ColourUpper = (10, 255, 255) # 64, 255, 255
buff_size = 64 # max buffer size
pts = deque(maxlen=buff_size)
# vs = VideoStream(src=2).start()
start =time.time()


def resizeImage(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

def output(x,y,depth):
    centerDistance = depth[int(depth.shape[0]/2)][int(depth.shape[1]/2)]/10
    print(f"Ball is at position ({x},{y}) and it is {depth[int(y)][int(x)]/10} cm away.")
    print(f"Distance in the center: {centerDistance} cm")
    # print(f"Wall is {centerDistance} away")

def frameWithTarget(color_image, targetImg, position):
    # position (x,y): is the position of the target, will put a target in the middle of that
    # TODO: add blur? to make it looks more smooth
    # TODO: !!! Check for out of range --> use the half of the target in the corner, if needed.

    composition = color_image.copy()
    composition = cv2.cvtColor(composition, cv2.COLOR_RGB2RGBA)
    #composition[:, :, 3] = np.zeros((composition.shape[0], composition.shape[1]))
    alpha_composition = composition[:,:,3] / 255.0
    alpha_targetImg = targetImg[:,:,3] / 255.0

    # needed variables:
    target_shp = targetImg.shape
    # start positions of the targe, from the left
    startPosX = position[0] - target_shp[0]//2 
    startPosY = position[1] - target_shp[1]//2

    endPosX = startPosX+targetImg.shape[0] 
    endPosY = startPosY+targetImg.shape[1]
    # set adjusted colors
    # https://en.wikipedia.org/wiki/Alpha_compositing
    for color in range(0, 3):
        composition[startPosX:endPosX,startPosY:endPosY,color] = alpha_targetImg * targetImg[:,:,color] + \
            alpha_composition[startPosX:endPosX,startPosY:endPosY] * composition[startPosX:endPosX,startPosY:endPosY,color] * (1 - alpha_targetImg)

    # set adjusted alpha and denormalize back to 0-255
    composition[startPosX:endPosX,startPosY:endPosY,3] = (1 - (1 - alpha_targetImg) * (1 - alpha_composition[startPosX:endPosX,startPosY:endPosY])) * 255

    return composition


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config) # Start streaming

targetImg_original = cv2.imread('target.png', cv2.IMREAD_UNCHANGED)
targetImg_original = resizeImage(targetImg_original, 30) # 30% of the size

# needed to make transition from scale to scale smooth (FIFO)
# can change number 15. Higher number means that changes for scale will happens
# slower but smoother
lastFewScales = [100]*15

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        de_sh = depth_image.shape 
        centerDistance = depth_image[int(de_sh[0]/2)][int(de_sh[1]/2)]/10
        # print(f"Distance in the center: {centerDistance} cm")
        # print(f"Wall is {depth_image[int(de_sh[0]/2)][int(de_sh[1]/2)]/10} away")
        color_image = np.asanyarray(color_frame.get_data())
        # print(color_image)

        """ADDED"""
        # resize the frame[removed this part] blur it, and convert it to the HSV color space
        blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "red", then perform
	    # a series of dilations and erosions to remove any small
	    # blobs left in the mask
        mask = cv2.inRange(hsv, ColourLower, ColourUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
	    # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            # print(radius)
            # print(x,y)
            if radius > 5:
                if time.time()-start > 1:
                    start=time.time()
                    output(x,y,depth_image) # prints to terminal info from function
                    balldist = depth_image[int(y)][int(x)]/10
                    walldist = centerDistance
                    print(balldist,walldist)
                    if balldist< walldist or balldist < depth_image[int(de_sh[0]/2)+30][int(de_sh[1]/2)+30]/10:
                        print("front")
                    # exit()
                # draw the circle and centroid on the color_image,
                # then update the list of tracked points
                cv2.circle(color_image, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(color_image, center, 5, (0, 0, 255), -1)
        # update the points queue
        pts.appendleft(center)
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(buff_size / float(i + 1)) * 2.5)
            cv2.line(color_image, pts[i - 1], pts[i], (0, 0, 255), thickness)
        cv2.imshow("Frame", color_image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        # if key == ord("q"):
        #     break
        # exit()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # TODO refresh scale less, so it will be more stable. OR save scale each 5 frames and use the mean of it, to make it even more smooth
        # calculate needed size given distance. Lowest distance to the target 20, highest 1000
        # 20 will have original_scale, 1000 will have the 5% from the original
        targetImg = targetImg_original
        if not centerDistance <= 20:
            lastFewScales.pop(0) # remove oldest one
            scale = (centerDistance-1000)/-10.3157894737
            lastFewScales.append(scale)
            targetImg = resizeImage(targetImg_original, np.mean(lastFewScales))
        composition = frameWithTarget(color_image, targetImg, [int(de_sh[0]/2),int(de_sh[1]/2)]) # [300,300]
        # display the image
        cv2.imshow("Composited image", composition)

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1) & 0xFF
	    # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

finally:
    # Stop streaming
    pipeline.stop()


