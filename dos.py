#https://medium.com/@alexppppp/adding-objects-to-image-in-python-133f165b9a01
from lib2to3.pytree import NegatedPattern
import pyrealsense2 as rs
import numpy as np
import cv2
import imutils
import time
import random
import threading
import simpleaudio as sa

OOF = sa.WaveObject.from_wave_file("Uffsound.wav")
OK = sa.WaveObject.from_wave_file("Ok_Sound_Effect.wav")
play_obj = None
# def playMusic():
#     for i in range (3):
#         bmusic = sa.WaveObject.from_wave_file("background.wav")
#         play_obj = bmusic.play()
#         play_obj.wait_done()

# TbackgroundMusic = threading.Thread(target = playMusic)
# TbackgroundMusic.start()

"""CONFIG"""
# HSV values to pick colour for the ball to track
ColourLower = (20, 86, 122)
ColourUpper = (133, 255, 255)
wall = 10000
HIT = False
ready = 0
points = 0
negativepoints = 0
GOAL = 0
oldTarget=(int(480/2),int(640/2))
maxscore = 100
currentTargetShape = None


def resizeImage(image, scale_percent):
    """Scales the image with a scale_percent"""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image


def output(x,y,depth):
    """Prints Ball and Target information to the terminal"""
    centerDistance = depth[oldTarget[0]][oldTarget[1]]/10
    print(f"Ball is at position ({x},{y}) and it is {depth[int(y)][int(x)]/10} cm away.")
    print(f"Distance in the center: {centerDistance} cm. Position: {oldTarget[0]} {oldTarget[1]}")


def frameWithTarget(color_image, targetImg, targetImg_mask, medianScale, position, ballCenter, radius):
    """Adds target to composite image"""
    # position (x,y): is the position of the target, will put a target in the middle of that
    # TODO: add blur? to make it looks more smooth
    # TODO: Check for out of range --> use the half of the target in the corner, if needed.
    targetImg = resizeImage(targetImg, medianScale)
    global currentTargetShape
    currentTargetShape = targetImg.shape

    targetImg_mask = resizeImage(targetImg_mask, medianScale)
    composition = color_image.copy()

    target_shp = targetImg.shape
    startPosX = 0
    startPosY = 0
    if target_shp != color_image.shape:
        # start positions of the targe, from the left
        startPosX = position[0] - target_shp[0]//2 
        startPosY = position[1] - target_shp[1]//2
    
    endPosX = startPosX+targetImg.shape[0] 
    endPosY = startPosY+targetImg.shape[1]
    composition[startPosX:endPosX,startPosY:endPosY] = composition[startPosX:endPosX,startPosY:endPosY] * targetImg_mask + \
            targetImg*(1-targetImg_mask)

    # Check if we detected ball. If yes, put ball on highest layer (over target)
    if radius is not None:
        (x,y) = ballCenter
        w0,h0,c0 = color_image.shape
        mask = np.zeros((w0,h0,c0),dtype=np.uint8)
        cv2.circle(mask,(int(x),int(y)),int(radius),(255,255,255),-1)
        composition[mask == 255] = color_image[mask == 255]
    return composition


def newTarget():
    """Picks new coordinates for where the target can appear"""
    first = random.randint(140,340)
    last = random.randint(140,500)
    return (first,last)


def check(x,y,depth_image):
    """Checks whether or not the ball has hit the target or missed the target. Main factor
    here is the depth as that will trigger the hit or miss check"""
    global ready, negativepoints
    balldepth = depth_image[int(y)][int(x)]/10
    row,col = (oldTarget[0],oldTarget[1])
    if balldepth > wall - 10 and balldepth < wall + 10:
        if int(y) > row - currentTargetShape[0]//2 and int(y) < row + currentTargetShape[0]//2:
            if int(x) > col - currentTargetShape[1]//2 and int(x) < col + currentTargetShape[1]//2:
                if time.time() - ready> 0.5:
                    ready = time.time()
                    print("HIT")
                    play_obj = OK.play()
                    play_obj.wait_done()
                    global HIT, points, GOAL
                    HIT=True
                    points+=1
                    GOAL += 1/np.linalg.norm(np.array([int(x),int(y)])-np.array(oldTarget))*balldepth*5
            else:
                if time.time()-ready > 0.5:
                    print("MISS")
                    play_obj = OOF.play()
                    play_obj.wait_done()
                    negativepoints+=1
                    ready=time.time()
        else:
            if time.time()-ready > 0.5:
                print("MISS")
                play_obj = OOF.play()
                play_obj.wait_done()
                negativepoints+=1
                ready=time.time()


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
#device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Programre quires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config) # Start streaming

targetImg_original = cv2.imread('target.png', cv2.IMREAD_UNCHANGED)
targetImg_original = resizeImage(targetImg_original, 30)[:,:,:3] # 30% of the size
targetImg_mask_original = cv2.imread('target_mask.png', cv2.IMREAD_UNCHANGED)/255 # normalize
targetImg_mask_original = resizeImage(targetImg_mask_original, 30)[:,:,:3] # don't need last channel

# Needed to make transition from scale to scale smooth (FIFO)
# Can change number 15. Higher number means that changes for scale will
# happens slower but smoother
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
        de_sh = oldTarget 
        centerDistance = depth_image[oldTarget[0]][oldTarget[1]]/10
        color_image = np.asanyarray(color_frame.get_data())


        blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, ColourLower, ColourUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        x = None
        y = None
        center = None
        radius = None
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
            # TODO: Use center of the ball, to find the distance, and use distance to calculate minimal radius. 
            #       Will be different for differnet sizes of the ball.
            if radius > 2:
                if wall != 10000 and wall != 0.0:
                    check(x,y,depth_image)
                # draw the circle and centroid on the color_image,
                # then update the list of tracked points
                cv2.circle(color_image, (int(x), int(y)), int(radius),(0, 255, 255), 2)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        targetImg = targetImg_original
        targetImg_mask = targetImg_mask_original
        medianScale = 100
        if not centerDistance <= 20:
            lastFewScales.pop(0) # remove oldest one
            if centerDistance > 1000: # no more than 10 meter of the distance
                centerDistance = 1000
            scale = (centerDistance-1100)/-10.3157894737 # Here we use 1100 because we are getting very small scale otherwise
            lastFewScales.append(scale)
            medianScale = np.median(lastFewScales)
        if HIT == True:
            HIT=False
            oldTarget = newTarget()
            print("---------------------------")
            print(f"WELL DONE, hit|miss is now: {points} | {negativepoints}")
            time.sleep(1)
            print("NEW TARGETS ARE;",oldTarget)
            box = []
            for n in range(10):
                for m in range(10):
                    a=depth_image[oldTarget[0]-5+n][oldTarget[1]-5+m]/10
                    if a != 0:
                        box.append(a)

            wall = np.median(box)
            print("New wall is",wall,"cm away")
            print(f"Total score is {np.round(GOAL,2)} out of {maxscore}")
        composition = frameWithTarget(color_image, targetImg, targetImg_mask, medianScale, [oldTarget[0],oldTarget[1]], (x,y), radius)
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
        if GOAL >= maxscore:
            print("\n==========================================================")
            print(f"You got {np.round(GOAL,2)} points and Hit: {points} and Missed: {negativepoints}")
            print("==========================================================\n")
            exit(1)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            # play_obj.stop() # stop music
            # TbackgroundMusic.join() # wait for thread
            break
        if key == ord("e"):
            # print("Wall was:", wall,"cm away")
            box = []
            # print(depth_image.shape)
            for n in range(10):
                for m in range(10):
                    a=depth_image[oldTarget[0]-5+n][oldTarget[1]-5+m]/10
                    if a != 0:
                        box.append(a)
            print("Wall is ",np.median(box),"cm away")
            wall = np.median(box)

finally:
    # Stop streaming
    pipeline.stop()


