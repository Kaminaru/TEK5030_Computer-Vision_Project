import pyrealsense2 as rs
import numpy as np
import cv2
import imutils
import time
from cv2 import aruco
import random
import tkinter as tk


# HSV values to pick colour for the ball to track
ColourLower = (20, 86, 122)
ColourUpper = (133, 255, 255)
HIT = False 
ready = 0 # used for timer
points = 0
negativepoints = 0
GOAL = 0
maxScore = 100



def resizeImage(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

def output(x,y,depth):
    centerDistance = depth[int(depth.shape[0]/2)][int(depth.shape[1]/2)]/10
    print(f"Ball is at position ({x},{y}) and it is {depth[int(y)][int(x)]/10} cm away.")
    print(f"Distance in the center: {centerDistance} cm. Position: {int(depth.shape[0]/2)} {int(depth.shape[1]/2)}")

def check(x,y,depth_image, depthList, xl_yh_list, currentActTargets):
    global ready, negativepoints, HIT, points, GOAL
    ballDepth = depth_image[int(y)][int(x)]/10
    
    for disToTarget, cords, id in zip(depthList,xl_yh_list, currentActTargets):
        if ballDepth > disToTarget - 10 and ballDepth < disToTarget + 10:
            print(ballDepth , " | " , disToTarget)

            # center of the target
            if int(y) > cords[2] and int(y) < cords[3]:
                if int(x) > cords[0] and int(x) < cords[1]:
                    if time.time() - ready > 1:
                        ready = time.time()
                        print("HIT")
                        print(f". ball:({int(x),int(y)}) - Depth of ball = {ballDepth} \n Depth of target ={disToTarget} TargetID: {id}")
                        HIT = True
                        points+=1
                        midToTheTarget = ((cords[1]-cords[0])/2 + cords[0],(cords[3]-cords[2])/2 + cords[2])
                        GOAL+=1/np.linalg.norm(np.array([int(x),int(y)])-np.array(midToTheTarget))*disToTarget*5
                else:
                    if time.time()-ready > 1:
                        print("MISS")
                        negativepoints+=1
                        ready=time.time()
            else:
                if time.time()-ready > 1:
                    print("MISS")
                    negativepoints+=1
                    ready=time.time()

def addImageOnMarker(corner, frame, targetImg):
    # TODO: does not work when sizeDiff negative number, need to check why
    # topLeft     = corner[0][0][0]-sizeDiff, corner[0][0][1]-sizeDiff
    # topRight    = corner[0][1][0]+sizeDiff, corner[0][1][1]-sizeDiff
    # bottomRight = corner[0][2][0]+sizeDiff, corner[0][2][1]+sizeDiff
    # bottomLeft  = corner[0][3][0]-sizeDiff, corner[0][3][1]+sizeDiff

    topLeft     = corner[0][0][0], corner[0][0][1]
    topRight    = corner[0][1][0], corner[0][1][1]
    bottomRight = corner[0][2][0], corner[0][2][1]
    bottomLeft  = corner[0][3][0], corner[0][3][1]

    h, w, c = targetImg.shape
    
    pts1 = np.array([topLeft, topRight, bottomRight, bottomLeft])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix, _ = cv2.findHomography(pts2, pts1) 
    imgOut = cv2.warpPerspective(targetImg, matrix, (frame.shape[1], frame.shape[0]))

    cv2.fillConvexPoly(frame, pts1.astype(int), (0,0,0))
    imgOut = frame + imgOut
    return imgOut


def addImage(corner, frame, fileName, depthImage):
    targetImg = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)[:,:,:3]
    frame = addImageOnMarker(corner, frame, targetImg)
    # To fix problem with rotation, I decided to find the lowest sum of x+y, in the each corner.
    # In our task we don't care about rotation of the image when we are trying to find depth to it.
    # But because how aruco implemented, it will rearange order of corners list to make if marker is rotated
    # So in theary upper left corenr will have lowest sum of x and y
    lowestSum = float('inf')
    lowestIndex = None
    for i in range(len(corner[0])):
        sum = corner[0][i][0] + corner[0][i][1]
        if sum < lowestSum:
            lowestSum = sum
            lowestIndex = i

    # makes list with need order
    sortedList = []
    sortedList.append(corner[0][lowestIndex])
    if lowestIndex == 3: # last element
        sortedList.append(corner[0][0]) # one in front
        sortedList.append(corner[0][lowestIndex-1]) # one behind
    else:
        sortedList.append(corner[0][lowestIndex+1]) # one in front (represent right top corner from left top)
        sortedList.append(corner[0][lowestIndex-1]) # one behind (represnet left bottom corner from left top)
    xl,xh,yl,yh = int(sortedList[0][1]), int(sortedList[2][1]), int(sortedList[0][0]), int(sortedList[1][0])
    squareDepth = depthImage[xl:xh,yl:yh]

    depthMedian = np.median(squareDepth)/10
    return frame, depthMedian, [xl,xh,yl,yh]


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

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


    # TODO: change it to parameter later on
    files = ["amongblue.png", "amongpink.png", "amongred.png", "amongnocolor.png"]
    # TODO: User input from terminal or GUI
    numOfActTargets = 2
    currentActTargets = [] # ids of the active markers for the game
    depthList = [] # list of the depth to every target
    xl_yh_list = [] # list of the square for each of the target area
    cornerList = []
    gameOn = False


    
    # def increaseNumOfTargets(testLabel):
    #     print("HEI")
    #     if numOfActTargets < 3:
    #         numOfActTargets += 1
    #         Window.update()
    #         testLabel['text'] = f"Number of Targets: {numOfActTargets}"

    # def decreaseNumOfTargets(testLabel):
    #     print("MINUss")
    #     if numOfActTargets > 0:
    #         numOfActTargets -= 1
    #         Window.update()
    #         testLabel['text'] = f"Number of Targets: {numOfActTargets}"

    # # set up GUI
    # Window = tk.Tk()
    # Window.geometry("250x550")
    # Window.title("Scoreboard")
    # testLabel = tk.Label(text=f"Number of Targets: {numOfActTargets}")
    # testLabel.place(x = 40, y = 60)
    # buttonMinus = tk.Button(Window, text ="-", command = lambda: decreaseNumOfTargets(testLabel))
    # buttonMinus.pack()
    # buttonPlus = tk.Button(Window, text ="+", command = lambda: increaseNumOfTargets(testLabel))
    # buttonPlus.place(x = 80, y = 100)

    # Window.update()
    


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
            color_image = np.asanyarray(color_frame.get_data())

            # Find markers
            frame = color_image.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters =  aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if len(corners) > 1 and len(corners) >= numOfActTargets: # if we have enough markers to work with
                if not gameOn: # set ut targets
                    corners_sub, ids_sub = zip(*random.sample(list(zip(corners, ids)), numOfActTargets))
                    gameOn = True
                    for corner, id in zip(corners_sub, ids_sub):
                        if int(id) > 3:
                            fileName = files[-1] # no color among us
                        else:
                            fileName = files[int(id)]
                        frame, depthMedian, xl_yh = addImage(corner, frame, fileName, depth_image)
                        cornerList.append(corner)
                        depthList.append(depthMedian)
                        xl_yh_list.append(xl_yh)
                        currentActTargets.append(int(id))
                
                
                for corner, id in zip(corners, ids):
                    if id in currentActTargets:
                        if int(id) > 3:
                            fileName = files[-1] # no color among us
                        else:
                            fileName = files[int(id)]
                        targetImg = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)[:,:,:3]
                        frame, depthMedian, xl_yh = addImage(corner, frame, fileName, depth_image)

                        frame = cv2.putText(frame, f'{depthMedian}', (int(corner[0][0][0]),int(corner[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


                # if gameOn:
                #     for corner, id in zip(cornerList, currentActTargets):
                #         if int(id) > 3:
                #             fileName = files[-1] # no color among us
                #         else:
                #             fileName = files[int(id)]
                #         targetImg = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)[:,:,:3]
                #         frameNew = addImageOnMarker(corner, frame.copy(), targetImg)
                #         cv2.imshow('TESTING', frameNew)
                        

            blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # construct a mask for the color "green", then perform
            # a series of dilations and erosions to remove any small
            # blobs left in the mask
            mask = cv2.inRange(hsv, ColourLower, ColourUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            # find contours in the mask and initialize the current
            # (x, y) center of the ball
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
                # TODO: Use center of the ball, to find the distance, and use distance to calculate minimal radius. 
                # TODO: Will be different for differnet sizes of the ball.
                if radius > 5:
                    #check(x,y,depth_image, depthList, xl_yh_list, currentActTargets)
                    #output(x,y,depth_image) # prints to terminal info from function
                    # draw the circle and centroid on the color_image
                    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)

                    # Check if we detected ball. If yes, put ball on highest layer (over target)
                    # In this case not really needed becuase we still have problem when there is something
                    # over marker, marker won't work.
                    w0,h0,c0 = frame.shape
                    mask = np.zeros((w0,h0,c0),dtype=np.uint8)
                    cv2.circle(mask,(int(x),int(y)),int(radius),(255,255,255),-1)
                    frame[mask == 255] = frame[mask == 255]

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                #images = np.hstack((resized_color_image, depth_colormap))
                images = np.hstack((frame, depth_colormap))
            else:
                images = np.hstack((frame, depth_colormap))
                #images = np.hstack((frame, color_image, depth_colormap))
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            #cv2.imshow('Marker Window', frame)

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break
            elif key == ord("r"):
                gameOn = False
                cornerList = []
                depthList  = []
                xl_yh_list = []
                currentActTargets = []
                # userNumInp = input("Write the number of targets: ")
                # numOfActTargets = int(userNumInp)

    finally:
        # Stop streaming
        pipeline.stop()
































# def checkMarker(frame, depthImage, corner,id):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#     parameters =  aruco.DetectorParameters_create()
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
#     # if len(corners) > 0:
#     #     a = corners[0][0] # size 4: | 0: left upper | 1: right upper | 2: right lower | 3: left lower
#     #     # print(corners[0][0])
#     #     # print(int((a[2][1]-a[0][1])/2),int((a[2,0]-a[0][0])/2))
#     #     # cv2.circle(frame, (int((a[2][1]-a[0][1])/2),int((a[2,0]-a[0][0])/2)), 5, color=(0, 255, 0), thickness=-1)
#     #     # print(f"{int(a[0][0])}-{int(a[0][1])}")
    
#     #     # midX = int((a[2][0]-a[0][0])/2) + int(a[0][0])
#     #     # midY = int((a[2][1]-a[0][1])/2) + int(a[0][1])
#     #     # cv2.circle(frame, (midX,midY), 10, color=(0, 255, 0), thickness=-1) # green

#     #     cv2.circle(frame, (int(a[0][0]),int(a[0][1])), 5, color=(0, 0, 255), thickness=-1) # red
#     #     cv2.circle(frame, (int(a[1][0]),int(a[1][1])), 5, color=(0, 255, 0), thickness=-1) # green
#     #     cv2.circle(frame, (int(a[2][0]),int(a[2][1])), 5, color=(255, 0, 0), thickness=-1) # blue
#     #     cv2.circle(frame, (int(a[3][0]),int(a[3][1])), 5, color=(50, 50, 50), thickness=-1) # black ish
#     #     # cornerDepth = depthImage[int(a[0][0]):int(a[3][0]),int(a[0][1]):int(a[1][1])]

#         # print(f"ID1 Distance: {np.median(cornerDepth)/10}")
#         # print(f"{int(a[3][0])}-{int(a[0][0])}")
#         # print(int(a[3][0])-int(a[0][0]))
#         # print(int(a[1][1])-int(a[0][1]))

#     #frameOut = frame.copy()
#     # TODO: change it to parameter later on
#     files = ["amongblue.png", "amongpink.png", "amongred.png"]
#     distances = []
#     if len(corners) > 0:
#         for corner, id in zip(corners,ids):
#             if id < 3: # in case there is more markers, but we only have 3 images to use
#                 targetImg = cv2.imread(files[int(id)], cv2.IMREAD_UNCHANGED)
#                 targetImg = resizeImage(targetImg, 30)[:,:,:3]
#                 frame = addImageOnMarker(corner, frame, targetImg)
#                 # To fix problem with rotation, I decided to find the lowest sum of x+y, in the each corner.
#                 # In our task we don't care about rotation of the image when we are trying to find depth to it.
#                 # But because how aruco implemented, it will rearange order of corners list to make if marker is rotated
#                 # So in theary upper left corenr will have lowest sum of x and y
#                 lowestSum = float('inf')
#                 lowestIndex = None
#                 for i in range(len(corner[0])):
#                     sum = corner[0][i][0] + corner[0][i][1]
#                     if sum < lowestSum:
#                         lowestSum = sum
#                         lowestIndex = i

#                 # makes list with need order
#                 sortedList = []
#                 sortedList.append(corner[0][lowestIndex])
#                 if lowestIndex == 3: # last element
#                     sortedList.append(corner[0][0]) # one in front
#                     sortedList.append(corner[0][lowestIndex-1]) # one behind
#                 else:
#                     sortedList.append(corner[0][lowestIndex+1]) # one in front (represent right top corner from left top)
#                     sortedList.append(corner[0][lowestIndex-1]) # one behind (represnet left bottom corner from left top)
#                 squareDepth = depthImage[int(sortedList[0][1]):int(sortedList[2][1]),int(sortedList[0][0]):int(sortedList[1][0])]

#                 #squareDepth = depthImage[int(a[0][0][1]):int(a[0][3][1]),int(a[0][0][0]):int(a[0][1][0])]


#                 # imageTest = frame.copy()
#                 # org = (int(corner[0][0][0]), int(corner[0][0][1]))
#                 # imageTest = cv2.putText(imageTest, str(org), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # blue
#                 # org = (int(corner[0][1][0]), int(corner[0][1][1]))
#                 # imageTest = cv2.putText(imageTest, str(org), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # red
#                 # org = (int(corner[0][2][0]), int(corner[0][2][1]))
#                 # imageTest = cv2.putText(imageTest, str(org), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # green
#                 # org = (int(corner[0][3][0]), int(corner[0][3][1]))
#                 # imageTest = cv2.putText(imageTest, str(org), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # white
#                 # cv2.imshow("TEXT", imageTest)
#                 # print(int(corner[0][0][1]) , "-" , int(corner[0][3][1]))
#                 # print(int(corner[0][0][0]) , "-" , int(corner[0][1][0]))
#                 # print("------------------------")
#                 # cv2.waitKey(1)
#                 # import time
#                 # time.sleep(1)
#                 depthMedian = np.median(squareDepth)/10
#                 distances.append(depthMedian)
#     # print(distances)
#     # cv2.imshow("TESTING", frame)
#     # cv2.waitKey(1)

#     return frame, distances, ids