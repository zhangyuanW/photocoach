# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
# Reading the required image in 
# which operations are to be done. 
# Make sure that the image is in the same 
# directory in which this python program is
#13565
def lines(image):
     
    # Convert the img to grayscale
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
     
    # Apply edge detection method on the image
    edges = cv2.Canny(image,200, 450,apertureSize = 3)
    #print edges[1] 
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges,1,np.pi/180, 200)
    goodline = 0
    returnValue = 0.0
    hLineNum = 0
    vLineNum = 0
    hRad = 0.0
    vRad = 0.0
    zero = False
    xPos = -1
    yPos = -1

    if isinstance (lines, np.ndarray):
        total= len(lines[0])
        #print total
        for r, theta in lines[0]:
            if (theta > 6.19592) or (theta < 0.0872665)  \
            or (theta > 1.48353 and theta < 1.65806) \
            or (theta > 3.05433 and theta < 3.22886) \
            or (theta > 4.62512 and theta < 4.79966):
                goodline += 1
            #print theta 
            
            if (theta > 3.92699 and theta < 5.49779) \
            or(theta > 0.785398 and theta < 2.35619):

                # Stores the value of cos(theta) in a
                a = np.cos(theta)

                # Stores the value of sin(theta) in b
                b = np.sin(theta)


                # y0 stores the value rsin(theta)
                y0 = b*r

                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1000*(a))
                #print y1
                
                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1000*(a))
                #print y2
                
                yPos += (y1+y2)/2
                
                hLineNum += 1
                hRad += theta
            
            if theta > 5.49779 or theta< 0.785398 \
            or(theta > 2.35619 and theta< 3.92699):

                a = np.cos(theta)

                # Stores the value of sin(theta) in b
                b = np.sin(theta)

                # x0 stores the value rcos(theta)
                x0 = a*r

                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1000*(-b))
                #print x1

                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1000*(-b))
                #print x2
                xPos += (x1+x2)/2
                vLineNum += 1
                vRad += theta
            
                
    else:
        zero = True
        f0 = 0
        
    if hLineNum == 0:
        f1 = -1
    else:
        f1 = hRad *1.0/hLineNum
        
    if vLineNum == 0:
        f2 = -1
    else:
        f2 = vRad *1.0/vLineNum

    if zero == False:
        f0 = goodline * 1.0 / total

    #print hLineNum
    #print vLineNum

    if xPos == -1:
        f4 = -1
    else:
        f4 =xPos/vLineNum

    if yPos == -1:
        f3 = -1
    else:
        f3=yPos/hLineNum

    #print f3
    #print f4
    #print lines
    #print lines[0]
     
    # The below for loop runs till r and theta values 
    # are in the range of the 2d array
    #print lines[0][1]
    """
    if isinstance (lines, np.ndarray):
      
        for r,theta in lines[0]:

            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a*r

            # y0 stores the value rsin(theta)
            y0 = b*r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000*(-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000*(a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000*(-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000*(a))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be 
            #drawn. In this case, it is red. 
            cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)
    else:
        print 000


    print f0
    print f1
    print f2
    print f3
    print f4
    """
    return np.array([f0, f1, f2, f3, f4]);
    #print f0 
    #print f1
    #print f2
    # All the changes made in the input image are finally
    # written on a new image houghlines.jpg
    """
    cv2.imwrite('/Users/ziranzhang/Desktop/cs221/PhotoQualityDataset/houghlines16.jpg', img)

    #plt.figure(figsize=(30,60)),plt.imshow(edges)
    plt.figure(figsize=(30,40)), plt.imshow(img)
    plt.show()
    """