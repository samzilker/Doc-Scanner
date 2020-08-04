# imports
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import pytesseract
import os

# Comment block --------
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
# args = vars(ap.parse_args())
# ----------------------

# load the image and resize it
image = cv2.imread('/Users/samzilker/Documents/image.jpeg')
ratio = image.shape[0] / 500
orig = image.copy()
image = imutils.resize(image, height=500)

# convert to greyscale, blur, and find edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the orginal and edge deteched image
print("Step 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours and keep the largest
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
    else:
        print("Error")

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert to grey scale then threshold it
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

# show original image and scanned image
print("Step 3: Final Image")
cv2.imshow("Orignal", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)

# convert scanned to pdf
print("Step 4: PDF created")
pdf = pytesseract.image_to_pdf_or_hocr(warped, extension='pdf')
with open(os.path.join('/Users/samzilker/Documents/', 'Scan.pdf'), 'w+b') as f:
    f.write(pdf)
