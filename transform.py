# imports
import numpy as np
import cv2


def order_points(pts):
    # reorders points into a list with (TopLeft, TopRight, BottomRight, BottomLeft)
    pts_list = np.zeros((4, 2), dtype="float32")

    # add together the X and Y cord for each point
    s = np.sum(pts, axis=1)
    # smallest X+Y is top left, largest is top right
    pts_list[0] = pts[np.argmin(s)]
    pts_list[2] = pts[np.argmax(s)]

    # find the difference between X and Y for each point
    d = np.diff(pts, axis=1)
    # smallest X-Y is top right, largest is bottom left
    pts_list[1] = pts[np.argmin(d)]
    pts_list[3] = pts[np.argmax(d)]

    # return the ordered list
    return pts_list


def four_point_transform(image, pts):
    # order the points
    pts_correct = order_points(pts)
    (tl, tr, br, bl) = pts_correct

    # calculate the width of the top and bottom, then find which is larger
    width_top = np.sqrt(((tr[0]-tl[0]) ** 2) + ((tr[1]-tl[1]) ** 2))
    width_bot = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bot))

    # calculate the height of the right and left, then find which is larger
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_right), int(height_left))

    # create an array with dimension of the max width and height forming a rectangle
    dst = np.array([
        [0,0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(pts_correct, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))

    # return the warped image
    return warped
