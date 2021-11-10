import cv2
import numpy as np
import matplotlib.pyplot as plt


points = [(60, 285), (70, 502), (320, 481), (295, 275)]
cat_points = []

# Mouse callback function. Appends the x,y location of mouse click to a list. 
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cat_points.append((y, x))


image = cv2.imread("Guy-Holding-Cardboard-Sign.jpg")
cat_image = cv2.imread("cat.jpg")

c = image.copy()

def inside_bounds(top_left, top_right, bot_left, bot_right, x, y):
    Y = 0
    X = 1
    top_slope = lambda a: ((top_left[Y] - top_right[Y]) / (top_left[X] - top_right[X])) * a
    top_slope_bias = top_left[Y] - top_slope(top_left[X])
    top_slopewb = lambda a: ((top_left[Y] - top_right[Y]) / (top_left[X] - top_right[X])) * a + top_slope_bias

    bot_slope = lambda a: ((bot_left[Y] - bot_right[Y]) / (bot_left[X] - bot_right[X])) * a
    bot_slope_bias = bot_left[Y] - bot_slope(bot_left[X])
    bot_slopewb = lambda a: ((bot_left[Y] - bot_right[Y]) / (bot_left[X] - bot_right[X])) * a + bot_slope_bias

    left_slope = lambda a: ((bot_left[X] - top_left[X]) / (bot_left[Y] - top_left[Y])) * a
    left_slope_bias = top_left[X] - left_slope(top_left[Y])
    left_slopewb = lambda a: ((bot_left[X] - top_left[X]) / (bot_left[Y] - top_left[Y])) * a + left_slope_bias

    right_slope = lambda a: ((bot_right[X] - top_right[X]) / (bot_right[Y] - top_right[Y])) * a
    right_slope_bias = top_right[X] - right_slope(top_right[Y])
    right_slopewb = lambda a: ((bot_right[X] - top_right[X]) / (bot_right[Y] - top_right[Y])) * a + right_slope_bias
    
    # (y >= top_slopewb(x)) and (y <= bot_slopewb(x)) and 
    if (x >= left_slopewb(y)) and (y >= top_slopewb(x)) and (y <= bot_slopewb(x)) and (x <= right_slopewb(y)): 
        return True
    return False

height, width = c.shape[:2]


y_offset = points[0][0]
x_offset = points[0][1]

b = c.copy()
cv2_points = []
for i in points:
    cv2_points.append((i[1], i[0]))

cv2_points = np.asarray(cv2_points)


cv2_points[:,0] -= cv2_points[0,0]
cv2_points[:,1] -= cv2_points[0,1]
H, _ = cv2.findHomography(np.asarray([[0,0], [cat_image.shape[1],0],  [cat_image.shape[1], cat_image.shape[0]], [0, cat_image.shape[0]]]), np.asarray(cv2_points))

output_width = 415
output_height = 642

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', (0,0,0))
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

bgr_ortho = cv2.warpPerspective(cat_image, H, (output_width, output_height))

empty = b.copy()
for i in range(empty.shape[0]):
    for j in range(empty.shape[1]):
        empty[i,j] = (0,0,0)

empty[y_offset:y_offset+bgr_ortho.shape[0], x_offset:x_offset+bgr_ortho.shape[1]] = bgr_ortho

overlay_bool = np.zeros((empty.shape[0], empty.shape[1]))
overlay_bool[y_offset:y_offset+bgr_ortho.shape[0], x_offset:x_offset+bgr_ortho.shape[1]] = 1


for y in range(height):
    for x in range(width):
        if overlay_bool[y, x] == 0 or not inside_bounds(points[0], points[1], points[3], points[2], x, y):
            empty[y, x] = c[y,x]


cv2.imshow("OUTPUT IMAGE", empty)
cv2.waitKey(0)

cv2.imwrite("cardboard_cat.jpg", empty)
