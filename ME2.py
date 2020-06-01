# Machine Exercise 2

def distort(input_image, output_image, patch):
    import cv2
    import random
    import math
    import numpy as np

    # determine bounds
    in_img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    size = in_img.shape[:2]
    upper_bound_x = size[0]
    upper_bound_y = size[1]

    # randomize distortion
    source_points = patch
    destination_points = []
    for entry in source_points:
        x = entry[0]
        y = entry[1]
        not_ok = True
        while(not_ok):
            x_dest = random.randrange(x + math.floor((x * 0.1)/2), x + math.floor(x * 0.4/2))
            y_dest = random.randrange(y + math.floor((y * 0.1)/2), y + math.floor(y * 0.4/2))
            if x_dest <= upper_bound_x and y_dest <= upper_bound_y:
                not_ok = False
                x_dest = math.floor(x_dest*0.7)
                y_dest = math.floor(y_dest*0.7)
        destination_points.append([x_dest, y_dest])

    # convert to numpy arrays
    source_points = np.asarray(source_points)
    destination_points = np.asarray(destination_points)

    h, status = cv2.findHomography(source_points, destination_points)
    out_img = cv2.warpPerspective(in_img, h, size)

    return out_img
