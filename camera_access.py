import cv2
import sys

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

    

source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


# Previews until escape key is hit.
while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

    key = cv2.waitKey(1)

    # exits loops when escape is pressed
    if key == 27:
        break

    # else if spacebar is pressed
    elif key == 32:
        # save current frame as an image
        img_filename = 'captured_image.jpg'
        is_saved = cv2.imwrite(img_filename, frame)

        # if plastic do this
        if is_saved:
            print("This is plastic type 1")
        else:
            print("This isnt plastic")


# CAPTURE AND 

source.release()
cv2.destroyWindow(win_name)

