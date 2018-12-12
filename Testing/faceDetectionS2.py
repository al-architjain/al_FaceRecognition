import sys
import dlib
from skimage import io
import cv2

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
file_name = sys.argv[1]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

# Load the image
image = io.imread(file_name)

# Run the HOG face detector on the image data
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

# Show the desktop window with the image
# win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):
    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                             face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    # win.add_overlay(face_rect)
    cv2.rectangle(image, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (255, 51, 51), 3)
    # Get the the face's pose
    pose_landmarks = face_pose_predictor(image, face_rect)
    # print(pose_landmarks)
    # cv2.line(image, pose_landmarks, (255, 51, 51), 3)
    # Draw the face landmarks on the screen.
    win.add_overlay(pose_landmarks)

win.set_image(image)
dlib.hit_enter_to_continue()