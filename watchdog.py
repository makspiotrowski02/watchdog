import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Loading google's lightning model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
# Extracting default
movenet = model.signatures['serving_default']

# Added according to documentation 
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Drawing 'joint' points
def draw_points(frame, points, confidence):
  # Scaling to frame dimensions
  a,b,c = frame.shape
  shaped = np.squeeze(np.multiply(points, [a,b,1]))
# Drawing points
  for p in shaped:
    y, x, p_conf = p
    if p_conf > confidence:
      cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)

# Drawing lines between points
def draw_lines(frame, points, edges, confidence):
    # Scaling to frame dimensions
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(points, [y,x,1]))
    # Defining lines
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        # Drawing lines
        if (c1 > confidence) & (c2 > confidence):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

# Iterating through all detected people
def multi_pose(frame, detected_points, edges, confidence):
  for person in detected_points:
    draw_points(frame, person, confidence)
    draw_lines(frame, person, edges, confidence)

# Main function
def track_pose(input):
  # Loading video
  capture = cv2.VideoCapture(input)  
  ret, frame = capture.read()

  # Iterating through all frames 
  while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    # Scaling down for easier processing
    frm = frame.copy()
    frm = tf.image.resize_with_pad(tf.expand_dims(frm, axis=0),352,640)
    input_frm = tf.cast(frm, dtype=tf.int32)
    # Detecting and drawing lines
    results = movenet(input_frm)
    detectedPoints = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    multi_pose(frame, detectedPoints, EDGES, 0.30)
    # Displaying results
    cv2.imshow('Detection video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  # Freeing up resources
  capture.release()
  cv2.destroyAllWindows()

def main():
   track_pose("test1.mp4")

if __name__=="__main__":
    main()
