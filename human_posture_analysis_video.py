import cv2
import time
import math as m
import mediapipe as mp
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""


def sendWarning(x):
    pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# ===============================================================================================#

def drawText(_image, text, position, color=(255, 255, 255), font_size=32, stroke_width=1):
    fontpath = "./simsun.ttc"  # <== 这里是宋体路径
    fontZh = ImageFont.truetype(fontpath, font_size)
    img_pil = Image.fromarray(_image)

    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=fontZh, fill=(color[2], color[1], color[0]), stroke_width=stroke_width)
    return np.array(img_pil)


if __name__ == "__main__":
    # For webcam input replace file name with 0.
    file_name = 'input.mp4'
    cap = cv2.VideoCapture(0)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Process the image.
        keypoints = pose.process(image)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # Acquire the landmark coordinates.
        # Once aligned properly, left or right should not be a concern.      
        # Left shoulder.
        if not hasattr(lm, 'landmark'):
            continue
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        # Right shoulder
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        # Left ear.
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        # Left hip.
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # Calculate distance between left shoulder and right shoulder points.
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        if offset < 100:
            image = drawText(image, str(int(offset)) + ' 对齐', (w - 150, 20), (0, 255, 127), 18)
        else:
            image = drawText(image, str(int(offset)) + ' 不对齐', (w - 150, 20), (255, 0, 0), 18)
            # cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

        # Calculate angles.
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string = '颈部 : ' + str(int(neck_inclination)) + '°  躯干 : ' + str(int(torso_inclination)) + '°'

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1

            image = drawText(image, angle_text_string, (10, 20), (0, 255, 127), 24)
            image = drawText(image, str(int(neck_inclination)) + "°", (l_shldr_x + 10, l_shldr_y), (0, 255, 127), 24)
            image = drawText(image, str(int(torso_inclination)) + "°", (l_hip_x + 10, l_hip_y), (0, 255, 127), 24)

            # Join landmarks.
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

        else:
            good_frames = 0
            bad_frames += 1

            image = drawText(image, angle_text_string, (10, 20), (255, 0, 0), 24)
            image = drawText(image, str(int(neck_inclination)) + "°", (l_shldr_x + 10, l_shldr_y), (255, 0, 0), 24)
            image = drawText(image, str(int(torso_inclination)) + "°", (l_hip_x + 10, l_hip_y), (255, 0, 0), 24)
            # Join landmarks.
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = '姿势准确 : ' + str(round(good_time, 1)) + '秒'
            image = drawText(image, time_string_good, (10, h - 30), (0, 255, 127), 22)
        else:
            time_string_bad = '姿势较差 : ' + str(round(bad_time, 1)) + '秒'
            image = drawText(image, time_string_bad, (10, h - 30), (255, 0, 0), 22)

        # If you stay in bad posture for more than 3 minutes (180s) send an alert.
        if bad_time > 180:
            sendWarning()

        logo = cv2.imread('logo.png')
        width = 123 * 2
        height = int(width / 4.3)
        logo = cv2.resize(logo, (width, height))
        img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        roi = image[-height - 10:-10, -width - 10:-10]
        roi[np.where(mask)] = 0
        roi += logo

        # Write frames.
        video_output.write(image)

        # Display.
        cv2.imshow('HealBone Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
