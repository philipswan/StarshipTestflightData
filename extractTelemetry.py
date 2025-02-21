import cv2
import pytesseract
import numpy as np
import os
import json

# Load video
video_path = "StarshipIFT7.mp4"  # Change this to your actual video file

# Extract base filename without extension
base_filename = os.path.splitext(os.path.basename(video_path))[0]
output_js_file = f"{base_filename}.js"

# Open the file in write mode initially
with open(output_js_file, "w") as js_file:
    js_file.write(f"export const {base_filename} = \[\n")  # Start array

# Function to append data after each frame
def append_to_file(data, is_last=False):
    with open(output_js_file, "a") as js_file:
        json.dump(data, js_file, indent=4)
        if not is_last:
            js_file.write(",\n")  # Separate entries with commas
        else:
            js_file.write("\n\];\n")  # Close the array at the end

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
num_frames = 1000  # Number of frames to process
liftoff = False
startFrame = 0

# Define bounding boxes (x0, y0, x1, y1)
regions = {
    "time": (856, 946, 1062, 985),
    "booster_speed": (333, 912, 446, 941),
    "booster_altitude": (362, 948, 450, 974),
    "ship_speed": (1518, 912, 1631, 941),
    "ship_altitude": (1538, 948, 1631, 974)
}

while frame_count < num_frames:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # print(f"\nProcessing Frame {frame_count}...")

    # Dictionary to store extracted values for this frame
    extracted_data = {}
    extracted_data["frame"] = frame_count

    for label, (x0, y0, x1, y1) in regions.items():
        roi = frame[y0:y1, x0:x1]  # Crop the region

        # Save debug images only for the first frame
        if frame_count == 0:
            cv2.imwrite(f"debug_{label}.png", roi)

        # Convert to grayscale and apply binary inversion
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if ((label == "time") or liftoff):
            # Perform OCR
            text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789T+-:').strip()

            # Convert text to integer (default to -1 if OCR fails)
            try:
                if (label == "time"):
                    extracted_data[label] = text
                else:
                    extracted_data[label] = int(text)
            except ValueError:
                extracted_data[label] = "NaN"  # Assign a default invalid value if OCR fails

            if (not liftoff and (label == "time")):
                # Compare the first two characters to check for liftoff
                if (text[0:2] == "T+"):
                    print("Liftoff detected!")
                    liftoff = True
                    startFrame = frame_count

            if (liftoff):
                extracted_data["timeInSec"] = max(0, (frame_count - startFrame)*0.0333)
                append_to_file(extracted_data)

    print(extracted_data)
    liftoff_data = extracted_data

    frame_count += 1

with open(output_js_file, "a") as js_file:
    js_file.write("];\n")
print(f"Saved extracted data to {output_js_file}")

cap.release()  # Release the video capture
