import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pytesseract

# Tesseract executable path 
pytesseract.pytesseract.tesseract_cmd = r'path_to_your_tesseract_executable' # Chnage to correct path

# Path to your uploaded image
image_path = 'path_to_your_image' # Change to correct image path
image = cv2.imread(image_path)

# Extract base filename without extension
base_filename = os.path.splitext(os.path.basename(image_path))[0]

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully.")
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.show()
    
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Converted to grayscale.")
plt.figure(figsize=(10, 10))
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Apply adaptive thresholding to create a binary image
if image is not None:
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    print("Adaptive thresholding applied.")
    

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    print("Dilation applied.")
    

    plt.figure(figsize=(10, 10))
    plt.imshow(dilated, cmap='gray')
    plt.title('Dilated Image')
    plt.show()

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours found: {len(contours)}")

# Draw contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('Contours Detected')
plt.show()

# Filter contours based on area
filtered_contours = [c for c in contours if cv2.contourArea(c) > 10000]  # Adjust the threshold as needed
print(f"Number of filtered contours: {len(filtered_contours)}")

# Output directory
output_folder = 'path_to_your_output_folder' # Change to correct  path
os.makedirs(output_folder, exist_ok=True)

if image is not None:
    #Measurements for cropping
    top_offset = 0 # Pixels to include above the  contour
    left_offset = 0  # Pixels to include to the left of the contour
    right_offset = 0 # Pixels to include to the right of the contour
    bottom_offset = 0 # No pixels to include to the bottom of the contour

    try:
        for i, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Contour {i+1}: x={x}, y={y}, w={w}, h={h}")

            x_start = max(0, x - left_offset)
            y_start = max(0, y - top_offset)
            x_end = min(image.shape[1], x + w + right_offset)
            y_end = min(image.shape[0], y + h // 2 + bottom_offset)  

            cropped_blot = image[y_start:y_end, x_start:x_end]

            # Save the cropped image with the name 
            output_path = os.path.join(output_folder, f'{base_filename}_{i+1}.jpg')
            print(f"Saving cropped image to: {output_path}")
            cv2.imwrite(output_path, cropped_blot)
            print("Image saved successfully.")

            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(cropped_blot, cv2.COLOR_BGR2RGB))
            plt.title(f'Saved Blot {i+1}')
            plt.show()
    except Exception as e:
        print(f"Error saving image: {e}")
