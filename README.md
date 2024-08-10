## Image Cropping Script 

This repository has a Python script that uses Tesseract OCR and OpenCV to process images. Several operations are carried out by the script such as loading the image, converting it to grayscale, adaptive thresholding, dilation, contour detection, contour filtering, and cropping the regions of interest according to contours.

## Requirements 

Make sure you have the following installed before running the script:

1. Python 3.x
2. OpenCV ('cv2')
3. NumPy
4. Matplotlib
5. Tesseract ORC

## Usage 
   1. Open the script `image_cropping.py` in a text editor.

3. Modify the following paths in the script:

- Set the path to your Tesseract executable:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'path_to_your_tesseract_executable'
  ```

- Set the path to your input image:
  ```python
  image_path = 'path_to_your_image'
  ```

- Set the path to your desired output folder:
  ```python
  output_folder = 'path_to_your_output_folder'
  ```

3. Save the script and run it using Python:

## Output

The script will display several images during processing:
- Original image
- Grayscale image
- Dilated image
- Image with detected contours
- Cropped images (if any contours are detected)

Cropped images will be saved in the specified output folder with names in the format: `{original_filename}_{contour_number}.jpg`


## Customization

You can adjust the following parameters in the script:
- Contour area threshold (default is 10000)
- Cropping offsets (top, left, right, bottom)

## Disclaimer 

This project is part of an academic assessment and is designed for educational purposes only.
