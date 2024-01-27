import numpy as np
import cv2
import pickle

# Set the desired width and height for the video capture
width = 640
height = 480
threshold = 0.8

# Open the video capture device (using the default camera, index 0)
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Load the trained model from the pickle file
try:
    with open("model_trained.p", "rb") as pickle_file:
        model = pickle.load(pickle_file)
except FileNotFoundError:
    print("Error: Model file not found.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Define font for displaying text on images
font = cv2.FONT_HERSHEY_SIMPLEX

def pre_processing(img):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    img = cv2.equalizeHist(img)
    # Normalize the pixel values to the range [0, 1]
    img = img / 255.0
    return img

def getClassName(classNo):
    class_names = [
        'Speed limit(5 km/h)',
        'Speed limit(15 km/h)',
        'Speed limit(30 km/h)',
        'Speed limit(40 km/h)',
        'Speed limit(50 km/h)',
        'Speed limit(60 km/h)',
        'Speed limit(70 km/h)',
        'Speed limit(80 km/h)',
        'Donot go straight or left',
        'Donot go straight or right',
        'Donot go straight',
        'Donot go left',
        'Donot go left or right',
        'Donot go right',
        'Donot overtake from left',
        'No U-turn',
        'No car',
        'No horn',
        'Speed limit(40 km/h)',
        'Speed limit(50 km/h)',
        'Go straight or right',
        'Go straight',
        'Go left',
        'Go left or right',
        'Go right',
        'Keep left',
        'Keep right',
        'Roundabout mandatory',
        'Watch out for cars',
        'Horn',
        'Bicycles crossing',
        'U-turn',
        'Road divider',
        'Traffic signals',
        'Danger ahead',
        'Zebra crossing',
        'Children crossing',
        'Dangerous curve to the left',
        'Dangerous curve to the right',
        'Unknown 1',
        'Unknown 2',
        'Unknown 3',
        'Go right or straight',
        'Go left or straight',
        'Unknown 4',
        'Zigzag curve',
        'Train crossing',
        'Under construction',
        'Unknown 5',
        'Fences',
        'Heavy vehicle accidents',
        'Unknown 6',
        'Give way',
        'No stopping',
        'No entry',
        'Unknown 7',
        'Unknown 8']

    if classNo < len(class_names):
        return class_names[classNo]
    else:
        return "Unknown Class"

# Main loop for video capture and prediction
while True:
    # Read a frame from the video capture device
    success, img_original = cap.read()

    # Check if the frame was successfully read
    if not success:
        print("Failed to read frame from the camera.")
        break

    # Resize the frame to the desired dimensions
    img = cv2.resize(img_original, (32, 32))

    # Preprocess the image
    img_processed = pre_processing(img)

    # Display the processed image
    cv2.imshow("Processed Image", img_processed)

    # Reshape the image for prediction
    img_input = img_processed.reshape(1, 32, 32, 1)

    # Predict the class probabilities
    try:
        predictions = model.predict(img_input)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        continue

    # Get the predicted class index and probability
    class_index = np.argmax(predictions)
    prob_val = np.max(predictions)

    print(class_index, prob_val)

    # Display the original image with class information if probability is above threshold
    if prob_val > threshold:
        class_name = getClassName(class_index)
        cv2.putText(img_original, class_name, (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img_original, f"{round(prob_val * 100, 2)}%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", img_original)

    # Display the original image
    cv2.imshow("Original Image", img_original)

    # Check for user input to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
