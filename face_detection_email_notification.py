import cv2 as cv  # Import OpenCV library for image processing
import imutils  # Import imutils for additional image processing utilities
import smtplib  # Import smtplib for sending emails
import os  # Import os to access environment variables securely
from datetime import datetime  # Import datetime for handling date and time operations
from email.message import EmailMessage  # Import for creating email messages with attachments
import mimetypes  # Import to automatically detect file types when attaching files

# Function to detect motion using the webcam
def movement_detected():
    # Initialize video capture from the default camera (0 represents the first camera)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Set the camera frame width to 640 pixels
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Set the camera frame height to 480 pixels

    # Capture the initial frame to use as a reference for detecting changes
    _, start_frame = cap.read()
    start_frame = imutils.resize(start_frame, width=640)  # Resize for consistent frame dimensions
    start_frame_gray = cv.cvtColor(start_frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    start_frame_blur = cv.GaussianBlur(start_frame_gray, (21, 21), 0)  # Apply Gaussian blur for noise reduction

    # Loop to continuously detect motion in each new frame
    while True:
        # Capture a new frame and process it for comparison
        _, frame = cap.read()
        frame = imutils.resize(frame, width=640)  # Resize for consistency
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        blur = cv.GaussianBlur(gray, (21, 21), 0)  # Apply Gaussian blur for smoothing

        # Calculate the difference between the reference frame and the new frame
        diff = cv.absdiff(start_frame_blur, blur)
        _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)  # Threshold the difference for a binary result

        # Check if the number of changed pixels exceeds the threshold (indicating motion)
        if cv.countNonZero(thresh) > 5000:  # Threshold (5000) can be adjusted for sensitivity
            cap.release()  # Release the camera
            cv.destroyAllWindows()  # Close any OpenCV windows
            return True  # Return True if movement is detected

        start_frame_blur = blur  # Update the reference frame to the current frame for continuous detection
        cv.imshow("Live Feed", frame)  # Show the current frame in a window

        # Allow the user to quit the live feed by pressing 'q'
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()  # Release the camera if the loop is exited
    cv.destroyAllWindows()  # Close any OpenCV windows
    return False  # Return False if no movement is detected

# Function to continuously detect faces within the specified time range
def continuous_face_detection():
    # Load the pre-trained face detection model (Haar cascade for frontal faces)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv.VideoCapture(0)  # Start video capture from the default camera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Set the camera frame width
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Set the camera frame height

    while True:
        # Get the current time and check if it's within the detection window (9:00 AM - 9:15 AM)
        current_time = datetime.now().strftime("%H:%M:%S")
        if current_time < "09:00:00" or current_time > "09:15:00":
            print("Outside of detection window. Stopping face detection.")
            break  # Exit the function if outside the time window

        # Read the next frame from the camera
        _, frame = cap.read()
        frame = imutils.resize(frame, width=640)  # Resize for consistency
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale for detection

        # Detect faces in the frame using the Haar cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces and save images
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around each face

            # Save the captured image with a timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"face_capture_{timestamp}.jpg"
            cv.imwrite(image_filename, frame)  # Save the frame with the detected face
            print(f"Face detected and saved as {image_filename}")
            
            # Send the saved image via email
            send_email(image_filename)

        cv.imshow("Face Detection", frame)  # Display the frame with face detection

        # Allow the user to exit the loop by pressing 'q'
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()  # Release the camera
    cv.destroyAllWindows()  # Close any OpenCV windows

# Function to send an email notification with an image attachment
def send_email(image_path):
    # Retrieve email credentials and recipient from environment variables for security
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")

    # Set email subject and body content
    subject = "Face Detected Notification"
    body = "A face has been detected by the camera."

    # Create the email message and set its content
    message = EmailMessage()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.set_content(body)

    # Attach the image if the file path exists
    if image_path and os.path.exists(image_path):
        mime_type, _ = mimetypes.guess_type(image_path)  # Guess the file type of the image
        mime_main, mime_sub = mime_type.split('/')
        with open(image_path, 'rb') as img_file:
            message.add_attachment(img_file.read(), maintype=mime_main, subtype=mime_sub, filename=os.path.basename(image_path))

    # Try to send the email via Gmail's SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Start TLS encryption for security
        server.login(sender_email, sender_password)  # Log in to the sender's email account
        server.send_message(message)  # Send the email message
        server.quit()  # Close the connection to the server
        print("Email sent successfully with attachment.")
    except Exception as e:
        print(f"Failed to send email: {e}")  # Print any error encountered during sending

# Main logic to trigger face detection only if motion is detected within the specified time window
if movement_detected() and datetime.now().strftime("%H:%M:%S") >= "09:00:00" and datetime.now().strftime("%H:%M:%S") <= "09:15:00":
    continuous_face_detection()  # Start face detection only if it's between 9:00 AM and 9:15 AM
