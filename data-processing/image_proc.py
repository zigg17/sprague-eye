import cv2
# Load the image
image = cv2.imread('Unknown.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Get bounding box coordinates for the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the cage area
cropped_cage = image[y:y+h, x:x+w]

# Save the cropped cage image
cv2.imwrite('cropped_cage.jpg', cropped_cage)

# Display the results (optional)
cv2.imshow("Cropped Cage", cropped_cage)
cv2.waitKey(0)
cv2.destroyAllWindows()
