import cv2
import numpy as np

from jetson_inference import segNet
from jetson_utils import cudaOverlay, cudaDeviceSynchronize, Log
from segnet_utils import segmentationBuffers

# Set your input and output video file paths here
input_uri = "your_input_video.mp4"
output_uri = "your_output_video.mp4"

# Parameters
network_model = "fcn-resnet18-voc"
filter_mode = "linear"
visualize = "overlay,mask"
ignore_class = "void"
alpha = 150.0
stats = False  # Change to True if you want to compute statistics about segmentation mask class output

# Load the segmentation network
net = segNet(network_model)

# Set the alpha blending value
net.SetOverlayAlpha(alpha)

# Create video output
output = cv2.VideoWriter(output_uri, cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))  # Adjust resolution as needed

# Create buffer manager
buffers = segmentationBuffers(net)

# Create video source
cap = cv2.VideoCapture(input_uri)

while cap.isOpened():
    ret, img_input = cap.read()

    if not ret:
        break

    # Process the segmentation network
    buffers.Alloc(img_input.shape, img_input.format)
    net.Process(img_input, ignore_class=ignore_class)

    # Generate the overlay
    if buffers.overlay:
        net.Overlay(buffers.overlay, filter_mode=filter_mode)

    # Generate the mask
    if buffers.mask:
        net.Mask(buffers.mask, filter_mode=filter_mode)

    # Composite the images
    if buffers.composite:
        cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
        cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

    # Render the output image
    output_image = np.zeros_like(buffers.output)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR by default
    output.Render(output_image)

    # Update the title bar
    # You can print this information to the console if needed
    title_bar_info = f"{network_model} | Network {net.GetNetworkFPS():.0f} FPS"

    # Print out performance info
    cudaDeviceSynchronize()
    net.PrintProfilerTimes()

    # Compute segmentation class stats
    if stats:
        buffers.ComputeStats()

    # Write the output frame to the video file
    output.write(output_image)

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()
