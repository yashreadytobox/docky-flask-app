import sys
import cv2
import numpy as np

from jetson_inference import segNet
from jetson_utils import cudaOverlay, cudaDeviceSynchronize, Log
from jetson_utils import videoSource, videoOutput

def allocate_buffers(net, visualize):
    overlay = None
    mask = None
    composite = None
    class_mask = None

    use_mask = "mask" in visualize
    use_overlay = "overlay" in visualize
    use_composite = use_mask and use_overlay

    if use_overlay:
        overlay = cudaAllocMapped(width=640, height=480, format='rgba8')

    if use_mask:
        mask_downsample = 2 if use_overlay else 1
        mask = cudaAllocMapped(width=640 // mask_downsample, height=480 // mask_downsample, format='gray8')

    if use_composite:
        composite = cudaAllocMapped(width=overlay.width + mask.width, height=overlay.height, format='rgba8')

    if use_stats:
        class_mask = cudaAllocMapped(width=net.GetGridSize()[0], height=net.GetGridSize()[1], format='gray8')

    return overlay, mask, composite, class_mask

def compute_stats(net, class_mask):
    if class_mask is None:
        return

    net.Mask(class_mask, class_mask.width, class_mask.height)
    class_mask_np = cudaToNumpy(class_mask)
    num_classes = net.GetNumClasses()

    class_histogram, _ = np.histogram(class_mask_np, bins=num_classes, range=(0, num_classes - 1))

    print('grid size:   {:d}x{:d}'.format(class_mask.width, class_mask.height))
    print('num classes: {:d}'.format(num_classes))
    print('-----------------------------------------')
    print(' ID  class name        count     %')
    print('-----------------------------------------')

    for n in range(num_classes):
        percentage = float(class_histogram[n]) / float(class_mask.width * class_mask.height)
        print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(n, net.GetClassDesc(n), class_histogram[n], percentage))

def main(input_uri, output_uri, network_model, filter_mode, visualize, ignore_class, alpha, stats):
    # Load the segmentation network
    net = segNet(network_model)

    # Set the alpha blending value
    net.SetOverlayAlpha(alpha)

    # Create video output
    output = videoOutput(output_uri)

    # Create buffer manager
    overlay, mask, composite, class_mask = allocate_buffers(net, visualize)

    # Create video source
    input = videoSource(input_uri)

    while True:
        ret, img_input = input.Capture()

        if not ret:
            break

        if overlay is not None and (overlay.height != img_input.shape[0] or overlay.width != img_input.shape[1]):
            overlay = None

        if mask is not None and (mask.height != img_input.shape[0] or mask.width != img_input.shape[1]):
            mask = None

        if composite is not None and (composite.height != img_input.shape[0] or composite.width != img_input.shape[1]):
            composite = None

        if class_mask is not None and (
                class_mask.height != net.GetGridSize()[1] or class_mask.width != net.GetGridSize()[0]):
            class_mask = None

        if overlay is None and visualize.find("overlay") != -1:
            overlay = cudaAllocMapped(width=img_input.shape[1], height=img_input.shape[0], format=img_input.format)

        if mask is None and visualize.find("mask") != -1:
            mask_downsample = 2 if overlay else 1
            mask = cudaAllocMapped(width=img_input.shape[1] // mask_downsample,
                                   height=img_input.shape[0] // mask_downsample, format=img_input.format)

        if composite is None and overlay and mask:
            composite = cudaAllocMapped(width=overlay.width + mask.width, height=overlay.height, format=img_input.format)

        if class_mask is None and stats:
            class_mask = cudaAllocMapped(width=net.GetGridSize()[0], height=net.GetGridSize()[1], format='gray8')

        if overlay:
            overlay = cudaAllocMapped(width=img_input.shape[1], height=img_input.shape[0], format=img_input.format)

        # Process the segmentation network
        net.Process(img_input, ignore_class=ignore_class)

        # Generate the overlay
        if overlay:
            net.Overlay(overlay, filter_mode=filter_mode)

        # Generate the mask
        if mask:
            net.Mask(mask, filter_mode=filter_mode)

        # Composite the images
        if composite:
            cudaOverlay(overlay, composite, 0, 0)
            cudaOverlay(mask, composite, overlay.width, 0)

        # Render the output image
        output.Render(composite if composite else overlay if overlay else mask)

        # Update the title bar
        output.SetStatus("{:s} | Network {:.0f} FPS".format(network_model, net.GetNetworkFPS()))

        # Print out performance info
        cudaDeviceSynchronize()
        net.PrintProfilerTimes()

        # Compute segmentation class stats
        if stats:
            compute_stats(net, class_mask)

        # Exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            break

    # Release resources
    input.Close()
    output.Close()

if __name__ == "__main__":
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

    main(input_uri, output_uri, network_model, filter_mode, visualize, ignore_class, alpha, stats)
