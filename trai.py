import cv2
import numpy as np
import jetson.inference
import jetson.utils

net = jetson.inference.segNet("fcn-resnet18-Cityscapes-512x256")
camera = jetson.utils.gstCamera(2560,720,"/dev/video0")
display = jetson.utils.glDisplay()
net.SetOverlayAlpha(150)

while True:
    cuda_frame, width, height = camera.CaptureRGBA(zeroCopy=1)
    net.Process(cuda_frame, width, height)
    net.Mask(cuda_frame, width, height)
    jetson.utils.cudaDeviceSynchronize ()
    frame = jetson.utils.cudaToNumpy (cuda_frame, width, height, 4)
    lo=np.array([0,0,0,255])
    hi=np.array([255,255,255,255])
    mask=cv2.inRange(frame,lo,hi)
    frame[mask>0] = (255,255,255,255)
    cv2.cvtColor(frame.astype (np.uint8), cv2.COLOR_RGBA2RGB)
    cv2.imshow("frame", frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
