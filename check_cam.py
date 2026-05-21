import cv2
cap = cv2.VideoCapture(0)
real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头真实分辨率：{real_w}x{real_h}")
cap.release()
