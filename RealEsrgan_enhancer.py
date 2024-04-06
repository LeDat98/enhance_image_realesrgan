from realesrgan_ncnn_py import Realesrgan
import cv2
import numpy as np

def enhancerfunc(file_path):

    realesrgan = Realesrgan(gpuid=0)
    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = realesrgan.process_cv2(image)
    file_name = file_path.split('/')[-1].split('.')[0]
    cv2.imencode(".jpg", image)[1].tofile(f"{file_name}_enhanced.jpg")

enhancerfunc("images.jpeg")
