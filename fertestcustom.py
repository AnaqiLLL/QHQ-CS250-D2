import os
import sys
import numpy as np
import cv2
from keras.models import model_from_json
from PIL import Image

try:
    # 获取图像文件路径
    image_path = sys.argv[1]

    # 设置模型文件的绝对路径
    model_json_path = 'E://Study_File/University/MUST/Studying/G3.1/SoftwareEngineering/FinalProject/QHQSystem/fer2013-master/fer.json'
    # model_json_path = 'E://Part-time_Project/fer2013-master/fer.json'
    # model_weights_path = 'E://Part-time_Project/fer2013-master/fer.h5'
    model_weights_path = 'E://Study_File/University/MUST/Studying/G3.1/SoftwareEngineering/FinalProject/QHQSystem/fer2013-master/fer.h5'
    face_cascade_path = 'E://Study_File/University/MUST/Studying/G3.1/SoftwareEngineering/FinalProject/QHQSystem/fer2013-master/haarcascade_frontalface_default.xml'

    # 检查模型文件是否存在
    assert os.path.exists(model_json_path), "Model JSON file not found!"
    assert os.path.exists(model_weights_path), "Model weights file not found!"

    # 加载模型
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    print("Loaded model from disk")

    # 设置图像调整大小的参数
    WIDTH = 48
    HEIGHT = 48
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # 从文件加载图像
    img = Image.open(image_path)
    full_size_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    assert full_size_image is not None, "Image not loaded correctly!"
    print("Image Loaded")

    # 转换为灰度图像
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)

    # 检查面部级联文件是否存在
    assert os.path.exists(face_cascade_path), "Face cascade file not found!"

    # 检测面部
    face = cv2.CascadeClassifier(face_cascade_path)
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Detected faces: {len(faces)}")

    # 处理每个面部
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # 预测情感
        yhat = loaded_model.predict(cropped_img)
        emotion = labels[int(np.argmax(yhat))]
        cv2.putText(full_size_image, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: " + emotion)

    cv2.imshow('Emotion', full_size_image)
    cv2.waitKey()

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)