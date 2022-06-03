import requests
import base64
import os

path_img = "D:\Workspace\Real_Project\QAI\\akaocr\\api\sbi_api\idcard_segmentation\\test_images\data01.jpg"
with open(path_img, 'rb') as img:
  name_img= os.path.basename(path_img)
  files= {'image': (name_img, img, 'multipart/form-data', {'Expires': '0'}) }
  url = ' http://192.168.1.6:5000/imageSegment'
  res = requests.post(url, files=files)
  print(res.json())