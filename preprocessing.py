import os
import glob
import numpy as np
from PIL import Image
import IPython
import cv2
import sys
import time
import mediapipe as mp
# import matplotlib.pyplot as plt
import U2Net
from U2Net import u2net_load
from U2Net import u2net_run
from predict_pose import generate_pose_keypoints

import requests
from io import BytesIO

import boto3
import base64
from boto3 import client

import config
from config import get_bucket_name, get_root_path, get_cloth_name, get_image_name, get_test_color, get_colormask, get_test_edge, get_test_img, get_test_label, get_test_mask, get_test_pose, get_inputs_path, get_person_img_path, get_cloth_img_path, get_test_pairs

BUCKET_NAME = get_bucket_name()
ROOT_PATH = get_root_path()

INPUT = get_inputs_path()
PERSON_IMG = get_person_img_path()
CLOTH_IMG = get_cloth_img_path()
IMG_NAME = get_image_name()
CLOTH_NAME = get_cloth_name()

TEST_COLOR = get_test_color()
COLORMASK = get_colormask()
TEST_EDGE = get_test_edge()
TEST_IMG = get_test_img()
TEST_LABEL = get_test_label()
TEST_MASK = get_test_mask()
TEST_POSE = get_test_pose()
TEST_PAIRS = get_test_pairs()

def make_directories():
    os.makedirs(ROOT_PATH + TEST_COLOR+ "/", exist_ok=True)    
    os.makedirs(ROOT_PATH + COLORMASK+ "/", exist_ok=True)    
    os.makedirs(ROOT_PATH + TEST_EDGE+ "/", exist_ok=True)    
    os.makedirs(ROOT_PATH + TEST_IMG+ "/", exist_ok=True)    
    os.makedirs(ROOT_PATH + TEST_LABEL+ "/", exist_ok=True)    
    os.makedirs(ROOT_PATH + TEST_MASK + "/", exist_ok=True)    
    os.makedirs(ROOT_PATH + TEST_POSE+ "/", exist_ok=True)
    os.makedirs(ROOT_PATH + INPUT + "/", exist_ok=True)   
    os.makedirs(ROOT_PATH + PERSON_IMG+ "/", exist_ok=True)  
    os.makedirs(ROOT_PATH + CLOTH_IMG+ "/", exist_ok=True)  
    os.makedirs(ROOT_PATH + "results/", exist_ok=True)
    
    print("Print current folders in data preprocessing after creating folders",os.listdir(ROOT_PATH + 'Data_preprocessing'))  
    print("Print current folders in code after creating folders",os.listdir(ROOT_PATH))  

def save_input_data(input_data):
    person_image = "user_images/"+input_data[0] 
    cloth_image = "products/"+input_data[1]   
    
    
    s3 = boto3.resource('s3')
    s3.meta.client.download_file(BUCKET_NAME ,person_image, ROOT_PATH + PERSON_IMG+ "/person_img.png")
    s3.meta.client.download_file(BUCKET_NAME ,cloth_image , ROOT_PATH + CLOTH_IMG + "/cloth_img.png")
    
    print("Print inputs cloth folder data bucket images --- ",os.listdir(ROOT_PATH+CLOTH_IMG))    
    print("Print inputs img folder data bucket images --- ",os.listdir(ROOT_PATH+PERSON_IMG))  
    
    
def generate_cloth_mask():
    u2netload_start_time = time.time()
    u2net = u2net_load.model(model_name='u2netp')
    u2netload_end_time = time.time()
    print('U2NEt load in {}s'.format(u2netload_end_time-u2netload_start_time))
    
    # Cloth mask generation
    cloth_start_time = time.time()
    
    cloth_path = os.path.join(ROOT_PATH + CLOTH_IMG, sorted(os.listdir(ROOT_PATH + CLOTH_IMG))[0])

    print("Cloth path",cloth_path)
    cloth = Image.open(cloth_path)
    
    cloth = cloth.resize((192, 256), Image.BICUBIC).convert('RGB')
    print("After resize")
    
    cloth.save(os.path.join(ROOT_PATH + TEST_COLOR, CLOTH_NAME))
    print("Print current folder after cloth image resizing in test color",os.listdir(ROOT_PATH + TEST_COLOR))
    
    u2net_run.infer(u2net, 'code/'+TEST_COLOR ,'code/'+TEST_EDGE)
    clothmask_time = time.time()
    print('cloth mask image in {}s'.format(clothmask_time-cloth_start_time))
    print("Print current folder after u2net infer in test edge",os.listdir(ROOT_PATH+TEST_EDGE))
    
# def background_removal():
#     change_background_mp = mp.solutions.selfie_segmentation
#     change_bg_segment = change_background_mp.Selget_image_namefieSegmentation()

#     img_path = os.path.join(ROOT_PATH+PERSON_IMG, sorted(os.listdir(ROOT_PATH+PERSON_IMG))[0])
#     sample_img = cv2.imread(img_path)
#     RGB_sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
#     result = change_bg_segment.process(RGB_sample_img)
#     binary_mask = result.segmentation_mask > 0.9
#     binary_mask_3 = np.dstack((binary_mask, binary_mask, binary_mask))
#     output_image = np.where(binary_mask_3, sample_img, 255)
#     os.system("rm -rf /.sagemaker/mms/models/model/code/inputs/img/*")
#     image = Image.fromarray(output_image[:, :, ::-1].astype('uint8'))
#     image.save("/.sagemaker/mms/models/model/code/inputs/img/New.png", "png")
    
def resize_human_image():
    start_time = time.time()
    
    img_path = os.path.join(ROOT_PATH + PERSON_IMG, sorted(os.listdir(ROOT_PATH + PERSON_IMG))[0])
    img = Image.open(img_path)
    img = img.resize((192, 256), Image.BICUBIC)
    img_path = os.path.join(ROOT_PATH + TEST_IMG, IMG_NAME)
    img.save(img_path)
    print("Print current folder after save person test_img",os.listdir(ROOT_PATH+TEST_IMG))
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(ROOT_PATH+TEST_IMG+"/000001_0.png",BUCKET_NAME,"test_img.png")
    
    
    resize_time = time.time()
    print('Resized image in {}s'.format(resize_time-start_time))
    
def self_correction_human_parsing():
    start_time = time.time()
    schp_input_dir= ROOT_PATH + 'Data_preprocessing/test_img'
    schp_output_dir= ROOT_PATH + 'Data_preprocessing/test_label'
    
    print("print schp_input_dir path : " , schp_input_dir)
    print("print schp_output_dir path : ", schp_output_dir)
    
    print("print schp_input_dir DYNAMIC path : " , ROOT_PATH + TEST_IMG)
    print("print schp_output_dir DYNAMIC  path : ", ROOT_PATH + TEST_LABEL)
    
    # os.system("python3 code/Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'code/lip_final.pth' --input-dir schp_input_dir --output-dir schp_output_dir")
    os.system("python3 code/Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'code/lip_final.pth' --input-dir '/.sagemaker/mms/models/model/code/Data_preprocessing/test_img' --output-dir '/.sagemaker/mms/models/model/code/Data_preprocessing/test_label'")
    
    print("Print current folder after human parsing in test_label",os.listdir(schp_output_dir))
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(ROOT_PATH + TEST_LABEL+"/000001_0.png",BUCKET_NAME,"test_label.png")
    
    
    parse_time = time.time()
    print('Parsing generated in {}s'.format(parse_time-start_time))
    
def generate_keypoints():
    start_time = time.time()
    img_path = os.path.join(ROOT_PATH + PERSON_IMG, sorted(os.listdir(ROOT_PATH + PERSON_IMG))[0])
    pose_path = os.path.join(ROOT_PATH + TEST_POSE, IMG_NAME.replace('.png', '_keypoints.json'))
    generate_pose_keypoints(img_path, pose_path)
    pose_time = time.time()
    print('Pose map generated in {}s'.format(pose_time-start_time))
    print("Print current folder after pose generation in test_pose",os.listdir(ROOT_PATH+TEST_POSE))
    
def generate_test_pairs():
    with open(ROOT_PATH + TEST_PAIRS, 'w') as f:
        f.write('000001_0.png 000001_1.png')
        
    with open(ROOT_PATH + TEST_PAIRS, 'r') as f:
        print("Content inside test-pairs.txt " , f.read())
    

# def inference(input_data): 
    #getting bucket name, root path
    # BUCKET_NAME = get_bucket_name()
    # ROOT_PATH= get_root_path()
    # os.makedirs(ROOT_PATH + "Data_preprocessing/test_color/", exist_ok=True)    
    # os.makedirs(ROOT_PATH + "Data_preprocessing/test_colormask/", exist_ok=True)    
    # os.makedirs(ROOT_PATH + "Data_preprocessing/test_edge/", exist_ok=True)    
    # os.makedirs(ROOT_PATH + "Data_preprocessing/test_img/", exist_ok=True)    
    # os.makedirs(ROOT_PATH + "Data_preprocessing/test_label/", exist_ok=True)    
    # os.makedirs(ROOT_PATH + "Data_preprocessing/test_mask/", exist_ok=True)    
    # os.makedirs(ROOT_PATH + "Data_preprocessing/test_pose/", exist_ok=True)
    # os.makedirs(ROOT_PATH + "inputs/", exist_ok=True)   
    # os.makedirs(ROOT_PATH + "inputs/img/", exist_ok=True)  
    # os.makedirs(ROOT_PATH + "inputs/cloth/", exist_ok=True)  
    # os.makedirs(ROOT_PATH + "results", exist_ok=True)
    

#     person_image = "trial_images/person/"+input_data[0] 
#     cloth_image = "trial_images/cloth/"+input_data[1]
    
#     s3 = boto3.resource('s3')
#     s3.meta.client.download_file(BUCKET_NAME ,person_image, ROOT_PATH + "inputs/img/person_img.png")
#     s3.meta.client.download_file(BUCKET_NAME,cloth_image , ROOT_PATH + "inputs/cloth/cloth_img.png")
    
#     print("Print inputs cloth folder data bucket images --- ",os.listdir('/.sagemaker/mms/models/model/code/inputs/cloth'))    
#     print("Print inputs img folder data bucket images --- ",os.listdir('/.sagemaker/mms/models/model/code/inputs/img'))   
    
#     u2netload_start_time = time.time()
#     u2net = u2net_load.model(model_name='u2netp')
#     u2netload_end_time = time.time()
#     print('U2NEt load in {}s'.format(u2netload_end_time-u2netload_start_time))
    
#     # Cloth mask generation
#     cloth_start_time = time.time()
#     CLOTH_NAME = get_cloth_name()
#     cloth_path = os.path.join(ROOT_PATH + 'inputs/cloth', sorted(os.listdir(ROOT_PATH + 'inputs/cloth'))[0])
#     print("Cloth path",cloth_path)
#     cloth = Image.open(cloth_path)
    
#     cloth = cloth.resize((192, 256), Image.BICUBIC).convert('RGB')
#     print("After resize")
#     cloth.save(os.path.join(ROOT_PATH + 'Data_preprocessing/test_color', CLOTH_NAME))
#     print("Print current folder after cloth image resizing in test color",os.listdir(ROOT_PATH + 'Data_preprocessing/test_color'))
    
#     u2net_run.infer(u2net, 'code/Data_preprocessing/test_color','code/Data_preprocessing/test_edge')
#     clothmask_time = time.time()
#     print('cloth mask image in {}s'.format(clothmask_time-cloth_start_time))
#     print("Print current folder after u2net infer in test edge",os.listdir('/.sagemaker/mms/models/model/code/Data_preprocessing/test_edge'))

    #Background removal code
#     change_background_mp = mp.solutions.selfie_segmentation
#     change_bg_segment = change_background_mp.Selget_image_namefieSegmentation()

#     img_path = os.path.join('/.sagemaker/mms/models/model/code/inputs/img', sorted(os.listdir('/.sagemaker/mms/models/model/code/inputs/img'))[0])
#     sample_img = cv2.imread(img_path)
#     RGB_sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
#     result = change_bg_segment.process(RGB_sample_img)
#     binary_mask = result.segmentation_mask > 0.9
#     binary_mask_3 = np.dstack((binary_mask, binary_mask, binary_mask))
#     output_image = np.where(binary_mask_3, sample_img, 255)
#     os.system("rm -rf /.sagemaker/mms/models/model/code/inputs/img/*")
#     image = Image.fromarray(output_image[:, :, ::-1].astype('uint8'))
#     image.save("/.sagemaker/mms/models/model/code/inputs/img/New.png", "png")

    #Person image resizing and SCHP code

#     start_time = time.time()
#     IMG_NAME = get_image_name()
#     img_path = os.path.join(ROOT_PATH + 'inputs/img', sorted(os.listdir(ROOT_PATH + 'inputs/img'))[0])
#     img = Image.open(img_path)
#     img = img.resize((192, 256), Image.BICUBIC)
#     img_path = os.path.join(ROOT_PATH + 'Data_preprocessing/test_img', IMG_NAME)
#     img.save(img_path)
#     print("Print current folder after save person test_img",os.listdir('/.sagemaker/mms/models/model/code/Data_preprocessing/test_img'))
    
#     resize_time = time.time()
#     print('Resized image in {}s'.format(resize_time-start_time))
    # SCHP
#     schp_input_dir= ROOT_PATH + 'Data_preprocessing/test_img'
#     schp_output_dir= ROOT_PATH + 'Data_preprocessing/test_label'
    
#     os.system("python3 code/Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'code/lip_final.pth' --input-dir schp_input_dir --output-dir schp_output_dir")
    
#     print("Print current folder after human parsing in test_label",os.listdir('/.sagemaker/mms/models/model/code/Data_preprocessing/test_label'))
    
#     parse_time = time.time()
#     print('Parsing generated in {}s'.format(parse_time-resize_time))


    # pose_path = os.path.join(ROOT_PATH + 'Data_preprocessing/test_pose', IMG_NAME.replace('.png', '_keypoints.json'))
    # generate_pose_keypoints(img_path, pose_path)
    # pose_time = time.time()
    # print('Pose map generated in {}s'.format(pose_time-parse_time))
    # print("Print current folder after pose generation in test_pose",os.listdir('/.sagemaker/mms/models/model/code/Data_preprocessing/test_pose'))
    

    # Writing in test_pairs.txt
    # os.system("rm -rf Data_preprocessing/test_pairs.txt")
#     with open(ROOT_PATH + 'Data_preprocessing/test_pairs.txt', 'w') as f:
#         f.write('000001_0.png 000001_1.png')
        
#     with open(ROOT_PATH + 'Data_preprocessing/test_pairs.txt', 'r') as f:
#         # f.write('000001_0.png 000001_1.png')
#         print("Content inside test-pairs.txt " , f.read())


def inference(input_data): 
    
    make_directories()
    save_input_data(input_data)  
    generate_cloth_mask()
    # background_removal()
    resize_human_image()
    self_correction_human_parsing()
    generate_keypoints()
    generate_test_pairs()