#bucket name
def get_bucket_name():
    bucket_name= "cp6-digitalinstore-bucket"
    return bucket_name

#root path
def get_root_path():
    root_path= "/.sagemaker/mms/models/model/code/"
    return root_path

#cloth name
def get_cloth_name():
    cloth_name= '000001_1.png'
    return cloth_name

#image name
def get_image_name():
    image_name= '000001_0.png'
    return image_name

def get_test_color():
    test_color = "Data_preprocessing/test_color"
    return test_color

def get_colormask():
    colormask = "Data_preprocessing/test_colormask"
    return colormask

def get_test_edge():
    test_edge = "Data_preprocessing/test_edge"
    return test_edge

def get_test_img():
    test_img = "Data_preprocessing/test_img"
    return test_img

def get_test_label():
    test_label = "Data_preprocessing/test_label"
    return test_label

def get_test_mask():
    test_mask = "Data_preprocessing/test_mask"
    return test_mask

def get_test_pose():
    test_pose = "Data_preprocessing/test_pose"
    return test_pose

def get_test_pairs():
    test_pairs = 'Data_preprocessing/test_pairs.txt'
    return test_pairs

def get_inputs_path():
    input_path = "inputs"
    return input_path

def get_person_img_path():
    img_path = "inputs/img"
    return img_path

def get_cloth_img_path():
    cloth_path = "inputs/cloth"
    return cloth_path

#result directory path
def get_result_path():
    result_path= "results/"
    return result_path