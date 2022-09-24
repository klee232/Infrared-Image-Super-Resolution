import cv2
import numpy as np
from scipy import misc
def edge_convert(input_dir):
    out_path = f'ncan_edge_data/'
    #os.makedirs('can_edge_data/Thermal_train_LR_bicubic/X4')
    out1 = 'ncan_edge_data/Thermal_train_LR_bicubic/X4'

    for image_dir in os.listdir(input_dir):
        # create the full input path and read the file
        input_path = os.path.join(input_dir, image_dir)
        # print(input_path)
        # input initial image for converting later
        initial_image = cv2.imread(input_path)
        # deprecate noise effect
        source = cv2.GaussianBlur(initial_image, (3, 3), 0)
        n_initial_image=cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        ##################################################
        # Laplacian Edge Extractor #######################
        ##################################################
        lap_x = cv2.Laplacian(n_initial_image,cv2.CV_64F)
        ##################################################
        # Sobel Edge Extractor ###########################
        ##################################################
        sobelx = cv2.Sobel(n_initial_image,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(n_initial_image,cv2.CV_64F,0,1,ksize=5)
        sobel = magnitude_computation(sobelx, sobely)
        ##################################################
        # Canny Edge Extractor ###########################
        ##################################################
        canny = cv2.Canny(n_initial_image,10,30,3)
        # concatenate together
        x = np.concatenate((initial_image, lap_x, sobel, canny), axis=2)        
        #print(out1)
        fullpath = os.path.join(out1,image_dir)
        cv2.imwrite(fullpath,x)

#===============================================================================================================================        
def edge_convert_ht(input_dir):
    out_path = f'ncan_edge_data/'
    #os.makedirs('can_edge_data/Thermal_train_HR')
    out1 = 'ncan_edge_data/Thermal_train_HR'
    
    for image_dir in os.listdir(input_dir):
        # create the full input path and read the file
        input_path = os.path.join(input_dir, image_dir)
        # input initial image for converting later
        initial_image = cv2.imread(input_path)
        if initial_image is None:
            continue
        for image_dir in os.listdir(input_dir):
        # create the full input path and read the file
        input_path = os.path.join(input_dir, image_dir)
        # print(input_path)
        # input initial image for converting later
        initial_image = cv2.imread(input_path)
        # deprecate noise effect
        source = cv2.GaussianBlur(initial_image, (3, 3), 0)
        n_initial_image=cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        ##################################################
        # Laplacian Edge Extractor #######################
        ##################################################
        lap_x = cv2.Laplacian(n_initial_image,cv2.CV_64F)
        ##################################################
        # Sobel Edge Extractor ###########################
        ##################################################
        sobelx = cv2.Sobel(n_initial_image,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(n_initial_image,cv2.CV_64F,0,1,ksize=5)
        sobel = magnitude_computation(sobelx, sobely)
        ##################################################
        # Canny Edge Extractor ###########################
        ##################################################
        canny = cv2.Canny(n_initial_image,10,30,3)
        # concatenate together
        x = np.concatenate((initial_image, lap_x, sobel, canny), axis=2)        
        #print(out1)
        fullpath = os.path.join(out1,image_dir)
        cv2.imwrite(fullpath,x)


#================================================================================================================================
def edge_convert_lv(input_dir):
    out_path = f'ncan_edge_data/'
    #os.makedirs('can_edge_data/Thermal_valid_LR_bicubic/X4')
    out1 = 'ncan_edge_data/Thermal_valid_LR_bicubic/X4'
    
    for image_dir in os.listdir(input_dir):
        # create the full input path and read the file
        input_path = os.path.join(input_dir, image_dir)
        # print(input_path)
        # input initial image for converting later
        initial_image = cv2.imread(input_path)
        # deprecate noise effect
        source = cv2.GaussianBlur(initial_image, (3, 3), 0)
        n_initial_image=cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        ##################################################
        # Laplacian Edge Extractor #######################
        ##################################################
        lap_x = cv2.Laplacian(n_initial_image,cv2.CV_64F)
        ##################################################
        # Sobel Edge Extractor ###########################
        ##################################################
        sobelx = cv2.Sobel(n_initial_image,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(n_initial_image,cv2.CV_64F,0,1,ksize=5)
        sobel = magnitude_computation(sobelx, sobely)
        ##################################################
        # Canny Edge Extractor ###########################
        ##################################################
        canny = cv2.Canny(n_initial_image,10,30,3)
        # concatenate together
        x = np.concatenate((initial_image, lap_x, sobel, canny), axis=2)        
        #print(out1)
        fullpath = os.path.join(out1,image_dir)
        cv2.imwrite(fullpath,x)

#================================================================================================================================================
def edge_convert_hv(input_dir):
    out_path = f'ncan_edge_data/'
    #os.makedirs('ncan_edge_data/Thermal_valid_HR')
    out1 = 'ncan_edge_data/Thermal_valid_HR'
    
    for image_dir in os.listdir(input_dir):
        # create the full input path and read the file
        input_path = os.path.join(input_dir, image_dir)
        # print(input_path)
        # input initial image for converting later
        initial_image = cv2.imread(input_path)
        # deprecate noise effect
        source = cv2.GaussianBlur(initial_image, (3, 3), 0)
        n_initial_image=cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        ##################################################
        # Laplacian Edge Extractor #######################
        ##################################################
        lap_x = cv2.Laplacian(n_initial_image,cv2.CV_64F)
        ##################################################
        # Sobel Edge Extractor ###########################
        ##################################################
        sobelx = cv2.Sobel(n_initial_image,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(n_initial_image,cv2.CV_64F,0,1,ksize=5)
        sobel = magnitude_computation(sobelx, sobely)
        ##################################################
        # Canny Edge Extractor ###########################
        ##################################################
        canny = cv2.Canny(n_initial_image,10,30,3)
        # concatenate together
        x = np.concatenate((initial_image, lap_x, sobel, canny), axis=2)        
        #print(out1)
        fullpath = os.path.join(out1,image_dir)
        cv2.imwrite(fullpath,x)

#============================================================================================================================

def magnitude_computation(input_x, input_y):
    return np.sqrt(np.square(input_x)+np.square(input_y))