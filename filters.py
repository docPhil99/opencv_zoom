import cv2
import numpy as np
import sys
import logging
logger = logging.getLogger(__name__)


def none(image):
    return image

#
# def temporal_feedback(image):
#     if temporal_feedback.buffer is None:
#         temporal_feedback.buffer=image.copy()
#
#     if temporal_feedback.counter == 10:
#         temporal_feedback.counter = 0
#         alpha=0.5
#         beta = (1.0 - alpha)
#         temporal_feedback.buffer = cv2.addWeighted(image, alpha, temporal_feedback.buffer, beta, 0.0)
#     temporal_feedback.counter += 1
#     return temporal_feedback.buffer
#
#
# temporal_feedback.buffer=None
# temporal_feedback.counter = 0

def add_noiseAWG(image):
    sigma=.001
    gn = np.random.normal(0,sigma,image.shape)*255
    img = image+gn.astype(np.uint8)
    return image+img

def sobelxy(image):
    p_depth=cv2.CV_8U
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgx=cv2.Sobel(img,p_depth,1,0,3,scale=1)
    imgx=cv2.convertScaleAbs(imgx)
    imgy=cv2.Sobel(img,p_depth,0,1,3,scale=1)
    imgy=cv2.convertScaleAbs(imgy)
    img=cv2.addWeighted(imgx,0.5,imgy,0.5,0)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = cv2.flip(img, 1)
    cv2.putText(img,"sobelxy",(40,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2, cv2.LINE_AA)
    img = cv2.flip(img, 1)
    return img



def sobely(image):
    p_depth=cv2.CV_8U
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgx=cv2.Sobel(img,p_depth,0,1,3,scale=1)
    img=cv2.cvtColor(imgx,cv2.COLOR_GRAY2RGB)
    img = cv2.flip(img, 1)
    cv2.putText(img,"sobely",(40,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2, cv2.LINE_AA)
    img = cv2.flip(img, 1)
    return img


def sobelx(image):
    p_depth=cv2.CV_8U
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgx=cv2.Sobel(img,p_depth,1,0,3,scale=1)
    img=cv2.cvtColor(imgx,cv2.COLOR_GRAY2RGB)
    img = cv2.flip(img, 1)
    cv2.putText(img,"sobelx",(40,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2, cv2.LINE_AA)
    img = cv2.flip(img, 1)
    return img


def laplace(image):
    p_depth=cv2.CV_8U
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgx=cv2.Laplacian(img,p_depth,3)
    img=cv2.cvtColor(imgx,cv2.COLOR_GRAY2RGB)
    img = cv2.flip(img, 1)
    cv2.putText(img,"laplace",(40,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2, cv2.LINE_AA)
    img = cv2.flip(img, 1)
    return img

def dog(image):
    p_depth=cv2.CV_8U
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Apply 3x3 and 7x7 Gaussian blur
    low_sigma = cv2.GaussianBlur(img, (3, 3), 0)
    high_sigma = cv2.GaussianBlur(img, (5, 5), 0)
    imgx = low_sigma - high_sigma
    #imgx=cv2.Laplacian(img,p_depth,3)
    img=cv2.cvtColor(imgx,cv2.COLOR_GRAY2RGB)
    img = cv2.flip(img, 1)
    cv2.putText(img,"dog 1",(40,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2, cv2.LINE_AA)
    img = cv2.flip(img, 1)
    return img



def dog2(image):
    p_depth=cv2.CV_8U
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Apply 3x3 and 7x7 Gaussian blur
    low_sigma = cv2.GaussianBlur(img, (7, 7), 0)
    high_sigma = cv2.GaussianBlur(img, (11, 11), 0)
    imgx = low_sigma - high_sigma
    #imgx=cv2.Laplacian(img,p_depth,3)
    img=cv2.cvtColor(imgx,cv2.COLOR_GRAY2RGB)
    img = cv2.flip(img, 1)
    cv2.putText(img,"dog 1",(40,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2, cv2.LINE_AA)
    img = cv2.flip(img, 1)
    return img



def laplace_of_gauss(image):
    p_depth=cv2.CV_8U
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgx = cv2.GaussianBlur(img,(5,5),1)
    imgx=cv2.Laplacian(imgx,p_depth,3)
    img=cv2.cvtColor(imgx,cv2.COLOR_GRAY2RGB)
    img = cv2.flip(img, 1)
    cv2.putText(img,"LoG",(40,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2, cv2.LINE_AA)
    img = cv2.flip(img, 1)
    return img
#
# def oil_paint(frame):
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (36, 36))
#     morph = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
#     result = cv2.normalize(morph, None, 20, 255, cv2.NORM_MINMAX)
#     return result
#
#
# def adaptive_threshold(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     return frame
#
# def global_threshold(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _,frame = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     return frame

def colormap(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

# def chroma(frame):
#     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     p1=(20,20)
#     p2=(620,60)
#     #draw rectangle
#     sample_region = hsv[p1[1]:p2[1],p1[0]:p2[0],:]
#     #hsv[p1[1]:p2[1], p1[0]:p2[0], 0]=1
#     frame=cv2.rectangle(frame,p1,p2,(255,0,0),1)
#
#     hm=[]
#     hstd=[]
#     for ind in range(3):
#         sample_region = hsv[p1[1]:p2[1], p1[0]:p2[0], ind].flatten()
#         mean = np.mean(sample_region)
#         std = np.std(sample_region)
#         hm.append(mean)
#         hstd.append(std)
#     logger.debug(f'mean {hm}, std {hstd}')
#     min_hue = np.array(hm) - np.array(std)*1
#     max_hue = np.array(hm) + np.array(std)*1
#     min_hue[2]=0
#     max_hue[2]=255
#
#     logger.debug(f'min {min_hue}, max {max_hue}')
#
#     mask = cv2.inRange(frame, min_hue, max_hue)
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     #return mask
#     output=frame.copy()
#     output[mask == 0] = [0, 0, 0]
#     return output


#def deep_dream(frame):
#    if deep_dream.dream is None:
#        from dream import DeepDream
#        deep_dream.dream=DeepDream()
#    return deep_dream.dream.process_frame(frame)

#deep_dream.dream = None
