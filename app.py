import cv2
import streamlit as st
from PIL import Image,ImageOps
import numpy as np

st.header("Image Cartonifier web app using OpenCV")
st.text("Upload the image to be cartonified")



def edge_mask(img,line_size,blur_value):
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        gray_blur=cv2.medianBlur(gray,blur_value)
        edges=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_value)
        return edges

def colour_quantization(image,k):
    #Transform the image
    data=np.float32(image).reshape((-1,3))
    #Determine Criteria
    criteria=(cv2.TermCriteria_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,0.01)
    # Implementing K-Means
    ret,label,center=cv2.kmeans(data,k, None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    result=center[label.flatten()]
    result=result.reshape(image.shape)

    return result


    

def main():
    
   file= st.file_uploader('Choose the file',type=['jpg','png','jpeg'])
   if file is not None:
      image=Image.open(file)
      img=np.array(image)
      line_size,blur_value=7,7
      edges=edge_mask(img,line_size,blur_value)
      img=colour_quantization(img,k=5 )
      blurred=cv2.bilateralFilter(img,d=7,sigmaColor=200,sigmaSpace=200)
      c=cv2.bitwise_and(blurred,blurred,mask=edges)
    
      st.image (image, caption='Original Image', width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
      st.image(c,caption='Cartonified Image',width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')



if __name__=='__main__':
    main()