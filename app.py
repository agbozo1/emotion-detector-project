#import imp
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import threading
from typing import Union
import av


import streamlit as st
import tensorflow as tf
np.set_printoptions(precision=1)


class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_image = frame.to_ndarray(format="bgr24")

            out_image = in_image[:, ::-1, :]  # Simple flipping for example.

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image

            return out_image

def emotion_classifier(image_file, model_location):

    #image pre-processing
    image = ImageOps.fit(image_file, (224, 224), Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    image_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_data[0] = normalized_image_array

    #predict
    emotion_model = tf.keras.models.load_model(model_location)
    prediction = emotion_model.predict(image_data)

    
    st.write(":neutral_face: - Neutral") 
    st.progress(round(np.float(prediction[0][0]),2))

    st.write(":smile: - Happy") 
    st.progress(round(np.float(prediction[0][1]),2))
    
    st.write(":pensive: - Sad") 
    st.progress(round(np.float(prediction[0][2]),2))

    st.write(":angry: - Angry") 
    st.progress(round(np.float(prediction[0][3]),2))

    st.write(":anguished: - Scared") 
    st.progress(round(np.float(prediction[0][4]),2))
    
    #st.progress(prediction[0][1] * 100)
    #st.progress(prediction[0][2] * 100)
    #st.progress(prediction[0][3] * 100)
    #st.progress(prediction[0][4] * 100)
    return np.argmax(prediction)


def take_a_shot(img):

    img_check, img_buffer = cv2.imencode(".jpg", img)
    io_bytes_buf = io.BytesIO(img_buffer)

    decoded_img = cv2.imdecode(np.frombuffer(io_bytes_buf.getbuffer(), np.uint8), -1)
    colorized_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    
    pil_image=Image.fromarray(colorized_img) 

    return pil_image

#teachable machine

st.set_page_config(layout='wide')
with st.sidebar:
    st.markdown("## Simple Emotion Detection App")
    st.markdown("this project was built as part of the **#30DaysofStreamlit** in April-May 2022.")
    st.markdown("- Liu Xiaozhi")
    st.markdown("- Sun Xu")
    st.markdown("- Avinash Kumar")
    st.markdown("- Al-Samarrai Safa Shakir Awad")
    st.markdown("**_Supervised By: Ebenezer Agbozo_**")
    st.markdown("**_(eagbozo@urfu.ru | agbozo1@gmail.com)_**")
    st.markdown("**_Ural Federal University_**")
col1, col2 = st.columns(2)
with col1:
    st.markdown('### **Option 1: Uploader**')
    img_upload = st.file_uploader('Use File Uploader',type=['jpg','png','jpeg'])

    #IMAGE UPLOADER OPTION
    if img_upload is not None:
        im1 = Image.open(img_upload).convert('RGB')
        emotion_classifier(im1, "keras_model.h5")
        #st.image(im1)

    st.write("OR")

    #CAMERA OPTION
    #camera on/off switch
    switch = st.radio("", ("Off", "On"))

    st.markdown('### **Option 2: Camera**')
    cam = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    if switch == "On":
        st.success("Turn Camera On") #notification
        
        #st_frame = st.empty()
           
        #cam_state, img = cam.read()
        #if switch:
        #      
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #FRAME_WINDOW.image(img)  


        #STREAMER----------------    
        ctx  = webrtc_streamer(key="emotion-app", #video_frame_callback=VideoTransformer,
            video_processor_factory=VideoTransformer
        )

        btn_mood = st.button("Capture Mood")
            
        

        if btn_mood:    
            #FRAME_WINDOW.image(img)           
            out_image = ctx.video_transformer.out_image
            
            pil_image = take_a_shot(out_image)
            
            #pass image from memory into model
            emotion_classifier(pil_image, "keras_model.h5")
            
            pil_image=None
                
                
    #switch off camera
    else:
        st.warning("Camera Off")        
        cam.release()
    
    #test for existence of image
    #image = cv2.imread("opencv.png")
    #if image is None:
    #    st.write('No Image Located')
    #else:
    #    image = Image.open("opencv.png")

