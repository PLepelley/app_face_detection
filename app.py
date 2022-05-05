import streamlit as st
import cv2 
from PIL import Image
import numpy as np
import os
import pandas as pd
import datetime
import time



@st.cache
def load_image(img):
    im = Image.open(img)
    return im

cascade_path =  "./cascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces(image):
    color = (0, 110, 127) #La couleur du carré qui entoure le visage détecté
    src = np.array(image.convert('RGB'))
    colored = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    #Detect face
    rect = cascade.detectMultiScale(colored)

    #Draw rectangle
    if len(rect) > 0:
        #Creation of dataframe
        df_info = pd.DataFrame(columns = ['Personne', 'Date', 'Heure'])
        for i,[x, y, w, h] in enumerate(rect):
            #modification of the image
            cv2.rectangle(src, (x, y), (x+w, y+h), color)
            cv2.rectangle(src, (x, y - 30), (x + w, y), color, -1)
            cv2.putText(src, f"Personne {i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, .5, (0,0,0))

            #add rows in dataframe
            df_info = df_info.append({'Personne': f"Personne {i+1}",'Date':str(datetime.date.today()), 'Heure':str(datetime.datetime.today().time())}, ignore_index=True)
    return src, rect, df_info


def main():
    """ Face detection app"""

    st.title('Face Detection App')
    st.text('Build with streamlit and OpenCV')

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select activity", activities) 
    
    if choice == 'Detection' :
        st.subheader("Face Detection")

        image_file = st.file_uploader("Upload Image", type = ['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.markdown("Original Image")
            st.image(our_image)
        
            #Face detection            
            if st.button("Process"):
                result_img, result_face, df = detect_faces(our_image)
                #Export dataframe in excel
                df.to_excel("output.xlsx")

                #Modification in streamlit
                st.subheader("Image with face(s) detected")
                st.image(result_img)
                if len(result_face) > 1 :
                    st.markdown(f"{len(result_face)} faces were found")
                else :
                    st.markdown(f"{len(result_face)} face was found")
                #st.success(f"Found {len(result_face)} faces")
                st.subheader("Details of person detected")
                st.dataframe(df)


    elif choice == 'About':
        st.subheader('About')
    


if __name__ == '__main__':
    main()