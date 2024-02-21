import torch
import streamlit as st 
from PIL import Image
from diffusers import DiffusionPipeline as DP

h = st.header('My Wed Site')
s = st.subheader('เว็บไซต์ส่วนตัวของนี้')
e = st.write('ท้ายสุดแต่ไปท้ายที่สุด')
b = st.button('click me')
banner = st.image('http://picsum.photos/800/250')
text = st.text_input('prompt :')
if text:
    dp = DP.from_pretrained("runwayml/stable-diffusion-v1-5",
                            torch_dtype = torch.float16)
    image_data = dp(text).images[0] 
    image = Image.fromarray(image_data)
    image.show()
    #text
    st.image('http://picsum.photos/800/250')
    b = st.button('สิไปต่อหรือ...')