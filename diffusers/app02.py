import torch 
from diffusers import DiffusionPipeline as DP
from PIL import Image,ImageDraw,ImageFont

def text_to_image(text,diffuser_model):
    #diffuser = diffusers.load_diffusers(diffuser_model)
    #image_data = diffuser.generate(text)
    #image = image.fromarray(image_data)
    dp = DP.from_pretrained("runwayml/stable-diffusion-v1-5",
                            torch_dtype = torch.float16)
    image_data = dp(text).images[0] 
    image = Image.fromarray(image_data)
    image.show()

if __name__ == "__main__" :
    input_text = "Hello,World!"
    diffuser_model = "example_diffuser_madel"
    text_to_image(input_text,diffuser_model)