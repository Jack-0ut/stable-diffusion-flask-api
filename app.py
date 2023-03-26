import os
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, send_file
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

def generate_image(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token="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"
    ).to(device)

    with autocast(device):
        image = pipe(prompt)["sample"][0]

    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        try:
            image = generate_image(prompt)
            image.save('static/generated_image.png', 'PNG')
            return send_file('static/generated_image.png', mimetype='image/png')
        except Exception as e:
            return str(e), 400
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
