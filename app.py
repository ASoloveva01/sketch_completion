import flask
from PIL import Image
from flask import render_template, request, json
import base64
import torch
from models.cgans import Generator
from training import device
from torchvision import transforms
from io import BytesIO
from pathlib import Path

MODEL_NAME = "generator.pth"
pretrained_path = Path("pretrained/")

sketch_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),


])
generator = Generator().to(device)
generator.load_state_dict(torch.load(f= pretrained_path / MODEL_NAME))
# Иницализируем индекс, с которого начинается нужная часть в строке закодированного изображения
init_Base64 = 21;

app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def home():
	return render_template('draw.html')




@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
         # Принимаем изображение в виде json
         draw = request.get_json()["init_img"]
         draw = draw[init_Base64:]
         # Декодирование
         draw_decoded = base64.b64decode(draw)
         image = Image.open(BytesIO(draw_decoded))
         # Применяем соответсвующее преобразование
         image = sketch_transforms(image)
         # Сокращаем количество каналов до 1
         image = image[[3],:, :]
         # Добавляем дополнительное 4-е измерение
         image = torch.unsqueeze(image, 0)
         # Передаем в модель предобработанное изображение
         gen_image = generator(image)
         pil_img = transforms.ToPILImage()(torch.squeeze(gen_image, 0)).convert('L')
         img_io = BytesIO()
         pil_img.save(img_io, 'png')
         img_str = base64.b64encode(img_io.getvalue()).decode("utf-8")
         # Ответ сервера в виде json
         return json.dumps({'gen_img': img_str})



if __name__ == '__main__':
	app.run(host='0.0.0.0')