from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch
import torch.nn.functional as F

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
weights = ResNet50_Weights.IMAGENET1K_V2

model.eval()

def preprocess_image(image_path):
    input_image = Image.open(image_path)
    preprocess = weights.transforms()
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def predict_image(filepath):
    input_tensor = preprocess_image(filepath)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = F.softmax(output[0], dim=0)
    _, indices = torch.topk(probabilities, 3)
    results = [(str(weights.meta["categories"][idx]), model.__class__.__name__.lower(), str(probabilities[idx].item())) for idx in indices]
    return results

def index(request):
    return render(request, 'index.html')

def upload_file(request):
    if request.method == 'POST':
        file = request.FILES['file']
        if file:
            filename = settings.MEDIA_ROOT / 'images' / file.name
            print(filename)
            with open(filename, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            prediction = predict_image(filename)
            return render(request, 'index.html', {'filename': file.name, 'prediction': prediction})
    return HttpResponse('Invalid request')