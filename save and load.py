import torch
import torch.onnx as onnx
import torchvision.models as models
#MODEL WEIGHTS
# saving model weights
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
# loading model weights
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
#MODEL SHAPE
# save model shape
torch.save(model, 'model.pth')
# load model shape
model = torch.load('model.pth')