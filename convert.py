import os

import torch
from torch.autograd import Variable

from net import Net

model = Net()
model.load_state_dict(torch.load('models/mnist_cnn.pt',
                      map_location=torch.device('cpu')))

onnx_model_path = "Sudoku/models"
onnx_model_name = "mnist_cnn.onnx"

os.makedirs(onnx_model_path, exist_ok=True)
full_model_path = os.path.join(onnx_model_path, onnx_model_name)

generated_input = Variable(
    torch.randn(1, 1, 32, 32),
)
torch.onnx.export(
    model,
    generated_input,
    full_model_path,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
)
