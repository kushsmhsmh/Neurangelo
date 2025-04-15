import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

def positional_encoding(x, L=10):
    output=[]
    output.append(x[0])
    for l in range(0, L):
        freq=(2**l)*3.141*x[0]
        output.append(torch.sin(freq))
        output.append(torch.cos(freq))
    output.append(x[1])
    for l in range(0, L):
        freq=(2**l)*3.141*x[1]
        output.append(torch.sin(freq))
        output.append(torch.cos(freq))
    output=torch.tensor(output).to('cuda')  # Move tensor to GPU
    return output

model = nn.Sequential(
    nn.Linear(42, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 3),
    nn.Sigmoid(),
).to("cuda")
image = np.array(imageio.imread("C:/Users/kamle/Downloads/1706544850231.jpg"))
image_path="C:/Users/kamle/Downloads/1706544850231.jpg"

def randomSampling(image_path, N=10000):
    image = np.array(imageio.imread(image_path))
    coordinates=[]
    colors=[]
    for i in range(0, N):
        x=np.random.randint(0, image.shape[0])
        y=np.random.randint(0, image.shape[1])
        coordinates.append([x / image.shape[0], y / image.shape[1]])
        colors.append(image[x, y] / 255.0)
    coordinates=torch.tensor(coordinates).to('cuda')  # Move tensor to GPU
    colors=torch.tensor(colors).to('cuda')  # Move tensor to GPU
    return coordinates, colors

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

import matplotlib.pyplot as plt

def generate_all_coordinates(image_shape):
    height, width = image_shape[0], image_shape[1]
    x = torch.arange(0, height).to('cuda') / height  # Move tensor to GPU
    y = torch.arange(0, width).to('cuda') / width  # Move tensor to GPU
    xv, yv = torch.meshgrid(x, y)
    coordinates = torch.stack((xv.flatten(), yv.flatten()), dim=1)
    return coordinates

def train(image_path, epochs=1000, batch_size=10000):
    coordinates, colors = randomSampling(image_path, batch_size)
    # Convert tensors to float
    coordinates = coordinates.float()
    colors = colors.float()
    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded_coordinates = torch.stack([positional_encoding(c) for c in coordinates])
        output = model(encoded_coordinates)
        loss = loss_fn(output, colors)
        loss.backward()
        optimizer.step()
        if epoch % 1 ==0:
            print(epoch, loss.item())
        if epoch % 100 == 0:
            print(epoch, loss.item())
        if epoch % 200 == 0:
            # Generate an image using the model
            test_coordinates = torch.stack([positional_encoding(c) for c in generate_all_coordinates(image.shape[:2])])
            test_output = model(test_coordinates)
            # Reshape the output to the shape of the original image
            test_image = test_output.view(image.shape[0], image.shape[1], 3).detach().cpu().numpy()
            # Plot the image
            plt.imshow(test_image)
            plt.show()

train(image_path, 1000, 10000)