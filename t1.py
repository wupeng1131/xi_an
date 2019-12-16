# # from __future__ import print_function
# # import torch
# # x = torch.empty(5,3)
# # x = torch.rand(5,5)
# # #print(x)
# # #print(x.size())
# # #x = torch.randn(1)
# # print(x)
# # # print(x.item())
# # print (torch.cuda.is_available())
# # if torch.cuda.is_available():
# #     device = torch.device("cuda")
# #     y = torch.ones_like(x,device = device)
# #     print(y)
# #     x = x.to(device)
# #     print(x)
# #     z = x + y
# #     print(z)
# #     print(z.to("cpu", torch.double))
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 3x3 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#
#
# net = Net()
# print(net)
#
# params = list(net.parameters())
# print(len(params))
# print(params[2].size())
#
# # print(params)
# input = torch.randn(1,1,32,32)
# out = net(input)
# print(out)
# net.zero_grad()
# out.backward(torch.randn(1,10))
#
# output = net(input)
# target = torch.randn(10)
# print("before",target, target.size())
# target = target.view(1,-1)
# print("after", target,target.size())
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)
#
#
#
import numpy as np
from matplotlib import pyplot as plt
import math
x = np.linspace(0.001,0.999,50)
# y = np.sin(x-2) * np.sin(x-2) * np.exp(-x**2)
a = 10
y = -1/(1+np.e**(a*(x - 0.5)))*np.log(x)
y1 = -1/(1+np.e**(20*(x - 0.5)))*np.log(x)
y2 = -1/(1+np.e**(5*(x - 0.5)))*np.log(x)
y3 = -1/(1+np.e**(1*(x - 0.5)))*np.log(x)
y4 = (-1/(1+np.e**(0.005*(x - 0.5))))*np.log(x)

y5 = (-1*(1-x)**0)*np.log(x)
# y5 = -1*np.log(x)


plt.figure(1)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0, 5)
plt.title(r"$f(x) = sin^2(x-2)e^{-x^2}$")
plt.annotate('max', xy=(0.22, 0.9), xytext=(0.22, 0.5),
            arrowprops=dict(facecolor='black'))
plt.plot(x,y)
plt.plot(x, y1, color='green', label='y1')
plt.plot(x, y2, color='red', label='y2')
plt.plot(x, y3, color='yellow', label='y3')
plt.plot(x, y4, color='yellow', label='y3')
plt.plot(x, y5, color='orange', label='y3')
plt.show()
