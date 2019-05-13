from nn import nn
from mse import MSE
import numpy as np

x = []
while 1:
    try:
        b = float(input("Please input values -> "))
        x.append(b)
    except EOFError:
        break
print("\n")

a = []
while 1:
    try:
        b = float(input("Please input answers -> "))
        a.append(b)
    except EOFError:
        break
print("\n")

c = int(input("Please input number of layer -> "))
l = float(input("Please input value of LR rate -> "))
count = int(input("Please input value of learn number of times -> "))

length = len(x)
if len(a) != length:
    print("Match the number of values and answers.")

x = np.array(x)
a = np.array(a)

i = 0
f = []
for i in range(c):
    f.append(nn(length, l))

mse = MSE()

i = 0
j = 0
for i in range(count):
    j = 0
    for j in range(c):
        if j == 0:
            tmp = f[j].forward(x)
        else:
            tmp = f[j].forward(tmp)

    loss = mse.forward(tmp, a).mean()
    print(loss)

    new_w = mse.backward(tmp, a)

    j = 0
    for j in range(c):
        if j == 0:
            tmp = f[c - (j + 1)].topback(new_w)
        else:
            tmp = f[c - (j + 1)].backward()
    
    j = 0
    for j in range(c):
        f[j].update()


