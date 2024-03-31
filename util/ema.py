import sys 
sys.path.append("../") 

import torch

class EMA():
       def __init__(self, mu,name_list):
           self.mu = mu
           self.name = name_list #["Alpha_encoder","Alpha_decoder"]
           self.shadow = {}

       def register(self, name, val):
           self.shadow[name] = val.clone()
          # print(self.shadow)
       def getname(self):
           return self.name

       def __call__(self, name, x):
           assert name in self.name
           if name not in self.shadow:
                self.register(name,x)
                return x
           else:
                new_average = (1.0 - self.mu) * x +  self.mu* self.shadow[name]
                self.shadow[name] = new_average.detach().clone()
                return new_average
