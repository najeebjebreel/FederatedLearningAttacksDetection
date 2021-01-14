import models
from models import CNNMnist
model = CNNMnist()
model.train()
print(model)
print(len(model.state_dict()['fc2.weight']))
print(model.state_dict()['fc2.weight'][0].flatten())