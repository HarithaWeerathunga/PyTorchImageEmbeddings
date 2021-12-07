import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import glob
import os 
import shutil

import pymongo
from datetime import datetime
import pprint

import io
from tqdm import tqdm

# dirname = "data/train/"
dirname = "Cereal/"
#database configurations
mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
mongoDb = mongoClient["ImageEmbeddings_Cereal_EfficientNet"]
mongoDbRecords = mongoDb["Products"]


def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    # unsqueeze(0))  reshape our image from (3, 224, 224) to (1, 3, 224, 224)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1536)
#     print("My Embedding")
#     print(my_embedding)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
#         my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders


# Load the pretrained model
print("Initialize Model")
model = EfficientNet.from_pretrained('efficientnet-b3')
print(model)

#last layer is the fully connected layer where the features are being classified
# We want the features before the classification part of the CNN
# we want the first fully-connected layer which is the one before fc: the avgpool layer
# Use the model object to select the desired layer
layer = model._modules.get('_avg_pooling')
print(layer)


# Set model to evaluation mode
#turns off computationally expensive gradient tracking and siables training only features
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


subfolders = fast_scandir(dirname)
print(subfolders)


for folder in tqdm(subfolders):
    print(folder)
    s = folder.replace('/','')
    s = s.replace('Cereal','')
    count = 0
    print(s)
    mongoDbRecords = mongoDb[s]
    for filename in glob.glob( folder +'/*.jpg'):
        embeddings = get_vector(filename)
        embeddings = embeddings.tolist()
        embedding_data = {"ProductName": str(s), "Embedding": embeddings }
        output = mongoDbRecords.insert_one(embedding_data)
    print("Data Written to DB")

print("Process Finished")