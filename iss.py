import numpy as np
import torch
from torchvision import datasets, transforms
import faiss                  


#GNERATING IMAGE EMBEDDINGS

# create dataloader with required transforms
tc = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

image_datasets = datasets.ImageFolder("C:/Datasets/faiss", transform=tc)
dloader = torch.utils.data.DataLoader(image_datasets, batch_size=10, shuffle=False)

print(len(image_datasets))


#for runnig in console (in debug mode)
#file = open('items4.txt','w')
# for i in range(len(image_datasets.samples)):
# 	file.write(str(i) + "\n" + str(image_datasets.samples[i]) + "\n")
#file.close()



# fetch pretrained model
model = torch.hub.load('pytorch/vision', 'efficientnet_v2_s', pretrained=True)
                            #resnet152  efficientnet_b7  efficientnet_v2_l  ...

# Select the desired layer
layer = model._modules.get('avgpool')
#layer = model._modules.get('layer3')


def copy_embeddings(m, i, o):
    """Copy embeddings from the penultimate layer.
    """
    o = o[:, :, 0, 0].detach().numpy().tolist()
    outputs.append(o)

outputs = []
# attach hook to the penulimate layer
_ = layer.register_forward_hook(copy_embeddings)



# Generate image's embeddings for all images in dloader and saves
# them in the list outputs
for X, y in dloader:
    _ = model(X)
print(len(outputs))


# flatten list of embeddings to remove batches
list_embeddings = [item for sublist in outputs for item in sublist]

print(len(list_embeddings))
print(np.array(list_embeddings[0]).shape) #returns (512,)




#PERFORMING SIMILARITY SEARCH ON GENERATED IMAGE EMVBEDDINGS

index = faiss.IndexFlatL2(1280)  # build the index, d=size of vectors. size of vectors depends on the architecture that you are using, e.g. resnet18, resnet152, efficientnet_b0 and etc.
#index = faiss.IndexFlatL2(256)

# here we assume arr contains a n-by-d numpy matrix of type float32
arr = np.array(list_embeddings)

index.add(arr)                  # add vectors to the index
print(index.ntotal)


# query is a n2-by-d matrix with query vectors
query = np.expand_dims(arr, axis=1)

k = 4                          # we want 4 similar vectors
D, I = index.search(query[55], k)     # actual search      55 is just an example
print(I)
