from torchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import faiss                  



# create dataloader with required transforms
tc = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

image_datasets = datasets.ImageFolder("C:/Datasets/faiss", transform=tc)
dloader = torch.utils.data.DataLoader(image_datasets, batch_size=10, shuffle=False)

print(len(image_datasets))

#file = open('items4.txt','w')
# for i in range(len(image_datasets.samples)):
# 	file.write(str(i) + "\n" + str(image_datasets.samples[i]) + "\n")
#file.close()


#for img, label in dloader:
      #print(np.transpose(img[0], (1,2,0)).shape)
      #print(img[0])
      #plt.imshow((img[0].detach().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
      #plt.show()
      #i = i + 1
      #break


# fetch pretrained model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)


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



# Iterate through the images and their corresponding embeddings,
#and append them to hub dataset
#for i in tqdm(range(len(image_datasets))):
    #img = image_datasets[i][0].detach().numpy().transpose(1, 2, 0)
    #img = img * 255 # images are normalized
    #img = img.astype(np.uint8)

    #print("Image:")
    #print(img.shape)
    #plt.imshow(img)
    #plt.show()
    #print(list_embeddings[0:10])  # show only 10 first values of the image embedding






index = faiss.IndexFlatL2(512)  # build the index, d=size of vectors
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
