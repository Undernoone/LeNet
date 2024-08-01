import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from net import LeNet
from torch.autograd import Variable

# Define the transforms for the data
date_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Load the dataset
train_dataset = datasets.MNIST('../Deeplearing_data', train=True, transform=date_transforms, download=False)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = datasets.MNIST('../Deeplearing_data', train=True, transform=date_transforms, download=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# If you want to use GPU, uncomment the following line
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
model = LeNet().to(device)

model.load_state_dict(torch.load('D:/LeNet/saved_model/best_model.pth'))

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

for i in range(10):
    img, label = test_dataset[i][0], train_dataset[i][1]
    img = Variable(torch.unsqueeze(img, 0)).to(device)
    with torch.no_grad():
        output = model(img)
        predicted, actual = classes[torch.argmax(output[0])], classes[label]
        print('Predicted: ', predicted, 'Actual: ', actual)