try:
    from PIL import Image
except ImportError:
    import Image
import PIL.ImageOps
import cv2
import numpy
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
import os, os.path
from time import time
from torch.utils.data import Dataset
from torchvision import transforms

def predict_digit(img, model):
    img = img.view(1, 1944)
    with torch.no_grad():
        logps = model(img)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    #print(probab)
    predicted = probab.index(max(probab))
    return predicted


def save_training_images():
    for f in os.listdir(src_dir):
        img = Image.open(os.path.join(src_dir, f))
        f_name = f[:-4]
        for i in range(4):
            tl_x = x_init_t1 + x_spacing * i
            tl_y = y_init_t1 + y_spacing * i

            k_a_img = img.crop((tl_x, tl_y, tl_x + width, tl_y + height))
            spec_img = img.crop((tl_x + x_spec_offset, tl_y - int((4 - i) / 2), tl_x + width + x_spec_offset, tl_y + height - int((4 - i) / 2)))

            k_a_grey = k_a_grey.resize((k_a_grey.size[0] * 3, k_a_grey.size[1] * 3), PIL.Image.LANCZOS)
            k_a_grey = get_contours_from_image(k_a_img)
            k_a_grey_1 = k_a_grey.crop((0, 0, int(k_a_grey.size[0] / 2), k_a_grey.size[1]))
            k_a_grey_2= k_a_grey.crop((int(k_a_grey.size[0] / 2), 0, k_a_grey.size[0], k_a_grey.size[1]))
            extrema = k_a_grey_1.getextrema()
            if extrema[0] != extrema[1]:
                k_a_grey_1.save(tgt_dir + '/' + f_name + '_ka_'+str(i + 1)+'_1.png')
            k_a_grey_2.save(tgt_dir + '/' + f_name + '_ka_'+str(i + 1)+'_2.png')

            spec_grey = spec_grey.resize((spec_grey.size[0] * 3, spec_grey.size[1] * 3), PIL.Image.LANCZOS)
            spec_grey = get_contours_from_image(spec_img)
            spec_grey_1 = spec_grey.crop((0, 0, int(spec_grey.size[0] / 2), spec_grey.size[1]))
            spec_grey_2 = spec_grey.crop((int(spec_grey.size[0] / 2), 0, spec_grey.size[0], spec_grey.size[1]))
            extrema = spec_grey_1.getextrema()
            if extrema[0] != extrema[1]:
                spec_grey_1.save(tgt_dir + '/' + f_name + '_spec_'+str(i + 1)+'_1.png')
            spec_grey_2.save(tgt_dir + '/' + f_name + '_spec_'+str(i + 1)+'_2.png')


            tl_x_t2 = x_init_t2 - x_spacing * i
            tl_y_t2 = y_init_t2 + y_spacing * i

            k_a_img = img.crop((tl_x_t2, tl_y_t2, tl_x_t2 + width, tl_y_t2 + height))
            spec_img = img.crop((tl_x_t2 + x_spec_offset, tl_y_t2 + i, tl_x_t2 + width + x_spec_offset, tl_y_t2 + height + i))

            k_a_grey = k_a_grey.resize((k_a_grey.size[0] * 3, k_a_grey.size[1] * 3), PIL.Image.LANCZOS)
            k_a_grey = get_contours_from_image(k_a_img)
            k_a_grey_1 = k_a_grey.crop((0, 0, int(k_a_grey.size[0] / 2), k_a_grey.size[1]))
            k_a_grey_2= k_a_grey.crop((int(k_a_grey.size[0] / 2), 0, k_a_grey.size[0], k_a_grey.size[1]))
            extrema = k_a_grey_1.getextrema()
            if extrema[0] != extrema[1]:
                k_a_grey_1.save(tgt_dir + '/' + f_name + '_ka_'+str(5 + i)+'_1.png')
            k_a_grey_2.save(tgt_dir + '/' + f_name + '_ka_'+str(5 + i)+'_2.png')

            spec_grey = spec_grey.resize((spec_grey.size[0] * 3, spec_grey.size[1] * 3), PIL.Image.LANCZOS)
            spec_grey = get_contours_from_image(spec_img)
            spec_grey_1 = spec_grey.crop((0, 0, int(spec_grey.size[0] / 2), spec_grey.size[1]))
            spec_grey_2 = spec_grey.crop((int(spec_grey.size[0] / 2), 0, spec_grey.size[0], spec_grey.size[1]))
            extrema = spec_grey_1.getextrema()
            if extrema[0] != extrema[1]:
                spec_grey_1.save(tgt_dir + '/' + f_name + '_spec_'+str(5 + i)+'_1.png')
            spec_grey_2.save(tgt_dir + '/' + f_name + '_spec_'+str(5 + i)+'_2.png')


def get_contours_from_image(img, thresh=50, bilat=17):
    open_cv_img = numpy.array(img) 
    # Convert RGB to BGR 
    open_cv_img = open_cv_img[:, :, ::-1].copy()
    grey = cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2GRAY)
    grey = cv2.bitwise_not(grey)
    (thresh2, grey) = cv2.threshold(grey, thresh, 255, cv2.THRESH_BINARY)
    grey = cv2.bilateralFilter(grey, 11, bilat, bilat)
    grey = cv2.Canny(grey, 30, 200)
    return Image.fromarray(grey)


class ImageLabelDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):

        self.to_tensor = transforms.ToTensor()
        self.filepaths = filepaths
        self.labels = labels
        # Calculate len
        self.data_len = len(self.filepaths)
        self.transform = transform

    def __getitem__(self, index):
        single_image_name = self.filepaths[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        if self.transform:
            img_as_img = self.transform(img_as_img)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


def delete_extra_train_imgs():
    src_dir = 'C://Users/bijmb/Documents/splatoon related/ocr/training/train'
    labels = []
    with open('C://Users/bijmb/Documents/splatoon related/ocr/training/training_labels_alphabetical.csv', 'r') as file:
        labels = list(map(lambda x: int(x), file.read().split('\n')))
    count = [0] * 10
    img_files = list(map(lambda x: os.path.join(src_dir, x), sorted(os.listdir(src_dir))))
    for i in range(len(img_files)):
        img = img_files[i]
        label = labels[i]
        if count[label] < 36:
            count[label] = count[label] + 1
        else:
            labels[i] = -1
            os.remove(img)

    with open('C://Users/bijmb/Documents/splatoon related/ocr/training/training_labels_alphabetical_2.csv', 'w') as f:
        for item in filter(lambda x: x != -1, labels):
            f.write("%s\n" % item)   


def train_splatfont_model():
    labels = []
    with open('C://Users/bijmb/Documents/splatoon related/ocr/training/training_labels_alphabetical.csv', 'r') as file:
        labels = list(map(lambda x: int(x), file.read().split('\n')))
    src_dir = 'C://Users/bijmb/Documents/splatoon related/ocr/training/train'
    img_files = list(map(lambda x: os.path.join(src_dir, x), sorted(os.listdir(src_dir))))

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(6,scale=None, translate=(.05, .05), shear=None, resample=0, fillcolor=0)
    ]) 

    train_data = ImageLabelDataset(img_files, labels, transforms)



    labels2 = []
    with open('C://Users/bijmb/Documents/splatoon related/ocr/training/temp_labels_alphabetical.csv', 'r') as file:
        #        labels = list(map(lambda x: [1 if int(x) == y else 0 for y, i in enumerate([0] * 10)], file.read().split('\n')))
        labels2 = list(map(lambda x: int(x), file.read().split('\n')))
    src_dir2 = 'C://Users/bijmb/Documents/splatoon related/ocr/training/temp'
    img_files2 = list(map(lambda x: os.path.join(src_dir2, x), sorted(os.listdir(src_dir2))))

    test_data = ImageLabelDataset(img_files2, labels2, transforms)  
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=63)

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64)
    i1, l1 = next(iter(trainloader))
    print(i1.shape)
    # # Download and load the training data
    # trainset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=True, transform=transform)
    # valset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=False, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    # Layer details for the neural network

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)


    input_size = 1944
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))
    print(model)

    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)
    logps = model(images)
    loss = criterion(logps, labels)



    print('Before backward pass: \n', model[0].weight.grad)

    loss.backward()

    print('After backward pass: \n', model[0].weight.grad)

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    print('Initial weights - ', model[0].weight)

    images, labels = next(iter(trainloader))
    images.resize_(64, 1944)

    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    print('Gradient -', model[0].weight.grad)

    # Take an update step and few the new weights
    optimizer.step()
    print('Updated weights - ', model[0].weight)

    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)
    time0 = time()
    epochs = 1000
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    images, labels = next(iter(testloader))
    correct = 0
    total = 0
    for img, lab in zip(images, labels):
        total = total + 1
        predicted = predict_digit(img, model)
        label = lab.item()
        if predicted == label:
            correct = correct + 1
        else:
            print("Predicted Digit =", predicted)
            print("Actual Digit =", label)
    print("Total Correct: " + str(correct) + "/" + str(total))


    images, labels = next(iter(trainloader))
    correct = 0
    total = 0
    for img, lab in zip(images, labels):
        total = total + 1
        predicted = predict_digit(img, model)
        label = lab.item()
        if predicted == label:
            correct = correct + 1
        else:
            print("Predicted Digit =", predicted)
            print("Actual Digit =", label)
    print("Total Correct: " + str(correct) + "/" + str(total))
        #view_classify(img.view(1, 36, 54), ps)
    torch.save(model.state_dict(), './splatfont_model.pt')  

def train_special_model():
    labels = []
    with open('C://Users/bijmb/Documents/splatoon related/ocr/training/training_labels_alphabetical.csv', 'r') as file:
        labels = list(map(lambda x: int(x), file.read().split('\n')))
    src_dir = 'C://Users/bijmb/Documents/splatoon related/ocr/training/train'
    img_files = list(map(lambda x: os.path.join(src_dir, x), sorted(os.listdir(src_dir))))

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(6,scale=None, translate=(.05, .05), shear=None, resample=0, fillcolor=0)
    ]) 

    train_data = ImageLabelDataset(img_files, labels, transforms)



    labels2 = []
    with open('C://Users/bijmb/Documents/splatoon related/ocr/training/temp_labels_alphabetical.csv', 'r') as file:
        #        labels = list(map(lambda x: [1 if int(x) == y else 0 for y, i in enumerate([0] * 10)], file.read().split('\n')))
        labels2 = list(map(lambda x: int(x), file.read().split('\n')))
    src_dir2 = 'C://Users/bijmb/Documents/splatoon related/ocr/training/temp'
    img_files2 = list(map(lambda x: os.path.join(src_dir2, x), sorted(os.listdir(src_dir2))))

    test_data = ImageLabelDataset(img_files2, labels2, transforms)  
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=63)

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64)
    i1, l1 = next(iter(trainloader))
    print(i1.shape)
    # # Download and load the training data
    # trainset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=True, transform=transform)
    # valset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=False, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    # Layer details for the neural network

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)


    input_size = 1944
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))
    print(model)

    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)
    logps = model(images)
    loss = criterion(logps, labels)



    print('Before backward pass: \n', model[0].weight.grad)

    loss.backward()

    print('After backward pass: \n', model[0].weight.grad)

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    print('Initial weights - ', model[0].weight)

    images, labels = next(iter(trainloader))
    images.resize_(64, 1944)

    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    print('Gradient -', model[0].weight.grad)

    # Take an update step and few the new weights
    optimizer.step()
    print('Updated weights - ', model[0].weight)

    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)
    time0 = time()
    epochs = 1000
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    images, labels = next(iter(testloader))
    correct = 0
    total = 0
    for img, lab in zip(images, labels):
        total = total + 1
        predicted = predict_digit(img, model)
        label = lab.item()
        if predicted == label:
            correct = correct + 1
        else:
            print("Predicted Digit =", predicted)
            print("Actual Digit =", label)
    print("Total Correct: " + str(correct) + "/" + str(total))


    images, labels = next(iter(trainloader))
    correct = 0
    total = 0
    for img, lab in zip(images, labels):
        total = total + 1
        predicted = predict_digit(img, model)
        label = lab.item()
        if predicted == label:
            correct = correct + 1
        else:
            print("Predicted Digit =", predicted)
            print("Actual Digit =", label)
    print("Total Correct: " + str(correct) + "/" + str(total))
        #view_classify(img.view(1, 36, 54), ps)
    torch.save(model.state_dict(), './splatfont_model.pt') 




d1 ='C://Users/bijmb/Documents/splatoon related/ocr/special_icons/'
d2 ='C://Users/bijmb/Documents/splatoon related/ocr/special_icons_outline'
for f in os.listdir(d1):
    img = Image.open(os.path.join(d1, f))
    get_contours_from_image(img, 190).save(os.path.join(d2, f))

d3 ='C://Users/bijmb/Documents/splatoon related/ocr/splash_ex.png'
d4 ='C://Users/bijmb/Documents/splatoon related/ocr/splash_ex_outline.png'    
img = Image.open(d3)
get_contours_from_image(img, 190).save(d4)