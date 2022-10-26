import gzip
import numpy as np

# read the training images for training
with gzip.open('data/train-images-idx3-ubyte.gz', 'r') as fh:

    image_size = 28 # 28*28 pixel image
    num_images = 60000 # max 60000

    fh.read(16) # skip first 16 bytes (header)
    buf = fh.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size*image_size, 1)
    data = data/256

# read the training labels for training
with gzip.open('data/train-labels-idx1-ubyte.gz','r') as fh:

    fh.read(8) # skip first 8 bytes
    label = list()
    for i in range(0, num_images):   
        num = int.from_bytes(fh.read(1), "big")
        vec = np.zeros((10,1))
        vec[num] = 1
        label.append(vec)
    label = np.array(label)

# randomize the weight matrices and initialize bias weights
# we have two hidden layers here
num_hidden1 = 160 # number of neurons in the 1st layer
num_hidden2 = 40  # number of neurons in the 2nd layer

# bias weights
b_i_1 = np.zeros((num_hidden1, 1)) # input -> 1st layer
b_1_2 = np.zeros((num_hidden2, 1)) # 1st layer -> 2nd layer
b_2_o = np.zeros((10,1))           # 2nd layer -> output

# weight matrices
w_i_1 = np.random.uniform(-0.5, 0.5, (num_hidden1, image_size*image_size)) # input -> 1st layer
w_1_2 = np.random.uniform(-0.5, 0.5, (num_hidden2, num_hidden1))           # 1st layer -> 2nd layer
w_2_o = np.random.uniform(-0.5, 0.5, (10, num_hidden2))                    # 2nd layer -> 1st layer

# train the network until the required accuracy is achieved
# required accuracy as percentage
required_accu = float(input("Enter the accuracy you want (99.9 for 99.9%): "))

learn_rate = 0.01
epochs = 0 # count the number of epochs

while (True):
    epochs += 1
    
    num_correct = 0 # correct results counter

    for img, l in zip(data, label):

        # forward propagation: input -> 1st layer
        h1_pre = b_i_1 + w_i_1 @ img
        h1 = 1/(1+np.exp(-h1_pre)) # normalization with Sigmoid function
        
        # forward propagation: 1st layer -> 2nd layer
        h2_pre = b_1_2 + w_1_2 @ h1
        h2 = 1/(1+np.exp(-h2_pre))

        # forward propagation: 2nd layer -> output
        o_pre = b_2_o + w_2_o @ h2
        o = 1/(1+np.exp(-o_pre))

        # check if the output is correct
        num_correct += int(np.argmax(o) == np.argmax(l))

        # backward propagation: output -> 2nd layer
        delta_o = o-l
        w_2_o += -learn_rate * delta_o @ np.transpose(h2)
        b_2_o += -learn_rate * delta_o

        # backward propagation: 2nd layer -> 1st layer
        delta_h2 = np.transpose(w_2_o) @ delta_o * (h2 * (1-h2))
        w_1_2 += -learn_rate * delta_h2 @ np.transpose(h1)
        b_1_2 += -learn_rate * delta_h2
        
        # backward propagation: 1st layer -> input
        delta_h1 = np.transpose(w_1_2) @ delta_h2 * (h1 * (1-h1))
        w_i_1 += -learn_rate * delta_h1 @ np.transpose(img)
        b_i_1 += -learn_rate * delta_h1
    
    # display the result of this epoch
    accuracy = num_correct*100/num_images
    print(f"{epochs} epoch: got {num_correct} out of {num_images}.")
    print(f"Accuracy {accuracy:.2f}%")
    
    if (accuracy >= required_accu): break

print("The training is done. Testing the network with the test data...")

# read the test images for the test
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'r') as fh:

    image_size = 28 # 28*28 image
    num_images = 10000 # max 10000

    fh.read(16) # skip first 16 bytes
    buf = fh.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size*image_size, 1)
    data = data/256

# read the test labels for the test
with gzip.open('data/t10k-labels-idx1-ubyte.gz','r') as fh:
    fh.read(8) # skip first 8 bytes

    label = list()
    for i in range(0, num_images):   
        num = int.from_bytes(fh.read(1), "big")
        vec = np.zeros((10,1))
        vec[num] = 1
        label.append(vec)
    label = np.array(label)
    
# test the accuracy of the two-layer network
# correct result counter
num_correct = 0

for img, l in zip(data, label):

    # forward propagation: input -> 1st layer
    h1_pre = b_i_1 + w_i_1 @ img
    h1 = 1/(1+np.exp(-h1_pre))

    # forward propagation: 1st layer -> 2nd layer
    h2_pre = b_1_2 + w_1_2 @ h1
    h2 = 1/(1+np.exp(-h2_pre))

    # forward propagation: 2nd layer -> output
    o_pre = b_2_o + w_2_o @ h2
    o = 1/(1+np.exp(-o_pre))

    # check if the output is correct
    num_correct += int(np.argmax(o) == np.argmax(l))

# display the result
print(f"The network got {num_correct} correct out of {num_images}.")
print(f"Accuracy {num_correct*100/num_images:.2f}%")

# display an image and output the number recognized by the network
import matplotlib.pyplot as plt

while (True):
    idx = input("Enter an integer (0-9999) or q for exit: ")
    
    if (idx == 'q'): break
    
    img = data[int(idx)]
    
    # forward propagation: input -> 1st layer
    h1_pre = b_i_1 + w_i_1 @ img
    h1 = 1/(1+np.exp(-h1_pre))

    # forward propagation: 1st layer -> 2nd layer
    h2_pre = b_1_2 + w_1_2 @ h1
    h2 = 1/(1+np.exp(-h2_pre))

    # forward propagation: 2nd layer -> output
    o_pre = b_2_o + w_2_o @ h2
    o = 1/(1+np.exp(-o_pre))

    print(f"Is this a(n) {int(np.argmax(o))}?")

    dis = img.reshape(image_size, image_size, 1)
    image = np.asarray(dis).squeeze()
    plt.imshow(image)
    plt.show()
