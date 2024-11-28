import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dcgan_train import Generator, Discriminator, createNoise, transformLabels
from PIL import Image

device = torch.device("cuda:0")

def dLoss(d_output):
    
    real_prob = 0.0
        
    real_prob = d_output.item()
    
    return real_prob

def dLossBatch(d_outputs):
    
    loss = 0.0
    
    for d in d_outputs:
        
        loss += torch.log(1 - d).item()
        
    return loss


def createImage(g_image_list):
    
    class_tens_list = []
    
    for img_class in g_image_list:
        
        img_list = []
        
        for img in img_class:
            
            img = img.reshape(3,52,52)
            img_list.append(img)
        
        class_tens = torch.cat(img_list, dim=-1)
        class_tens_list.append(class_tens)
        
    whole_tens = torch.cat(class_tens_list, dim=1)
        
    whole_tens = whole_tens * 255
    whole_tens = whole_tens.byte()

    np_image = whole_tens.cpu().numpy() 
    np_image = np.transpose(np_image, (1, 2, 0))
    pil_image = Image.fromarray(np_image)
    path = "images/dcganImage" + ".png"
    pil_image.save(path)

def normalize(values): 
    
    min_val = min(values) 
    max_val = max(values) 
    
    return [(x - min_val) / (max_val - min_val) for x in values]

def evaluate():
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((52,52)),
        transforms.ToTensor()])
    
    testset = torchvision.datasets.CIFAR10(root='./test_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    gen = torch.load("dc_generator.pth", weights_only = True)
    new_gen = {k.replace('module.', ''): v for k,v in gen.items()}
    generator.load_state_dict(new_gen)

    disc = torch.load("dc_discriminator.pth", weights_only=True)
    new_disc = {k.replace('module.', ''): v for k,v in disc.items()}
    discriminator.load_state_dict(new_disc)
    
    generator.eval()
    discriminator.eval()
    
    best_image_list = []
    big_loss_list = []
    
    label_tens = torch.zeros(1,10)
    
    for j in range(10):
        
        label_tens[0,j] = 1
        g_class_list = []
        loss_list = []
    
        for i in range(500):

            noise = createNoise(1).to(device)

            g_image = generator(noise, label_tens.to(device))

            d_output = discriminator(g_image, label_tens.to(device))

            loss = dLoss(d_output)

            g_class_list.append(g_image.squeeze(0))
            loss_list.append(loss)
            big_loss_list.append(loss)
            
        loss_dict = {str(i) : loss_list[i] for i in range(500)}

        sorted_loss_list = sorted(loss_dict.items(), key=lambda item: item[1])
        best_ind_list = [int(key) for key, _ in sorted_loss_list[-10:]]
        best_class_imgs = [g_class_list[i] for i in best_ind_list]
        best_image_list.append(best_class_imgs)

        print(j)
                
        label_tens[0,j] = 0
    
    createImage(best_image_list)
    
    real_cnt = 1
    real_loss_list = []
    
    for real_imgs, real_labels in testloader:
        
        if real_cnt <= 500:
            
            label_tens = transformLabels(real_labels)
            d_output = discriminator(real_imgs.to(device), label_tens.to(device))
            loss = dLoss(d_output)
            real_loss_list.append(loss)
            
        elif real_cnt == 1000:
            
            real_cnt = 0
            print("Done with real images")
            
        real_cnt+=1
    
    
    precision = real_cnt / 5000
    precision_str = str(precision)
    
    big_norm_loss_list = normalize(big_loss_list)
    real_norm_loss_list = normalize(real_loss_list)
    
    mean = np.mean(big_norm_loss_list)
    std = np.std(big_norm_loss_list)
    rmean = np.mean(real_norm_loss_list)
    rstd = np.std(real_norm_loss_list)
    
    with open('precision.txt', 'w') as file:
        
        file.write(precision_str)
        file.write("\n")
        file.write(str(mean))
        file.write("\n")
        file.write(str(std))
        file.write("\n")
        file.write(str(rmean))
        file.write("\n")
        file.write(str(rstd))

if __name__ == "__main__":
    
    evaluate()
    
    
    