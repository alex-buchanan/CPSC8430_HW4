import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random

device = torch.device("cuda:0") 

def initWeights(layer):
    
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.normal_(layer.weight.data, mean = 0.0, std = 0.02)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight.data, mean = 0.0, std = 0.02)
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight.data, mean = 0.0, std = 0.02)
    elif isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.normal_(layer.weight.data, mean = 0.0, std = 0.02)

class Generator(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.l1 = torch.nn.Linear(10,25)
        self.l2 = torch.nn.Linear(125, 4*4*512)
        self.bn1 = torch.nn.BatchNorm2d(512, momentum=0.9)
        self.ct1 = torch.nn.ConvTranspose2d(512, 256, 13)
        self.bn2 = torch.nn.BatchNorm2d(256, momentum=0.9)
        self.ct2 = torch.nn.ConvTranspose2d(256, 128, 13)
        self.bn3 = torch.nn.BatchNorm2d(128, momentum=0.9)
        self.ct3 = torch.nn.ConvTranspose2d(128, 64, 13)
        self.bn4 = torch.nn.BatchNorm2d(64, momentum=0.9)
        self.ct4 = torch.nn.ConvTranspose2d(64, 3, 13)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, noise, text_tens):

        batch_dim = noise.size(0)
        text_emb = self.relu(self.l1(text_tens))
        x = torch.concat([noise, text_emb], dim = -1)
        x = self.relu(self.l2(x))
        x = x.reshape(batch_dim, 512, 4, 4)
        x = self.relu(self.bn1(x))
        x = self.ct1(x)
        x = self.relu(self.bn2(x))
        x = self.ct2(x)
        x = self.relu(self.bn3(x))
        x = self.ct3(x)
        x = self.relu(self.bn4(x))
        output = self.tanh(self.ct4(x))
        output = (output + 1)/2
        return output
    


class Discriminator(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 64, 13)
        self.l1 = torch.nn.Linear(10, 25)
        self.c2 = torch.nn.Conv2d(64, 128, 13)
        self.b1 = torch.nn.BatchNorm2d(128, momentum=0.9)
        self.c3 = torch.nn.Conv2d(128, 256, 13)
        self.b2 = torch.nn.BatchNorm2d(256, momentum=0.9)
        self.c4 = torch.nn.Conv2d(256, 512, 13)
        self.b3 = torch.nn.BatchNorm2d(512, momentum=0.9)
        self.c5 = torch.nn.Conv2d(537, 260, 4)
        self.l2 = torch.nn.Linear(260, 1)
        self.relu = torch.nn.ReLU()
        self.leaky = torch.nn.LeakyReLU(0.2)
        self.sig = torch.nn.Sigmoid()
        
        
    def forward(self, imgs, text):
        
        batch_dim = imgs.size(0)
        
        text_emb = self.relu(self.l1(text))
        
        img_x = self.leaky(self.c1(imgs))
        img_x = self.c2(img_x)
        img_x = self.leaky(self.b1(img_x))
        
        
        img_x = self.c3(img_x)
        img_x = self.leaky(self.b2(img_x))
        img_x = self.c4(img_x)
        img_x = self.leaky(self.b3(img_x))
        
        text_emb = text_emb.unsqueeze(-1)
        text_emb = text_emb.unsqueeze(-1)
        text_emb_tile = text_emb.repeat(1,1,4,4)
        
        x = torch.concat([img_x, text_emb_tile], dim = 1)
       
        x = self.c5(x)
        x = x.reshape(batch_dim, 260)
        x = self.sig(self.l2(x))
        
        return x
    

def dLoss(d_outputs, batch_size, index_list):
    loss = torch.tensor([0.0], requires_grad = True).to(device)
    for i in range(batch_size): 
        if (index_list[i] < 2500):
            loss = loss + torch.log(1-d_outputs[i])    
        else: 
            loss = loss + torch.log(d_outputs[i]) 
    loss = loss / batch_size
    loss = loss * -1
    return loss



def gLoss(fake_outputs, batch_size):
    loss = torch.tensor([0.0], requires_grad = True).to(device)
    for f_output in fake_outputs:
        loss = loss + torch.log(1-f_output) 
    loss = loss / batch_size
    return loss


def createNoise(batch_size):
    noise = torch.zeros(batch_size, 100).to(device)      
    torch.nn.init.uniform_(noise)      
    return noise

def getFakeRealIndexes(g_batch, r_batch, label_tens):
    index_list = [x for x in range(5000)]
    combined_tens = torch.cat([g_batch, r_batch], dim=0).to(device)
    tile_label_tens = label_tens.repeat(2,1)
    
    random.shuffle(index_list)
    
    new_tens_list = []
    new_label_list = []
    
    for index in index_list:
        
        new_tens_list.append(combined_tens[index].unsqueeze(0))
        new_label_list.append(tile_label_tens[index].unsqueeze(0))
        
    new_combined_tens = torch.cat(new_tens_list, dim=0)
    new_label_tens = torch.cat(new_label_list, dim = 0)
    
    return new_combined_tens, new_label_tens, index_list

def transformLabels(batch_labels):
    batch_num = batch_labels.size(0)
    label_tens = torch.zeros(batch_num, 10)
    for i in range(batch_num):
        index = batch_labels[i].item()
        label_tens[i, index] = 1
    return label_tens

def train(): 
    generator = Generator()
    generator = nn.DataParallel(generator)
    generator.to(device)
    discriminator = Discriminator()
    discriminator = nn.DataParallel(discriminator)
    discriminator.to(device)
            
    generator.apply(initWeights)
    discriminator.apply(initWeights)
    
    optim_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas = (0.5, 0.999))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas = (0.5, 0.999))
    
    transform = transforms.Compose([
        
        transforms.Resize((52,52)),
        transforms.ToTensor()])
    
    trainset = torchvision.datasets.CIFAR10(root='./train_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2500, shuffle=True)
            
    for epoch in range(5):
    
        cnt = 1
    
        for real_batch, batch_labels in trainloader:

            if cnt % 5 != 0:
                n_batch_size = real_batch.size(0)
                noise = createNoise(n_batch_size)
                label_tens = transformLabels(batch_labels.to(device))
                g_outputs = generator(noise, label_tens)
                combined_tens, combined_labels, fakeRealList = getFakeRealIndexes(g_outputs, real_batch.to(device), label_tens)
                d_outputs = discriminator(combined_tens, combined_labels)
                tot_batch_size = combined_tens.size(0)
                d_loss = dLoss(d_outputs, tot_batch_size, fakeRealList)
                optim_d.zero_grad()
                d_loss.backward()
                optim_d.step()
            
                print(f"d_loss: {d_loss.item()}")

            else:
                noise = createNoise(5000)
                label_tens = transformLabels(batch_labels)
                combined_label_tens = label_tens.repeat(2,1)
                g_outputs = generator(noise, combined_label_tens)
                d_outputs = discriminator(g_outputs, combined_label_tens)
                g_loss = gLoss(d_outputs, 5000)
                optim_g.zero_grad()
                g_loss.backward()
                optim_g.step()

                print(f"g_loss: {g_loss.item()}")
            
            cnt += 1
            torch.cuda.empty_cache()
            
    torch.save(generator.state_dict(), "dc_generator.pth")
    torch.save(discriminator.state_dict(), "dc_discriminator.pth")

if __name__ == "__main__":
            
    train()
