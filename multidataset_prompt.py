import os.path as osp
import argparse
import torch
import torch.nn.utils as utils
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
import glob 
import random
from trainer.cocoop import *

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device='cpu')

repeat_transform = transforms.Compose([
    transforms.ToTensor(),
])

with open('prompts/prompts_list_multi.txt', 'r') as file:
    prompt_list = file.readlines()

            
attri_embed = torch.from_numpy(np.load('./attributes/attribute_multi.npy')).to(device).to(torch.float32)
mask_embed = torch.from_numpy(np.load('./attributes/masks_multi.npy')).to(device).to(torch.bool)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(42)

class DataTrain(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    image = preprocess(Image.open(self.image_path[idx]))
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
 
    label_one_hot=F.one_hot(label,49)
  
    return image, domain, label, label_one_hot 


#################-------DATASET------#######################

# target_domains = [ 'painting','clipart', 'real','sketch']
parser = argparse.ArgumentParser(description='Run domain adaptation with specified source and target domains')
parser.add_argument('--source_domains', type=str, required=True, help='Comma-separated list of source domains')
parser.add_argument('--target_domain', type=str, required=True, help='Target domain')
# parser.add_argument('--shot', type=int, required=True, help='Target domain')
parser.add_argument('--shots', type=int, default=1, help='Number of shots')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    # Define fine argument as a boolean flag that defaults to False
parser.add_argument('--fine', action='store_true', help='Enable fine-tuning for the target domain')
args = parser.parse_args()

import yaml

# Load the configuration from the YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Split the source domains string into a list
source_domains = args.source_domains.split(',')
target_domains = args.target_domain

# Set the domains list
domains = source_domains + [target_domains]
target = domains[-1]
shots =args.shots

print(f"Source domains: {source_domains}")
print(f"Target domain: {target_domains}")
print(f"Shots: {shots}")
if args.fine:
        # Code for when fine-tuning is enabled
        print("Running with fine-tuning on the target domain.")
else:
        # Code for when fine-tuning is disabled
        print("Running without fine-tuning.")

'''
############### The source dataset 1 ##################
'''

image_path_dom1=[]
label_class_dom1=[]
label_dom1=[]
class_names1=[]
paths_list={}
# path_dom1='./data/office31/amazon'
# domain_name1 = path_dom1.split('/')[-1]
# dirs_dom1=os.listdir(path_dom1)
# class_names = dirs_dom1
# num_classes = len(class_names)
# class_names.sort()
# dirs_dom1.sort()
# c=0
# index=0
# for i in dirs_dom1:
#     class_names1.append(i)
#     impaths = path_dom1 + '/' + i
#     paths = glob.glob(impaths+'/**.jpg')
#     random.shuffle(paths)
#     image_path_dom1.extend(paths)
#     label_class_dom1.extend([c for _ in range(len(paths))])
#     c = c + 1
#     index = index + 1
# label_dom1=[0 for _ in range(len(image_path_dom1))] 
class_names4={}
all_classes={}
root1 = '/raid/biplab/hassan/office31'
paths_labels =[]
with open('/raid/biplab/hassan/office31/amazon_split.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        paths = line.strip().split()[0]
        if len(parts) >= 3:
            filename,label = parts[-1].split()
            class_name = parts[-2]
            if class_name not in class_names1 and (int)(label)<48:
                class_names1.append(class_name)
                class_names4[(int)(label)]=class_name         
            filename_with_class_name = f"{class_name}/{filename}"
            # dom1_filenames.append(filename_with_class_name)
            # label_class_dom1.append((int)(label))
            # image_path_dom1.append(os.path.join(root1,paths))
            # paths_labels.append((os.path.join(root1,paths),(int)(label)))
            paths_list.setdefault(class_name, [])
            paths_list[class_name].append((os.path.join(root1,paths),(int)(label)))
    # random.shuffle(paths_labels)
    # for p,l in paths_labels:
    #     label_class_dom1.append((int)(l))
    #     image_path_dom1.append(p) 
    # label_dom1.extend([0 for p in image_path_dom1])
for i in class_names1:
    lists_sample = random.sample(paths_list[i],shots)
    paths_labels.extend(lists_sample)
random.shuffle(paths_labels)
for p,l in paths_labels:
    label_class_dom1.append((int)(l))
    image_path_dom1.append(p)
label_dom1.extend([0 for p in image_path_dom1])

# print(len(class_names1))
print(class_names4.keys())




'''
############### The source dataset 2 ##################
'''
paths_list={}
image_path_dom2=[]
label_class_dom2=[]
label_dom2=[]
class_names2=[]
root2 = '/raid/biplab/hassan/visda'
paths_labels =[]
with open('/raid/biplab/hassan/visda/visda_split.txt', 'r') as file:
    for line in file:
        # print(line)
        parts = line.strip().split('/')
        # print(parts)
        paths = line.strip().split()[0]
        if len(parts) >= 2:
            filename,label = parts[-1].split()
            # print(label)
            class_name = parts[-2]
            if class_name not in class_names2 and (int)(label)<48 and (int)(label) not in class_names4.keys():
                class_names2.append(class_name)    
                class_names4[(int)(label)]=class_name       
            filename_with_class_name = f"{class_name}/{filename}"
            # dom1_filenames.append(filename_with_class_name)
            # label_class_dom1.append((int)(label))
            # image_path_dom1.append(os.path.join(root1,paths))
            # paths_labels.append((os.path.join(root2,paths),(int)(label)))
            paths_list.setdefault(class_name, [])
            paths_list[class_name].append((os.path.join(root2,paths),(int)(label)))
    # random.shuffle(paths_labels)
    # for p,l in paths_labels:
    #     label_class_dom2.append((int)(l))
    #     image_path_dom2.append(p) 
    # label_dom2.extend([1 for p in image_path_dom2])
for i in class_names2:
    lists_sample = random.sample(paths_list[i],shots)
    paths_labels.extend(lists_sample)
random.shuffle(paths_labels)
for p,l in paths_labels:
    label_class_dom2.append((int)(l))
    image_path_dom2.append(p)
label_dom2.extend([1 for p in image_path_dom2])
print(class_names4.keys())

'''
############### The source dataset 3 ##################
'''

paths_list={}
image_path_dom3=[]
label_class_dom3=[]
label_dom3=[]
class_names3=[]
root3 = '/raid/biplab/hassan/stl10/train'
paths_labels =[]
with open('/raid/biplab/hassan/stl10/stl10_split.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        paths = line.strip().split()[0]
        if len(parts) >= 2:
            filename,label = parts[-1].split()
            class_name = parts[-2]
            if class_name not in class_names3 and (int)(label)<48:
                class_names3.append(class_name)           
                class_names4[(int)(label)]=class_name
            filename_with_class_name = f"{class_name}/{filename}"
            # dom1_filenames.append(filename_with_class_name)
            # label_class_dom1.append((int)(label))
            # image_path_dom1.append(os.path.join(root1,paths))
            paths_list.setdefault(class_name, [])
            paths_list[class_name].append((os.path.join(root3,paths),(int)(label)))
            # paths_labels.append((os.path.join(root3,paths),(int)(label)))
for i in class_names3:
    lists_sample = random.sample(paths_list[i],shots)
    paths_labels.extend(lists_sample)
random.shuffle(paths_labels)
for p,l in paths_labels:
    label_class_dom3.append((int)(l))
    image_path_dom3.append(p)
label_dom3.extend([2 for p in image_path_dom3])
print(class_names4.keys())

'''
############### The combining the source dataset ##################
'''   
  
image_path_final=[]
image_path_final.extend(image_path_dom1)
image_path_final.extend(image_path_dom2)
image_path_final.extend(image_path_dom3)
label_class_final=[]
label_class_final.extend(label_class_dom1)
label_class_final.extend(label_class_dom2)
label_class_final.extend(label_class_dom3)
label_dom_final=[]
label_dom_final.extend(label_dom1)
label_dom_final.extend(label_dom2)
label_dom_final.extend(label_dom3)
domain_names=['Product image','grayscale view', 'A blurry photo']
domains_open = domain_names
print("domain_names",domain_names)

batchsize=config['batch_size']
train_prev_ds=DataTrain(image_path_final,label_dom_final,label_class_final)
print(f'length of train_prev_ds: {len(train_prev_ds)}')
train_dl=DataLoader(train_prev_ds,batch_size=batchsize, num_workers=2, shuffle=True)
img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))

'''
############### Test dataset ##################
'''


test_image_path_dom=[]
test_label_class_dom=[]
test_label_dom=[]
test_classnames=[]
# all_classes=[]
test_path_dom='./data/domainnet/'+target_domains[0]
test_domain_name = test_path_dom.split('/')[-1]
c=0
index=0
target_labels = [0, 1, 5, 6, 10, 11, 14, 17, 20, 26] + list(range(31, 37)) + list(range(39, 44)) + list(range(45, 47)) + list(range(48, 68))
known_index_dom = [0, 1, 5, 6, 10, 11, 14, 17, 20, 26] + list(range(31, 37)) + list(range(39, 44)) + list(range(45, 47))
root4 = f'/raid/biplab/hassan/domainnet'
with open(f'/raid/biplab/hassan/domainnet/splits/{target_domains}_test_new.txt', 'r') as file:
    paths_labels =[]
    for line in file:
        parts = line.strip().split('/')
        paths = line.strip().split()[0]
        if len(parts) >= 3:
            filename,label = parts[-1].split()
            class_name = parts[-2]
            if (int)(label) not in class_names4.keys() and (int)(label)<48:
                class_names4[(int)(label)]=class_name    
            if (int)(label) not in all_classes.keys() and (int)(label)>=48 :
                all_classes[(int)(label)]=class_name       
            filename_with_class_name = f"{class_name}/{filename}"
            # dom1_filenames.append(filename_with_class_name)
            # label_class_dom1.append((int)(label))
            # image_path_dom1.append(os.path.join(root1,paths))
            paths_labels.append((os.path.join(root4,paths),(int)(label)))
    random.shuffle(paths_labels)
    for p,l in paths_labels:
        test_label_class_dom.append((int)(l))
        test_image_path_dom.append(p) 
    test_label_dom.extend([3 for p in test_image_path_dom])

test_image_path_final=[]
test_image_path_final.extend(test_image_path_dom)
# test_classnames.extend(class_names4)

test_label_class_final=[]
test_label_class_final_modified = [label if label <= 47 else 48 for label in test_label_class_dom]
test_label_class_final.extend(test_label_class_final_modified)

test_label_dom_final=[]
test_label_dom_final.extend(test_label_dom)

test_domain_names = []
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name) 

# Known Classes
# known_class_names = test_classnames.sort()
known_class_names=[]
for i in range(48):
    known_class_names.append(class_names4[i])
known_classes = ",".join(known_class_names)
unknown_class_names=[]
for i in range(48,68):
    unknown_class_names.append(all_classes[i])
print(unknown_class_names)
train_prev_classnames = known_classes.split(",")

print("known_classes: ",known_classes)

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class ImageFilter(nn.Module):
    def __init__(self, brightness_threshold=0.01):
        super(ImageFilter, self).__init__()
        self.brightness_threshold = brightness_threshold

    def calculate_brightness(self, images):
        grayscale_images = torch.mean(images, dim=1, keepdim=True)  # Convert to grayscale
        return grayscale_images.mean((2, 3))  # Calculate the average pixel value for each image

    def forward(self, image_tensor):
        batch_size = image_tensor.size(0)
        brightness_values = self.calculate_brightness(image_tensor)

        fraction_to_select = 1.0
        
        num_images_to_select = int(batch_size * fraction_to_select)
        indices_with_brightness_condition = [i for i, value in enumerate(brightness_values) if value >= self.brightness_threshold]
        if len(indices_with_brightness_condition) < num_images_to_select:
           selected_indices = indices_with_brightness_condition
           num_black_images_to_select = num_images_to_select - len(indices_with_brightness_condition)
           all_indices = list(range(batch_size))
           black_indices = [i for i in all_indices if i not in indices_with_brightness_condition]
           random_black_indices = random.sample(black_indices, num_black_images_to_select)
           selected_indices += random_black_indices
           return selected_indices
        else:
           selected_indices = random.sample(indices_with_brightness_condition, num_images_to_select)
           return selected_indices

image_filter = ImageFilter(brightness_threshold=0.01)
def train_epoch(model,params, unknown_image_generator, domainnames, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for img_prev, domain_prev, label_prev, label_one_hot_prev in tqdm_object:
        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device)

        random_prompts = random.sample(prompt_list, 1)
        random_int = random.sample([0,1,2], 2)
        # random_prompts.append('unknown class')
        random.shuffle(random_prompts)
        label_prev = label_prev.to(device)
        label_one_hot_prev = label_one_hot_prev.to(device)
        batch = 1
        if args.fine:
            # unknown_posprompt1 = domains_open[0].replace("_", " ") + " of an " + random_prompts[0]
            # generated_unknown_images1 = unknown_image_generator(batch, unknown_posprompt1, known_classes)

            # unknown_posprompt2 = domains_open[1].replace("_", " ") + " of an " + random_prompts[1]
            # generated_unknown_images2 = unknown_image_generator(batch, unknown_posprompt2, known_classes)

            unknown_posprompt3 = domains_open[2].replace("_", " ") + " of an " + random_prompts[0]
            generated_unknown_images3 = unknown_image_generator(batch, unknown_posprompt3, known_classes)
        else:
            # unknown_posprompt1 = domainnames[0].replace("_", " ") + " of an unknown object"
            # generated_unknown_images1 = unknown_image_generator(batch, unknown_posprompt1, known_classes)

            # unknown_posprompt2 = domainnames[1].replace("_", " ") + " of an unknown object"
            # generated_unknown_images2 = unknown_image_generator(batch, unknown_posprompt2, known_classes)

            unknown_posprompt3 = domainnames[2].replace("_", " ") + " of an unknown object"
            generated_unknown_images3 = unknown_image_generator(batch, unknown_posprompt3, known_classes)
     
        unknown_label_rank = len(train_prev_classnames)
        unknown_label = torch.full((generated_unknown_images3.shape[0],), unknown_label_rank).to(device)
        
        # unknown_domain1 = torch.full((generated_unknown_images1.shape[0],), 0).to(device)
        # unknown_domain2 = torch.full((generated_unknown_images2.shape[0],), 1).to(device)
        unknown_domain3 = torch.full((generated_unknown_images3.shape[0],), 2).to(device)


        # generated_unknown_images = torch.cat((generated_unknown_images1, generated_unknown_images2,generated_unknown_images3), dim=0)
        # unknown_domains = torch.cat((unknown_domain1, unknown_domain2, unknown_domain3), dim=0)
        generated_unknown_images = generated_unknown_images3
        unknown_domains = unknown_domain3
        random_indices = image_filter(generated_unknown_images) 
        selected_images = generated_unknown_images[random_indices]
        selected_labels = unknown_label[random_indices]
        selected_domains = unknown_domains[random_indices]
        
        img = torch.cat((img_prev, selected_images), dim=0)
        img = img.to(device)

        label = torch.cat((label_prev, selected_labels), dim=0)
        label = label.to(device)
        # print(label)

        domain = torch.cat((domain_prev, selected_domains), dim=0)
        domain = domain.to(device)

        output,loss_sty,cos,margin_loss,invariant,feat = model(img,attri_embed,mask_embed,label,domain,len(random_indices))
        crossentropy_loss =   F.cross_entropy(output, label) +  loss_sty +(1-F.cosine_similarity(invariant,feat,dim=1)).mean()
        loss = crossentropy_loss 

        optimizer.zero_grad()
        loss.backward()
        # print(model.promptlearner.projector[0].linear.weight.grad)
        utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        # if step == "batch":
        # lr_scheduler.step()
        count = img.size(0)
        loss_meter.update(loss.item(), count)

        acc = compute_accuracy(output, label)[0].item()
        accuracy_meter.update(acc, count)
    
    return loss_meter, accuracy_meter.avg


unknown_image_generator = GenerateUnknownImages().to(device)

train_classnames = train_prev_classnames + ['unknown']
print(f'length of train_classnames : {len(train_classnames)}')

train_model = CustomCLIP(train_classnames, domains_open, clip_model,config)

for param in train_model.parameters():
            param.requires_grad_(False)
for p in train_model.cross_attention.parameters():
    p.requires_grad= True
train_model.projector.requires_grad = True
for p in train_model.promptlearner.parameters():
    p.requires_grad = True
params = [
            {"params": train_model.promptlearner.parameters(),'lr' : config["prompt_lr"]},
            {"params": train_model.projector.parameters(),'lr' : config["projector_lr"]},
            {"params": train_model.cross_attention.parameters(),'lr' : config["cross_attention_lr"]},
        ]
optimizer = torch.optim.AdamW(params,  weight_decay=config["weight_decay"])
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.8
        )
all_params=[]
for group in optimizer.param_groups:
        all_params += group['params']
scaler = GradScaler() 

test_ds=DataTrain(test_image_path_final,test_label_dom_final,test_label_class_final)
print(len(test_ds))
test_dl=DataLoader(test_ds,batch_size=32, num_workers=4, shuffle=True)
test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

num_epochs = 10
step = "epoch"
best_acc = 0
best_closed_set_acc = 0
best_open_set_acc = 0
best_avg_acc = 0
accuracy_file_path = f"./results/multidataset/{target_domains}_{shots}.txt"  
accuracy_dir = os.path.dirname(accuracy_file_path)
if not os.path.exists(accuracy_dir):
    os.makedirs(accuracy_dir)
accuracy_file = open(accuracy_file_path, "w")
torch.autograd.set_detect_anomaly(True)
test_model = CustomCLIP(train_classnames, test_domain_names, clip_model,config).to(device)
train_model = train_model.to(device)
for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}")
    train_model.train()
    train_loss, train_acc = train_epoch(train_model,all_params, unknown_image_generator, domain_names, train_dl, optimizer, lr_scheduler, step)
    print(f"epoch {epoch+1} : training accuracy: {train_acc}")
    # train_loss, train_acc = train_epoch(train_model,all_params, unknown_image_generator, domain_names, train_dl, optimizer, lr_scheduler, step)
    TRAIN_MODEL_PATH = Path(f"./train_models/multidataset/{target_domains}")
    TRAIN_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_MODEL_NAME = f"{target_domains}_{epoch+1}.pth"
    TRAIN_MODEL_SAVE_PATH = TRAIN_MODEL_PATH / TRAIN_MODEL_NAME
    print(f"Saving train_model to: {TRAIN_MODEL_SAVE_PATH}")
    torch.save(obj=train_model.state_dict(), f=TRAIN_MODEL_SAVE_PATH)

    MODEL_PATH = f"./train_models/multidataset/{target_domains}"
    MODEL_NAME = f"{target_domains}_{epoch+1}.pth"
    MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)
    
    test_model.load_state_dict(torch.load(MODEL_FILE))

    with torch.no_grad():
        test_probs_all = torch.empty(0).to(device)
        test_labels_all = torch.empty(0).to(device)
        test_class_all = torch.empty(0).to(device)
        test_tqdm_object = tqdm(test_dl, total=len(test_dl))

        total_correct_a = 0
        total_samples_a = 0
        total_correct_b = 0
        total_samples_b = 0
        
        for test_img, test_domain, test_label, test_label_one_hot in test_tqdm_object:
            test_img = test_img.to(device)
            test_domain =test_domain.to(device)
            test_label = test_label.to(device)
            test_label_one_hot = test_label_one_hot.to(device)
            
            test_output,_ = test_model(test_img.to(device),attri_embed,mask_embed,test_label)

            predictions = torch.argmax(test_output, dim=1)
            class_a_mask = (test_label <= 47) 
            class_b_mask = (test_label > 47)

            correct_predictions_a = (predictions[class_a_mask] == test_label[class_a_mask]).sum().item()
            correct_predictions_b = (predictions[class_b_mask] == test_label[class_b_mask]).sum().item()
            
            total_correct_a += correct_predictions_a
            total_samples_a += class_a_mask.sum().item()
            
            total_correct_b += correct_predictions_b
            total_samples_b += class_b_mask.sum().item()
        
        closed_set_accuracy = total_correct_a / total_samples_a if total_samples_a > 0 else 0.0
        closed_set_acc = closed_set_accuracy*100
        open_set_accuracy = total_correct_b / total_samples_b if total_samples_b > 0 else 0.0
        open_set_acc = open_set_accuracy*100

        print(f"epoch {epoch+1} : open set prediction accuracy: {open_set_acc}")
        print(f"epoch {epoch+1} : closed set prediction accuracy: {closed_set_acc}")

        average_acc = (2*closed_set_acc*open_set_acc)/(closed_set_acc + open_set_acc)
        print(f"epoch {epoch+1} : harmonic score: {average_acc}")

        accuracy_file.write(f"Epoch {epoch+1} - Open Set Accuracy: {open_set_acc}%\n")
        accuracy_file.write(f"Epoch {epoch+1} - Closed Set Accuracy: {closed_set_acc}%\n")
        accuracy_file.write(f"Epoch {epoch+1} - Harmonic Score: {average_acc}%\n")
        accuracy_file.write("\n") 
        accuracy_file.flush()
        
        if average_acc > best_avg_acc:
            best_closed_set_acc = closed_set_acc
            best_open_set_acc = open_set_acc
            best_avg_acc = average_acc
            TEST_MODEL_PATH = Path("./test_models/multidataset")
            TEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            TEST_MODEL_NAME = f"{target_domains}.pth"
            TEST_MODEL_SAVE_PATH = TEST_MODEL_PATH / TEST_MODEL_NAME
            print(f"Saving test_model with best harmonic score to: {TEST_MODEL_SAVE_PATH}")
            torch.save(obj=test_model.state_dict(), f=TEST_MODEL_SAVE_PATH) 
            
            print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
            print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
            print(f"Best harmonic score till now: {best_avg_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now : {best_open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Harmonic Score now: {best_avg_acc}%\n")
            accuracy_file.write("\n") 
            accuracy_file.flush()
        else:
            print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
            print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
            print(f"Best harmonic score till now: {best_avg_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now : {best_open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Harmonic Score now: {best_avg_acc}%\n")
            accuracy_file.write("\n") 
            accuracy_file.flush()

print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
print(f"Best harmonic score till now: {best_avg_acc}")