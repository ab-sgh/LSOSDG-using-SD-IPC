import os.path as osp
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
from dassl.optim.lr_scheduler import ConstantWarmupScheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
import glob
import time
import random
import argparse
from trainer.cocoop import *  # This import should now include SDIPCGenerateUnknownImages

import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import torchvision.transforms as transforms

# Set device (note: if CUDA is not available, you will run on CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
#clip_model = clip_model.half()

repeat_transform = transforms.Compose([
    transforms.ToTensor(),
])

# =========================
# Dataset Definition
# =========================
class DataTrain(Dataset):
    def __init__(self, train_image_paths, train_domain, train_labels):
        self.image_path = train_image_paths
        self.domain = train_domain
        self.labels = train_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_path = self.image_path[idx]
        # Check if file exists before opening it
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        try:
            image = Image.open(file_path)
        except Exception as e:
            raise RuntimeError(f"Error opening image {file_path}: {e}")
        image = preprocess(image)
        
        domain = self.domain[idx]
        domain = torch.from_numpy(np.array(domain))
        
        label = self.labels[idx]
        # Ensure label is a LongTensor for one_hot
        label = torch.from_numpy(np.array(label)).long()
        
        label_one_hot = F.one_hot(label, num_classes)
    
        return image, domain, label, label_one_hot



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

# Read prompt list (if needed elsewhere)
with open('prompts/prompt_list_mini_new.txt', 'r') as file:
    prompt_list = file.readlines()

# =========================
# Argument Parsing & Config
# =========================
parser = argparse.ArgumentParser(description='Run domain adaptation with specified source and target domains')
parser.add_argument('--source_domains', type=str, required=True, help='Comma-separated list of source domains')
parser.add_argument('--target_domain', type=str, required=True, help='Target domain')
parser.add_argument('--shots', type=int, default=1, help='Number of shots')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
parser.add_argument('--fine', action='store_true', help='Enable fine-tuning for the target domain')
args = parser.parse_args()

source_domains = args.source_domains.split(',')
target_domains = args.target_domain
domains = source_domains + [target_domains]
target = domains[-1]
shots = args.shots

import yaml
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# =========================
# Data Loading (DomainNet splits)
# =========================
image_path_dom1 = []
label_class_dom1 = []
label_dom1 = []
path_dom1 = '../domainnet/' + domains[0]
domain_name1 = path_dom1.split('/')[-1]
dom1_classnames = []
dom1_filenames = []
with open('../domainnet/splits_mini/' + domains[0] + '_train.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        if len(parts) >= 3:
            filename = parts[-1].split()[0]
            class_name = parts[-2]
            if class_name not in dom1_classnames:
                dom1_classnames.append(class_name)
            filename_with_class_name = f"{class_name}/{filename}"
            dom1_filenames.append(filename_with_class_name)
dom1_classnames.sort()
class_names = dom1_classnames
num_classes = len(class_names)
c = 0
index = 0
index_dom1 = list(range(0, 20)) + list(range(40, 60))
for i in dom1_classnames:
    if index in index_dom1:
        paths = random.sample([filename for filename in dom1_filenames if filename.startswith(i + '/')], shots)
        for j in paths:
            selected_paths = glob.glob(path_dom1 + '/' + j)
            random.shuffle(selected_paths)
            image_path_dom1.extend(selected_paths)
            label_class_dom1.extend([c for _ in range(len(selected_paths))])
    c = c + 1
    index = index + 1
label_dom1 = [0 for _ in range(len(image_path_dom1))]

# (Repeat similar loading for domain2 and domain3 if applicable)
# For brevity, assume image_path_final, label_class_final, and label_dom_final are assembled accordingly.
image_path_final = []
image_path_final.extend(image_path_dom1)
# ... add domain2 and domain3 paths similarly ...
label_class_final = []
label_class_final.extend(label_class_dom1)
# ... add domain2 and domain3 labels similarly ...
label_dom_final = []
label_dom_final.extend(label_dom1)
# ... add domain2 and domain3 domain labels similarly ...

# Set domain names and known classes
domain_names = [domain_name1, "domain2", "domain3"]  # Update as needed
domains_open = domain_names

# =========================
# Create DataLoader
# =========================
batchsize = config['batch_size']
train_prev_ds = DataTrain(image_path_final, label_dom_final, label_class_final)
train_dl = DataLoader(train_prev_ds, batch_size=batchsize, num_workers=2, shuffle=True)
img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))
domain_prev = domain_prev.to(device)
class_names.sort()
train_prev_classnames = class_names[:90]

# =========================
# Helper Functions
# =========================
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
        return f"{self.name}: {self.avg:.4f}"

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class ImageFilter(nn.Module):
    def __init__(self, brightness_threshold=0.01):
        super(ImageFilter, self).__init__()
        self.brightness_threshold = brightness_threshold
    def calculate_brightness(self, images):
        grayscale_images = torch.mean(images, dim=1, keepdim=True)
        return grayscale_images.mean((2, 3))
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
            return random.sample(indices_with_brightness_condition, num_images_to_select)

image_filter = ImageFilter(brightness_threshold=0.01)
attri_embed = torch.from_numpy(np.load('./attributes/attribute_mini_4.npy')).to(device).to(torch.float32)
mask_embed = torch.from_numpy(np.load('./attributes/masks_mini_4.npy')).to(device).to(torch.bool)

def train_epoch(model, unknown_image_generator, domainnames, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    for img_prev, domain_prev, label_prev, label_one_hot_prev in train_loader:
        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device)
        label_prev = label_prev.to(device)
        label_one_hot_prev = label_one_hot_prev.to(device)
        # Use first image in the batch as reference for unknown generation
        reference_image = img_prev[0].unsqueeze(0)
        neg_prompt = "low quality, blurry, text, error"
        generated_unknown_images = unknown_image_generator(reference_image, neg_prompt)
        unknown_label_rank = len(train_prev_classnames)
        unknown_label = torch.full((generated_unknown_images.shape[0],), unknown_label_rank).to(device)
        unknown_domain = torch.full((generated_unknown_images.shape[0],), 0).to(device)
        img = torch.cat((img_prev, generated_unknown_images), dim=0)
        label = torch.cat((label_prev, unknown_label), dim=0)
        domain = torch.cat((domain_prev, unknown_domain), dim=0)
        output, loss_sty, score, margin_loss, invariant, feat = model(img, attri_embed, mask_embed, label, domain, generated_unknown_images.shape[0])
        crossentropy_loss = F.cross_entropy(output, label) + loss_sty + (1 - F.cosine_similarity(invariant, feat, dim=1)).mean()
        loss = crossentropy_loss
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        acc = compute_accuracy(output, label)[0].item()
        accuracy_meter.update(acc, count)
    return loss_meter, accuracy_meter.avg

# =========================
# Instantiate Unknown Generator & Model
# =========================
# Use the new SD-IPC style generator (make sure SDIPCGenerateUnknownImages is defined in your cocoop.py)
unknown_image_generator = SDIPCGenerateUnknownImages(clip_model).to(device)
train_classnames = train_prev_classnames + ['unknown']
print(f'length of train_classnames : {len(train_classnames)}')

train_model = CustomCLIP(train_classnames, domains_open, clip_model, config)
for param in train_model.parameters():
    param.requires_grad_(False)
for p in train_model.cross_attention.parameters():
    p.requires_grad = True
train_model.projector.requires_grad = True
for p in train_model.promptlearner.parameters():
    p.requires_grad = True

num_epochs = 10
params = [
    {"params": train_model.promptlearner.parameters(), 'lr': config["prompt_lr"]},
    {"params": train_model.projector.parameters(), 'lr': config["projector_lr"]},
    {"params": train_model.cross_attention.parameters(), 'lr': config["cross_attention_lr"]},
]
optimizer = torch.optim.AdamW(params, weight_decay=config["weight_decay"])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(num_epochs))
all_params = []
for group in optimizer.param_groups:
    all_params += group['params']
scaler = GradScaler()

# =========================
# Set up Test Dataset (using DomainNet splits)
# =========================
test_image_path_dom = []
test_label_class_dom = []
test_label_dom = []
test_domain_names = []
test_path_dom = '../domainnet/' + domains[-1]
test_domain_name = test_path_dom.split('/')[-1]
testdom_classnames = []
test_filenames = []
with open(f'../domainnet/splits_mini/{domains[-1]}_test.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        if len(parts) >= 3:
            filename = parts[-1].split()[0]
            class_name = parts[-2]
            if class_name not in testdom_classnames:
                testdom_classnames.append(class_name)
            filename_with_class_name = f"{class_name}/{filename}"
            test_filenames.append(filename_with_class_name)
testdom_classnames.sort()
test_classnames = []
print(len(test_filenames))
c = 0
index = 0
test_index = list(range(0,5)) + list(range(8,18)) + list(range(25,35)) + list(range(43,48)) + list(range(75,80)) + list(range(83,88)) + list(range(90,126))
for i in testdom_classnames:
    if index in test_index:
        paths = [filename for filename in test_filenames if filename.startswith(i + '/')]
        for j in paths:
            selected_paths = glob.glob(test_path_dom + '/' + j)
            random.shuffle(selected_paths)
            test_image_path_dom.extend(selected_paths)
            test_label_class_dom.extend([c for _ in range(len(selected_paths))])
    c = c + 1
    index = index + 1
test_label_dom = [3 for _ in range(len(test_image_path_dom))]
test_image_path_final = []
test_image_path_final.extend(test_image_path_dom)
test_label_class_final = []
test_label_class_final_modified = [label if label <= 89 else 90 for label in test_label_class_dom]
test_label_class_final.extend(test_label_class_final_modified)
test_label_dom_final = []
test_label_dom_final.extend(test_label_dom)
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)
test_ds = DataTrain(test_image_path_final, test_label_dom_final, test_label_class_final)
test_dl = DataLoader(test_ds, batch_size=32, num_workers=4, shuffle=True)
test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

# =========================
# Training Loop (Main function)
# =========================
def main():
    best_closed_set_acc = 0
    best_open_set_acc = 0
    best_avg_acc = 0
    accuracy_file_path = f"./results/minidomainnet/{domains[-1]}_{shots}.txt"
    accuracy_dir = os.path.dirname(accuracy_file_path)
    if not os.path.exists(accuracy_dir):
        os.makedirs(accuracy_dir)
    accuracy_file = open(accuracy_file_path, "w")
    torch.autograd.set_detect_anomaly(True)
    test_model = CustomCLIP(train_classnames, test_domain_names, clip_model, config).to(device)
    train_model_local = train_model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        start_time = time.time()
        train_model_local.train()
        train_loss, train_acc = train_epoch(train_model_local, unknown_image_generator, domain_names, train_dl, optimizer, lr_scheduler, "epoch")
        print(f"epoch {epoch+1} : training accuracy: {train_acc}")
        TRAIN_MODEL_PATH = Path(f"./train_models/minidomainnet/{domains[-1]}")
        TRAIN_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        TRAIN_MODEL_NAME = f"{domains[-1]}_{epoch+1}.pth"
        TRAIN_MODEL_SAVE_PATH = TRAIN_MODEL_PATH / TRAIN_MODEL_NAME
        print(f"Saving train_model to: {TRAIN_MODEL_SAVE_PATH}")
        torch.save(obj=train_model_local.state_dict(), f=TRAIN_MODEL_SAVE_PATH)
        MODEL_PATH = f"./train_models/minidomainnet/{domains[-1]}"
        MODEL_NAME = f"{domains[-1]}_{epoch+1}.pth"
        MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)
        test_model.load_state_dict(torch.load(MODEL_FILE))
        with torch.no_grad():
            total_correct_a = 0
            total_samples_a = 0
            total_correct_b = 0
            total_samples_b = 0
            for test_img, test_domain, test_label, test_label_one_hot in tqdm(test_dl, total=len(test_dl)):
                test_img = test_img.to(device)
                test_domain = test_domain.to(device)
                test_label = test_label.to(device)
                test_label_one_hot = test_label_one_hot.to(device)
                test_output, _ = test_model(test_img, attri_embed, mask_embed, test_label)
                predictions = torch.argmax(test_output, dim=1)
                class_a_mask = (test_label <= 89)
                class_b_mask = (test_label > 89)
                correct_predictions_a = (predictions[class_a_mask] == test_label[class_a_mask]).sum().item()
                correct_predictions_b = (predictions[class_b_mask] == test_label[class_b_mask]).sum().item()
                total_correct_a += correct_predictions_a
                total_samples_a += class_a_mask.sum().item()
                total_correct_b += correct_predictions_b
                total_samples_b += class_b_mask.sum().item()
            closed_set_accuracy = total_correct_a / total_samples_a if total_samples_a > 0 else 0.0
            closed_set_acc = closed_set_accuracy * 100
            open_set_accuracy = total_correct_b / total_samples_b if total_samples_b > 0 else 0.0
            open_set_acc = open_set_accuracy * 100
            print(f"epoch {epoch+1} : open set prediction accuracy: {open_set_acc}")
            print(f"epoch {epoch+1} : closed set prediction accuracy: {closed_set_acc}")
            average_acc = (2 * closed_set_acc * open_set_acc) / (closed_set_acc + open_set_acc)
            print(f"epoch {epoch+1} : harmonic score: {average_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Open Set Accuracy: {open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Closed Set Accuracy: {closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Harmonic Score: {average_acc}%\n\n")
            accuracy_file.flush()
            if average_acc > best_avg_acc:
                best_closed_set_acc = closed_set_acc
                best_open_set_acc = open_set_acc
                best_avg_acc = average_acc
                TEST_MODEL_PATH = Path("./test_models/minidomainnet")
                TEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
                TEST_MODEL_NAME = f"{domains[-1]}.pth"
                TEST_MODEL_SAVE_PATH = TEST_MODEL_PATH / TEST_MODEL_NAME
                print(f"Saving test_model with best harmonic score to: {TEST_MODEL_SAVE_PATH}")
                torch.save(obj=test_model.state_dict(), f=TEST_MODEL_SAVE_PATH)
                print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
                print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
                print(f"Best harmonic score till now: {best_avg_acc}")
                accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now: {best_open_set_acc}%\n")
                accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
                accuracy_file.write(f"Epoch {epoch+1} - Best Harmonic Score now: {best_avg_acc}%\n\n")
                accuracy_file.flush()
            else:
                print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
                print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
                print(f"Best harmonic score till now: {best_avg_acc}")
                accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now: {best_open_set_acc}%\n")
                accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
                accuracy_file.write(f"Epoch {epoch+1} - Best Harmonic Score now: {best_avg_acc}%\n\n")
                accuracy_file.flush()
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1} took {epoch_duration:.2f} seconds")
    print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
    print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
    print(f"Best harmonic score till now: {best_avg_acc}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
