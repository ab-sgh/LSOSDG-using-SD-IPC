import os.path as osp
import torch
import torch.nn.utils as utils
import torch.nn as nn
import argparse
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from random import sample 
from torch.profiler import profile, ProfilerActivity
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import random
import umap
import numpy as np
import seaborn as sns
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
import glob 
import random
from trainer.cocoop import *

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import torchvision.transforms as transforms

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

def main():
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
    clip_model = clip_model.float()

    with open('prompts/prompts_list_pacs.txt', 'r') as file:
        prompt_list = file.readlines()
    random.shuffle(prompt_list)
    prompt_list = prompt_list[:12]

    attri_embed = torch.from_numpy(np.load('./attributes/attribute_pacs.npy')).to(device).to(torch.float32)
    mask_embed = torch.from_numpy(np.load('./attributes/masks_pacs.npy')).to(device).to(torch.bool)

    repeat_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    class DataTrain(Dataset):
        def __init__(self, train_image_paths, train_domain, train_labels):
            self.image_path = train_image_paths
            self.domain = train_domain
            self.labels = train_labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            image = preprocess(Image.open(self.image_path[idx]))
            domain = torch.from_numpy(np.array(self.domain[idx]))
            label = torch.from_numpy(np.array(self.labels[idx]))
            label_one_hot = F.one_hot(label, num_classes)
            return image, domain, label, label_one_hot

    # Argument parsing
    parser = argparse.ArgumentParser(description='Run PACS domain adaptation')
    parser.add_argument('--source_domains', type=str, required=True, help='Comma-separated list of source domains')
    parser.add_argument('--target_domain', type=str, required=True, help='Target domain')
    parser.add_argument('--shots', type=int, default=1, help='Number of shots')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    source_domains = args.source_domains.split(',')
    target_domain = args.target_domain
    domains = source_domains + [target_domain]
    target = domains[-1]
    shots = args.shots
    extensions = {'art_painting': '.jpg', 'cartoon': '.jpg', 'photo': '.jpg', 'sketch': '.png'}

    # Build source dataset for PACS
    image_path_dom1 = []
    label_class_dom1 = []
    label_dom1 = []
    class_names1 = []
    path_dom1 = './pacs_data/' + domains[0]
    domain_name1 = path_dom1.split('/')[-1]
    dirs_dom1 = os.listdir(path_dom1)
    class_names = dirs_dom1
    num_classes = len(class_names)
    class_names.sort()
    dirs_dom1.sort()
    c = 0
    index = 0
    index_dom1 = [3, 0, 1]
    for i in dirs_dom1:
        if index in index_dom1:
            class_names1.append(i)
            impaths = path_dom1 + '/' + i
            paths = glob.glob(impaths + f'/**{extensions[domains[0]]}')
            paths = sample(paths, min(shots, len(paths)))
            random.shuffle(paths)
            image_path_dom1.extend(paths)
            label_class_dom1.extend([c] * len(paths))
        c += 1
        index += 1
    label_dom1 = [0] * len(image_path_dom1)

    image_path_dom2 = []
    label_class_dom2 = []
    label_dom2 = []
    class_names2 = []
    path_dom2 = './pacs_data/' + domains[1]
    domain_name2 = path_dom2.split('/')[-1]
    dirs_dom2 = os.listdir(path_dom2)
    dirs_dom2.sort()
    c = 0
    index = 0
    index_dom2 = [4, 0, 2]
    for i in dirs_dom2:
        if index in index_dom2:
            class_names2.append(i)
            impaths = path_dom2 + '/' + i
            paths = glob.glob(impaths + f'*/**{extensions[domains[1]]}')
            paths = sample(paths, min(shots, len(paths)))
            random.shuffle(paths)
            image_path_dom2.extend(paths)
            label_class_dom2.extend([c] * len(paths))
        c += 1
        index += 1
    label_dom2 = [1] * len(image_path_dom2)

    image_path_dom3 = []
    label_class_dom3 = []
    label_dom3 = []
    class_names3 = []
    path_dom3 = './pacs_data/' + domains[2]
    domain_name3 = path_dom3.split('/')[-1]
    dirs_dom3 = os.listdir(path_dom3)
    dirs_dom3.sort()
    c = 0
    index = 0
    index_dom3 = [5, 1, 2]
    for i in dirs_dom3:
        if index in index_dom3:
            class_names3.append(i)
            impaths = path_dom3 + '/' + i
            paths = glob.glob(impaths + f'*/**{extensions[domains[2]]}')
            paths = sample(paths, min(shots, len(paths)))
            random.shuffle(paths)
            image_path_dom3.extend(paths)
            label_class_dom3.extend([c] * len(paths))
        c += 1
        index += 1
    label_dom3 = [2] * len(image_path_dom3)

    # Known classes
    index_dom = list(set(index_dom1 + index_dom2 + index_dom3))
    known_class_names = [class_names[idx] for idx in index_dom]
    known_classes = ",".join(known_class_names)

    image_path_final = image_path_dom1 + image_path_dom2 + image_path_dom3
    label_class_final = label_class_dom1 + label_class_dom2 + label_class_dom3
    label_dom_final = label_dom1 + label_dom2 + label_dom3
    domain_names = [domain_name1, domain_name2, domain_name3]
    domains_open = domain_names

    print("Domain names:", domain_names)
    print("Known classes:", known_classes)

    batchsize = config['batch_size']
    train_prev_ds = DataTrain(image_path_final, label_dom_final, label_class_final)
    print(f'Length of train_prev_ds: {len(train_prev_ds)}')
    train_dl = DataLoader(train_prev_ds, batch_size=batchsize, num_workers=2, shuffle=True)
    img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))
    domain_prev = domain_prev.to(device)
    class_names.sort()
    train_prev_classnames = class_names[:len(set(index_dom1 + index_dom2 + index_dom3))]
    
    # Helper functions
    class AvgMeter:
        def __init__(self, name="Metric"):
            self.name = name
            self.reset()
        def reset(self):
            self.avg, self.sum, self.count = 0, 0, 0
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
            num_images_to_select = int(batch_size * 1.0)
            indices = [i for i, value in enumerate(brightness_values) if value >= self.brightness_threshold]
            if len(indices) < num_images_to_select:
                return indices + random.sample([i for i in range(batch_size) if i not in indices],
                                               num_images_to_select - len(indices))
            else:
                return random.sample(indices, num_images_to_select)

    image_filter = ImageFilter(brightness_threshold=0.01)

    def plot_umap_and_save(closed_set_features, closed_set_labels, open_set_features, best_score, epoch, classnames):
        X = np.concatenate((closed_set_features, open_set_features), axis=0)
        y = np.concatenate((closed_set_labels, np.full(len(open_set_features), -1)), axis=0)
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)
        closed_umap = X_umap[:len(closed_set_features)]
        open_umap = X_umap[len(closed_set_features):]
        plt.figure(figsize=(10, 8))
        unique_classes, _ = np.unique(closed_set_labels, return_inverse=True)
        palette = sns.color_palette("tab20", len(unique_classes))
        for cls in unique_classes:
            mask = (closed_set_labels == cls)
            plt.scatter(closed_umap[mask, 0], closed_umap[mask, 1], label=f'{classnames[cls]}', alpha=0.6)
        plt.title(f'UMAP of Closed Set Samples')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f'./plots/umap_best_harmonic_score_epoch_{epoch}.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        print(f'Saved UMAP plot for epoch {epoch}')

    # Training epoch with SD-IPC style unknown generation
    def train_epoch(model, params, unknown_image_generator, domainnames, train_loader, optimizer, lr_scheduler, step):
        loss_meter = AvgMeter()
        accuracy_meter = AvgMeter()
        for img_prev, domain_prev, label_prev, label_one_hot_prev in tqdm(train_loader, total=len(train_loader)):
            img_prev = img_prev.to(device)
            domain_prev = domain_prev.to(device)
            label_prev = label_prev.to(device)
            label_one_hot_prev = label_one_hot_prev.to(device)
            # Use first image of batch as reference
            reference_image = img_prev[0].unsqueeze(0).float()
            neg_prompt = "low quality, blurry, text, error"
            generated_unknown_images = unknown_image_generator(reference_image, neg_prompt)
            unknown_label_index = len(train_prev_classnames)
            unknown_label = torch.full((generated_unknown_images.shape[0],), unknown_label_index).to(device)
            unknown_domain = torch.full((generated_unknown_images.shape[0],), 0).to(device)
            img = torch.cat((img_prev, generated_unknown_images), dim=0)
            label = torch.cat((label_prev, unknown_label), dim=0)
            domain = torch.cat((domain_prev, unknown_domain), dim=0)
            output, loss_sty, cos, margin_loss, invariant, feat = model(img, attri_embed, mask_embed, label, domain, generated_unknown_images.shape[0])
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

    from trainer.cocoop import SDIPCGenerateUnknownImages, CustomCLIP
    unknown_image_generator = SDIPCGenerateUnknownImages(clip_model).to(device)
    train_classnames = train_prev_classnames + ['unknown']
    print(f'Length of train_classnames: {len(train_classnames)}')

    train_model = CustomCLIP(train_classnames, domains_open, clip_model, config)
    train_model = train_model.float()
    for param in train_model.parameters():
        param.data = param.data.float()
        param.requires_grad_(False)
    for p in train_model.cross_attention.parameters():
        p.requires_grad = True
    train_model.projector.requires_grad = True
    for p in train_model.promptlearner.float().parameters():
        p.requires_grad = True

    params = [
        {"params": train_model.promptlearner.float().parameters(), 'lr': config["prompt_lr"]},
        {"params": train_model.projector.parameters(), 'lr': config["projector_lr"]},
        {"params": train_model.cross_attention.parameters(), 'lr': config["cross_attention_lr"]},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=config["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.8)
    all_params = []
    for group in optimizer.param_groups:
        all_params += group['params']
    scaler = GradScaler()

    # Test dataset
    test_image_path_dom = []
    test_label_class_dom = []
    test_label_dom = []
    test_domain_names = []
    test_path_dom = './pacs_data/' + target_domain
    test_domain_name = test_path_dom.split('/')[-1]
    test_dirs_dom = os.listdir(test_path_dom)
    test_dirs_dom.sort()
    c = 0
    index = 0
    text_index = [0, 1, 2, 3, 4, 5, 6]
    for i in test_dirs_dom:
        if index in text_index:
            impaths = test_path_dom + '/' + i
            paths = glob.glob(impaths + f'*/**{extensions[target_domain]}')
            test_image_path_dom.extend(paths)
            test_label_class_dom.extend([c] * len(paths))
        c += 1
        index += 1
    test_label_dom = [3] * len(test_image_path_dom)
    test_image_path_final = test_image_path_dom.copy()
    test_label_class_final = []
    test_label_class_final_modified = [label if label <= 5 else 6 for label in test_label_class_dom]
    test_label_class_final.extend(test_label_class_final_modified)
    test_label_dom_final = test_label_dom.copy()
    test_domain_names.extend([test_domain_name] * 3)
    test_ds = DataTrain(test_image_path_final, test_label_dom_final, test_label_class_final)
    print(f'Length of test_ds: {len(test_ds)}')
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=4, shuffle=True)
    test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

    num_epochs = 10
    step = "epoch"
    best_closed_set_acc = 0
    best_open_set_acc = 0
    best_avg_acc = 0
    accuracy_file_path = "./results/pacs/" + target_domain + "_" + str(shots) + "_ours.txt"
    accuracy_dir = os.path.dirname(accuracy_file_path)
    if not os.path.exists(accuracy_dir):
        os.makedirs(accuracy_dir)
    accuracy_file = open(accuracy_file_path, "w")
    torch.autograd.set_detect_anomaly(True)
    test_model = CustomCLIP(train_classnames, test_domain_names, clip_model, config).to(device)
    train_model = train_model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        train_model.train()
        train_loss, train_acc = train_epoch(train_model, all_params, unknown_image_generator, domains_open, train_dl, optimizer, lr_scheduler, step)
        print(f"Epoch {epoch+1} : training accuracy: {train_acc}")
        TRAIN_MODEL_PATH = Path("./train_models/pacs/" + target_domain)
        TRAIN_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        TRAIN_MODEL_NAME = f"{target_domain}ours_{epoch+1}.pth"
        TRAIN_MODEL_SAVE_PATH = TRAIN_MODEL_PATH / TRAIN_MODEL_NAME
        print(f"Saving train_model to: {TRAIN_MODEL_SAVE_PATH}")
        torch.save(obj=train_model.state_dict(), f=TRAIN_MODEL_SAVE_PATH)
        MODEL_PATH = "./train_models/pacs/" + target_domain
        MODEL_NAME = f"{target_domain}ours_{epoch+1}.pth"
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
                class_a_mask = (test_label <= 5)
                class_b_mask = (test_label > 5)
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
            print(f"Epoch {epoch+1} : open set prediction accuracy: {open_set_acc}")
            print(f"Epoch {epoch+1} : closed set prediction accuracy: {closed_set_acc}")
            average_acc = (2 * closed_set_acc * open_set_acc) / (closed_set_acc + open_set_acc)
            print(f"Epoch {epoch+1} : harmonic score: {average_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Open Set Accuracy: {open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Closed Set Accuracy: {closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Harmonic Score: {average_acc}%\n\n")
            accuracy_file.flush()
            if average_acc > best_avg_acc:
                best_closed_set_acc = closed_set_acc
                best_open_set_acc = open_set_acc
                best_avg_acc = average_acc
                TEST_MODEL_PATH = Path("./test_models/pacs")
                TEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
                TEST_MODEL_NAME = f"{target_domain}ours.pth"
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
        #print(f"Epoch {epoch+1} took {time.time() - start_time:.2f} seconds")
    print(f"Best open set prediction accuracy: {best_open_set_acc}")
    print(f"Best closed set prediction accuracy: {best_closed_set_acc}")
    print(f"Best harmonic score: {best_avg_acc}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
