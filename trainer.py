import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import pickle
from dataset import BatchData
from model import PreResNet, BiasLayer
from cifar import Cifar100
from readmat import GXWData
from exemplar import Exemplar
from copy import deepcopy
from sklearn.model_selection import train_test_split
from readmatrobusteval import ShiftDataLoader
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from torch.utils.data import TensorDataset
from collections import Counter


class Trainer:
    def __init__(self, init_cls):  # total_cls = 100 # modify to flexible value     ## changed total_cls to init_cls
        self.total_cls = init_cls                                                   ## changed total_cls to init_cls
        self.seen_cls = 0
        self.num_new_cls = []
        self.dataset = GXWData()  
        self.openset = GXWData()
        shift_loader = ShiftDataLoader()
        shift_groups = shift_loader.get_shift_data()

        if len(shift_groups) > 0:
            shift_x, shift_y = zip(*shift_groups)
        else:
            shift_x, shift_y = [], []


        # self.dataset = Cifar100()  # 这里可以选择不同的数据集
        self.model = PreResNet(47,init_cls).cuda()        # formerly 32 for basicblock                          
        # print(self.model)
        #self.model = nn.DataParallel(self.model, device_ids=[0,1])
        self.model = self.model.cuda()  # 将模型移动到 GPU 0
        self.lossdecay = []
        self.valacc    = []
        # 动态增加 bias layer 以适应增量训练的未知新任务
        # self.bias_layer1 = BiasLayer().cuda()
        # self.bias_layer2 = BiasLayer().cuda()
        # self.bias_layer3 = BiasLayer().cuda()
        # self.bias_layer4 = BiasLayer().cuda()
        # self.bias_layer5 = BiasLayer().cuda()
        # self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5]

        self.bias_layers = []
        # self.bias_layer = BiasLayer().cuda()  # 仅建立一个 BiasLayer，作用于所有旧类别
        # self.bias_layers = [self.bias_layer]  # 仅建立一个 BiasLayer，作用于所有旧类别

        # self.input_transform= Compose([
        #                         transforms.RandomHorizontalFlip(),
        #                         transforms.RandomCrop(32,padding=4),
        #                         ToTensor(),
        #                         Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        # self.input_transform_eval= Compose([
        #                         ToTensor(),
        #                         Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        self.input_transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize([0.5],[0.2]),  
        ])

        self.input_transform_eval = Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.2]),
        ])
        self.shift_data = DataLoader(BatchData(shift_x, shift_y, self.input_transform_eval),
                    batch_size=10, shuffle=False)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)


    def expand_model(self, new_cls):
        old_state_dict = self.model.state_dict()
        # define new model with expanded fc layer
        self.model = PreResNet(47, self.total_cls + new_cls).cuda()     
        new_state_dict = self.model.state_dict()

        for name, param in old_state_dict.items():
            if "fc" not in name:  # remain old params in other layers
                new_state_dict[name] = param

        self.model.load_state_dict(new_state_dict)  # 
        self.seen_cls += new_cls  # update seen_cls  

    def expand_model(self, new_cls):
        old_state_dict = self.model.state_dict()
        # define new model with expanded fc layer
        self.model = PreResNet(47, self.total_cls + new_cls).cuda()
        new_state_dict = self.model.state_dict()

        # Copy all old feature extractor weights
        for name, param in old_state_dict.items():
            if "fc" not in name: # remain all old params in other layers
                new_state_dict[name] = param
        
        # Copy fc weights of old classes
        old_fc_weight = old_state_dict['fc.weight']  # shape: [old_cls, dim]
        old_fc_bias = old_state_dict['fc.bias']      # shape: [old_cls]
        print('old class number:',self.seen_cls)
        new_state_dict['fc.weight'][:old_fc_weight.shape[0]] = old_fc_weight
        new_state_dict['fc.bias'][:old_fc_weight.shape[0]] = old_fc_bias

        # load weights
        self.model.load_state_dict(new_state_dict)

        # # Freeze all
        # self.model.fc.weight.requires_grad = False
        # self.model.fc.bias.requires_grad = False

        # # Unfreeze new classes
        # self.model.fc.weight[self.seen_cls:].requires_grad = True
        # self.model.fc.bias[self.seen_cls:].requires_grad = True
        # # cannot ""... = False" part of params in a layer

        # update seen_cls
        self.seen_cls += new_cls

    
    def test(self, testdata, inc_i, heatmap_name=None):  # used in phased test, producing heatmap
        print("test set size : ",len(testdata.dataset))
        per_class_correct = [0 for _ in range(self.seen_cls)]
        per_class_total = [0 for _ in range(self.seen_cls)]
        self.model.eval()  
        with torch.no_grad():
            correct = 0
            wrong = 0
            pred_labels = torch.zeros((self.seen_cls, self.seen_cls), dtype=torch.int32)
            for i, (image, label) in enumerate(testdata):
                image = image.cuda()
                label = label.view(-1).cuda()
                p = self.model(image)
                if inc_i > 0:   # if there is bias correction
                    p = self.bias_forward(p)
                pred = p[:,:self.seen_cls].argmax(dim=-1)
                correct += sum(pred == label).item()
                wrong += sum(pred != label).item()
                for p_i, l_i in zip(pred, label):
                    pred_labels[l_i.item(), p_i.item()] += 1
                    if p_i.item() == l_i.item():
                        correct += 1
                        per_class_correct[l_i.item()] += 1
                    else:
                        wrong += 1
                    per_class_total[l_i.item()] += 1
        acc = correct / (wrong + correct)
        print("Total Test Acc: {}".format(acc*100, '.2f'))
        if heatmap_name is not None:
            self.heat_map(pred_labels.cpu().numpy(),heatmap_name)
            # 各类准确率
            print("Per-class accuracy:")
            pc_accs = []
            for i in range(self.seen_cls):
                pc_total = per_class_total[i]
                pc_correct = per_class_correct[i]
                pc_acc= pc_correct / pc_total * 100 if pc_total > 0 else 0
                pc_accs.append(pc_acc)
                if pc_total > 0:
                    class_acc = pc_correct / pc_total * 100
                    print(f"  Class {i:02d} — {pc_correct}/{pc_total} = {class_acc:.2f}%")
                else:
                    print(f"  Class {i:02d} — No samples.")
            return acc,pc_accs
        self.model.train()
        print("---------------------------------------------")
        return acc

    def validation(self, evaldata, inc_i):
        print("Validation data number : ",len(evaldata.dataset))
        self.model.eval()
        with torch.no_grad():
            correct = 0
            wrong = 0
            for i, (image, label) in enumerate(evaldata):
                image = image.cuda()
                label = label.view(-1).cuda()
                p = self.model(image)
                if inc_i > 0:   # if there is bias correction
                    p = self.bias_forward(p)
                pred = p[:,:self.seen_cls].argmax(dim=-1)
                correct += sum(pred == label).item()
                wrong += sum(pred != label).item()
                pred = p[:,:self.seen_cls].argmax(dim=-1)
                correct += sum(pred == label).item()
                wrong += sum(pred != label).item()
                acc = correct/(correct+wrong)
        print("Validation Acc: {}".format(format(acc * 100, '.2f')))
        self.valacc.append(100*acc)
        self.model.train()
        return acc

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, bias_lr, max_size, T = 4, evt_threshold = 0.75, beta = 0.5, resume_task = 0):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)
        
        previous_model = None

        dataset = self.dataset
        test_xs, test_ys, train_xs, train_ys = [], [], [], []
        biastrain_xs, biastrain_ys = [], []
        opensettest_xs, opensettest_ys = [], []

        val_acc_noBiC = []
        val_acc = []
        test_acc_in_training_process = []
        test_accs = []
        test_accs_noBiC = []
        per_class_accuracies = []
        per_class_accuracies_noBiC = []
        distill_loss_all_tasks = []  
        ce_loss_all_tasks = []
        feature_loss_all_tasks = []
        stage2_losses_all_tasks = []

        opensettest_data = dataset.extract_small_balanced_set(split='test', per_class=100)
        opensettest_x, opensettest_y = zip(*opensettest_data)
        opensettest_xs.extend(opensettest_x)
        opensettest_ys.extend(opensettest_y)

        for inc_i in range(dataset.batch_num):
            if resume_task > 0:
                if inc_i < resume_task:
                    continue
                elif inc_i == resume_task:
                    print(f"Resuming from task {resume_task}")
                    with open(f'num_new_class{resume_task-1}.pkl', 'rb') as f:
                        self.num_new_cls = pickle.load(f)
                    with open(f'total_class{resume_task-1}.pkl', 'rb') as f:
                        self.total_cls = pickle.load(f)
                    # fc expand
                    if self.total_cls > 0:
                        self.expand_model(self.total_cls)
                    # recover model params
                    self.model.load_state_dict(torch.load(f'model{resume_task-1}.pth'))
                    self.previous_model = deepcopy(self.model)
                    # recover bias layers
                    for i in range(resume_task):
                        bias_layer = BiasLayer(self.num_new_cls[i]).cuda()
                        bias_layer.load_state_dict(torch.load(f'bias_layer{i}.pth'))
                        self.bias_layers.append(bias_layer)
                    # restore exemplar
                    with open(f'exemplar{resume_task-1}.pkl', 'rb') as f:
                        exemplar = pickle.load(f)
                    # testset
                    with open(f'test_data{resume_task - 1}.pkl', 'rb') as f:
                        test_data_dict = pickle.load(f)
                    test_xs = test_data_dict['test_xs']
                    test_ys = test_data_dict['test_ys']

                    print(f"Resume complete! total_cls = {self.total_cls}, num_new_cls = {self.num_new_cls}")


            print(f"Incremental num: {inc_i}")

            train, val, test = dataset.getNextClasses(inc_i)

            #当前 batch 的类别数
            new_cls = set(y for _, y in train + val + test) - set(range(self.total_cls))
            
            num_new_cls = len(new_cls)
            self.num_new_cls.append(num_new_cls)
            print(f"New classes detected: {num_new_cls}")
            print(f"New classes for former tasks: {self.num_new_cls}")
            #总类别数 update
            self.total_cls += num_new_cls

            if num_new_cls > 0:
                
                self.expand_model(num_new_cls)
            
            print(f"total class till task {inc_i} :{self.total_cls}")
            self.seen_cls = self.total_cls
            
            bias_layer = BiasLayer(num_new_cls).cuda()

            self.bias_layers.append(bias_layer) 
            print('current bias layers number : ', len(self.bias_layers))

            print('Training Samples:',len(train),'validation samples:', len(val), 'test samples:', len(test))
            
            train_class_count = self.count_class_samples(train)
            val_class_count = self.count_class_samples(val)
            test_class_count = self.count_class_samples(test)

            # valid sample number for each class
            print("Training samples count:")
            for cls, count in train_class_count.items():
                print(f"Class {cls}: {count} samples")

            print("Validation samples count:")
            for cls, count in val_class_count.items():
                print(f"Class {cls}: {count} samples")

            print("Test samples count:")
            for cls, count in test_class_count.items():
                print(f"Class {cls}: {count} samples")

            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)
            # exemplar get
            train_xs, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(train_x)
            train_xs.extend(val_x)
            train_ys.extend(train_y)
            train_ys.extend(val_y)
            cls_trainsample = Counter(train_ys)
            for cls, count in cls_trainsample.items():
                print(f"Train sample Class with exemplars\n Class {cls}: {count} samples")
            biastrain_data = dataset.extract_small_balanced_set(split='train', per_class=30)
            biastrain_x, biastrain_y = zip(*biastrain_data)
            biastrain_xs.extend(biastrain_x)
            biastrain_ys.extend(biastrain_y) 


            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            eval_data = DataLoader(BatchData(val_x, val_y, self.input_transform_eval),
                        batch_size=batch_size*10, shuffle=False)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size*10, shuffle=False)
            if inc_i == 0:
                optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
                # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
                # scheduler = CosineAnnealingLR(optimizer, T_max=40)
                scheduler = StepLR(optimizer, step_size=70, gamma=0.1)

            if inc_i > 0:              
                # Freeze layer1 + layer2
                # for param in self.model.layer1.parameters():
                #     param.requires_grad = False
                # for param in self.model.layer2.parameters():
                #     param.requires_grad = False
                # # Open layer3 + layer4
                # for param in self.model.layer3.parameters():
                #     param.requires_grad = True

                # bias_optimizer = optim.SGD(self.bias_layers[inc_i].parameters(), lr=lr, momentum=0.9)
                # bias_optimizer = optim.Adam(self.bias_layers[-1].parameters(), lr=0.001) #version 1: only train the last bias layer #inc-1 -> -1
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),lr=lr,momentum=0.9,weight_decay=2e-4)

                # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
                # scheduler = CosineAnnealingLR(optimizer, T_max=40)
                scheduler = StepLR(optimizer, step_size=70, gamma=0.1)
                bias_optimizer = optim.Adam([param for layer in self.bias_layers for param in layer.parameters()], lr=bias_lr)  #version2: train all the bias layers              
                # bias_scheduler = StepLR(bias_optimizer, step_size=70, gamma=0.1)
            # exemplar.update(total_cls//dataset.batch_num, (train_x, train_y), (val_x, val_y))
            # modify to flexible class number, especially for new tasks with old classes
            # adapt to tasks with repepitition of old classes
            # val_xs, val_ys = exemplar.get_exemplar_val()
            bias_data = DataLoader(BatchData(biastrain_xs, biastrain_ys, self.input_transform),
                        batch_size=batch_size*2, shuffle=True, drop_last=False)    

            val_acc.append([])
            val_acc_noBiC.append([])
            test_acc_in_training_process.append([])     
            if inc_i > 0:
                distill_loss_all_tasks.append([])
                feature_loss_all_tasks.append([])
                ce_loss_all_tasks.append([])
                stage2_losses_all_tasks.append([])
            
            print("Model FC out_features:", self.model.fc.out_features)

            for epoch in range(int(epoches)):
                print("---"*20)
                print("current incremental task : ", inc_i)
                print("Epoch", epoch)
                scheduler.step()
                cur_lr = self.get_lr(optimizer)
                print("Current Learning Rate : ", cur_lr)
                self.model.train()
                for _ in range(len(self.bias_layers)):
                    with torch.no_grad():
                        self.bias_layers[_].eval()
                    #self.bias_layers[_].eval()
                if inc_i > 0:
                    distill_losses, ce_losses, feature_loss = self.stage1_distill(train_data, criterion, optimizer, num_new_cls, T = T, beta = beta)
                    distill_loss_all_tasks[-1].append(distill_losses)   
                    ce_loss_all_tasks[-1].append(ce_losses)             
                    feature_loss_all_tasks[-1].append(feature_loss)
                else:
                    self.stage1_initial(train_data, criterion, optimizer)
                acc = self.validation(eval_data, inc_i)
                acc_test = self.test(test_data, inc_i)
                test_acc_in_training_process[-1].append(acc_test)
                val_acc_noBiC[-1].append(acc)

            features, labels, raw_images = self.extract_features(self.model, train_data, device="cuda")

            centroids = self.compute_class_centroids(features, labels,num_classes=self.seen_cls)
        
            exemplar.update(
                train=(train_x, train_y), val=(val_x, val_y), 
                features=features, labels=labels, centroids=centroids, model_images=raw_images)
            

            heatmap_name = f"heatmap_task_{inc_i}_before_BiC.png"
            test_acc_noBic, per_class_accuracy_noBiC = self.test(test_data, inc_i, heatmap_name=heatmap_name)
            test_accs_noBiC.append(test_acc_noBic)
            per_class_accuracies_noBiC.append(per_class_accuracy_noBiC)
            if inc_i > 0:
                for epoch in range(int(epoches)):
                    # bias_scheduler.step()
                    with torch.no_grad():
                        self.model.eval()
                    for _ in range(len(self.bias_layers)):
                        self.bias_layers[_].train()
                    loss_stage2 = self.stage2(bias_data, criterion, bias_optimizer)
                    stage2_losses_all_tasks[-1].append(loss_stage2)
                    print("train data number : ",len(biastrain_data))
                    acc = self.test(test_data, inc_i)
                    val_acc[-1].append(acc)
            for i, layer in enumerate(self.bias_layers):
                layer.printParam(i)
            self.previous_model = deepcopy(self.model)
            heatmap_name = f"heatmap_task_{inc_i}_after_BiC.png"
            test_acc, per_class_accuracy = self.test(test_data, inc_i, heatmap_name=heatmap_name)
            if self.seen_cls >= 6:
                shifttestacc, perclsshifttestacc = self.test(self.shift_data, inc_i, heatmap_name=f'shift_test{inc_i}.png')
            test_accs.append(test_acc)
            per_class_accuracies.append(per_class_accuracy)
            print("Test results on testset of model without BiC",test_accs_noBiC)
            print("Test results on testset after BiC training",test_accs)
            print("Per-class accuracies on testset without BiC",per_class_accuracies_noBiC)
            print("Per-class accuracies on testset after BiC training",per_class_accuracies)
            torch.save(self.model.state_dict(), f'model{inc_i}.pth')  
            torch.save(self.bias_layers[-1].state_dict(), f'bias_layer{inc_i}.pth')
            with open(f'total_class{inc_i}.pkl', 'wb') as f:
                pickle.dump(self.seen_cls, f)
            with open(f'num_new_class{inc_i}.pkl', 'wb') as f:
                pickle.dump(self.num_new_cls, f)
            with open(f'exemplar{inc_i}.pkl', 'wb') as f:
                pickle.dump(exemplar, f)
            with open(f'test_data{inc_i}.pkl', 'wb') as f:
                pickle.dump({'test_xs': test_xs, 'test_ys': test_ys}, f)
            #save model & bias layer
        print("number of new classes seen in each task : ", self.num_new_cls)
        print(f'initial learning rate: {lr}')
        print(f'Bias layer learning rate: {bias_lr}')
        print(f'epoches: {epoches}')
        print(f'Distillation temperature: {T}')
        print(f'maximum exemplar size: {max_size}')
        self.trainer_visual(val_acc_noBiC, val_acc, test_accs, test_accs_noBiC, test_acc_in_training_process,
                            distill_loss_all_tasks=distill_loss_all_tasks, ce_loss_all_tasks=ce_loss_all_tasks, feature_loss_all_tasks=feature_loss_all_tasks,
                            stage2_losses_all_tasks=stage2_losses_all_tasks
                            )

        
    def trainer_visual(self, val_acc_noBiC, val_acc, test_accs, test_accs_noBiC, test_acc_in_training_process,
                       distill_loss_all_tasks=None, ce_loss_all_tasks=None, feature_loss_all_tasks=None,
                       stage2_losses_all_tasks=None
                       ):
        # stage1 validation accuracy visualization
        save_dir = "output"
        os.makedirs(save_dir, exist_ok=True)  

        plt.figure(figsize=(10, 6))  
        for i in range(len(val_acc_noBiC)):
            epochs = range(len(val_acc_noBiC[i]))
            if i == 0:
                plt.plot(epochs, val_acc_noBiC[i], label=f"Task {i} validation accuracy", color='orange', linestyle=':')
                plt.plot(epochs, test_acc_in_training_process[i], label=f"Task {i} test accuracy", color='orange', linestyle='-') 
            if i > 0:
                plt.plot(epochs, val_acc_noBiC[i], label=f"Task {i} validation accuracy", color='blue', linestyle=':')
                plt.plot(epochs, test_acc_in_training_process[i], label=f"Task {i} test accuracy", color='blue', linestyle='-') 
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy/%")
        plt.title("Accuracy Changes Over Epochs for Each Incremental Task in stage 1")
        plt.legend()  
        plt.grid(True)  
        save_path = os.path.join(save_dir, "Stage1_accuracy_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        print(f"Plot saved at: {save_path}")

        # loss visualization 
        if distill_loss_all_tasks is not None and ce_loss_all_tasks is not None and feature_loss_all_tasks is not None:
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            losses = [distill_loss_all_tasks, ce_loss_all_tasks, feature_loss_all_tasks]
            titles = ["Distill loss", "CE loss", "Feature loss"]
            ylabels = ["Distill loss", "CE loss", "Feature loss"]

            for ax, loss_task_list, title, ylabel in zip(axs, losses, titles, ylabels):
                for i in range(len(loss_task_list)):
                    task_losses = loss_task_list[i]   # shape = [epoch losses in task i]
                    epochs = range(len(task_losses))
                    ax.plot(epochs, task_losses, marker='o', label=f'Task {i+1}')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend()
                ax.grid(True)

            axs[-1].set_xlabel('Epochs')
            fig.suptitle('Loss Changes Over Epochs for Each Incremental Task in stage 1', fontsize=14)
            loss_save_path = os.path.join(save_dir, "Stage1_loss_plot.png")
            plt.savefig(loss_save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved at: {loss_save_path}")


        # stage2 test accuracy visualization
        plt.figure(figsize=(10, 6))  
        for i in range(len(val_acc)):
            epochs = range(len(val_acc[i])) 
            plt.plot(epochs , val_acc[i], label=f"Task {i} validation accuracy", color='orange')  
            
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy/%")
        plt.title("Accuracy Changes Over Epochs for Each Incremental Task in stage 2")
        plt.legend()  
        plt.grid(True)  
        save_path = os.path.join(save_dir, "Stage2_accuracy_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")

        # stage2 loss visualization
        if stage2_losses_all_tasks is not None:
            for i in range(len(stage2_losses_all_tasks)):
                plt.figure(figsize=(10, 6))
                task_losses = stage2_losses_all_tasks[i] 
                plt.plot(task_losses, marker='o', label=f'Task {i+1}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Stage 2 Loss Changes for Task {i+1}')
            plt.legend()
            plt.grid(True)
            loss_save_path = os.path.join(save_dir, f"Stage2_loss_plot_task_{i+1}.png")
            plt.savefig(loss_save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved at: {loss_save_path}")  



        plt.figure(figsize=(10, 6))
        # task accuracy decay visualization
        for i in range(len(test_accs_noBiC)):
            tasks = range(len(test_accs_noBiC))
            plt.plot(tasks, test_accs_noBiC, label=f"Task {i}", color='orange')
            plt.plot(test_accs, label=f"Task {i} with BiC", color='blue')
        plt.xlabel("Task number")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title('Task Accuracy Decay')
        plt.grid(True)
        save_path = os.path.join(save_dir, "Task_Accuracy_Decay.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")

    def heat_map(self, data, name="heatmap.png"):
        data = data/data.max(axis=0) # normalization
        plt.clf()
        plt.imshow(data, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Heat Map")
        save_path = os.path.join("output", name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")

    def heat_map(self, data, name="heatmap.png"):
        plt.clf()
        plt.imshow(data, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
        plt.title("Heat Map")
        # 类别标签
        n = data.shape[0]  
        class_labels = [f'class_{i}' for i in range(n)] 

        img = plt.imshow(data, cmap='Blues', interpolation='nearest')
        plt.colorbar(img)
        plt.xticks(ticks=range(n), labels=class_labels, rotation=45, ha='right')
        plt.yticks(ticks=range(n), labels=class_labels, rotation=0)
        plt.xlabel("Predicted Label", fontsize=12)  
        plt.ylabel("True Label", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14)
        threshold = (data.max() + data.min()) / 2
        for i in range(n):
            for j in range(n):
                colour = 'white' if data[i,j] > threshold else 'black'
                plt.text(j, i, int(data[i, j]), ha='center', va='center', color=colour, fontsize=16-n)

        save_path = os.path.join("output", name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")

    def count_class_samples(self, data):
        class_count = {}
        for _, label in data:
            if label not in class_count:
                class_count[label] = 0
            class_count[label] += 1
        return class_count

    def bias_forward(self, input):
        num_new_cls = self.num_new_cls  
        bias_layers = self.bias_layers  # 每个任务对应的 bias 层
        out_list = []  
        start = 0

        for i in range(len(bias_layers)):  
            end = start + num_new_cls[i]  # 计算当前任务的起始和结束位置
            out_list.append(bias_layers[i](input[:, start:end]))  # 逐个 task 进行 bias 修正
            # out_list.append(bias_layers[i](input[:,:end]))  # 逐个 task 进行 bias 修正,但是每个任务的输出都是从头开始的
            start = end

        return torch.cat(out_list, dim=1)  # 拼接所有任务的输出

    # def bias_forward(self, input):
    #     num_new_cls = self.num_new_cls  
    #     bias_layers = self.bias_layers  # 存储 Bias 层（不包含第一个任务）
    #     out_list = []  
    #     start = 0

    #     for i in range(len(num_new_cls)):  
    #         end = start + num_new_cls[i]  
    #         prev_classes = sum(num_new_cls[:i])  # 旧类别数
    #         out_list.append(bias_layers[i - 1](input[:, :prev_classes]))  # 作用于所有旧类别
    #         out_list.append(input[:, start:end])  # 直接添加当前任务的原始输出

    #         start = end  

    #     return torch.cat(out_list, dim=1)  

# # 仅建立一个 BiasLayer，作用于所有旧类别     
#     def bias_forward(self, input):
#         num_new_cls = self.num_new_cls  
#         out_list = []
#         out_list.append(self.bias_layers[0](input[:, :num_new_cls[-1]]))  # Bias 作用于所有旧类别
#         out_list.append((input[:, num_new_cls[-1]:]))  # 最新任务的类别不经过 Bias 修正
#         return torch.cat(out_list, dim=1)  



    def stage1_initial(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        optimizer.zero_grad()
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            # p = self.bias_forward(p)    #no meed for bias_forward in initial training
            loss = criterion(p[:,:self.seen_cls], label)
            loss.backward(retain_graph=True)
            if (i + 1) % 13 == 0:
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer, num_new_cls, T=4, beta=0.5):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        feature_losses = []
        accumulation_steps = 13
        alpha = (self.seen_cls - self.num_new_cls[-1]) / (self.seen_cls) 
        # alpha = 0
        beta = beta
        print("classification proportion 1-alpha = ", 1-alpha)
        optimizer.zero_grad()
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p, feats = self.model(image, return_features=True, return_attentions=False)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p, pre_feats = self.previous_model(image, return_features=True, return_attentions=False)
                pre_p = self.bias_forward(pre_p)
                pre_p = F.softmax(pre_p[:,:self.seen_cls-num_new_cls]/T, dim=1)  
            logp = F.log_softmax(p[:,:self.seen_cls-num_new_cls]/T, dim=1)       
            # loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_soft_target = F.kl_div(logp, pre_p, reduction='batchmean')
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            # attn_loss = 0
            # for f, pf in zip(feats, pre_feats):
            #     attn = self.get_attention_map(f)
            #     pre_attn = self.get_attention_map(pf)
            #     attn_loss += F.mse_loss(attn, pre_attn)
            #     if not torch.isfinite(attn).all():
            #         print("Non-finite student_attn:", attn)
            #     if not torch.isfinite(pre_attn).all():
            #         print("Non-finite teacher_attn:", pre_attn)
            feat_old = pre_feats
            feat_new = feats
        ##    feature_loss = F.mse_loss(feat_new, feat_old)
            feature_loss = 0
            # attn_loss /= len(feats)
            loss = alpha * loss_soft_target * T * T + alpha * feature_loss + (1 - alpha) * loss_hard_target 

            loss.backward(retain_graph=True)
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        ##    feature_losses.append(feature_loss.item())
            distill_loss_epoch_mean = np.mean(distill_losses)
            ce_loss_epoch_mean = np.mean(ce_losses)
        ##    feature_loss_epoch_mean = np.mean(feature_losses)
            feature_loss_epoch_mean = 0
        print("stage1 distill loss :", distill_loss_epoch_mean, "ce loss :", ce_loss_epoch_mean, "feature loss :", feature_loss_epoch_mean)
        return distill_loss_epoch_mean, ce_loss_epoch_mean, feature_loss_epoch_mean

    def stage2(self, bias_data, criterion, optimizer):
        print("Evaluating ... ")
        losses = []
        optimizer.zero_grad()
        accumulation_steps = 3  # 累积3次
        self.model.train()
        for i, (image, label) in enumerate(tqdm(bias_data,leave = True)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            # 除以累积步数，防止梯度过大
            (loss / accumulation_steps).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(bias_data):
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))
        return np.mean(losses)

    def get_attention_map(self, feature_map, eps=1e-6):
         # feature_map: [B, C, H, W]
        attn_map = torch.norm(feature_map, p=2, dim=1)  # [B, H, W]
        attn_map = attn_map.view(attn_map.size(0), -1)  # [B, H*W]
 
        attn_map = attn_map - attn_map.min(dim=1, keepdim=True)[0]  # 归一化前减最小值
        attn_map = attn_map / (attn_map.max(dim=1, keepdim=True)[0] + eps)  # 避免除以0
 
        if not torch.isfinite(attn_map).all():
            print("attention map contains NaN/Inf")
            attn_map = torch.nan_to_num(attn_map, nan=0.0, posinf=0.0, neginf=0.0)
         
        return attn_map

    def extract_features(self, model, dataloader, device):
        """
        提取 dataloader 中所有样本的特征和标签

        Returns:
            features: Tensor [N, D]
            labels: Tensor [N]
            raw_images: List[N] 原始输入（用于 exemplar 记录）
        """
        model.eval()
        features_list = []
        labels_list = []
        raw_images_list = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(device)
                labels = labels.to(device)
                logits, feats = model(images, return_features=True)
                features_list.append(feats.cpu())
                labels_list.append(labels.cpu())
                raw_images_list.extend(images.cpu())  # 注意 raw images 是 list

        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return features, labels, raw_images_list

    def compute_class_centroids(self, features, labels, num_classes):
         """
         features: tensor(样本数 x 特征维度) , 所有样本的特征
         labels: tensor(样本数）
         num_classes: seen 总类别数
         返回: tensor(类别数 x 特征维度)，每个类别的中心向量
         """
         centroids = []
         labels = labels.squeeze()
         for cls in range(num_classes):
             cls_feats = features[labels == cls,:]
             centroid = cls_feats.mean(dim=0)
             centroids.append(centroid)
         return torch.stack(centroids)
