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
from torch.optim.lr_scheduler import LambdaLR, StepLR
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
from resnetPack import BiasLayer, resnet50, resnet152, resnet34
from cifar import Cifar100
from exemplar import Exemplar
from copy import deepcopy


class Trainer:
    def __init__(self, init_cls):  # total_cls = 100 # modify to flexible value     ## changed total_cls to init_cls
        self.total_cls = init_cls                                                   ## changed total_cls to init_cls
        self.seen_cls = 0
        self.num_new_cls = []
        self.dataset = Cifar100()  #for display
        #self.dataset = SubCifar100() #for debug
        self.lossdecay = []
        self.valacc    = []
        self.model = PreResNet(47,init_cls).cuda()        # formerly 32 for basicblock
## deeper model usage
        # self.model = resnet34(init_cls).cuda()                          
        print(self.model)
        #self.model = nn.DataParallel(self.model, device_ids=[0,1])
        self.model = self.model.cuda()  # 将模型移动到 GPU 0

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

        self.input_transform= Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32,padding=4),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        self.input_transform_eval= Compose([
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)


    def expand_model(self, new_cls):
        # 记录原始模型参数
        old_state_dict = self.model.state_dict()
        # 创建新的模型，但仅修改 fc 层
        self.model = PreResNet(47, self.total_cls + new_cls).cuda()     # formerly 32 for basicblock
## deeper model usage
        # self.model = resnet34(self.total_cls + new_cls).cuda()  # 创建新的模型，但仅修改 fc 层
        new_state_dict = self.model.state_dict()

        # 载入原有模型的参数（除 fc 层外）
        for name, param in old_state_dict.items():
            if "fc" not in name:  # 只更新非 fc 层的参数
                new_state_dict[name] = param

        self.model.load_state_dict(new_state_dict)  # 应用参数更新
        self.seen_cls += new_cls  # 这里修正 seen_cls，避免 bias_forward 出错 

        # if self.model is None:
        #     self.model = PreResNet(32, self.total_cls).cuda()  # 第一次初始化
        # else:
            # # 备份旧模型
            # old_model = self.model
            # old_fc_weights = old_model.fc.weight.data
            # old_fc_bias = old_model.fc.bias.data if old_model.fc.bias is not None else None
            
            # # 重新初始化模型
            # self.model = PreResNet(32, self.total_cls).cuda()

            # # 复制旧参数
            # with torch.no_grad():
            #     self.model.fc.weight[:old_fc_weights.shape[0], :] = old_fc_weights
            #     if old_fc_bias is not None:
            #         self.model.fc.bias[:old_fc_bias.shape[0]] = old_fc_bias
            
            #         # 记录原始模型参数
            # old_state_dict = self.model.state_dict()

            # 创建新的模型，但仅修改 fc 层
        #save model
        torch.save(self.model.state_dict(), 'model.pth')
            
               



    def test(self, testdata, inc_i):     # task number added for deciding whether to use bias correction
        print("test data number : ",len(testdata))   
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            if inc_i > 0:   # if there is bias correction
                p = self.bias_forward(p)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc


    def eval(self, criterion, evaldata):
        self.model.eval()
        losses = []
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(evaldata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p, label)
            losses.append(loss.item())
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        print("Validation Loss: {}".format(np.mean(losses)))
        print("Validation Acc: {}".format(100*correct/(correct+wrong)))
        self.model.train()
        self.epoch_loss_mean = np.mean(losses)
        self.valacc.append(100*correct/(correct+wrong))
        return

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)
        
        previous_model = None

        dataset = self.dataset
        test_xs, test_ys, train_xs, train_ys = [], [], [], []

        test_acc_noBiC = []
        test_acc = []
        test_accs = []
        test_accs_noBiC = []
        
        for inc_i in range(dataset.batch_num):
            print(f"Incremental num: {inc_i}")
            bias_layer = BiasLayer().cuda()

            # 为每个增量任务添加一个 bias layer 在第二个任务时才加上第一个任务的layer
            # if inc_i == 1 :
            #     self.bias_layers.append(bias_layer)
            #     self.bias_layers.append(bias_layer)
            # if inc_i > 1:
                # bias correction layer creation for each Incremental task
            
            #if inc_i > 0:
            self.bias_layers.append(bias_layer) 
            print('current bias layers number : ', len(self.bias_layers))

            train, val, test = dataset.getNextClasses(inc_i)

            # 计算当前 batch 的类别数
            new_cls = set(y for _, y in train + val + test) - set(range(self.total_cls))
            
            num_new_cls = len(new_cls)
            self.num_new_cls.append(num_new_cls)
            print(f"New classes detected: {num_new_cls}")
            print(f"New classes for former tasks: {self.num_new_cls}")
            # 1. 记录总类别数
            self.total_cls += num_new_cls

            if num_new_cls > 0:
                
                # 2. 扩展模型
                self.expand_model(num_new_cls)
            
            print(f"total class till task {inc_i} :{self.total_cls}")
            self.seen_cls = self.total_cls

            print('Training Samples:',len(train),'validation samples:', len(val), 'test samples:', len(test))
            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            train_xs, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(train_x)
            train_xs.extend(val_x)
            train_ys.extend(train_y)
            train_ys.extend(val_y)


            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            val_data = DataLoader(BatchData(val_x, val_y, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            optimizer = optim.AdamW(self.model.parameters(), lr=lr/100, betas=(0.9, 0.999), weight_decay=4e-2)


            if inc_i > 0:               # Biaslayer trained only if there is bias correction
                # bias_optimizer = optim.SGD(self.bias_layers[inc_i].parameters(), lr=lr, momentum=0.9)
                # bias_optimizer = optim.Adam(self.bias_layers[-1].parameters(), lr=0.001) #version 1: only train the last bias layer #inc-1 -> -1
                bias_optimizer = optim.Adam([param for layer in self.bias_layers for param in layer.parameters()], lr=0.001)  #version2: train all the bias layers              
                # bias_scheduler = StepLR(bias_optimizer, step_size=70, gamma=0.1)
            # exemplar.update(total_cls//dataset.batch_num, (train_x, train_y), (val_x, val_y))
            exemplar.update(len(set(train_y) | set(val_y)), (train_x, train_y), (val_x, val_y)) 
            # modify to flexible class number, especially for new tasks with old classes
            # adapt to tasks with repepitition of old classes
            self.seen_cls = exemplar.get_cur_cls()
            print("seen cls number : ", self.seen_cls)
            val_xs, val_ys = exemplar.get_exemplar_val()
            val_bias_data = DataLoader(BatchData(val_xs, val_ys, self.input_transform),
                        batch_size=100, shuffle=True, drop_last=False)

            test_acc.append([])
            test_acc_noBiC.append([])


            for epoch in range(epoches):
                print("---"*50)
                print("current incremental task : ", inc_i)
                print("Epoch", epoch)
                # scheduler.step()
                cur_lr = self.get_lr(optimizer)
                print("Current Learning Rate : ", cur_lr)
                self.model.train()
                for _ in range(len(self.bias_layers)):
                    self.bias_layers[_].eval()
                if inc_i > 0:
                    self.stage1_distill(train_data, criterion, optimizer, num_new_cls)
                else:
                    self.stage1_initial(train_data, criterion, optimizer)
                
                acc = self.test(test_data, inc_i)
                test_acc_noBiC[-1].append(acc)

            test_accs_noBiC.append(max(test_acc_noBiC[-1]))

            if inc_i > 0:
                for epoch in range(4*epoches):
                    # bias_scheduler.step()
                    self.model.eval()
                    for _ in range(len(self.bias_layers)):
                        self.bias_layers[_].train()
                    self.stage2(val_bias_data, criterion, bias_optimizer)
                    if epoch % 25 == 0:
                        acc = self.test(test_data, inc_i)
                        test_acc[-1].append(acc)
            for i, layer in enumerate(self.bias_layers):
                layer.printParam(i)
            self.previous_model = deepcopy(self.model)
            acc = self.test(test_data, inc_i)
            test_acc[-1].append(acc)
            test_accs.append(max(test_acc[-1]))
            print("Test results on testset of model without BiC",test_accs_noBiC)
            print("Test results on testset after BiC training",test_accs)
    
        print("number of new classes seen in each task : ", self.num_new_cls)

        # stage1 test accuracy 可视化
        save_dir = "output"  # 当前文件夹下的 output 目录
        os.makedirs(save_dir, exist_ok=True)  


        plt.figure(figsize=(10, 6))  

        for i in range(len(test_acc_noBiC)):
            epochs = range(len(test_acc_noBiC[i]))
            plt.plot(epochs, test_acc_noBiC[i], label=f"Task {i}")  

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Changes Over Epochs for Each Incremental Task in stage 1")
        plt.legend()  
        plt.grid(True)  

        save_path = os.path.join(save_dir, "Stage1_accuracy_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        print(f"Plot saved at: {save_path}")

        # stage2 test accuracy 可视化
        plt.figure(figsize=(10, 6))  

        for i in range(len(test_acc)):
            epochs = range(len(test_acc[i]))
            plt.plot(epochs, test_acc[i], label=f"Task {i}")  

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Changes Over Epochs for Each Incremental Task in stage 2")
        plt.legend()  
        plt.grid(True)  
        save_path = os.path.join(save_dir, "Stage2_accuracy_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")

            
            


    # def bias_forward(self, input):  #V1
    #     in1 = input[:, :20]
    #     in2 = input[:, 20:40]
    #     in3 = input[:, 40:60]
    #     in4 = input[:, 60:80]
    #     in5 = input[:, 80:100]
    #     out1 = self.bias_layer1(in1)
    #     out2 = self.bias_layer2(in2)
    #     out3 = self.bias_layer3(in3)
    #     out4 = self.bias_layer4(in4)
    #     out5 = self.bias_layer5(in5)
    #     return torch.cat([out1, out2, out3, out4, out5], dim = 1)

#     def bias_forward(self, input): # modify to flexible class number   #V2
#         out_list = []
#         #### NEEDS TO BE modify to flexible class number
#         steps = [self.total_cls // len(self.bias_layers)] * len(self.bias_layers)   # 计算每个 bias_layer 处理多少类
#         for i in range(self.total_cls % len(self.bias_layers)):
#             steps[i] += 1
#         out_list = []
#         start = 0
#         for i, step in enumerate(steps):
#             out_list.append(self.bias_layers[i](input[:, start:start + step]))
#             start += step
#         return torch.cat(out_list, dim=1)
    # def bias_forward(self, input): #V3  
    #     # modified to different new class number in each task 
    #     # modified to only train the last bias layer
    #     num_new_cls = self.num_new_cls
    #     bias_layers = self.bias_layers
    #     out = input
    #     for i in range(len(bias_layers)):
    #         out[:,sum(num_new_cls[:i]):sum(num_new_cls[:i+1])]=bias_layers[i](out[:,sum(num_new_cls[:i]):sum(num_new_cls[:i+1])])
    #     return torch.cat(out, dim=1)

    # def bias_forward(self, input):  # V4
    #     # modified for better performance on task 2
    #     num_new_cls = self.num_new_cls
    #     bias_layers = self.bias_layers
    #     new_input = input[:, :self.seen_cls]
    #     old_input = input[:, self.seen_cls:]
    #     out_list = torch.zeros(1,self.total_cls)
    #     new_output = new_input
    #     old_output = []
    #     start = 0
    #     #bias_forward only used for i > 0
    #     for i in range(len(bias_layers)):
    #         end = start + num_new_cls[i]

    #         if i == 0:
    #             old_output = bias_layers[i](old_input[:, start:end]) 
    #         else:            
    #             old_output = torch.cat((old_output,bias_layers[i](old_input[:, start:end])),dim=1)
    #         start = end

    #     old_output = torch.tensor(old_output)
    #     out_list = torch.cat((old_output,new_output), dim=1)
    #     return out_list

    def bias_forward(self, input):
        num_new_cls = self.num_new_cls  # 每个任务的新类别数
        bias_layers = self.bias_layers  # 每个任务对应的 bias 层
        out_list = []  # 存储每个任务经过 bias 处理后的输出
        start = 0

        for i in range(len(bias_layers)):  
            end = start + num_new_cls[i]  # 计算当前任务的起始和结束位置
            out_list.append(bias_layers[i](input[:, start:end]))  # 逐个 task 进行 bias 修正
            # out_list.append(bias_layers[i](input[:,:end]))  # 逐个 task 进行 bias 修正,但是每个任务的输出都是从头开始的
            start = end

        return torch.cat(out_list, dim=1)  # 拼接所有任务的输出

    # def bias_forward(self, input):
    #     num_new_cls = self.num_new_cls  # 每个任务的新类别数
    #     bias_layers = self.bias_layers  # 存储 Bias 层（不包含第一个任务）
    #     out_list = []  # 存储经过 bias 处理后的输出
    #     start = 0

    #     for i in range(len(num_new_cls)):  
    #         end = start + num_new_cls[i]  
    #         prev_classes = sum(num_new_cls[:i])  # 旧类别数
    #         out_list.append(bias_layers[i - 1](input[:, :prev_classes]))  # 作用于所有旧类别
    #         out_list.append(input[:, start:end])  # 直接添加当前任务的原始输出

    #         start = end  # 更新起始索引

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
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            # p = self.bias_forward(p)    #no meed for bias_forward in initial training
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer, num_new_cls):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        #alpha = (self.seen_cls - 20)/ self.seen_cls
                #alpha = (self.seen_cls - (self.total_cls // self.dataset.batch_num)) / self.seen_cls
                # modify to flexible class number 
        ###### Might be problems on alpha calculation 
        alpha = (self.seen_cls - self.num_new_cls[-1]) / (self.seen_cls) 
#########################################################################################################################################        
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p = self.previous_model(image)
                pre_p = self.bias_forward(pre_p)
                pre_p = F.softmax(pre_p[:,:self.seen_cls-num_new_cls]/T, dim=1)  # modified to flexible class number (-20 -> -num_new_cls)
            logp = F.log_softmax(p[:,:self.seen_cls-num_new_cls]/T, dim=1)       # modified to flexible class number (-20 -> -num_new_cls) 
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target * T * T + (1-alpha) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))
        

    def stage2(self, val_bias_data, criterion, optimizer):
        print("Evaluating ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(val_bias_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))
