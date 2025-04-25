import torch


class Exemplar:
    def __init__(self, max_size, total_cls):
        self.val = {}
        self.train = {}
        self.cur_cls = 0
        self.max_size = max_size    # 每类 exemplar 最大数目
        self.total_classes = total_cls

    def update(self, train, val, features=None, labels=None, centroids=None, model_images=None):
        """
        train/val: (x, y) 格式（列表）
        features, labels: Tensor，训练集图像提取后的特征和标签（可选）
        centroids: Tensor[C, D]，每类中心（可选）
        model_images: List，训练图像（与 features 对应）
        """
        train_x, train_y = train
        val_x, val_y = val

        # 更新类别总数
        all_existing_classes = set(self.val.keys()) | set(val_y) | set(train_y)
        self.cur_cls = len(all_existing_classes)

        total_store_num = self.max_size / self.cur_cls if self.cur_cls != 0 else self.max_size
        train_store_num = int(total_store_num * 0.9)
        val_store_num = int(total_store_num * 0.1)

        # 裁剪旧类
        for key in self.val:
            self.val[key] = self.val[key][:val_store_num]

        # 添加 val 样本
        for x, y in zip(val_x, val_y):
            if y not in self.val:
                self.val[y] = []
            if len(self.val[y]) < val_store_num:
                self.val[y].append(x)

        # 添加 train 样本：改为 iCaRL herding 策略（特征选择）
        if features is not None and labels is not None and centroids is not None and model_images is not None:
            selected = self.select_exemplars_icarl_from_features(
                features, labels, centroids, train_store_num, model_images
            )
            for cls, img_list in selected.items():
                if cls not in self.train:
                    self.train[cls] = []
                self.train[cls].extend(img_list)
        else:
            # 如果没传特征就用原来的方法
            for x, y in zip(train_x, train_y):
                if y not in self.train:
                    self.train[y] = []
                if len(self.train[y]) < train_store_num:
                    self.train[y].append(x)

        print(f"Current classes: {self.cur_cls}, Train size: {len(self.train)}, Val size: {len(self.val)}")
        print(f"len(self.val): {len(self.val)}, len(self.train): {len(self.train)}")
        print(f"train classes = {set(train_y)} \n validation classes =  {set(val_y)} \n all classes = {all_existing_classes}\n current classes = {self.cur_cls}")
        assert len(self.val) == self.cur_cls
        assert len(self.train) == self.cur_cls

    def select_exemplars_icarl_from_features(self, features, labels, centroids, num_exemplars_per_class, raw_images):
        """
        基于 iCaRL 的 herding 策略，从已有特征中选择样本
        """
        exemplars_by_class = {}
        features = features / features.norm(dim=1, keepdim=True)
        centroids = centroids / centroids.norm(dim=1, keepdim=True)
        labels = labels.squeeze()
        for cls in torch.unique(labels):
            cls = cls.item()
            cls_mask = (labels == cls)
            cls_feats = features[cls_mask]
            cls_imgs = [img for i, img in enumerate(raw_images) if cls_mask[i]]
            cls_center = centroids[cls]

            selected_imgs = []
            selected_feats = []
            available_feats = cls_feats.clone()
            available_imgs = cls_imgs.copy()

            for _ in range(min(num_exemplars_per_class, len(available_imgs))):
                if selected_feats:
                    mean_selected = torch.stack(selected_feats).mean(dim=0)
                    mean_diff = cls_center - mean_selected
                else:
                    mean_diff = cls_center

                sims = torch.matmul(available_feats, mean_diff)
                idx = torch.argmax(sims).item()
                selected_imgs.append(available_imgs[idx])
                selected_feats.append(available_feats[idx])

                # 移除已选
                available_feats = torch.cat([available_feats[:idx], available_feats[idx+1:]], dim=0)
                del available_imgs[idx]

            exemplars_by_class[cls] = selected_imgs
        return exemplars_by_class

    def get_exemplar_train(self):
        exemplar_train_x = []
        exemplar_train_y = []
        for key, value in self.train.items():
            for train_x in value:
                exemplar_train_x.append(train_x)
                exemplar_train_y.append(key)
        return exemplar_train_x, exemplar_train_y

    def get_exemplar_val(self):
        exemplar_val_x = []
        exemplar_val_y = []
        for key, value in self.val.items():
            for val_x in value:
                exemplar_val_x.append(val_x)
                exemplar_val_y.append(key)
        return exemplar_val_x, exemplar_val_y

    def get_cur_cls(self):
        return self.cur_cls
