import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import torch

class Exemplar:
    def __init__(self, max_size, total_cls):
        self.val = {}
        self.train = {}
        self.cur_cls = 0
        self.max_size = max_size    # storage number of exemplar per class
        self.total_classes = total_cls

    # def update(self, cls_num, train, val):
    #     train_x, train_y = train
    #     val_x, val_y = val
    #     assert self.cur_cls == len(list(self.val.keys()))
    #     assert self.cur_cls == len(list(self.train.keys()))
    #     cur_keys = list(set(val_y))
    #     self.cur_cls += cls_num
    #     total_store_num = self.max_size / self.cur_cls if self.cur_cls != 0 else max_size
    #     train_store_num = int(total_store_num * 0.9)
    #     val_store_num = int(total_store_num * 0.1)
    #     for key, value in self.val.items():
    #         self.val[key] = value[:val_store_num]
    #     for key, value in self.train.items():
    #         self.train[key] = value[:train_store_num]

    #     for x, y in zip(val_x, val_y):
    #         if y not in self.val:
    #             self.val[y] = [x]
    #         else:
    #             if len(self.val[y]) < val_store_num:
    #                 self.val[y].append(x)
    #     assert self.cur_cls == len(list(self.val.keys()))
    #     for key, value in self.val.items():
    #         assert len(self.val[key]) == val_store_num

    #     for x, y in zip(train_x, train_y):
    #         if y not in self.train:
    #             self.train[y] = [x]
    #         else:
    #             if len(self.train[y]) < train_store_num:
    #                 self.train[y].append(x)
    #     assert self.cur_cls == len(list(self.train.keys()))
    #     for key, value in self.train.items():
    #         assert len(self.train[key]) == train_store_num

# modify to flexible class number, especially for new tasks with old classes
# adapt to tasks with repepitition of old classes

    def update(self, train, val, features=None, labels=None, centroids=None, model_images=None):
        """
        train/val: (x, y) 格式（列表）
        features, labels: Tensor，训练集图像提取后的特征和标签
        centroids: Tensor[C, D]，每类中心
        model_images: List，训练图像（与 features 对应）
        """

        train_x, train_y = train
        val_x, val_y = val

        # 已存储的类别 + 新任务的新类别
        all_existing_classes = set(self.val.keys()) | set(val_y) | set(train_y)      
        self.cur_cls = len(all_existing_classes)  

        total_store_num = self.max_size / self.cur_cls if self.cur_cls != 0 else self.max_size 
        train_store_num = int(total_store_num * 0.9)
        val_store_num = int(total_store_num * 0.1)

        
        # cutting
        for key in self.val:
            self.val[key] = self.val[key][:val_store_num]

        # validation set
        for x, y in zip(val_x, val_y):
            if y not in self.val:
                self.val[y] = []
            if len(self.val[y]) < val_store_num:
                self.val[y].append(x)

        # iCaRL herding : feature centroid l2 
        if features is not None and labels is not None and centroids is not None and model_images is not None:
            self.select_exemplars_icarl_from_features(features, labels, centroids, train_store_num, model_images)
            # for cls, img_list in selected.items():
            #     if cls not in self.train:
            #         self.train[cls] = []
            #     self.train[cls][0].extend(img_list)  # extend images
            #     self.train[cls][1].extend([cls]*len(img_list))  # extend labels

        else:
            # no feature input
            for x, y in zip(train_x, train_y):
                if y not in self.train:
                    self.train[y] = []
                if len(self.train[y]) < train_store_num:
                    self.train[y].append(x)

        # 确保所有类别的样本数量符合存储限制
        print(f"Current classes: {self.cur_cls}, Train size: {len(self.train)}, Val size: {len(self.val)}")
        print(f"len(self.val): {len(self.val)}, len(self.train): {len(self.train)}")
        print(f"train classes = {set(train_y)} \n validation classes =  {set(val_y)} \n all classes = {all_existing_classes}\n current classes = {self.cur_cls}")
        assert len(self.val) == self.cur_cls
        assert len(self.train) == self.cur_cls
        


    def get_exemplar_train(self):
        exemplar_train_x = []
        exemplar_train_y = []
        for key, (images, labels) in self.train.items():
            exemplar_train_x.extend(images)
            exemplar_train_y.extend(labels)
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

    def select_exemplars_icarl_from_features(self, features, labels, centroids, num_exemplars_per_class, raw_images):
        """
        iCaRL herding-based exemplar selection using features and centroids.

        Args:
            features (Tensor): [N, D] Feature vectors normalized.
            labels (Tensor): [N] Class labels.
            centroids (Tensor): [C, D] Class centroids normalized.
            num_exemplars_per_class (int): Number of exemplars to select per class.
            raw_images (List): Raw images aligned with features/labels.

        Returns:
            dict: {class_id: [selected_raw_images]}
        """
        exemplars_by_class = {}

        # Normalize features and centroids
        features = features / features.norm(dim=1, keepdim=True)
        centroids = centroids / centroids.norm(dim=1, keepdim=True)
        labels = labels.view(-1)

        unique_classes = torch.unique(labels)

        for cls in unique_classes:
            cls = cls.item()

            # Get mask and indices for current class
            cls_mask = (labels == cls)
            cls_indices = cls_mask.nonzero(as_tuple=True)[0]  # shape [n_cls]

            cls_feats = features[cls_indices]  # shape [n_cls, D]
            cls_imgs = [raw_images[i] for i in cls_indices.tolist()]  # List of images

            cls_center = centroids[cls]  # shape [D]

            # Initialize selection sets
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

                # Cosine similarity equivalent (since normalized): dot product with mean_diff
                sims = torch.matmul(available_feats, mean_diff)
                idx = torch.argmax(sims).item()

                selected_imgs.append(available_imgs[idx].squeeze(0))
                selected_feats.append(available_feats[idx])

                available_feats = torch.cat([available_feats[:idx], available_feats[idx+1:]], dim=0)
                del available_imgs[idx]

            exemplars_by_class[cls] = (selected_imgs, [cls]*len(selected_imgs))            
            for cls, (imgs, labels) in exemplars_by_class.items():
                if cls in self.train:
                    old_imgs, old_labels = self.train[cls]
                    self.train[cls] = (old_imgs + imgs, old_labels + labels)
                else:
                    self.train[cls] = (imgs, labels)
        
        return exemplars_by_class


    def select_exemplars_icarl_from_features(self, features, labels, centroids, num_exemplars_per_class, raw_images):
        """
        iCaRL原始herding选样 + 可视化（改正版）
        """
        import torch
        import os
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np

        # Step 1: 归一化
        device = features.device
        features = features / features.norm(dim=1, keepdim=True)
        centroids = centroids / centroids.norm(dim=1, keepdim=True)
        labels = labels.view(-1)

        exemplars_by_class = {}
        selected_mask = torch.zeros(len(raw_images), dtype=torch.bool, device=device)
        unique_classes = torch.unique(labels)

        # Step 2: 主循环
        for cls in unique_classes:
            cls = cls.item()
            cls_mask = (labels == cls)
            cls_indices = cls_mask.nonzero(as_tuple=True)[0]  # 当前类样本索引

            cls_feats = features[cls_indices]  # shape [n_cls, D]
            cls_imgs = [raw_images[i] for i in cls_indices.tolist()]
            cls_center = centroids[cls]

            # 计算和质心余弦相似度（即 dot product，因为 normalized）
            sims = torch.matmul(cls_feats, cls_center)
            topk = torch.topk(sims, k=min(num_exemplars_per_class, len(cls_feats)))

            selected_indices = cls_indices[topk.indices]  # 全局索引
            
            selected_imgs = [raw_images[i].squeeze(0) for i in selected_indices.tolist()]

            # 更新掩码
            selected_mask[selected_indices] = True

            # 存储 (imgs, labels) tuple
            exemplars_by_class[cls] = (selected_imgs, [cls]*len(selected_imgs))

        # Step 3: 可视化
        self.visualize_feature_space(
            features=features.cpu(),
            all_labels=labels.cpu(),
            selected_mask=selected_mask.cpu(),
            centroids=centroids.cpu(),
            method='tsne'
        )

        # Step 4: 存 self.train （保持和旧版一致）
        self.train = exemplars_by_class

        return exemplars_by_class
    
    def select_exemplars_icarl_herding_diversity(self, features, labels, centroids, num_exemplars_per_class, raw_images, diversity_lambda=0.75):
        """
        iCaRL exemplar selection with Herding + explicit MMD-based Diversity

        Args:
            features: Tensor [N, D] — normalized feature embeddings
            labels: Tensor [N] — class labels
            centroids: Tensor [C, D] — class means
            num_exemplars_per_class: int — number of exemplars per class
            raw_images: list — corresponding raw data samples
            diversity_lambda: float — balance factor for diversity

        Returns:
            exemplars_by_class: dict {class_id: (exemplar_imgs, labels)}
        """
        import torch
        import numpy as np

        device = features.device
        features = features / features.norm(dim=1, keepdim=True)
        centroids = centroids / centroids.norm(dim=1, keepdim=True)
        labels = labels.view(-1)

        exemplars_by_class = {}
        selected_mask = torch.zeros(len(raw_images), dtype=torch.bool, device=device)
        unique_classes = torch.unique(labels)

        for cls in unique_classes:
            cls = cls.item()
            cls_mask = (labels == cls)
            cls_indices = cls_mask.nonzero(as_tuple=True)[0]  # 当前类样本索引

            cls_feats = features[cls_indices]  # shape [n_cls, D]
            cls_imgs = [raw_images[i] for i in cls_indices.tolist()]

            n_samples = len(cls_feats)
            n_exemplars = min(num_exemplars_per_class, n_samples)

            mu = centroids[cls].unsqueeze(0)  # [1, D]

            # Herding + Diversity selection
            selected_local_indices = []
            selected_feats = []  # 已选样本的特征

            candidates = list(range(n_samples))

            for _ in range(n_exemplars):
                best_score = None
                best_idx = None

                for idx in candidates:
                    feat = cls_feats[idx].unsqueeze(0)  # [1, D]
                    
                    # representativeness: 距离均值越近越好 (负平方距离)
                    rep_score = -torch.norm(feat - mu, p=2).item() ** 2

                    # diversity: 距离已有样本越远越好
                    if selected_feats:
                        dists = [torch.norm(feat - sf, p=2).item() ** 2 for sf in selected_feats]
                        diversity_score = np.mean(dists)
                    else:
                        diversity_score = 0

                    total_score = rep_score + diversity_lambda * diversity_score

                    if (best_score is None) or (total_score > best_score):
                        best_score = total_score
                        best_idx = idx

                selected_local_indices.append(best_idx)
                selected_feats.append(cls_feats[best_idx])
                candidates.remove(best_idx)

            selected_indices = cls_indices[torch.tensor(selected_local_indices, device=device)]
            selected_imgs = [cls_imgs[i] for i in selected_local_indices]

            # 更新掩码
            selected_mask[selected_indices] = True

            exemplars_by_class[cls] = ([img.squeeze(0) for img in selected_imgs], [cls] * len(selected_imgs))

        # 可视化
        self.visualize_feature_space(
            features=features.cpu(),
            all_labels=labels.cpu(),
            selected_mask=selected_mask.cpu(),
            centroids=centroids.cpu(),
            method='tsne'
        )

        self.train = exemplars_by_class
        return exemplars_by_class


    def visualize_feature_space(self, features, all_labels, selected_mask, centroids, method='tsne'):
        import os
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import numpy as np

        combined = np.concatenate([features.numpy(), centroids.numpy()], axis=0)

        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError("Unsupported method")

        low_dim = reducer.fit_transform(combined)
        feat_dim = low_dim[:len(features)]
        centroid_dim = low_dim[len(features):]

        plt.figure(figsize=(15, 12))
        plt.scatter(feat_dim[~selected_mask, 0], feat_dim[~selected_mask, 1],
                    c=all_labels[~selected_mask], cmap='tab20', alpha=0.4, label='Unselected', edgecolor='w')
        sc = plt.scatter(feat_dim[selected_mask, 0], feat_dim[selected_mask, 1],
                        c=all_labels[selected_mask], cmap='tab20', marker='*',
                        s=100, label='Selected', edgecolor='k')
        plt.scatter(centroid_dim[:, 0], centroid_dim[:, 1], c='black',
                    marker='X', s=200, label='Centroids', edgecolor='gold',
                    linewidth=1.5)

        plt.title(f'Feature Space Visualization ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(title='Legend')
        plt.colorbar(sc, label='Class ID')
        plt.tight_layout()
        os.makedirs('output', exist_ok=True)
        plt.savefig(f'output/feature_visualization_{method}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
