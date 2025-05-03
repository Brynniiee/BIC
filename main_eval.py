import torch
from torch.utils.data import DataLoader
from trainer import Trainer
from dataset import BatchData
import argparse

# 1. 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='model.pth', help='Path to saved model (.pth)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--shift', action='store_true', help='Use shift dataset')
args = parser.parse_args()

# 2. 初始化 trainer (不需要训练，只用数据接口)
trainer = Trainer(init_cls=0)  
trainer.model.cuda()

# 3. 准备数据集
if args.shift:
    shift_xs, shift_ys = zip(*trainer.shift_groups)
    eval_xs = list(shift_xs)
    eval_ys = list(shift_ys)
    print(f"[INFO] Using shift set: {len(eval_xs)} samples")
else:
    test_xs, test_ys = zip(*trainer.dataset.test_groups)
    eval_xs = list(test_xs)
    eval_ys = list(test_ys)
    print(f"[INFO] Using test set: {len(eval_xs)} samples")

eval_data = DataLoader(BatchData(eval_xs, eval_ys, trainer.input_transform_eval),
                        batch_size=args.batch_size, shuffle=False)

# 4. 评估模型
acc = trainer.eval_model(eval_data, model_path=args.model_path, heatmap_name='eval_heatmap.png')
print(f"[FINAL ACC] {acc*100:.2f}%")
