## Open-set Class-incremental Learning for a phi-OTDR dataset

#### Project Origin  
This repository is based on [BIC](https://github.com/sairin1202/BIC) created by [sairin1202]. 

#### Dataset

6 classes in total, each being .mat file of   $15\times10240$

with sample numbers 

| Class | Samples | proportion |
| :-----: | :------------------: | :------: |
| 0            |                       578 |  25.8%   |
| 1            |                       738 |  33.0%   |
| 2            |                       194 |   8.7%   |
| 3            |                        40 |   1.8%   |
| 4            |                       241 |  10.8%   |
| 5            |                       459 |  20.5%   |
| **Total** |                  **2250** | **100%** |
#### feature extractor
ResNet47 (3 layers * 5 bottleneck blocks)
#### CIL method
BIC: exemplar replay + knowledge distill + bias correction 
#### OSR method
EVT