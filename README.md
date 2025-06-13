# Unification of Closed-Open Industrial detection scenarios: New Large-Scale Benchmarks, Challenges and Baselines

[AAAI2025] The extened version of dataset and code of "Zero-Shot learning in industrial scenarios: New Large-Scale Benchmark, Challenges and Baseline".

## The extened version of abstract
Large-scale Visual-Language models (LVLMs) have achieved remarkable success in visual tasks. However, the significant differences between industrial and natural scenes make the application of LVLMs challenging. On the one hand, the scarcity of large-scale industrial defect data makes the application of LVLMs in industrial scenarios unexplored. On the other hand, existing LVLMs rely on user-provided prompts to segment objects. This usually leads to poor performance due to the inclusion of irrelevant semantics. To fill this gap, this paper proposes the first large-scale industrial multi-modal dataset and a refined text-visual prompts network for industrial defect detection (RTVPNet). First, this paper constructs a multi-modal industrial unified open-closed dataset (MMIOC-1M) with 1M+ samples. MMIOC-1M supports open and closed scene tasks and has rich industry categories, including 17 super categories, 31 scenes, and 351 subcategories. MMIOC-1M is the first large-scale multi-modal pre-training dataset for industrial zero-shot and fully supervised learning, providing valuable training data for large models in future industrial scenarios. Based on MMIO-1M, this paper provides RTVPNet specifically for industrial zero-shot, fully supervised learning, and visual question-answering tasks. RTVPNet has two significant advantages: First, we design an expert-guided domain projection mechanism for large models and an industrial zero-shot method based on Mobile-SAM, which enhances the generalization of large models in industrial scenarios. Second, RTVPNet automatically generates refined visual prompts directly from images and considers the text-visual prompt interactions that were ignored by previous LVLM, improving visual and textual content understanding. RTVPNet achieves SOTA with 42.2\% and 24.7\% AP in open and closed scenarios on MMIOC-1M, respectively. Parts of this paper were originally published in its AAAI-2025 conference version. This paper extends earlier work in terms of dataset, network architecture and analysis, etc.

## MMIOC-1M data has been upload to the Google Drive: 

To the best of our knowledge, this paper constructs the first multi-modal object detection dataset (MMIOC-1M) for industrial open and closed scenes. MMIOC-1M is extended based on MMIO-80K and contains more than 1M samples and 31 industrial scenes, effectively alleviating the lack of domain expertise in industrial open scenes. The comparison with mainstream defect data is as follows:
![MMIOC-1M_dataset_compare](https://github.com/hellozzk/MMIO/blob/main/datasetcompare_01.jpg#pic_center)
The dataset is visualized as follows:
![RTVPNet Architect](https://github.com/hellozzk/MMIO/blob/main/datasetvis_01.jpg#pic_center)
The statistics of the dataset categories are as follows:
![RTVPNet Architect](https://github.com/hellozzk/MMIO/blob/main/statistic_01.jpg#pic_center)
This paper proposes a refined learnable text-visual prompt to improve the detection ability of visual language models in industrial open and closed scenes. Compared with the previous version of RTVP, RTVPNet designs a new text-visual bidirectional prompt interaction and an energy-based refined visual prompt method. RTVPNet automatically provides specific text-visual prompts for each image, which reduces the noise and effectively improves the knowledge and understanding ability of LVLMs in industrial domains.
![RTVPNet Architect](https://github.com/hellozzk/MMIO/blob/main/Architect.png#pic_center)



# 前言

`提示：这里可以添加本文要记录的大概内容：`

例如：随着人工智能的不断发展，机器学习这门技术也越来越重要，很多人都开启了学习机器学习，本文就介绍了机器学习的基础内容。

---

`提示：以下是本篇文章正文内容，下面案例可供参考`

# 一、pandas是什么？

示例：pandas 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。

# 二、使用步骤
## 1.引入库
>代码如下（示例）：

```c
Name_of_Dataset
|-- Images
|-----|----- train
|-----|--------|------ defect images1
|-----|--------|------ ...
|-----|--------|------ ...
|-----|----- validation
|-----|--------|------ defect images1
|-----|--------|------ ...
|-----|--------|------ ...
|-- labels
|-----|----- train
|--------|--------|------txt
|-----|--------|--------|------ defect.txt1
|-----|--------|--------|------ ...
|-----|--------|--------|------ ...
|--------|--------|------annotations
|-----|--------|--------|------ train.json
|-----|----- validation
|-----|--------|--------|------ defect.txt1
|-----|--------|--------|------ ...
|-----|--------|--------|------ ...
|--------|--------|------annotations
|-----|--------|--------|------ train.json
```

## 2.读入数据

代码如下（示例）：

```c
data = pd.read_csv(
    'https://labfile.oss.aliyuncs.com/courses/1283/adult.data.csv')
print(data.head())
```

该处使用的url网络请求的数据。

---

# 总结
`提示：这里对文章进行总结：`

例如：以上就是今天要讲的内容，本文仅仅简单介绍了pandas的使用，而pandas提供了大量能使我们快速便捷地处理数据的函数和方法。

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 # Enhanced Glass Defect Localization through Polarization Information Fusion and Reciprocal Feature Evolution: New Challenge, Dataset and Baseline

## Abstract
Glass containers are widely used in daily life, but existing defect detection methods are still difficult to detect accurately. Firstly, the extreme scarcity of glass container defect samples makes it difficult to train the model accurately. Secondly, due to the characteristics of transparent reflection, it is challenging to obtain blurred defect separation edge and context information. Finally, the existing positioning loss does not consider the shape accuracy of the predicted box, the accurate positioning information cannot be obtained. This paper introduces polarization and RGB information to create the glass container defect dataset locations containing 60,000+ samples. Subsequently, this paper proposes a novel interactive decoupled feature evolution network (IDFE-Net) by decoupling edge and context information in a feature interactive coevolution method. Finally, with the demand for accurate positioning in industrial defect detection scenarios, this paper proposes a novel Inforced-IoU, which can obtain more precise position information by adaptively adjusting the scale of the predicted box. Experiments show that our method only uses 18.1 GFLOPs and achieves 94.61 % and 67.43 % mAP on glass container and wood defects datasets, better than the current state-of-the-art method.


## GCD data statistical distribution
| Class            | bubble              | oil                   | Plastering thread         | Black spot        | Quenched grain               |
|------------------|---------------------|-----------------------|---------------------------|-------------------|------------------------------|
| S                | 22905               | 3279                  | 2239                      | 19463             | 3104                         |
| M                | 823                 | 2142                  | 3850                      | 219               | 2610                         |
| L                | 572                 | 6489                  | 13356                     | 188               | 2066                         |
| Feature describe | bright round hollow | dark irregular shapes | Irregular striped pattern | dark round shapes | blurred and irregular shapes |

## GCD data Part-1 sample has been upload to the Google Drive: https://drive.google.com/file/d/1aXztYIRyDEiJJlhpE6tDQT88GeSJ-M-0/view?usp=drive_link


![Relationship between different defect](https://github.com/hellozzk/GCD.github.io/blob/main/img/Rekationship.png#pic_center)

![GCD samples](https://github.com/hellozzk/GCD.github.io/blob/main/img/GCDsample.png#pic_center)

## Code
The code has been open-sourced, and users can choose between the basic version and the more powerful version IDFE-Net.

## Requirements
GPU: NVIDIA 3090   CPU: Intel i7-12700KF      Running Memory: 64GB+      PyTorch 1.11.0      Python 3.8

Command: pip install -r requirements.txt



## Multi-GPU Run train on GCD-Part-1

python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py  --data /data/GCD-Part-1.yaml --cfg /modles/IDFE-Net-Enhanced version.yaml --epochs 300 --img 640 --device 0,1,2,3

## single-GPU Run train

python train.py  --data /data/GCD-Part-1.yaml ----cfg /modles/IDFE-Net-Enhanced version.yaml --epochs 300 --img 640 --device 0,1,2,3

## Run val

python val.py

## Run detect

python detect.py
