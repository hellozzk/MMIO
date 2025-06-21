# Unification of Closed-Open Industrial detection scenarios: New Large-Scale Benchmarks, Challenges and Baselines

[AAAI2025] The extened version of dataset and code of "Zero-Shot learning in industrial scenarios: New Large-Scale Benchmark, Challenges and Baseline". The paper will be Submitting to IEEE TPAMI.

## The extened version of abstract
Large-scale Visual-Language models (LVLMs) have achieved remarkable success in visual tasks. However, the significant differences between industrial and natural scenes make the application of LVLMs challenging. On the one hand, the scarcity of large-scale industrial defect data makes the application of LVLMs in industrial scenarios unexplored. On the other hand, existing LVLMs rely on user-provided prompts to segment objects. This usually leads to poor performance due to the inclusion of irrelevant semantics. To fill this gap, this paper proposes the first large-scale industrial multi-modal dataset and a refined text-visual prompts network for industrial defect detection (RTVPNet). First, this paper constructs a multi-modal industrial unified open-closed dataset (MMIOC-1M) with 1M+ samples. MMIOC-1M supports open and closed scene tasks and has rich industry categories, including 17 super categories, 31 scenes, and 351 subcategories. MMIOC-1M is the first large-scale multi-modal pre-training dataset for industrial zero-shot and fully supervised learning, providing valuable training data for large models in future industrial scenarios. Based on MMIO-1M, this paper provides RTVPNet specifically for industrial zero-shot, fully supervised learning, and visual question-answering tasks. RTVPNet has two significant advantages: First, we design an expert-guided domain projection mechanism for large models and an industrial zero-shot method based on Mobile-SAM, which enhances the generalization of large models in industrial scenarios. Second, RTVPNet automatically generates refined visual prompts directly from images and considers the text-visual prompt interactions that were ignored by previous LVLM, improving visual and textual content understanding. RTVPNet achieves SOTA with 42.2\% and 24.7\% AP in open and closed scenarios on MMIOC-1M, respectively. Parts of this paper were originally published in its AAAI-2025 conference version. This paper extends earlier work in terms of dataset, network architecture and analysis, etc.

## MMIOC-1M data has been upload to the BaiduDisk. To access the data, please contact 2673679261@qq.com and sign the open source agreement to obtain it. 
## In order to facilitate the verification of RTVPNet accuracy, the verification set of MMIOC-1M is as follows:
## MMIOC-Closed validation dataset link: https://pan.baidu.com/s/114tXzzYIt2B1_ncoImENgg?pwd=ifau 
## MMIOC-Open validation dataset link: https://pan.baidu.com/s/1CfBehOiRNVD6I8GkAje2JQ?pwd=01j5
To the best of our knowledge, this paper constructs the first multi-modal object detection dataset (MMIOC-1M) for industrial open and closed scenes. MMIOC-1M is extended based on MMIO-80K and contains more than 1M samples and 31 industrial scenes, effectively alleviating the lack of domain expertise in industrial open scenes. The comparison with mainstream defect data is as follows:
![MMIOC-1M_dataset_compare](https://github.com/hellozzk/MMIO/blob/main/datasetcompare_01.jpg#pic_center)
The dataset is visualized as follows:
![RTVPNet Architect](https://github.com/hellozzk/MMIO/blob/main/datasetvis_01.jpg#pic_center)
The statistics of the dataset categories are as follows:
![RTVPNet Architect](https://github.com/hellozzk/MMIO/blob/main/statistic_01.jpg#pic_center)
This paper proposes a refined learnable text-visual prompt to improve the detection ability of visual language models in industrial open and closed scenes. Compared with the previous version of RTVP, RTVPNet designs a new text-visual bidirectional prompt interaction and an energy-based refined visual prompt method. RTVPNet automatically provides specific text-visual prompts for each image, which reduces the noise and effectively improves the knowledge and understanding ability of LVLMs in industrial domains.
![RTVPNet Architect](https://github.com/hellozzk/MMIO/blob/main/Architect.png#pic_center)

## MMIOC-1M dataset construct 

```c
Name_of_Dataset
|-- Images
|-----|----- train
|-----|--------|------ defect images
|-----|--------|------ ...
|-----|--------|------ ...
|-----|----- validation
|-----|--------|------ defect images
|-----|--------|------ ...
|-----|--------|------ ...
|-- labels
|-----|----- train
|--------|--------|------txt
|-----|--------|--------|------ defect.txt
|-----|--------|--------|------ ...
|-----|--------|--------|------ ...
|--------|--------|------annotations
|-----|--------|--------|------ train.json
|-----|----- validation
|-----|--------|--------|------ defect.txt
|-----|--------|--------|------ ...
|-----|--------|--------|------ ...
|--------|--------|------annotations
|-----|--------|--------|------ train.json
```

## Code
The code has been open-sourced, and users can choose between the basic version and the more powerful version RTVPNet.

## Requirements
GPU: 8xNVIDIA A100-SXM4-40GB   CPU: Intel(R) Xeon(R) Platinum 8473C      Running Memory: 256GB+      PyTorch 2.1.0      Python 3.9

Command: pip install -r requirements.txt


## Multi-GPU Run train on MMIOC-1M

./tools/dist_train.sh configs/pretrain/Closed.py --device cuda:0 --amp

## single-GPU Run train on MMIOC-1M

./tools/dist_train.sh configs/pretrain/Closed.py --device cuda:0,1,2 --amp

## Run val

bash dist_test.sh --device cuda:0,1,2
