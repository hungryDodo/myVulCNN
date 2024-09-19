## VulCNN: An Image-inspired Scalable Vulnerability Detection System

Since deep learning (DL) can automatically learn features from
source code, it has been widely used to detect source code vulnerability. To achieve scalable vulnerability scanning, some prior studies intend to process the source code directly by treating them as
text. To achieve accurate vulnerability detection, other approaches
consider distilling the program semantics into graph representations and using them to detect vulnerability. In practice, text-based
techniques are scalable but not accurate due to the lack of program
semantics. Graph-based methods are accurate but not scalable since
graph analysis is typically time-consuming.

In this paper, we aim to achieve both scalability and accuracy on
scanning large-scale source code vulnerabilities. Inspired by existing DL-based image classification which has the ability to analyze
millions of images accurately, we prefer to use these techniques
to accomplish our purpose. Specifically, we propose a novel idea
that can efficiently convert the source code of a function into an
image while preserving the program details. We implement VulCNN and evaluate it on a dataset of 13,687 vulnerable functions and
26,970 non-vulnerable functions. Experimental results report that
VulCNN can achieve better accuracy than eight state-of-the-art vulnerability detectors (i.e., Checkmarx, FlawFinder, RATS, TokenCNN,
VulDeePecker, SySeVR, VulDeeLocator, and Devign). As for scalability,
VulCNN is about four times faster than VulDeePecker and SySeVR,
about 15 times faster than VulDeeLocator, and about six times faster
than Devign. Furthermore, we conduct a case study on more than 25 million lines of code and the result indicates that VulCNN can
detect large-scale vulnerability. Through the scanning reports, we
finally discover 73 vulnerabilities that are not reported in NVD.

在本文中，我们的目标是实现可扩展性和准确性 扫描大规模源代码漏洞。灵感来自现有的基于 DL 的图像分类，该分类具有分析能力 数以百万计的图像，我们更喜欢使用这些技术 实现我们的目的。具体来说，我们提出了一个新的想法 可以有效地将函数的源代码转换为 图像，同时保留程序详细信息。我们实现了 VulCNN，并在 13,687 个易受攻击的函数和 26,970 个不易受攻击的函数。实验结果表明 VulCNN 可以达到比八种最先进的漏洞检测器（即 Checkmarx、FlawFinder、RATS、TokenCNN、 VulDeePecker、SySeVR、VulDeeLocator 和 Devign）。至于可扩展性， VulCNN 比 VulDeePecker 和 SySeVR 快四倍， 比 VulDeeLocator 快约 15 倍，快约 6 倍 比德维恩。此外，我们对超过 2500 万行代码进行了案例研究，结果表明 VulCNN 可以 检测大规模漏洞。通过扫描报告，我们 最后发现 NVD 中未报告的 73 个漏洞。

## Design of VulCNN
 <img src="overview.png" width = "800" height = "300" alt="图片名称" align=center />

VulCNN consists of four main phases:
Graph Extraction, Sentence Embedding, Image Generation, and
Classification.

VulCNN由四个主要阶段组成： 图提取、句子嵌入、图像生成和 分类。

1. Graph Extraction: Given the source code of a function,
    we first normalize them and then perform static analysis to
    extract the program dependency graph of the function.

图提取：给定函数的源代码， 我们首先对它们进行归一化，然后执行静态分析，以 提取函数的程序依赖关系图。

1. Sentence Embedding: Each node in the program depen-
    dency graph corresponds to a line of code in the function.
    We regard a line of code as a sentence and embed them into
    a vector.

句子嵌入：程序中的每个节点都 depen- DENCY 图对应于函数中的一行代码。 我们将一行代码视为一个句子，并将它们嵌入到 向量。

1. Image Generation: After sentence embedding, we apply
    centrality analysis to obtain the importance of all lines of code and multiply them by the vectors one by one. The
    output of this phase is an image.

图像生成：句子嵌入后，我们应用 中心性分析，以获得所有代码行的重要性，并将它们逐个乘以向量。这 此阶段的输出是图像。

1. Classification: Our final phase focuses on classification.
    Given generated images, we first train a CNN model and
    then use it to detect vulnerability.

分类：我们的最后阶段侧重于分类。 给定生成的图像，我们首先训练一个 CNN 模型，然后 然后使用它来检测漏洞

## Dataset
We first collect a dataset from Software Assurance Reference Dataset
(SARD) ( https://samate.nist.gov/SRD/index.php) which is a project maintained by National Institute
of Standards and Technology (NIST) (https://www.nist.gov/). SARD contains a large
number of production, synthetic（合成）, and academic security flaws or vulnerabilities (i.e., bad functions) and many good functions. In our
paper, we focus on detecting vulnerability in C/C++, therefore, we
only select functions written in C/C++ in SARD. Data obtained
from SARD consists of 12,303 vulnerable functions and 21,057
non-vulnerable functions. 

我们首先从软件保障参考数据集中收集数据集 （萨德）（https://samate.nist.gov/SRD/index.php）这是国家研究所维护的一个项目 标准与技术 （NIST） （https://www.nist.gov/）。SARD 包含一个大 生产、合成和学术安全缺陷或漏洞（即不良功能）的数量以及许多良好的功能。在我们的 论文中，我们专注于检测 C/C++ 中的漏洞，因此，我们 仅选择在 SARD 中用 C/C++ 编写的函数。获得的数据 来自 SARD 的 12,303 个易受攻击的函数和 21,057 个 不易受攻击的功能。

Moreover, since the synthetic programs
in SARD may not be realistic, we collect another dataset from
real-world software. For real-world vulnerabilities, we consider
National Vulnerability Database (NVD) (https://nvd.nist.gov) as our collection
source. We finally obtain 1,384 vulnerable functions that belong to
different open-source software written in C/C++. For real-world
non-vulnerable functions, we randomly select a part of the dataset
in *"Deep learning-based vulnerable function detection: A benchmark"* which contains non-vulnerable functions from several open-
source projects. Our final dataset consists of 13,687 vulnerable
functions and 26,970 non-vulnerable functions.

此外，由于合成程序 在 SARD 中可能不现实，我们从中收集另一个数据集 真实世界的软件。对于现实世界的漏洞，我们考虑 国家漏洞数据库 （[NVD](https://nvd.nist.gov/)） （https://nvd.nist.gov） 作为我们的集合 源。我们最终获得了 1,384 个易受攻击的函数，这些函数属于 用 C/C++ 编写的不同开源软件。对于现实世界 非易受攻击的函数，我们随机选择数据集的一部分 在*“基于深度学习的易受攻击函数检测：基准测试”*中，其中包含来自多个开放 源项目。我们的最终数据集由 13,687 个易受攻击的人组成 函数和 26,970 个不易受攻击的函数。

## Source Code

#### Step 1: Code normalization

第 1 步：代码规范化

Normalize the code with normalization.py (This operation will overwrite the data file, please <font color="red">make a backup(做备份)</font>)

用 normalization.py 规范化代码（此操作会覆盖数据文件，请做备份）

```
python ./normalization.py -i ./data/sard
```
#### Step 2: Generate pdgs with the help of joern

第 2 步：在 joern 的帮助下生成 pdg

Prepare the environment refering to: [joern](https://github.com/joernio/joern) you can try the version between 1.1.995 to 1.1.1125

准备环境参考： [joern](https://github.com/joernio/joern) 您可以尝试 1.1.995 到 1.1.1125 之间的版本

```powershell
# first generate .bin files
python joern_graph_gen.py  -i ./data/sard/Vul -o ./data/sard/bins/Vul -t parse
python joern_graph_gen.py  -i ./data/sard/No-Vul -o ./data/sard/bins/No-Vul -t parse

#使用下面的全局路径进行 生成.bon files
python joern_graph_gen.py  -i /mnt/f/Code/VulCNN/VulCNN/data/sard/Vul -o /mnt/f/Code/VulCNN/VulCNN/data/sard/bins/Vul -t parse

# then generate pdgs (.dot files)
python joern_graph_gen.py  -i ./data/sard/bins/Vul -o ./data/sard/pdgs/Vul -t export -r pdg
python joern_graph_gen.py  -i ./data/sard/bins/Vul -o ./data/sard/pdgs/No-Vul -t export -r pdg

#全局路径
python joern_graph_gen.py  -i /mnt/f/Code/VulCNN/VulCNN/data/sard/bins/Vul -o /mnt/f/Code/VulCNN/VulCNN/data/sard/pdgs/Vul -t export -r pdg
python joern_graph_gen.py  -i /mnt/f/Code/VulCNN/VulCNN/data/sard/bins/No-Vul -o /mnt/f/Code/VulCNN/VulCNN/data/sard/pdgs/No-Vul -t export -r pdg
```
#### Step 3: Train a sent2vec model

步骤 3：训练 sent2vec 模型

Refer to [sent2vec](https://github.com/epfml/sent2vec#train-a-new-sent2vec-model)

请参阅 [sent2vec](https://github.com/epfml/sent2vec#train-a-new-sent2vec-model)

```
./fasttext sent2vec -input ./data/data.txt -output ./data/data_model -minCount 8 -dim 128 -epoch 9 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000 -maxVocabSize 750000 -numCheckPoints 10
```
(For convenience, we share a simple sent2vec model [here|baidu](https://pan.baidu.com/s/1i4TQP8gSk5_0WlD34yDHwg?pwd=6666) or [here|google](https://drive.google.com/file/d/1p4X4PH9tqFbKByTHGnUiIwtjvmYL8VsL/view?usp=share_link) trained by using our sard dataset. If you want to achieve better performance of VulCNN, you'd better train a new sent2vec by using larger dataset such as Linux Kernel.)

（为方便起见，[我们在这里|百度](https://pan.baidu.com/s/1i4TQP8gSk5_0WlD34yDHwg?pwd=6666)或[这里|谷歌](https://drive.google.com/file/d/1p4X4PH9tqFbKByTHGnUiIwtjvmYL8VsL/view?usp=share_link)使用我们的 sard 数据集训练了一个简单的 sent2vec 模型。如果你想获得更好的 VulCNN 性能，你最好使用更大的数据集（如 Linux 内核）来训练一个新的 sent2vec。

#### Step 4: Generate images from the pdgs

第 4 步：从 pdgs 生成图像

Generate Images from the pdgs with ImageGeneration.py, this step will output a .pkl file for each .dot file.

使用 ImageGeneration.py 从 pdgs 生成图像，此步骤将为每个 .dot 文件输出一个 .pkl 文件。

```
python ImageGeneration.py -i ./data/sard/pdgs/Vul -o ./data/sard/outputs/Vul -m ./data/data_model.bin
python ImageGeneration.py -i ./data/sard/pdgs/No-Vul -o ./data/sard/outputs/No-Vul  -m ./data/data_model.bin

#全局地址
python ImageGeneration.py -i /mnt/f/Code/VulCNN/VulCNN/data/sard/pdgs/Vul -o /mnt/f/Code/VulCNN/VulCNN/data/sard/outputs/Vul -m /mnt/f/Code/VulCNN/VulCNN/data/data_model.bin
python ImageGeneration.py -i /mnt/f/Code/VulCNN/VulCNN/data/sard/pdgs/No-Vul -o /mnt/f/Code/VulCNN/VulCNN/data/sard/outputs/No-Vul  -m /mnt/f/Code/VulCNN/VulCNN/data/data_model.bin
```
#### Step 5: Integrate the data and divide the training and testing datasets

第 5 步：整合数据并划分训练和测试数据集

Integrate the data and divide the training and testing datasets with generate_train_test_data.py, this step will output a train.pkl and a test.pkl file.

集成数据，用generate_train_test_data.py划分训练和测试数据集，此步骤将输出 train.pkl 和 test.pkl 文件。

```powershell
# n denotes the number of kfold, i.e., n=10 then the training set and test set are divided according to 9:1 and 10 sets of experiments will be performed
python generate_train_test_data.py -i ./data/sard/outputs -o ./data/sard/pkl -n 5
```
#### Step 6: Train with CNN

第 6 步：使用 CNN 进行培训

```
python VulCNN.py -i ./data/sard/pkl
```

## Publication
Yueming Wu, Deqing Zou, Shihan Dou, Wei Yang, Duo Xu, and Hai Jin.
2022. VulCNN: An Image-inspired Scalable Vulnerability Detection System.
In 44th International Conference on Software Engineering (ICSE ’22), May
21–29, 2022, Pittsburgh, PA, USA. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3510003.3510229

If you use our dataset or source code, please kindly cite our paper:

如果您使用我们的数据集或源代码，请引用我们的论文：

```
@INPROCEEDINGS{vulcnn2022,
  author={Wu, Yueming and Zou, Deqing and Dou, Shihan and Yang, Wei and Xu, Duo and Jin, Hai},
  booktitle={2022 IEEE/ACM 44th International Conference on Software Engineering (ICSE)}, 
  title={VulCNN: An Image-inspired Scalable Vulnerability Detection System}, 
  year={2022},
  pages={2365-2376},
  doi={10.1145/3510003.3510229}}
```
