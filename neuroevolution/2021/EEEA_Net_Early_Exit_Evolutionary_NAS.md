---

# EEEA-NET: AN EARLY EXIT EVOLUTIONARY NEURAL ARCHITECTURE SEARCH

---

PUBLISHED AT ENGINEERING APPLICATIONS OF ARTIFICIAL INTELLIGENCE. DOI:  
[HTTPS://DOI.ORG/10.1016/J.ENGAPPAI.2021.104397](https://doi.org/10.1016/j.engappai.2021.104397)

**Chakkrit Termritthikun\***  
STEM, University of South Australia  
Adelaide, SA, 5095, Australia  
chakkritt60@nu.ac.th

**Yeshi Jamtsho**  
College of Science and Technology  
Royal University of Bhutan  
Phuentsholing, 21101, Bhutan  
yjamtsho.cst@rub.edu.bt

**Jirarat Ieamsaard**  
Department of Electrical and Computer Engineering  
Faculty of Engineering, Naresuan University  
Phitsanulok, 65000, Thailand  
jirarati@nu.ac.th

**Paisarn Muneesawang**  
Department of Electrical and Computer Engineering  
Faculty of Engineering, Naresuan University  
Phitsanulok, 65000, Thailand  
paisarnmu@nu.ac.th

**Ivan Lee**  
STEM, University of South Australia  
Adelaide, SA, 5095, Australia  
ivan.lee@unisa.edu.au

## ABSTRACT

The goals of this research were to search for Convolutional Neural Network (CNN) architectures, suitable for an on-device processor with limited computing resources, performing at substantially lower Network Architecture Search (NAS) costs. A new algorithm entitled an Early Exit Population Initialisation (EE-PI) for Evolutionary Algorithm (EA) was developed to achieve both goals. The EE-PI reduces the total number of parameters in the search process by filtering the models with fewer parameters than the maximum threshold. It will look for a new model to replace those models with parameters more than the threshold. Thereby, reducing the number of parameters, memory usage for model storage and processing time while maintaining the same performance or accuracy. The search time was reduced to 0.52 GPU day. This is a huge and significant achievement compared to the NAS of 4 GPU days achieved using NSGA-Net, 3,150 GPU days by the AmoebaNet model, and the 2,000 GPU days by the NASNet model. As well, Early Exit Evolutionary Algorithm networks (EEEA-Nets) yield network architectures with minimal error and computational cost suitable for a given dataset as a class of network algorithms. Using EEEA-Net on CIFAR-10, CIFAR-100, and ImageNet datasets, our experiments showed that EEEA-Net achieved the lowest error rate among state-of-the-art NAS models, with 2.46% for CIFAR-10, 15.02% for CIFAR-100, and 23.8% for ImageNet dataset. Further, we implemented this image recognition architecture for other tasks, such as object detection, semantic segmentation, and keypoint detection tasks, and, in our experiments, EEEA-Net-C2 outperformed MobileNet-V3 on all of these various tasks. (The algorithm code is available at <https://github.com/chakkritte/EEEA-Net>).

**Keywords** Deep learning · Neural Architecture Search · Multi-Objective Evolutionary Algorithms · Image classification

---

\*This work is done when Chakkrit Termritthikun works as a visiting research student in University of South Australia## 1 Introduction

Deep convolutional neural networks (CNNs) have been widely used in computer vision applications, including image recognition, image detection, and image segmentation. In the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) Russakovsky et al. [2015], the AlexNet Krizhevsky et al. [2017], GoogLeNet Szegedy et al. [2015], ResNet He et al. [2016], and SENet Hu et al. [2018] were represented models that had been widely used in various applications. The SqueezeNet Iandola et al. [2016], MobileNets Howard et al. [2017], NUF-Net Termritthikun et al. [2019, 2020], and ShuffleNet Zhang et al. [2018] models were simultaneously developed to be used on devices with limited resources. All of these architecture networks have been strengthened and advanced by developers for many years.

However, a significant drawback of the useability of these models, and to the development of the efficient CNNs models, was the dependence on the designer's expertise and experience, including utilising resources such as high-performance computing (HPC) for the experimentation. The datasets used for analysis also affect the model efficiency, depending on the different features that different datasets manifest, and all image recognition datasets require specialised research knowledge for each dataset when modelling. One algorithm, the NAS Zoph and Le [2016] was designed to search the CNN network architecture for different datasets and thereby avoided the hitherto human intervention or design activity except during the definition of the initial hyper-parameters.

The CNN network architecture is flexible, allowing the model to be developed in different structures, with the module structure consisting of layers, linked in sequence with different parameters. Models obtained from NAS methods differ in structure and parameters, making NAS searches more efficient in finding dataset models, resulting in finding a model for each particular dataset which have a unique structure and parameter set.

Reinforcement Learning (RL) and gradient descent algorithms, automate the search for models of deep learning. Searching a model with an RL algorithm takes 2,000 GPU days to uncover an effective model. However, when the gradient descent algorithm is focused on searching a model with the highest accuracy for only one objective; it takes 4 GPU days. Both algorithms have difficulty in dealing with a multi-objective problem. EA, however, can solve multi-objective optimisation problems.

EA apply optimisation methods that mimic the evolution of living organisms in nature, including reproduction, mutation, recombination, and selection. EA can find the most suitable candidate solution with quality function rates. EA-based NAS approaches are very robust with shallow error values for experiments with CIFAR-10 and CIFAR-100 datasets. However, past search algorithms in models of the EA-based NAS approaches have taken up to 3,150 GPU days in Real et al. [2019], 300 GPU days in Liu et al. [2018a], and 4 GPU days in Liu et al. [2018b]. Many network architectures built from NAS have a high number of parameters and high computing costs. It is obvious that extensive network architectures must be avoided when attempting to identify a network architecture that is suitable for other applications.

A network architecture, built from the DARTS Liu et al. [2018b] search space, is a multi-path NAS. However, excessive path-level selection is a problem for multi-path NASs where every operation of the super network of a multi-path NAS takes a separate path. This means that all connected weights across all paths require considerable memory.

The single-path NAS Stamoulis et al. [2019] was introduced to solve the NAS problem by finding a subset of kernel weights in the single layer. This layer is called a super kernel. We called the model built from the super kernel, the Supernet. Based on the Supernet, which has many subnets, whole subnets are trained at the same time through weight sharing. The Supernet can be sampled and deployed without re-training.

In our current work, we developed the EE-PI method for Evolutionary NASs. EE-PI was applied to the Multi-Objective Evolutionary Algorithm (MOEA) in the first generation to locate the newly created model with a certain number of parameters, discarding models with parameters more than the set criteria. This process iteratively continues until the model with fewer parameters than the initial set criteria is found. Thus, EE-PI complements MOEA. We created this add-on to prevent models with a high number of parameters and a high number of Floating point Operations Per Second (FLOPS). Also, using a small number of parameters helps reduce model search costs.

The key contributions of this paper are:

- • We introduce Evolutionary Neural Architecture Search (EA-Net), which adopts a multi-objective evolutionary algorithm for neural architecture search with three objectives: minimisation of errors, minimal number of parameters, and lowest computing costs.
- • We proposed a simple method called Early Exit Population Initialisation (EE-PI) to avoid a model with a high number of parameters and high computational cost, by filtering the neural network architecture based on the number of parameters in the first-generation of the evolution. The architectures obtained by this method are called Early Exit Evolutionary Algorithm networks (EEEA-Net).The diagram illustrates two neural network architectures. On the left, the NASNet architecture is shown as a vertical stack of layers: an initial channel, followed by a sequence of Normal, Reduction, Normal, Reduction, and Normal layers, each Normal layer repeated  $N$  times, and finally a Softmax layer. On the right, the NSGA-Net architecture is shown as a vertical stack of Normal layers. Each Normal layer receives an input from the previous Normal layer (labeled  $x_1$ ) and a skip connection from the input of the previous Normal layer (labeled  $x_2$ ). A dashed box labeled 'Channel increment' indicates that the input  $x_1$  is incremented at each stage.

Figure 1: NASNet Zoph and Le [2016] network architecture (left) and NSGA-Net Lu et al. [2019] network architecture (right).

- • We conduct extensive experiments to evaluate the effectiveness of an EEEA-Net by outperforming MobileNet-V3 for all of the image recognition, object detection, semantic segmentation, and keypoint detection. Also, the EEEA-Net was widely tested on standard CIFAR-10 Krizhevsky [2009], CIFAR-100 Krizhevsky [2009], ImageNet Russakovsky et al. [2015], PASCAL VOC Everingham et al. [2010], Cityscapes Cordts et al. [2016], and MS COCO Lin et al. [2014] datasets.

## 2 Related Work

NAS was designed to search and design the model structure that suits the best to the applied dataset. Thus, the model obtained by NAS has a small number of parameters with high performance. NAS can find models suitable for both small and large datasets. The NAS can be of single-objective NAS and multi-objective NAS: a single-objective NAS considers models from a single objective such as error rate, number of parameters, or FLOPS. The multi-objective NAS considers models considering more than one objective, which we have adopted in this paper to optimise the model performance.

### 2.1 Network Architecture Search

Setting optimised parameters in each layer, such as kernel size, kernel scroll position (stride), zero paddings, as well as the output size, is the main challenge in creating CNN architectures efficiently for a given dataset. The total parameters are directly proportional to the number of layers. Manually designing a model takes too long and requires considerable experimentation to achieve optimal performance, which is why an automated model discovery is essential.

The ability of NASs to find automated models suitable for datasets has proved a popular area of experimentation. Deep learning also is gaining popularity and is now being widely used. The NAS model has also been designed and expanded to enable applications in new tasks such as a NAS for semantic image segmentation, NAS for object detection, and NAS for skin lesion classification. The approaches used in NAS are RL Zoph and Le [2016], EA Real et al. [2019], Liu et al. [2018a], Lu et al. [2019], and relaxation Liu et al. [2018b]. The three critical steps for NAS are:

Step 1: Search the CNN model’s search space to find the suitable architecture. A CNN architecture search model contains many search spaces which are optimised in the image classification application. Google Brain has launched a search space for the NASNet model. It is a feed-forward network in which layers are broken into subgroups called cells. The normal cell can learn from images, and the normal cell maintains an image output equal to the input size. However, the kernel stride in each reduction cell is 2, halving the input image size. Normal and reduction cells are linked. Each cell is stacked during modelling, where  $N$  normal cells are connected. Reduction cells are added between the  $N$  normal cells, as shown in Fig. 1 (left), to halve the image size, helping the next normal cell to process faster.The output from the search space thereby includes normal cells and reduction cells used in the evaluation. NAS has directed acyclic graphs (DAGs) connection between input  $X1$  and  $X2$  of cells, as shown in Fig. 1 (right). In each normal cell, there are two input activations and one output activation. In the first normal cell, input  $X1$  and  $X2$  are copied from the input (image). The next normal cell uses, as input, both the  $X1$  from the last normal cell, and the  $X2$  from the second to last normal cell. All cells are connected the same way until the end of the model. Also, each cell has a greater number of cell output channels in each layer.

Step 2: Evaluate the CNN model on a standard dataset for benchmarks. These benchmarks include number of errors, number of parameters, and search cost. The normal cells and reduction cells that are found in the search space are evaluated to measure the error rate, the number of parameters, and the computing costs, using CIFAR-10. Due to limited processor resources and GPU memory, parameters such as cell count ( $N$ ), number of epochs, initial channel, and channel increment, are different for each search space and evaluation.

Step 3: Evaluate with a large-scale dataset. When the model from the search space has been identified, it is evaluated with a larger dataset. Model evaluation with the CIFAR-10 dataset cannot be compared with other models because the CIFAR-10 dataset contains only ten classes. Given this constraint, CIFAR-100 datasets with 100 classes are required.

The search space of NASNet used RL and was tested with the CIFAR-10 dataset, which takes up to 2,000 GPU days to model. The AmoebaNet [Real et al. \[2019\]](#), based on an evolutionary algorithm model, takes up to 3,150 GPU days for the same dataset. Also, the search space of NASNet was designed to use shorter search times. However, the sequential model-based optimisation (SMBO) method [Liu et al. \[2018c\]](#) takes 335 GPU days, the gradient descent method [Liu et al. \[2018b\]](#) takes just 4 GPU days, whereas weight-sharing across different structures [Pham et al. \[2018\]](#) takes only 0.5 GPU days.

As indicated, the AmoebaNet takes 3,150 GPU days, whereas the NSGA-Net [Lu et al. \[2019\]](#), which uses a multi-objective evolutionary algorithm to find models, takes 4 GPU days. However, although the error rate of NSGA-Net is higher than that of AmoebaNet, based on a standard CIFAR-10 evaluation, the main focus of this area of research has been the reduction of search costs.

## 2.2 Multi-objective Network Architecture Search

NAS aims to minimise errors, hyper-parameters, FLOPS, and delays, making it challenging to identify a network architecture suitable for each objective simultaneously. Thus, the best network architecture should reduce or minimise all of these dimensions. For the evolution-based NASs, NSGA-Nets [Lu et al. \[2019\]](#) considers FLOPS and error count, CARS [Yang et al. \[2020\]](#) and LEMONADE [Elsken et al. \[2018\]](#) consider device-agnostic and device-aware objectives. In our work, however, we sought the achievement of the three goals; minimising errors, and reducing parameters and FLOPS, simultaneously.

The NASs mostly focus on creating a network architecture for image recognition, then transferring that architecture to other tasks. However, for object detection and semantic segmentation, the same network architecture can be used as a backbone.

Many network architectures, such as the EfficientNet [Tan and Le \[2019\]](#), FBNetV2 [Wan et al. \[2020\]](#), DARTS [Liu et al. \[2018b\]](#), P-DARTS [Chen et al. \[2019\]](#), CDARTS [Yu and Peng \[2020\]](#), CARS [Yang et al. \[2020\]](#), LEMONADE [Elsken et al. \[2018\]](#), NSGA-Net [Lu et al. \[2019\]](#) and NSGA-NetV2 [Lu et al. \[2020a\]](#) were tested only on image recognition datasets. It is challenging to design and evaluate a network architecture for general purposes.

Table 1 shows the research objectives of the various NASs, illustrating that the image identification architectures were, in some cases, transferred to object detection, with one, MobileNetV3 [Howard et al. \[2019\]](#) also being applied to transfer specifically researched image identification architecture to both object detection and semantic segmentation.

Our objective was to extend this image identification architecture, using the ImageNet dataset, to object detection, semantic segmentation, as well the further purpose of keypoint detection.

## 2.3 Inverted Residuals Network (IRN)

The Inverted Residuals Network (IRN) [Tan et al. \[2019\]](#) concept is needed to reduce the Residuals Network (RN) parameters. In contrast, the RN concept integrates data from the previous layer into the last layer. Fig. 2 (left) shows that the RN structure has three layers: wide, narrow, and wide approach layers. The wide layers have  $N \times 16$  output channels whereas the narrow layers have  $N \times 16$  channels each. The wide approach layer has  $N \times 32$  output channels ( $N$  is the input channels in each case). However, all the convolution layers used the standard convolution. Batch normalisation (BN) and activation functions (ReLU) were also added into each convolution layer.<table border="1">
<thead>
<tr>
<th>Methods</th>
<th>Search Method</th>
<th>Multiple Objective</th>
<th>Dataset Searched</th>
<th>Architecture transfer<sup>†</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileNetV3 Howard et al. [2019]</td>
<td>RL + expert</td>
<td>-</td>
<td>ImageNet</td>
<td>IR, OD, SS</td>
</tr>
<tr>
<td>EfficientNet Tan and Le [2019]</td>
<td>RL + scaling</td>
<td>-</td>
<td>ImageNet</td>
<td>IR</td>
</tr>
<tr>
<td>FBNetV2 Wan et al. [2020]</td>
<td>gradient</td>
<td>-</td>
<td>ImageNet</td>
<td>IR</td>
</tr>
<tr>
<td>DARTS Liu et al. [2018b]</td>
<td>gradient</td>
<td>-</td>
<td>CIFAR-10</td>
<td>IR</td>
</tr>
<tr>
<td>P-DARTS Chen et al. [2019]</td>
<td>gradient</td>
<td>-</td>
<td>CIFAR-10, CIFAR-100</td>
<td>IR</td>
</tr>
<tr>
<td>PC-DARTS Xu et al. [2020]</td>
<td>gradient</td>
<td>-</td>
<td>CIFAR-10, ImageNet</td>
<td>IR, OD</td>
</tr>
<tr>
<td>CDARTS Yu and Peng [2020]</td>
<td>gradient</td>
<td>-</td>
<td>CIFAR-10, ImageNet</td>
<td>IR</td>
</tr>
<tr>
<td>CARS Yang et al. [2020]</td>
<td>EA</td>
<td>Yes</td>
<td>CIFAR-10</td>
<td>IR</td>
</tr>
<tr>
<td>LEMONADE Elskens et al. [2018]</td>
<td>EA</td>
<td>Yes</td>
<td>CIFAR-10, CIFAR-100, ImageNet64</td>
<td>IR</td>
</tr>
<tr>
<td>NSGA-Net Lu et al. [2019]</td>
<td>EA</td>
<td>Yes</td>
<td>CIFAR-10</td>
<td>IR</td>
</tr>
<tr>
<td>NSGA-NetV2 Lu et al. [2020a]</td>
<td>EA</td>
<td>Yes</td>
<td>ImageNet</td>
<td>IR</td>
</tr>
<tr>
<td>EEEA-Net (this paper)</td>
<td>EA+ EE-PI</td>
<td>Yes</td>
<td>CIFAR-10, ImageNet</td>
<td>IR, OD, SS, KD</td>
</tr>
</tbody>
</table>

<sup>†</sup> IR = Image Recognition, OD = Object Detection, SS = Semantic Segmentation, KD = Keypoint Detection.

Table 1: Comparison of different NAS search method with multi-objectives.

The diagram illustrates the structural differences between a standard Residual Network (RN) and an Inverted Residuals Network (IRN). Both networks take an input  $X_i$  with  $N$  channels.

**Residual Network (Left):** The input  $X_i$  is split into two paths. The main path consists of a 'wide' layer (Conv 1x1, f=Nx16, BN, ReLU) followed by a 'narrow' layer (Conv 3x3, f=Nx16, BN, ReLU), and finally a 'wide approach' layer (Conv 1x1, f=Nx32, BN, ReLU). The output of the main path has  $N \times 32$  channels. A skip connection labeled 'Add' combines the original input  $X_i$  with the output of the main path to produce the final output  $X_{i+1}$ .

**Inverted Residuals Network (Right):** The input  $X_i$  is split into two paths. The main path consists of a 'narrow' layer (Conv 1x1, f=Nx16, BN, ReLU) followed by a 'wide' layer (Depthwise 3x3, BN, ReLU), and finally a 'narrow approach' layer (Conv 1x1, f=N, BN). The output of the main path has  $N$  channels. A skip connection labeled 'Add' combines the original input  $X_i$  with the output of the main path to produce the final output  $X_{i+1}$ .

Figure 2: The difference between the residual network (left) and the inverted residual network (right).

The RN structure is modified and reversed to obtain the IRN. The layers in IRN are defined as narrow layer, wide layer and narrow approach layer. In IRN, the number of output channels obtained is equal to the number of input channels,  $N$ , as shown in Fig. 2 (right). When the data is fed into the  $1 \times 1$  convolution layer, the number of channels will be expanded to  $N \times 16$ . The wide layer changes to a  $3 \times 3$  depth-wise separable convolution instead of a  $3 \times 3$  standard convolution, reducing the FLOPS and number of parameters. There are  $N \times 16$  channels in a wide layer, which is equal to the previous layer. Also,  $1 \times 1$  standard convolution is used in the narrow approach to reduce the channels' size to be equal to the input channel  $N$ . Then, the input data ( $X_i$ ) is combined with the IRN's output to get data ( $X_{i+1}$ ).

Convolutional layers can be defined in various formats to find a model structure with EA. Convolutional layers are also called cells. The types of convolutional layers used in our experiment are presented in Table 2.

### 3 Method

The most commonly used methods to develop NAS are RL and gradient descent algorithms. However, these algorithms possess limitations in solving multi-objective problems. EA automates the model search process, is easier to implement, and enables discovery of solutions while considering multiple objectives.

A general description of an EA, including encoding, a presentation of the multi-objective genetic algorithm, and genetic operations used with NAS, is provided in Section 3.1. The Early Exit Population Initialisation concept and method, its simple, yet effective application in EA, mitigating the complexity and parameters used in the previous models, while maintaining the same accuracy, are described in Section 3.2.

#### 3.1 Evolutionary Neural Architecture Search

The Genetic Algorithm (GA) is an algorithm based on Darwinian concepts of evolution. The GA is part of a random-based EA Xie and Yuille [2017], Baldominos et al. [2017], Real et al. [2017]. GA's search-based solutions stem from the genetic selection of robust members that can survive. The population is integral to GA because GA solutions are<table border="1">
<thead>
<tr>
<th>kernel</th>
<th>Type</th>
<th>Layer</th>
</tr>
</thead>
<tbody>
<tr>
<td>3×3</td>
<td>max and average</td>
<td>pooling</td>
</tr>
<tr>
<td>3×3 and 5×5</td>
<td>depth-wise separable</td>
<td>convolution</td>
</tr>
<tr>
<td>3×3 and 5×5</td>
<td>dilated</td>
<td>convolution</td>
</tr>
<tr>
<td>3×3 and 5×5</td>
<td>inverted residuals</td>
<td>convolution</td>
</tr>
<tr>
<td>-</td>
<td>skip connection</td>
<td>-</td>
</tr>
</tbody>
</table>

 Table 2: Search space of EA-Net and EEEA-Net.

Chromosome =  $LA_1LA_2LB_1LB_2LC_1LC_2LD_1LD_2$   
 = 3080-0121-1202-6373

L = Type of conv layers [0,1,2,3,4,5,6,7,8]  
 A = Index of cells [0]  
 B = Index of cells [0,1]  
 C = Index of cells [0,1,2]  
 D = Index of cells [0,1,2,3]

 Figure 3: The chromosome structure of the Evolutionary Algorithm.

like organisms that evolve after the environment. The most suitable solutions need to rely on genetic diversity. Thus, a greater number of genetically diverse populations enables more effective GA solutions.

The initial phase of GA creates the first population for candidate solutions. The population is determined by population size, which describes total solutions. Each solution is called an individual, where an individual consists of a chromosome. The chromosome is a mix of genes. In the initial population, it is possible to provide unique information about each gene to all other genes, at random. A fitness function computes the fitness value for each individual. CNN’s model structure is searched with the NAS search space, defining error rate as a fitness function where fitness value represents dataset error value. As shown in Equation 1, the fitness value is calculated where  $n$  is the number of individuals.

$$fitness(i) = f(x_i), i = 1, 2, 3, \dots, n \quad (1)$$

Organisms consist of different phenotypes and genotypes. Appearances such as foreign features (such as eye colour) and internal features (such as blood types) are called phenotypes. Genes of different organisms, called genotypes, can be transferred from model to model by gene transfer.

The CNN model’s architecture is represented as a genotype in the NAS search space. A subset of the NAS search space includes normal cells and reduction cells. The cells are stacked in the complete architecture. Normal cells or reduction cells consist of connected layers such as convolution layers, average pooling, max pooling, and skip connection. A complete model must connect cells to create a genotype for training or testing.

### 3.1.1 Encoding

The genotype of the NAS model consists of normal cells and reduction cells, called chromosomes. There are various genes linked to chromosomes. The number of genes is defined as  $LA_1LA_2$ ,  $LB_1LB_2$ ,  $LC_1LC_2$ , and  $LD_1LD_2$  as in Equation 2. The gene consists of operations ( $L$ ) and indices of operations ( $A, B, C, D$ ). Operation ( $L$ ) can be considered a type of CNN layer, such as max pooling, average pooling, depth-wise separable convolution, dilated convolution, inverted residuals block, and skip connection.

$$chromosome(x) = LA_1LA_2, LB_1LB_2, LC_1LC_2, LD_1LD_2 \quad (2)$$

For example, consider nine different operations ( $L$ ) in the experiment, as [0, 1, 2, 3, 4, 5, 6, 7, 8]. Moreover, the operation index ( $A, B, C, D$ ) refers to the operation location to be connected with other operations ( $L$ ). From Fig.3 (left), theindex is defined as follows:  $A = [0]$ ,  $B = [0, 1]$ ,  $C = [0, 1, 2]$ , and  $D = [0, 1, 2, 3]$ . The connection between operations (L) and the operation index ( $A, B, C, D$ ) determines the location of the connection between operations. For example, LA's gene code,  $LA_1LA_2$  is 30,80, meaning output data processed in operation 3 and 8 will be linked to index 0.

Similarly,  $LB_1LB_2$  equals 01,21, meaning data processed by operation 0 and 1 are connected at index 1. However, in the genes of  $LC_1LC_2$  and  $LD_1LD_2$ , those genes are linked sequentially to the output of  $LA_1LA_2$  and  $LB_1LB_2$ , to help reduce the number of model parameters. If the  $LC_1LC_2$  and  $LD_1LD_2$  is connected to the same input as  $LA_1LA_2$ , and  $LB_1LB_2$  (parallel network) will increase the processing time and the parameters.

$$PreviousIndex = index - 2 \quad (3)$$

The position of the previous index can be computed from Equation 3, where the index is greater than 1. If the index is an even number, it is linked to the even previous index; otherwise, it is linked to the odd previous index. Thus, the  $LC_1LC_2$  gene is 12-02, which has an index of 2, and the  $LC_1LC_2$  gene is linked from index 0. While the  $LD_1LD_2$  gene is 63-73, it has an index of 3. The  $LD_1LD_2$  gene is linked from index 1. However, if there are different indices in a gene, for example, a gene 63-72, operator 6 is connected from index 1 and operator 7 is from index 0.

### 3.1.2 Multi-objective Genetic Algorithm

Initially, GA was used for single-objective optimisation problems (SOOP) and, later, GA was developed to solve the multi-objective optimisation problem (MOOP) [Deb et al. \[2002\]](#), which has more than one objective function to minimise fitness values. The GA that can solve the MOOP problems [Carrau et al. \[2017\]](#), [Hasan et al. \[2019\]](#) is called a multi-objective genetic algorithm (MOGA).

$$\begin{aligned} \min \quad & \{f_1(x), f_2(x), \dots, f_k(x)\} \\ \text{s.t.} \quad & x \in X \end{aligned} \quad (4)$$

The optimisation of the CNN model is generally a problem with more than one objective. As illustrated in Equation 4, where  $f$  is fitness values, the integer  $k \geq 2$  is the number of objectives,  $x$  is individual, and  $X$  is the set of individuals. All these objectives must be as small as possible.

Indicators used to measure CNNs model performance include model accuracy, model size, and processing speed. There are three objectives to consider during a model search: lowest validation error, minimum parameters, and computational cost.

$$\begin{aligned} \min \quad & \{Error(x), FLOPS(x), Params(x)\} \\ \text{s.t.} \quad & x \in X \\ & w_{error} + w_{flops} + w_{params} = 1 \\ & w_{error}, w_{flops}, w_{params} \geq 0 \end{aligned} \quad (5)$$

The evolutionary algorithm finds the most effective model for each objective by finding the lowest objective values of the entire population. We defined the three objective values as being equally important. Thus, it is necessary to set the weight of each of the three objective values to 1/3 to find the best model for each value. As illustrated in Equation 5, where  $x$  is individual,  $X$  is the individual's set and  $w_{error}, w_{flops}, w_{params}$  weighs each objective's weight.

For the MOOP problem, it is almost impossible to find one solution that provides the optimal value for each objective function. For each solution given by the MOOP, the best solution group is called nondominated or Pareto optimal because these solutions are compared using the Pareto Domination principle. Many solutions can be obtained from the search using MOOP. These solutions will be reviewed again to find the best solution within the searched solution group.

The best solution group should not dominate when compared to other solutions. For example, any solution  $v$  overwhelming a better solution can be represented as  $v \prec w$ . If no solution  $v$  is worse than solution  $w$ , then solutions  $v$  is better than  $w$ .

### 3.1.3 Genetic Operations

The processes used to create offspring in the new generation are called genetic operations. A new population must replace an ancestor group that cannot survive. The population can be created in two ways, by crossover or mutation.Figure 4 illustrates the Crossover and Mutation operations in the EEEA. The top part shows the Crossover operation, where two parent networks (Parent 1 and Parent 2) are combined to form a child network. The bottom part shows the Mutation operation, where a normal chromosome is replaced by a mutant one.

**Crossover:** Parent 1  $\otimes$  Parent 2 = Child

Parent 1: 40-30-61-31-00-60-42-13  
 Parent 2: 40-30-21-61-22-72-53-11  
 (Child): 40-30-61-61-02-72-52-13

**Mutation:** Normal  $\rightarrow$  Mutant

(Original): 40-30-61-61-02-72-52-13  
 (Mutant): 40-30-61-61-02-33-52-13

Legend:  
 ■ : Common  
 ■ : Parent 1  
 ■ : Parent 2  
 ■ : Common  
 ■ : Normal  
 ■ : Mutant

Figure 4: Crossover operation (top): the parent has two different network architectures; the chromosome of each parent architecture can be visualised as a digits string. Child architecture was mixed with a chromosome index from their parent's chromosome. Mutation operation (bottom): The location of a normal chromosome was randomly selected. Then this pair of genes will be replaced by a new random pair.

Crossover is the creation of a new population by switching genes of different chromosomes from two populations. The genotype of the parent chromosomes will be recombined to create a novel chromosome, which can be done in various ways. For example, a point crossover that performs random cutting points or chromosomes to produce offspring is a crossover between two chromosomes with a random probability of 0.5. Crossover creates offspring using random genes from parents.

Fig. 4 (top) demonstrates the uniform crossover operation used in this implementation, requiring two-parent architectures. We visualised the architectures as follows: 40-30-61-31-00-60-42-13, as parent 1 and 40-30-21-61-22-72-53-11, as parent 2. Then, in the crossover operation, the random probability of 0.5 is defined. The fifty-fifty chance was used to cross the gene between the two parent architectures for child modelling (40-30-61-61-02-72-52-13). The common parent gene is coloured black, but if the gene derived from the first parent is red, then the gene derived from the second parent is represented by blue.

A mutation is an operation to reduce population uniformity and contribute to genetic diversity. The mutation changes data in the gene by randomly locating the gene and replacing the original gene with random new genes. The mutation causes offspring chromosomes to be different from parents. The individual being mutated is called a mutant.

Fig. 4 (bottom) shows the mutation operation used during implementation; the location of a chromosome of architecture was determined by randomly selecting only one pair of gene locations (72, orange). Then it was replaced with a random pair of gene value (33, magenta) of the newly mutated gene.

Figure 5 illustrates the Early Exit Evolutionary Algorithm. The flowchart shows the process from 'Begin' to 'Output', including 'Generate the initial populations', 'Encoding', 'Evaluate populations', 'Multi-Objective Selection', 'Crossover', and 'Mutation'. It also includes an 'Early Exit' loop for 'searching model?' and 'return the model with the less parameter'. A 3D plot shows the 'Output' space with axes for 'Error', 'FLOPs', and 'Params'.

Flowchart:  
 Begin  $\rightarrow$  Generate the initial populations  $\rightarrow$  Encoding  $\rightarrow$  Evaluate populations  $\rightarrow$  Multi-Objective Selection  $\rightarrow$  Crossover  $\rightarrow$  Mutation  $\rightarrow$  Stop ?  
 If No: Loop back to Generate the initial populations.  
 If Yes: Output.

Early Exit:  
 Define maximum of parameters ( $\beta$ )  $\rightarrow$  searching model?  $\rightarrow$  Param  $< \beta$ ?  
 If No: Loop back to searching model?  
 If Yes: return the model with the less parameter

Output Plot:  
 3D plot showing 'Error' (vertical axis), 'FLOPs' (horizontal axis), and 'Params' (depth axis). The plot shows 'Non-dominated' (blue dots) and 'Population' (red dots) points. The axes are labeled 'Minimize'.

Figure 5: An Early Exit Evolutionary Algorithm.---

**Algorithm 1** Multi-objective evolutionary algorithm with an Early Exit Population Initialisation.
 

---

**Input:** The number of generations  $G$ , population size  $n$ , validation dataset  $D$ , objectives  $w$ .

**Output:** A set of  $K$  individuals on the Pareto front.

**Initialisation:** An Early Exit Population Initialisation  $P_1$  and  $Q_1$ .

**for**  $i = 1$  **to**  $G$  **do**

$R_i = P_i \cup Q_i$

**for all**  $p \in R_i$  **do**

Train model  $p$  on  $D$

Evaluate model  $p$  on  $D$

**end for**

$M = \text{tournament-selection}(P_{i+1})$

$F = \text{non-dominated-sorting}(R_i)$

Pick  $n$  individuals to form  $P_{i+1}$  by ranks and the crowding distance **weighted** by  $w$  based on Equation 5

$Q_{i+1} = \text{crossover}(M) \cup \text{mutation}(M)$

**end for**

Select  $K$  models at an equal distance near Pareto front from  $P_G$

---

### 3.2 Early Exit Population Initialisation (EE-PI)

EA can find cell patterns by selecting the best model with the lowest error rate. However, the discovery process takes longer to search and select cells in each generation. Each population has to be trained and evaluated, which increases the time needed to find cell patterns. The single-objective EA uses the error value to find network architecture. However, the network architecture with the lowest error may have too many parameters. In our experiment, the maximum number of generations was set to 30 with 40 populations per generation due to the limited resources of a single GPU.

The population is the only factor affecting processing time. The evaluation must examine the entire population to select the population with the lowest error rate, thus obtaining an effective model structure. However, a longer search time is required to evaluate every population using a single processor unit. Therefore, the EE-PI method was introduced into the evolutionary algorithm to reduce the search time and control the number of parameters, as illustrated in Fig. 5 and detailed in Algorithm 1.

The EE-PI method filters the CNN models based on the number of parameters in the network architecture, which is iteratively compared to a pre-set maximum value ( $\beta$ ). The EE-PI obtains the CNN network architecture which has less parameters than the maximum number of parameters attached to the EA, as illustrated in Fig. 5, which shows the Early Exit as the dashed-line block.

$$\text{EarlyExit}(\alpha, \beta) \triangleq \begin{cases} 1, & \text{if } \alpha \leq \beta \\ 0, & \text{otherwise} \end{cases} \quad (6)$$

In Equation 6, where  $\alpha$  is the parameter of the model which is discovered,  $\beta$  is the maximum number of parameters. If the number of parameters found in the model are less than or equal to the maximum number of parameters ( $\alpha \leq \beta$ ), then the model will be considered as a part of the first-generation population.

For example, to select a network architecture with a maximum of 3 million parameters ( $\beta = 3$ ), EA selects the model by considering the number of parameters lower than the maximum number of parameters. Suppose network architecture is not considered, because it has more than maximum parameters ( $\beta$ ). In this case, it chooses a new structure with less than 3 million parameters. Therefore, in conjunction with the EA in the selection process, Early Exit facilitates the filtering out of the model with the number of parameters greater than the maximum number of parameters. The best network architecture is also discovered using the EA with Early Exit by considering the error rate and the number of parameters.

## 4 Experiments and Results

Experiments were carried out in three parts: First, finding and evaluating the network architecture with EEEA on CIFAR-10 and CIFAR-100. The second part was the finding and evaluation of the EEEA-Net using the ImageNet datasets. In the third part, the EEEA-Net obtained from the second part is applied for other tasks such as object detection, semantic segmentation, and keypoint detection. The PyTorch deep learning library was used in the experiments. Theexperiment was carried out on Intel(R) Xeon(R) W-3235 CPU @ 3.30GHz 12 Core CPU, 192 GB RAM and NVIDIA RTX 2080 Ti GPU, running on the Ubuntu 18.04.3 operating systems.

#### 4.1 CIFAR-10 and CIFAR-100 datasets

This subsection searched for a model with the CIFAR-10 dataset; it was evaluated on CIFAR-10 and CIFAR-100 datasets. Both CIFAR-10 and CIFAR-100 datasets consisted of 60,000 images, with 50,000 images and 10,000 images in the training set and test set, respectively. The CIFAR-10 and CIFAR-100 have 10 and 100 classes, respectively, with 600 images in each class.

##### 4.1.1 Architecture Search on CIFAR-10

Thirty generations with 40 populations in each generation were defined to locate the network architecture with EA. The first-generation populations were randomly generated, with subsequent populations in generations 2-30 being evolved with EA. Each population was defined with a depth of two normal cells instead of the usual six normal cells. Thus, it reduced the search time of the network architecture. The search and evolution process happens more rapidly when Early Exit is used in the initial populations' process. Early Exit selects the network architecture having less than the pre-specified maximum of parameters ( $\beta$ ). Thus, population evolution will choose only network architectures that are efficient and have fewer parameters.

The hyper-parameters for the search process were defined as: the total number of cells (normal cells and reduce cells) equal to eight layers with 32 initial channels by training the network from scratch for one epoch on the CIFAR-10 dataset. The hyper-parameters used included a batch size of 128, with SGD optimiser with weight decay equal to 0.0003 and momentum equal to 0.9. The initial learning rate was 0.05. Using the cosine rule scheduler, the Cutout regularisation had a length set to 16, a drop-path of the probability of 0.2, and the maximum number of parameters equal to 3, 4, and 5 million.

The evolutionary algorithm (EA-Net,  $\beta = 0$ ) took 0.57 GPU days to find the network architecture with NVIDIA RTX 2080 Ti. However, the early exit evolutionary algorithm (EEEA-Net-A,  $\beta = 3$ ) took 0.38 GPU days, EEEA-Net-B ( $\beta = 4$ ) took up to 0.36 GPU days, and EEEA-Net-C ( $\beta = 5$ ) took up to 0.52 GPU days. These architectures are used for performance evaluation in the next section.

##### 4.1.2 Architecture Evaluation on the CIFAR-10 dataset

The network architecture had to be changed to find the normal and reduced cells with the Early Exit evolutionary algorithms. The CIFAR-10 dataset was used for the evaluation. The hyper-parameters were defined with the number of all cells (normal and reduce cells) set to 20 layers with 32 initial channels, the network was trained from scratch with 600 epochs with a batch size of 96, SGD optimiser with weight decay was 0.0003 and momentum 0.9, and the initial learning rate set to 0.025. Using the cosine rule scheduler, the Cutout regularisation had a length set to 16, a drop-path of the probability of 0.2, and auxiliary towers of weight equal to 0.4.

Table 3 shows the comparisons and evaluations of EEEA-Net with other state-of-the-art models. EEEA-Net-C was evaluated with the test dataset, giving an error rate of 2.46% for CIFAR-10. It took 0.52 GPU days to find normal and reduce cells.

By comparison, our EEEA-Net-C model achieved a lower error rate and search time than all of those other models.

AmoebaNet-B was the lowest of the other state-of-the-art models: NASNet-A model, PNAS, both DARTS versions, and NSGA-Net. However, the AmoebaNet-B model required 3,150 GPU days to complete, according to [Real et al. \[2019\]](#). This is clearly a hugely greater amount of search resources than required in our model (0.52 GPU days).

##### 4.1.3 Performance and Computational Complexity Analysis

The multi-objective search tested error rate, number of FLOPS, and parameters. Optimisation on the effectiveness of a multi-objective uses Hypervolume (HV) as a measure of performance that computes the dominated area, using a reference point (Nadir point) with the most significant objective value from the first-generation population. Then, the Pareto-frontier solution computes the area between the reference point and Pareto. The higher HV shows that a multi-objective solution performs better in all objectives.

After the model search, the HV for all the solutions obtained from the search is calculated to compare the performance of two variants: a model that uses Early Exit (EA-Net) and a model that does not use Early Exit. In Fig. 6, the values shown in the vertical axis are normalised HV, and the horizontal axis is generations. When we closely look at the HV value, it was found that the search using the EA-Net model yielded HV values greater than the three EEEA-Net models.<table border="1">
<thead>
<tr>
<th>Architecture</th>
<th>CIFAR-10 Error (%)</th>
<th>CIFAR-100 Error (%)</th>
<th>Params (M)</th>
<th>Search cost (GPU days)</th>
<th>Search Method</th>
</tr>
</thead>
<tbody>
<tr>
<td>NASNet-A + CO Zoph and Le [2016]</td>
<td>2.83</td>
<td>16.58</td>
<td>3.1</td>
<td>2,000</td>
<td>RL</td>
</tr>
<tr>
<td>ENAS + CO Pham et al. [2018]</td>
<td>2.89</td>
<td>17.27</td>
<td>4.6</td>
<td>0.5</td>
<td>RL</td>
</tr>
<tr>
<td>PNAS Liu et al. [2018c]</td>
<td>3.41</td>
<td>17.63</td>
<td>3.2</td>
<td>225</td>
<td>SMBO</td>
</tr>
<tr>
<td>DARTS-V1 + CO Liu et al. [2018b]</td>
<td>2.94</td>
<td>-</td>
<td>2.9</td>
<td>1.5</td>
<td>gradient</td>
</tr>
<tr>
<td>DARTS-V2 + CO Liu et al. [2018b]</td>
<td>2.83</td>
<td>17.54</td>
<td>3.4</td>
<td>4</td>
<td>gradient</td>
</tr>
<tr>
<td>P-DARTS + CO Chen et al. [2019]</td>
<td>2.50</td>
<td>15.92</td>
<td>3.4</td>
<td>0.3</td>
<td>gradient</td>
</tr>
<tr>
<td>PC-DARTS + CO Xu et al. [2020]</td>
<td>2.57</td>
<td>17.36</td>
<td>3.6</td>
<td><b>0.1</b></td>
<td>gradient</td>
</tr>
<tr>
<td>CDARTS + CO Yu and Peng [2020]</td>
<td>2.48</td>
<td>15.69</td>
<td>3.8</td>
<td>0.3</td>
<td>gradient</td>
</tr>
<tr>
<td>AmoebaNet-A + CO Real et al. [2019]</td>
<td>3.12</td>
<td>18.93</td>
<td>3.1</td>
<td>3,150</td>
<td>evolution</td>
</tr>
<tr>
<td>AmoebaNet-B + CO Real et al. [2019]</td>
<td>2.55</td>
<td>-</td>
<td>2.8</td>
<td>3,150</td>
<td>evolution</td>
</tr>
<tr>
<td>LEMONADE Elskens et al. [2018]</td>
<td>3.05</td>
<td>-</td>
<td>4.7</td>
<td>80</td>
<td>evolution</td>
</tr>
<tr>
<td>NSGA-Net + CO Lu et al. [2019]</td>
<td>2.75</td>
<td>20.74</td>
<td>3.3</td>
<td>4</td>
<td>evolution</td>
</tr>
<tr>
<td>CARS-I + CO Yang et al. [2020]</td>
<td>2.62</td>
<td>16.00</td>
<td>3.6</td>
<td>0.4</td>
<td>evolution</td>
</tr>
<tr>
<td>EA-Net (<math>\beta = 0</math>) + CO</td>
<td>3.30</td>
<td>17.58</td>
<td>2.9</td>
<td>0.57</td>
<td>evolution</td>
</tr>
<tr>
<td>EEEA-Net-A (<math>\beta = 3</math>) + CO</td>
<td>3.69</td>
<td>20.16</td>
<td><b>1.8</b></td>
<td>0.34</td>
<td>evolution</td>
</tr>
<tr>
<td>EEEA-Net-B (<math>\beta = 4</math>) + CO</td>
<td>2.88</td>
<td>16.90</td>
<td><b>1.8</b></td>
<td>0.36</td>
<td>evolution</td>
</tr>
<tr>
<td>EEEA-Net-C (<math>\beta = 5</math>) + CO</td>
<td><b>2.46</b></td>
<td><b>15.02</b></td>
<td>3.6</td>
<td>0.52</td>
<td>evolution</td>
</tr>
</tbody>
</table>

Table 3: Comparing EEEA-Net with other architectures from RL, SMBO, gradient, and evolution search method on CIFAR-10 and CIFAR-100 datasets.

Figure 6: Performance Metric of EA-Net and EEEA-Nets.

However, considering only the model with an early exit, it was found that searches using  $\beta$  equal to 5 performed better than  $\beta$  equal to 3 and 4, since  $\beta$  is a parameter that determines the model size by the number of parameters. Consequently, creating larger models by increasing the size of  $\beta$ , gave superior performance.

In addition, when considering a model without an Early Exit (EA-Net) and a model that used an Early Exit ( $\beta=5$ , EEEA-Net-C), it was found that the search efficiency of the EEEA-Net-C model was nearly similar to that of EA-Net because the EA-Net search does not control the model size while searching for the model. Therefore, the model obtained by EA-Net’s search may be likely to obtain a model of a large parameter size. On the other hand, the model with an Early Exit better controls the model size, and the resulting model provides similar performance than achievable in an uncontrolled search.(a) CIFAR-10 Accuracy vs Generations.

(b) CIFAR-10 Accuracy vs FLOPs.

(c) CIFAR-10 Accuracy vs Generations.

(d) CIFAR-10 Accuracy vs FLOPs.

(e) CIFAR-10 Accuracy vs Generations.

(f) CIFAR-10 Accuracy vs FLOPs.

(g) CIFAR-10 Accuracy vs Generations.

(h) CIFAR-10 Accuracy vs FLOPs.

Figure 7: Progress of trade-offs after each generation of EA-Net and EEEA-Nets.We present progress trade-offs after each generation of the EA-Net and EEEA-Nets search through Fig. 7. The whole population is demonstrated by two-dimensional coordinates such as CIFAR-10 accuracy vs. Generations and CIFAR-10 accuracy vs. FLOPS.

#### 4.1.4 Data augmentation

The results from the evaluation of EEEA-Net are represented in precision floating-point (FP32). Our experimental goals were to create the most effective EEEA-Net without modifying the model structure. In the evaluation, we added the AutoAugment (AA) [Cubuk et al. \[2019\]](#) technique to the data augmentation process. AA created a more diverse set of data, making the model more effective. When Cutout [DeVries and Taylor \[2017\]](#) and AA [Cubuk et al. \[2019\]](#) were used, we observed that the error rate of EEEA-Net-C was reduced to 2.42%. Without AA, however, an error rate of 2.46% occurred, as shown in Table 4.

#### 4.1.5 Architecture Evaluation on CIFAR-100 dataset

The AA technique was used to optimise the search and evaluation process of the EEEA-Net. The EEEA-Net-C was trained using the CIFAR-10 dataset, which was not sufficient for our purposes. Consequently, the EEEA-Net architectures obtained from CIFAR-10 dataset were used with CIFAR-100.

The hyper-parameters used in the training process were changed to evaluate the EEEA-Net with the CIFAR-100 dataset, where the number of all cells (normal and reduce cells) was set to 20 layers with 36 initial channels. This was the outcome of training the network from scratch in 600 epochs with a batch size of 128, setting the SGD optimiser with a weight decay of 0.0003 and momentum of 0.9, and the initial learning rate set to 0.025 and running with the cosine rule scheduler. The Cutout regularisation length was equal to 16, and the drop-path of probability was 0.2, with auxiliary towers of the weight of 0.4.

When the EEEA-Net-C (same model structure) was evaluated with CIFAR-100 datasets, it showed an error rate of 15.02%, as shown in Table 3. Further, this evaluation, with 3.6 million parameters, resulted in the lowest error rate of all the state-of-the-art models.

### 4.2 ImageNet dataset

In this subsection, we used the ImageNet dataset for the search and model evaluation. The ImageNet dataset is a large-scale standard dataset for benchmarking performance for image recognition for 1,000 classes with 1,281,167 images for the training set, 50,000 images for the test set, divided into 1,000 classes.

#### 4.2.1 Architecture Search on ImageNet

Early Exit was used to discover a network architecture using the CIFAR-10 dataset. However, this network architecture was constructed from a multi-path NAS, which requires considerable memory. Given this, we used a single-path NAS to find the network architecture on ImageNet to reduce this search time, which also allows a multi-objective search with early exit population initialisation to be used on the OnceForAll [Cai et al. \[2020\]](#) super-network (called Supernet) to discover all network architectures that offer the best trade-off. Supernet can also search for the four dimensions of the network architecture, including kernel size, width (number of channels), depth (number of layers), and input resolution resize. We set all hyper-parameters for our architecture searches following the process in NSGA-NetV2 [Lu et al. \[2020a\]](#).

The two objectives of accuracy and FLOPS were the criteria for searching for 300 high accuracy samples with low FLOPS. However, these sample architectures have a diverse number of parameters. The number of parameters affects

<table border="1">
<thead>
<tr>
<th>Architecture</th>
<th>CIFAR-10 Error (%)</th>
<th>Training time (GPU Hours)</th>
<th>AutoAugment</th>
</tr>
</thead>
<tbody>
<tr>
<td>EEEA-Net-A (<math>\beta = 3</math>) + CO</td>
<td>3.69</td>
<td>25.75</td>
<td>-</td>
</tr>
<tr>
<td>EEEA-Net-B (<math>\beta = 4</math>) + CO</td>
<td>2.88</td>
<td>25.95</td>
<td>-</td>
</tr>
<tr>
<td>EEEA-Net-C (<math>\beta = 5</math>) + CO</td>
<td><b>2.46</b></td>
<td>48.05</td>
<td>-</td>
</tr>
<tr>
<td>EEEA-Net-A (<math>\beta = 3</math>) + CO</td>
<td>3.35</td>
<td>30.38</td>
<td>Yes</td>
</tr>
<tr>
<td>EEEA-Net-B (<math>\beta = 4</math>) + CO</td>
<td>2.87</td>
<td>31.26</td>
<td>Yes</td>
</tr>
<tr>
<td>EEEA-Net-C (<math>\beta = 5</math>) + CO</td>
<td><b>2.42</b></td>
<td>54.26</td>
<td>Yes</td>
</tr>
</tbody>
</table>

Table 4: Results of CIFAR-10 using Cutout (CO) and AutoAugment (AA).architecture size when running on devices that may have memory constraints. Thus, to prevent the architecture from having too many parameters, we appended the Early Exit to create the first population with limited parameters.

In this experiment, we compiled the number of architecture parameters shown in Table 5 to calculate the average number of parameters equal to 5. Thus, the maximum number of parameters ( $\beta$ ) where  $\beta$  equals 5, 6 or 7, was defined as follows: EEEA-Net-A ( $\beta = 5$ ), EEEA-Net-B ( $\beta = 6$ ), EEEA-Net-C ( $\beta = 7$ ). For a fair comparison, we set  $\beta$  equal to 0, and called that EA-Net-N ( $\beta = 0$ ). We categorised our networks using the number of MobilenetV3 FLOPS to define the network size architectures of EEEA-Net-A1, EEEA-Net-B1, EEEA-Net-C1, and EEEA-Net-N1 as small-scale architectures ( $<155$  FLOPS). The large-scale architectures ( $<219$  FLOPS) are EEEA-Net-A2, EEEA-Net-B2, EEEA-Net-C2, and EEEA-Net-N2.

#### 4.2.2 Architecture Evaluation on ImageNet dataset

The discovery of architecture from Supernet is the separation of some layers from Supernet called subnets. Since Supernet and the subnets have different network architectures, the accuracy of the subnets, with pre-trained weight from Supernet, is very low when they were tested on the validation dataset. So, the subnets have calibrated the statistics of batch normalisation (BN) after searching on Supernet. The new BN statistics from the subnets were calculated using the validation dataset and updating the BN of all of the subnets. Thus, BN calibration can improve test accuracy value efficiency with the ImageNet dataset.

Table 5 shows the comparison of EEEA-Net performance with other models, using three main comparison factors: error rate, number of parameters, and FLOPS. We classify models using architectural search methods such as auto, manual, or a combination. When comparing our small-scale architectures (EEEA-Net-A1, EEEA-Net-B1, EEEA-Net-C1, and EEEA-Net-N1) with GhostNet 1.0 Han et al. [2020], we found that all our architectures outperform GhostNet 1.0. Also, EEEA-Net-A1, EEEA-Net-B1, EEEA-Net-C1, and EEEA-Net-N1 provide lower error and FLOPS counts than MobileNetsV3 Large 0.75 Howard et al. [2019]. However, the MobileNetsV3 Large 0.75 has fewer parameters than our models.

Similarly, when we compared our large-scale architectures with other architectures, we found that EEEA-Net-C2 ( $\beta = 7$ ) has a Top-1 error, and FLOPS were lower than all other architectures, as shown in Table 4. When we compare our architecture with MobileNetsV3 Large 1.0, EEEA-Net-C2 provides a 1% less error value than MobileNetsV3 [28], and the FLOPS count of EEEA-Net-C2 is reduced by 2 Million from MobileNetsV3. However, EEEA-Net-C2 had 0.6 Million more parameters than MobileNetsV3.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Top-1 Error (%)</th>
<th>Top-5 Error (%)</th>
<th>Params (M)</th>
<th>FLOPS (M)</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td>GhostNet 1.0 Han et al. [2020]</td>
<td>26.1</td>
<td>8.6</td>
<td>5.2</td>
<td>141</td>
<td>manual</td>
</tr>
<tr>
<td>MobileNetsV3 Large 0.75 Howard et al. [2019]</td>
<td>26.7</td>
<td>-</td>
<td><b>4.0</b></td>
<td>155</td>
<td>combined</td>
</tr>
<tr>
<td>EA-Net-N1 (ours)</td>
<td>26.1</td>
<td>8.6</td>
<td><b>4.4</b></td>
<td>140</td>
<td>auto</td>
</tr>
<tr>
<td>EEEA-Net-A1 (<math>\beta = 5</math>)(our)</td>
<td>26.3</td>
<td>8.8</td>
<td>5.0</td>
<td><b>127</b></td>
<td>auto</td>
</tr>
<tr>
<td>EEEA-Net-B1 (<math>\beta = 6</math>) (our)</td>
<td>26.0</td>
<td><b>8.5</b></td>
<td>5.0</td>
<td><b>138</b></td>
<td>auto</td>
</tr>
<tr>
<td>EEEA-Net-C1 (<math>\beta = 7</math>) (our)</td>
<td><b>25.7</b></td>
<td><b>8.5</b></td>
<td>5.1</td>
<td>137</td>
<td>auto</td>
</tr>
<tr>
<td>MobileNetsV1 Howard et al. [2017]</td>
<td>29.4</td>
<td>-</td>
<td>4.2</td>
<td>575</td>
<td>manual</td>
</tr>
<tr>
<td>MobileNetsV2 Sandler et al. [2018]</td>
<td>28.0</td>
<td>-</td>
<td>3.4</td>
<td>300</td>
<td>manual</td>
</tr>
<tr>
<td>GhostNet 1.3 Han et al. [2020]</td>
<td>24.3</td>
<td>7.3</td>
<td>7.3</td>
<td>226</td>
<td>manual</td>
</tr>
<tr>
<td>MobileNetsV3 Large 1.0 Howard et al. [2019]</td>
<td>24.8</td>
<td>-</td>
<td>5.4</td>
<td>219</td>
<td>combined</td>
</tr>
<tr>
<td>NASNet-A Zoph and Le [2016]</td>
<td>26.0</td>
<td>8.4</td>
<td>5.3</td>
<td>564</td>
<td>auto</td>
</tr>
<tr>
<td>MnasNet-A1 Tan et al. [2019]</td>
<td>24.8</td>
<td>7.5</td>
<td><b>3.9</b></td>
<td>312</td>
<td>auto</td>
</tr>
<tr>
<td>FBNet-C Wu et al. [2019]</td>
<td>25.1</td>
<td>-</td>
<td>5.5</td>
<td>375</td>
<td>auto</td>
</tr>
<tr>
<td>MOGA-A Chu et al. [2020]</td>
<td>24.1</td>
<td><b>7.2</b></td>
<td>5.1</td>
<td>304</td>
<td>auto</td>
</tr>
<tr>
<td>FairNAS-A Chu et al. [2019]</td>
<td>24.7</td>
<td>7.6</td>
<td>4.6</td>
<td>388</td>
<td>auto</td>
</tr>
<tr>
<td>PNASNet-5 Liu et al. [2018c]</td>
<td>25.8</td>
<td>8.1</td>
<td>5.1</td>
<td>588</td>
<td>auto</td>
</tr>
<tr>
<td>NSGANetV1-A2 Lu et al. [2020b]</td>
<td>25.5</td>
<td>8.0</td>
<td>4.1</td>
<td>466</td>
<td>auto</td>
</tr>
<tr>
<td>OnceForAll Cai et al. [2020]</td>
<td>24.0</td>
<td>-</td>
<td>6.1</td>
<td>230</td>
<td>auto</td>
</tr>
<tr>
<td>MSuNAS Cai et al. [2020]</td>
<td>24.1</td>
<td>-</td>
<td>6.1</td>
<td>225</td>
<td>auto</td>
</tr>
<tr>
<td>EA-Net-N2 (ours)</td>
<td>24.4</td>
<td>7.6</td>
<td>5.9</td>
<td>226</td>
<td>auto</td>
</tr>
<tr>
<td>EEEA-Net-A2 (<math>\beta = 5</math>) (our)</td>
<td>24.1</td>
<td>7.4</td>
<td>5.6</td>
<td><b>198</b></td>
<td>auto</td>
</tr>
<tr>
<td>EEEA-Net-B2 (<math>\beta = 6</math>) (our)</td>
<td>24.0</td>
<td>7.5</td>
<td>5.7</td>
<td><b>219</b></td>
<td>auto</td>
</tr>
<tr>
<td>EEEA-Net-C2 (<math>\beta = 7</math>) (our)</td>
<td><b>23.8</b></td>
<td>7.3</td>
<td>6.0</td>
<td><b>217</b></td>
<td>auto</td>
</tr>
</tbody>
</table>

Table 5: Comparing EEEA-Net with other architectures from manual, combined, and auto search method on ImageNet datasets.Figure 8: Comparison of Top-1 accuracy, FLOPS (left), and parameters (right) between EEEA-Nets and MobileNetV3 [28] and GhostNet [40] on ImageNet dataset.

We chose MobileNetV3 and GhostNet, including both small and large versions, to compare with our architecture, as shown in Fig. 8. Overall, we observed that EEEA-Net-C ( $\beta = 7$ ) significantly outperforms MobileNetV3 and GhostNet for Top-1 accuracy and FLOPS. Furthermore, EEEA-Net-C ( $\beta = 7$ ) achieves lower parameters than GhostNet.

### 4.3 Architecture Transfer

After searching and evaluating the model using the ImageNet dataset for image recognition, the models trained with the ImageNet dataset can be further developed and applied to object detection, semantic segmentation and human keypoint detection applications.

#### 4.3.1 Object detection

EEEA-Net-C2 ( $\beta = 7$ ) was used as the backbone for the object detection task to compare the effectiveness of our architecture on a real-world application. We utilised the same architecture trained with ImageNet datasets on the firmware called Single-Shot Detectors (SSD) Liu et al. [2016] and You Only Look Once version four (YOLOv4) Bochkovskiy et al. [2020].

PASCAL VOC is a standard set of data used to measure an architecture’s performance with object detection datasets. It consists of 20 classes, with bottles and plants being small objects with the lowest Average Precision (AP) of all classes. We used the SSDLite framework Sandler et al. [2018] for fast and optimised processing on mobile devices. We also used the YOLOv4 framework Bochkovskiy et al. [2020] for high precision object detection.

All models were trained on the PASCAL VOC 2007 and VOC 2012 Everingham et al. [2010] train set by training the network for 200 epochs with a batch size of 32, SGD optimiser with weight decay equal to 0.0005 and momentum equal to 0.9, the initial learning rate is 0.01. It uses the scheduler with the cosine rule without a restart. All input images are resized to  $320 \times 320$  pixels, and these models were used to evaluate the PASCAL VOC test set.

For YOLOv4, we adopted MobileNet-V2 Sandler et al. [2018], MobileNet-V3 Howard et al. [2019] and EEEA-Net-C2 as the backbone of YOLOv4. All models were trained with 140 epochs and a batch size of 4. All inputs are random scales with multi-scale images ranging from 320 to 640 pixels. The label smoothing is 0.1, SGD optimiser with weight decay equal to 0.0005 and momentum equal to 0.9. The initial learning rate was set to 0.01 for a scheduler with a cosine rule with the warm-up strategy performed twice.

Table 6 shows the performance of our architecture for object detection. EEEA-Net-C2 achieved a higher AP than NAS-Net, DARTS, ShuffleNet-V2, MobileNet-V2, MobileNet-V3, and MnasNet for the SSDLite framework. Nonetheless, EEEA-Net-C2 has 152 more million FLOPS than MobileNet-V3. For fairness, we used MobileNet-V2, MobileNet-V3 and EEEA-Net-C2 for training and evaluated these models using the PASCAL VOC test dataset via the YOLOv4 framework. The EEEA-Net-C2 significantly outperformed both MobileNet-V2 and MobileNet-V3.<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Framework</th>
<th rowspan="2">Params (M)</th>
<th rowspan="2">FLOPS (M)</th>
<th colspan="2">Small Objects AP (%)</th>
<th rowspan="2">VOC2007 mAP (%)</th>
</tr>
<tr>
<th>Bottle</th>
<th>Plant</th>
</tr>
</thead>
<tbody>
<tr>
<td>NASNet <a href="#">Zoph and Le [2016]</a></td>
<td>SSDLite</td>
<td>5.22</td>
<td>1238</td>
<td>41.5</td>
<td>46.1</td>
<td>71.6</td>
</tr>
<tr>
<td>DARTS <a href="#">Liu et al. [2018b]</a></td>
<td>SSDLite</td>
<td>4.73</td>
<td>1138</td>
<td>38.3</td>
<td>49.3</td>
<td>71.2</td>
</tr>
<tr>
<td>ShuffleNet-V2 <a href="#">Ma et al. [2018]</a></td>
<td>SSDLite</td>
<td><b>2.17</b></td>
<td><b>355</b></td>
<td>29.9</td>
<td>38.1</td>
<td>65.4</td>
</tr>
<tr>
<td>MobileNet-V2 <a href="#">Sandler et al. [2018]</a></td>
<td>SSDLite</td>
<td>3.30</td>
<td>680</td>
<td>37.9</td>
<td>43.9</td>
<td>69.4</td>
</tr>
<tr>
<td>MobileNet-V3 <a href="#">Howard et al. [2019]</a></td>
<td>SSDLite</td>
<td>3.82</td>
<td>485</td>
<td>38.1</td>
<td>45.6</td>
<td>69.2</td>
</tr>
<tr>
<td>MnasNet <a href="#">Tan et al. [2019]</a></td>
<td>SSDLite</td>
<td>4.18</td>
<td>708</td>
<td>37.7</td>
<td>44.4</td>
<td>69.6</td>
</tr>
<tr>
<td>EEEA-Net-C2 (ours)</td>
<td>SSDLite</td>
<td>5.57</td>
<td>637</td>
<td><b>40.9</b></td>
<td><b>48.9</b></td>
<td><b>71.7</b></td>
</tr>
<tr>
<td>MobileNet-V2 <a href="#">Sandler et al. [2018]</a></td>
<td>YOLOv4</td>
<td>46.34</td>
<td>8740</td>
<td>66.4</td>
<td><b>58.4</b></td>
<td>81.5</td>
</tr>
<tr>
<td>MobileNet-V3 <a href="#">Howard et al. [2019]</a></td>
<td>YOLOv4</td>
<td>47.30</td>
<td>8520</td>
<td>68.2</td>
<td>50.7</td>
<td>78.9</td>
</tr>
<tr>
<td>EEEA-Net-C2 (ours)</td>
<td>YOLOv4</td>
<td><b>31.15</b></td>
<td><b>5540</b></td>
<td><b>68.6</b></td>
<td>56.7</td>
<td><b>81.8</b></td>
</tr>
</tbody>
</table>

Table 6: Result of Object detection with different backbones on PASCAL VOC 2007 test set.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Params (M)</th>
<th>FLOPS (G)</th>
<th>mIoU (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>NASNet <a href="#">Zoph and Le [2016]</a></td>
<td>7.46</td>
<td>36.51</td>
<td>77.9</td>
</tr>
<tr>
<td>DARTS <a href="#">Liu et al. [2018b]</a></td>
<td>6.64</td>
<td>34.77</td>
<td>77.5</td>
</tr>
<tr>
<td>ShuffleNet-V2 <a href="#">Ma et al. [2018]</a></td>
<td><b>4.10</b></td>
<td><b>26.30</b></td>
<td>73.0</td>
</tr>
<tr>
<td>MobileNet-V2 <a href="#">Sandler et al. [2018]</a></td>
<td>5.24</td>
<td>29.21</td>
<td><b>77.1</b></td>
</tr>
<tr>
<td>MobileNet-V3 <a href="#">Howard et al. [2019]</a></td>
<td>5.60</td>
<td>27.09</td>
<td>75.9</td>
</tr>
<tr>
<td>MnasNet <a href="#">Tan et al. [2019]</a></td>
<td>6.12</td>
<td>29.50</td>
<td>76.8</td>
</tr>
<tr>
<td>EEEA-Net-C2 (ours)</td>
<td>7.34</td>
<td>28.65</td>
<td>76.8</td>
</tr>
</tbody>
</table>

Table 7: Results of BiSeNet with different backbones on Cityscapes validation set. (single scale and no flipping).

### 4.3.2 Semantic Segmentation

The cityscape dataset [Cordts et al. \[2016\]](#) was chosen to experiment with semantic segmentation. It is a large-scale dataset of street scenes in 50 cities. Cityscapes provide dense pixel annotations of 5,000 images. These images were divided into three groups of 2,975, 500, 1,525 images for training, validation, and testing. We used BiSeNet [Yu et al. \[2018\]](#) with different backbones to evaluate our architecture’s performance for semantic segmentation on the Cityscapes dataset. NASNet, DARTS, ShuffleNet-V2, MobileNet-V2, MobileNet-V3, MnasNet, and EEEA-Net-C2 have trained 80,000 iterations with a poly learning scheduler at an initial learning rate of 0.01 and batch size equals 16. All training images were resized to  $1024 \times 1024$  pixels using image augmentation using colour jitter, random scale, and random horizontal flip.

Table 7 shows that ShuffleNet-V2 achieved a smaller number of parameters and lower FLOPS than other architectures. However, MobileNet-V2 achieved a greater Mean Intersection over Union (mIoU) than ShuffleNet-V2, MobileNet-V3, MnasNet, and EEEA-Net-C2. The mIoU of EEEA-Net-C2 is the same as MnasNet. It is better than ShuffleNet-V2 and MobileNet-V3.

### 4.3.3 Keypoint Detection

Human keypoint detection, also known as human pose estimation, is the visual sensing of human gestures from keypoints such as the head, hips, or ankles. MS COCO [Lin et al. \[2014\]](#) is a comprehensive dataset to measure keypoint detection performance, consisting of data for 250,000 persons, with the data labelled at 17 keypoints. SimpleBaseline [Xiao et al. \[2018\]](#) is a framework for keypoint detection, enabling easier changes to backbones. Given this, it allowed us to adapt to other architectures more simply.

All architectures were trained on the MS COCO train2017 set by training the network for 140 epochs with a batch size of 128, Adam optimiser, the initial learning rate is 0.001, which is reduced to 0.0001 at 90th epoch and reduced to 0.00001 at 120th epoch. The training set is resized to  $256 \times 192$  pixels using random rotation, scale, and flipping data.

Table 8 shows the experimental result of SimpleBaseline with different backbones. Our EEEA-Net-C2 performed better than other backbones in the number of parameters. As well, EEEA-Net-C2 outperforms small architectures (excluding NASNet and DARTS).<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Params (M)</th>
<th>FLOPS (M)</th>
<th>AP (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>NASNet <a href="#">Zoph and Le [2016]</a></td>
<td>10.66</td>
<td>569.11</td>
<td>67.9</td>
</tr>
<tr>
<td>DARTS <a href="#">Liu et al. [2018b]</a></td>
<td>9.20</td>
<td>531.77</td>
<td>66.9</td>
</tr>
<tr>
<td>ShuffleNet-V2 <a href="#">Ma et al. [2018]</a></td>
<td>7.55</td>
<td><b>154.37</b></td>
<td>60.4</td>
</tr>
<tr>
<td>MobileNet-V2 <a href="#">Sandler et al. [2018]</a></td>
<td>9.57</td>
<td>306.80</td>
<td>64.9</td>
</tr>
<tr>
<td>MobileNet-V3 <a href="#">Howard et al. [2019]</a></td>
<td>9.01</td>
<td>223.16</td>
<td>65.3</td>
</tr>
<tr>
<td>MnasNet <a href="#">Tan et al. [2019]</a></td>
<td>10.45</td>
<td>320.17</td>
<td>62.5</td>
</tr>
<tr>
<td>EEEA-Net-C2 (ours)</td>
<td><b>7.47</b></td>
<td>297.49</td>
<td><b>66.7</b></td>
</tr>
</tbody>
</table>

Table 8: Results of SimpleBaseline with different backbone settings on MS COCO2017 validation set. Flip is used during validation.

#### 4.4 Limitations

The development of a NAS search with only one GPU processor was a challenge for the reasons set out below. Setting the appropriate number of populations, the number of generations of each population, and the number of search epochs suitable for one GPU processor presents considerable difficulties. All of these parameters affect the model search time.

Increasing the number of generations increases the computing cost, but increasing the number of generations provides an opportunity for greater recombination of populations, thereby maximising the efficiency of discovering new populations. Moreover, an increased number of search epoch helps improve each population’s error fitness value.

All these restrictions help to improve the NAS search. However, increasing these numbers influences search time. For example, increasing the number of search epochs from 1 epoch to 10 epochs results in a  $10\times$  increase in search time.

## 5 Conclusion

We achieved our research goals by successfully developing a CNN architecture suitable for an on-device processor with limited computing resources and applying it in real-world applications.

This outcome was achieved by significantly reducing the computational cost of a neural architecture search. We introduced the Early Exit Population Initialisation (EE-PI) for Evolutionary Algorithm method to create the EEEA-Nets model. Our method achieved a massive reduction in search time on CIFAR-10 dataset; 0.34 to 0.52 GPU days. This must be seen as an outstanding outcome compared against other state-of-the-art models, such as the NSGA-Net model, which required 4 GPU days, the 2,000 GPU days of the NASNet model and the 3,150 GPU days of the AmoebaNet model.

In the EEEA-Nets architecture, our emphasis was on reducing the number of parameters, the error rate and the computing cost. We were able to achieve this by introducing an Early Exit step into the Evolutionary Algorithm.

Our EEEA-Nets architectures were searched on image recognition task, transferring architectures to other tasks. Experimentally, EEEA-Net-C2 is significantly better than MobileNet-V3 on image recognition, object detection, semantic segmentation, and keypoint detection tasks. Addressing this latter task had not been achieved or even attempted in any other CNN model. Therefore, our architectures can be deployed on devices with limited memory and processing capacity by achieving these significant reductions, allowing real-time processing on smartphones or on-device systems.

The task of optimising the search for multi-objective evolutionary algorithms shall be continued as our future work to find better-performing models. In addition, we will consider applying a multi-objective evolutionary algorithm with EE-PI to find mobile-suitable models in other applications such as marine detection or pest detection.

## Acknowledgements

The authors would like to acknowledge the Thailand Research Fund’s financial support through the Royal Golden Jubilee PhD. Program (Grant No. PHD/0101/2559). The study was undertaken using the National Computational Infrastructure (NCI) in Australia under the National Computational Merit Allocation Scheme (NCMAS). Further, we would like to extend our appreciation to Mr Roy I. Morien of the Naresuan University Graduate School for his assistance in editing the English grammar, syntax, and expression in the paper.## References

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3):211–252, 2015.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. *Communications of The ACM*, 60(6):84–90, 2017.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In *2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1–9, 2015.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 770–778, 2016.

Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7132–7141, 2018.

Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and <0.5mb model size. *arXiv preprint arXiv:1602.07360*, 2016.

Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*, 2017.

Chakkrit Termritthikun, Yeshi Jamtsho, and Paisarn Muneesawang. On-device facial verification using nuf-net model of deep learning. *Engineering Applications of Artificial Intelligence*, 85:579–589, 2019.

Chakkrit Termritthikun, Yeshi Jamtsho, and Paisarn Muneesawang. An improved residual network model for image recognition using a combination of snapshot ensembles and the cutout technique. *Multimedia Tools and Applications*, 79(1):1475–1495, 2020.

Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6848–6856, 2018.

Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In *ICLR*, 2016.

Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Regularized evolution for image classifier architecture search. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, pages 4780–4789, 2019.

Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, and Koray Kavukcuoglu. Hierarchical representations for efficient architecture search. In *International Conference on Learning Representations*, 2018a.

Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. In *International Conference on Learning Representations*, 2018b.

Dimitrios Stamouli, Ruizhou Ding, Di Wang, Dimitrios Lymberopoulos, Bodhi Priyantha, Jie Liu, and Diana Marculescu. Single-path nas: Designing hardware-efficient convnets in less than 4 hours. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*, pages 481–497, 2019.

Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.

Mark Everingham, Luc Gool, Christopher K. Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. *International Journal of Computer Vision*, 88(2):303–338, 2010.

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 3213–3223, 2016.

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In *European Conference on Computer Vision*, pages 740–755, 2014.

Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman, and Wolfgang Banzhaf. Nsga-net: neural architecture search using multi-objective genetic algorithm. In *Proceedings of the Genetic and Evolutionary Computation Conference on*, pages 419–427, 2019.

Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan L. Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural architecture search. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 19–35, 2018c.Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, and Jeff Dean. Efficient neural architecture search via parameter sharing. *arXiv preprint arXiv:1802.03268*, 2018.

Zhaohui Yang, Yunhe Wang, Xinghao Chen, Boxin Shi, Chao Xu, Chunjing Xu, Qi Tian, and Chang Xu. Cars: Continuous evolution for efficient neural architecture search. In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1829–1838, 2020.

Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Efficient multi-objective neural architecture search via lamarckian evolution. In *International Conference on Learning Representations*, 2018.

Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In *International Conference on Machine Learning*, pages 6105–6114, 2019.

Alvin Wan, Xiaoliang Dai, Peizhao Zhang, Zijian He, Yuandong Tian, Saining Xie, Bichen Wu, Matthew Yu, Tao Xu, Kan Chen, Peter Vajda, and Joseph E. Gonzalez. Fbnetv2: Differentiable neural architecture search for spatial and channel dimensions. In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 12965–12974, 2020.

Xin Chen, Lingxi Xie, Jun Wu, and Qi Tian. Progressive differentiable architecture search: Bridging the depth gap between search and evaluation. In *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 1294–1303, 2019.

Hongyuan Yu and Houwen Peng. Cyclic differentiable architecture search. *arXiv preprint arXiv:2006.10724*, 2020.

Zhichao Lu, Kalyanmoy Deb, Erik D. Goodman, Wolfgang Banzhaf, and Vishnu Naresh Boddeti. Nsganetv2: Evolutionary multi-objective surrogate-assisted neural architecture search. In *European Conference on Computer Vision*, pages 35–51, 2020a.

Andrew Howard, Ruoming Pang, Hartwig Adam, Quoc Le, Mark Sandler, Bo Chen, Weijun Wang, Liang-Chieh Chen, Mingxing Tan, Grace Chu, Vijay Vasudevan, and Yukun Zhu. Searching for mobilenetv3. In *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 1314–1324, 2019.

Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi, Qi Tian, and Hongkai Xiong. Pc-darts: Partial channel connections for memory-efficient architecture search. In *ICLR 2020 : Eighth International Conference on Learning Representations*, 2020.

Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V. Le. Mnasnet: Platform-aware neural architecture search for mobile. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 2820–2828, 2019.

Lingxi Xie and Alan Yuille. Genetic cnn. In *2017 IEEE International Conference on Computer Vision (ICCV)*, 2017.

Alejandro Baldominos, Yago Saez, and Pedro Isasi. Evolutionary convolutional neural networks: An application to handwriting recognition. *Neurocomputing*, 283:38–52, 2017.

Esteban Real, Sherry Moore, Andrew Selle, Saurabh Saxena, Yutaka Leon Suematsu, Jie Tan, Quoc V. Le, and Alexey Kurakin. Large-scale evolution of image classifiers. In *ICML'17 Proceedings of the 34th International Conference on Machine Learning - Volume 70*, pages 2902–2911, 2017.

K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A fast and elitist multiobjective genetic algorithm: Nsga-ii. *IEEE Transactions on Evolutionary Computation*, 6(2):182–197, 2002.

Jesús Velasco Carrau, Gilberto Reynoso-Meza, Sergio García-Nieto, and Xavier Blasco. Enhancing controller's tuning reliability with multi-objective optimisation: From model in the loop to hardware in the loop. *Engineering Applications of Artificial Intelligence*, 64:52–66, 2017.

Mahmudul Hasan, Khin Lwin, Maryam Imani, Antesar M. Shabut, Luiz F. Bittencourt, and Mohammed Alamgir Hossain. Dynamic multi-objective optimisation using deep reinforcement learning: benchmark, algorithm and an application to identify vulnerable zones based on water quality. *Engineering Applications of Artificial Intelligence*, 86:107–135, 2019.

Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V. Le. Autoaugment: Learning augmentation strategies from data. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 113–123, 2019.

Terrance DeVries and Graham W. Taylor. Improved regularization of convolutional neural networks with cutout. *arXiv preprint arXiv:1708.04552*, 2017.

Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, and Chang Xu. Ghostnet: More features from cheap operations. In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1580–1589, 2020.Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4510–4520, 2018.

Bichen Wu, Kurt Keutzer, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, and Yangqing Jia. Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 10734–10742, 2019.

Xiangxiang Chu, Bo Zhang, and Ruijun Xu. Moga: Searching beyond mobilenetv3. In *ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 4042–4046, 2020.

Xiangxiang Chu, Bo Zhang, Ruijun Xu, and Jixiang Li. Fairnas: Rethinking evaluation fairness of weight sharing neural architecture search. *arXiv preprint arXiv:1907.01845*, 2019.

Zhichao Lu, Ian Whalen, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman, Wolfgang Banzhaf, and Vishnu Naresh Boddeti. Multi-objective evolutionary design of deep convolutional neural networks for image classification. *IEEE Transactions on Evolutionary Computation*, pages 1–1, 2020b.

Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once for all: Train one network and specialize it for efficient deployment. In *ICLR 2020 : Eighth International Conference on Learning Representations*, 2020.

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott E. Reed, Cheng-Yang Fu, and Alexander C. Berg. Ssd: Single shot multibox detector. In *14th European Conference on Computer Vision, ECCV 2016*, pages 21–37, 2016.

Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. *arXiv preprint arXiv:2004.10934*, 2020.

Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. Shufflenet v2: Practical guidelines for efficient cnn architecture design. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 122–138, 2018.

Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. Bisenet: Bilateral segmentation network for real-time semantic segmentation. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 325–341, 2018.

Bin Xiao, Haiping Wu, and Yichen Wei. Simple baselines for human pose estimation and tracking. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 472–487, 2018.

Chris Ying, Aaron Klein, Eric Christiansen, Esteban Real, Kevin Murphy, and Frank Hutter. Nas-bench-101: Towards reproducible neural architecture search. In *International Conference on Machine Learning*, pages 7105–7114, 2019.

Arber Zela, Julien Siems, and Frank Hutter. Nas-bench-1shot1: Benchmarking and dissecting one-shot neural architecture search. In *ICLR 2020 : Eighth International Conference on Learning Representations*, 2020.

Xuanyi Dong and Yi Yang. Nas-bench-201: Extending the scope of reproducible neural architecture search. In *Eighth International Conference on Learning Representations*, 2020.

Liam Li and Ameet Talwalkar. Random search and reproducibility for neural architecture search. In *Uncertainty in Artificial Intelligence*, pages 367–377, 2019.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3):229–256, 1992.

Xuanyi Dong and Yi Yang. Searching for a robust neural architecture in four gpu hours. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1761–1770, 2019.

Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. Snas: stochastic neural architecture search. In *International Conference on Learning Representations*, 2018.

Shoukang Hu, Sirui Xie, Hehui Zheng, Chunxiao Liu, Jianping Shi, Xunying Liu, and Dahua Lin. Dsnas: Direct neural architecture search without parameter retraining. In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 12084–12092, 2020.

## 6 Appendix

### 6.1 Architecture Visualisation

This section visualises the architectures obtained by searching for EEEA-Nets with CIFAR-10 datasets, as shown in Fig. 9, and EEEA-Nets with ImageNet datasets shown in Fig. 10. These architectures are the most reliable, minimising three goals.(a) EA-Net ( $\beta = 0$ ) Normal Cell.

(b) EA-Net ( $\beta = 0$ ) Reduction Cell.

(c) EEEA-Net-A ( $\beta=3.0$ ) Normal Cell.

(d) EEEA-Net-A ( $\beta=3.0$ ) Reduction Cell.

(e) EEEA-Net-B ( $\beta=4.0$ ) Normal Cell.

(f) EEEA-Net-B ( $\beta=4.0$ ) Reduction Cell.

(g) EEEA-Net-C ( $\beta=5.0$ ) Normal Cell.

(h) EEEA-Net-C ( $\beta=5.0$ ) Reduction Cell.

Figure 9: Normal and Reduction cells learned on CIFAR-10: EA-Net ( $\beta=0.0$ ), EEEA-Net-A ( $\beta=3.0$ ), EEEA-Net-B ( $\beta=4.0$ ), and EEEA-Net-C ( $\beta=5.0$ ).Figure 10: EA-Nets and EEEA-Nets architectures were searched from ImageNet datasets. The stem and tail layers in all architectures are the same.

Figure 11: An example of object detection results of MobileNetv2, MobileNetv3, and EEEA-Net-C2 models.

## 6.2 Error Analysis of EEEA-Net-C2

Our experiment applied EEEA-Net-C2 for detection, semantic segmentation, and human keypoint detection, where we concluded that the EEEA-Net-C2 model was better than the MobileNet-V3 model. For error analysis of the EEEA-Net-C2 model, images for each application were processed to check the correct results. In this appendix, error analysis of the EEEA-Net-C2 model is divided into three parts: error analysis of object detection, error analysis of semantic segmentation errors and error analysis of human keypoint detection.

### 6.2.1 Object detection

Object detection results from MobileNetv2, MobileNetv3 and EEEA-Net-C2 models are shown in Fig. 11. Given the error from the images in the first column, the MobileNetv2 model was found to have mistakenly identified “bird” as “person”, while the MobileNetv3 model was unable to find “bird”. However, the EEEA-Net-C2 model detected the location of all “birds”.

As with the second column in Fig. 11, the EEEA-Net-C2 model can identify all “persons” positions. However, only the EEEA-Net-C2 model could not locate the hidden “person” behind the middle woman in the third column image. Additionally, in the fourth column pictures, the EEEA-Net-C2 model was able to identify more “plant pots” than the MobileNetv2 and MobileNetv3 models.Figure 12: An example of semantic segmentation results of MobileNetv2, MobileNetv3 and EEEA-Net-C2 models.

Figure 13: An example of human keypoint detection results of MobileNetv2, MobileNetv3 and EEEA-Net-C2 models.

### 6.2.2 Semantic segmentation

The results of visual image segmentation using MobileNetv3 and EEEA-Net-C2 models are shown in Fig. 12. The error of the semantic segmentation results can be determined from the pictures of the first column. It was found that MobileNetv3 models could segment only “Traffic sign pole”. However, the MobileNetv3 model cannot segment the left “traffic sign”, while the EEEA-Net-C2 model can segment both “pole and sign”.

The pictures in the second column from Fig. 12 depicts that the EEEA-Net-C2 model segmented from the “traffic island” was less than the MobileNetv3 model. Next, in the third column, the EEEA-Net-C2 model segmented the “footpath” more precisely than the MobileNetv3 model.

### 6.2.3 Human keypoint detection

The human keypoint detection results from the MobileNetv2, MobileNetv3 and EEEA-Net-C2 models are shown in Fig. 13. When considering the error from the pictures in the first column, the MobileNetv3 model was found to indicate the “left arm” position, while the MobileNetv2 and EEEA-Net-C2 models were able to locate.Figure 14: Comparison of latency between EEEA-Net and other state-of-the-art models on non-GPU processing.

Only the MobileNetv3 model can pinpoint the “leg” of the person sitting in the second column pictures. However, in the third column pictures, the EEEA-Net-C2 model can locate the middle person’s “arms and legs”, while the MobileNetv3 model identifies the person’s wrong location. Additionally, in the fourth column pictures, the EEEA-Net-C2 model could locate the “arm and leg” more accurately than the MobileNetv2 and MobileNetv3 models.

The above data shows that the EEEA-Net-C2 model is accurate as well as inaccurate. The EEEA-Net-C2 model was designed and searched with the ImageNet dataset. Thus, the EEEA-Net-C2 model may have errors when used with other dataset or tasks. However, the EEEA-Net-C2 model has a performance higher than the MobileNetv2 and MobileNetv3 models on the same dataset and framework used in the three applications.

### 6.3 Mobile Processing

This appendix measures the performance of our EEEA-Net-C2 model and other state-of-the-art models on the smartphone and only CPU. All trained models with the ImageNet dataset are converted to the PyTorch JIT version to enable easy implementation on different platforms.

Fig. 14 shows the latency performance with 100 images with 224x224 pixels on the Google Pixel 3 XL smartphone (blue bars) and Intel i7-6700HQ CPU (red bars) devices with non-GPU resources by DARTSv2, P-DARTS, NASNet, RelativeNAS, ShuffleNetv2, MNASNet 1.0, MobileNetV2, MobileNetV3, and EEEA-Net-C2 models.

On the Google Pixel 3 XL, the EEEA-Net-C2 model processed each image in 86 milliseconds per image, whereas the MobileNetV2 and MobileNetV3 models took 90 and 88 milliseconds, respectively. The EEEA-Net-C2 model has a shorter latency time than state-of-the-art models (including DARTSv2, P-DARTS, NASNet, and RelativeNAS), and MobileNets models, primarily models for smartphones, are compared to EEEA-Net-C2.

Also, on the Intel i7-6700HQ CPU, the latency time of the EEEA-Net-C2 model has shorter latency than state-of-the-art models and lightweight models (including MNASNet 1.0, MobileNetV2, and MobileNetV3).

### 6.4 NAS-Bench dataset

Experimental results with CIFAR-10, CIFAR-100, and ImageNet datasets were compared between NAS methods. The results obtained from different methods are achieved with different settings such as hyperparameters (e.g., learning rate and batch size), data augmentation (e.g., Cutout and AutoAugment). Thus, the comparison may not be fair.Figure 15: Comparison of accuracy between random search, regularised evolution and Early Exit evolution algorithms on NAS-Bench-101.

This section has implemented an Early Exit method to model search from the NAS-Bench datasets, which avoids unfair comparisons and provides a uniform benchmark for NAS algorithms. The dataset used in this experiment was NAS-Bench-101, NAS-Bench-1Shot1 and NAS-Bench-201.

#### 6.4.1 NAS-Bench-101

The NAS-Bench-101 [Ying et al. \[2019\]](#) provides a table dataset of 423,624 unique architectures. These architectures have trained and evaluated the CIFAR-10 dataset to allow our work to search and query the mapped performance in the dataset in a few milliseconds.

We have re-implemented model search from the NAS-Bench-101 dataset by using random search, regularised evolution and Early Exit evolution algorithms to search and query the performance of the resulting models. We used a re-implemented regularised evolution with the Early Exit method by taking population size of 100, a tournament size of 10, and maximum parameters' Early Exit ( $\beta$ ) is 25 million.

The results in Fig. 15 show that our early exit evolution algorithm tends to be higher in accuracy than the regularised evolution from 2 million seconds to 5 million seconds. Overall, the regularised evolution algorithm appears to perform better than the random search algorithm. However, our early exit evolution tends to outperform both random search and regularised evolution algorithms.

#### 6.4.2 NAS-Bench-1Shot1

NAS-Bench-1Shot1 [Zela et al. \[2020\]](#) is the benchmark for a one-shot neural architecture search, developed from the NAS-Bench-101 search space by tracking the trajectory and performance of the obtained architectures for three search spaces: 6,240 architectures for search space 1, 29,160 architectures for search space 2, and 363,648 architectures for search space 3.

Fig. 16 shows the mean performance on validation regret of architectures obtained by random search, regularised evolution and Early Exit evolution algorithms. For search space 1, our algorithm achieves validation regret close to the regularised evolution algorithm. For search space 2, our algorithm converges better than the regularised evolution algorithm. Our algorithm outperforms random search and regularised evolution algorithms for search space 3, the most significant ( $100\times$  more architectures than space 1, and  $10\times$  more than space 2).Figure 16: Comparison of accuracy between random search (RS), regularised evolution (RE) and Early Exit evolution (EE) algorithms on NAS-Bench-1Shot1.

<table border="1">
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">CIFAR-10</th>
<th colspan="2">CIFAR-100</th>
<th colspan="2">ImageNet16-120</th>
</tr>
<tr>
<th>validation</th>
<th>test</th>
<th>validation</th>
<th>test</th>
<th>validation</th>
<th>test</th>
</tr>
</thead>
<tbody>
<tr>
<td>ResNet He et al. [2016]</td>
<td>90.83</td>
<td>93.97</td>
<td>70.42</td>
<td>70.86</td>
<td>44.53</td>
<td>43.63</td>
</tr>
<tr>
<td>RSPS Li and Talwalkar [2019]</td>
<td>84.16±1.69</td>
<td>87.66±1.69</td>
<td>45.78±6.33</td>
<td>46.60±6.57</td>
<td>31.09±5.65</td>
<td>30.78±6.12</td>
</tr>
<tr>
<td>Reinforce Williams [1992]</td>
<td>91.09±0.37</td>
<td>93.85±0.37</td>
<td>70.05±1.67</td>
<td>70.17±1.61</td>
<td>43.04±2.18</td>
<td>43.16±2.28</td>
</tr>
<tr>
<td>ENAS Pham et al. [2018]</td>
<td>39.77±0.00</td>
<td>54.30±0.00</td>
<td>10.23±0.12</td>
<td>10.62±0.27</td>
<td>16.43±0.00</td>
<td>16.32±0.00</td>
</tr>
<tr>
<td>DARTS Liu et al. [2018b]</td>
<td>39.77±0.00</td>
<td>54.30±0.00</td>
<td>38.57±0.00</td>
<td>38.97±0.00</td>
<td>18.87±0.00</td>
<td>18.41±0.00</td>
</tr>
<tr>
<td>GDAS Dong and Yang [2019]</td>
<td>90.01±0.46</td>
<td>93.23±0.23</td>
<td>24.05±8.12</td>
<td>24.20±8.08</td>
<td>40.66±0.00</td>
<td>41.02±0.00</td>
</tr>
<tr>
<td>SNAS Xie et al. [2018]</td>
<td>90.10±1.04</td>
<td>92.77±0.83</td>
<td>69.69±2.39</td>
<td>69.34±1.98</td>
<td>42.84±1.79</td>
<td>43.16±2.64</td>
</tr>
<tr>
<td>DSNAS Hu et al. [2020]</td>
<td>89.66±0.29</td>
<td>93.08±0.13</td>
<td>30.87±16.40</td>
<td>31.01±16.38</td>
<td>40.61±0.09</td>
<td>41.07±0.09</td>
</tr>
<tr>
<td>PC-DARTS Xu et al. [2020]</td>
<td>89.96±0.15</td>
<td>93.41±0.30</td>
<td>67.12±0.39</td>
<td>67.48±0.89</td>
<td>40.83±0.08</td>
<td>41.31±0.22</td>
</tr>
<tr>
<td>EA-Net (SO)</td>
<td>91.53±0.00</td>
<td>94.22±0.00</td>
<td>73.13±0.00</td>
<td>73.17±0.00</td>
<td>46.32±0.00</td>
<td>46.48±0.00</td>
</tr>
<tr>
<td>EA-Net (<math>\beta = 0</math>)</td>
<td>88.97±2.48</td>
<td>91.54±2.69</td>
<td>66.84±5.08</td>
<td>67.00±4.90</td>
<td>39.93±5.54</td>
<td>39.27±6.21</td>
</tr>
<tr>
<td>EEEA-Net (<math>\beta = 0.3</math>)</td>
<td>87.07±1.59</td>
<td>89.76±1.87</td>
<td>64.04±3.21</td>
<td>64.31±3.21</td>
<td>35.42±3.81</td>
<td>34.98±4.13</td>
</tr>
<tr>
<td>EEEA-Net (<math>\beta = 0.4</math>)</td>
<td>89.91±0.77</td>
<td>92.68±0.69</td>
<td>68.70±1.50</td>
<td>68.65±1.51</td>
<td>41.71±1.58</td>
<td>41.25±1.61</td>
</tr>
<tr>
<td>EEEA-Net (<math>\beta = 0.5</math>)</td>
<td>90.21±0.58</td>
<td>92.83±0.46</td>
<td>69.15±1.36</td>
<td>68.95±1.25</td>
<td>42.14±1.14</td>
<td>41.98±1.22</td>
</tr>
<tr>
<td>Optimal</td>
<td>91.61</td>
<td>94.37</td>
<td>73.49</td>
<td>73.51</td>
<td>46.77</td>
<td>47.31</td>
</tr>
</tbody>
</table>

Table 9: Comparison of a single objective and multi-objective evolution algorithm with the 8 NAS methods provided by NAS-Bench-201 benchmark. Optimal shows the best architecture in the search space.

### 6.4.3 NAS-Bench-201

NAS-Bench-201 Dong and Yang [2020] is an extension of NAS-Bench-101, which extends different search spaces, and it has a wide range of datasets, including CIFAR-10, CIFAR-100, and ImageNet-16-120. It contains 15,625 architectures by five operations, and 6-dimensional vectors indicate the operation in the cell. All architecture evaluated performance by validation and test sets on CIFAR-10, CIFAR-100, and ImageNet-16-120.

We compare our Early Exit Evolution Algorithm or EEEA-Net ( $\beta = 0.3, 0.4$  and  $0.5$ ) with the single objective evolution algorithms (SO) and multi-objective evolution algorithms ( $\beta = 0$ ). The hyper-parameters for this search process were defined as the generations of EA equal to 10 generations with 100 populations by retaining a probability of 0.5, a mutation probability of 0.1, and Early Exit is the maximum number of parameters equal to 0.3, 0.4 and 0.5 million.

The results are shown in Table 9, our EEEA-Net ( $\beta = 0.4$  and  $0.5$ ) outperforms EEEA-Net ( $\beta = 0$  and  $0.3$ ). However, the EA-Net (SO) using accuracy as the optimisation objective performed better than all EEEA-Nets.

Furthermore, when we compared our EEEA-Net ( $\beta = 0.5$ ) with 8 NAS methods, including RSPS Li and Talwalkar [2019], Reinforce Williams [1992], ENAS Pham et al. [2018], DARTS Liu et al. [2018b], GDAS Dong and Yang [2019], SNAS Xie et al. [2018], DSNAS, and PC-DARTS Xu et al. [2020], we found that EEEA-Net ( $\beta = 0.5$ ) has an accuracy was higher than all other NAS method except for the Reinforce method.

