Title: Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration

URL Source: https://arxiv.org/html/2404.01817

Published Time: Fri, 12 Apr 2024 00:34:36 GMT

Markdown Content:
(2024)

###### Abstract.

The NeuroEvolution of Augmenting Topologies (NEAT) algorithm has received considerable recognition in the field of neuroevolution. Its effectiveness is derived from initiating with simple networks and incrementally evolving both their topologies and weights. Although its capability across various challenges is evident, the algorithm’s computational efficiency remains an impediment, limiting its scalability potential. In response, this paper introduces a tensorization method for the NEAT algorithm, enabling the transformation of its diverse network topologies and associated operations into uniformly shaped tensors for computation. This advancement facilitates the execution of the NEAT algorithm in a parallelized manner across the entire population. Furthermore, we develop TensorNEAT, a library that implements the tensorized NEAT algorithm and its variants, such as CPPN and HyperNEAT. Building upon JAX, TensorNEAT promotes efficient parallel computations via automated function vectorization and hardware acceleration. Moreover, the TensorNEAT library supports various benchmark environments including Gym, Brax, and gymnax. Through evaluations across a spectrum of robotics control environments in Brax, TensorNEAT achieves up to 500x speedups compared to the existing implementations such as NEAT-Python. Source codes are available at: [https://github.com/EMI-Group/tensorneat](https://github.com/EMI-Group/tensorneat).

Neuroevolution, GPU Acceleration, Algorithm Library

††journalyear: 2024††copyright: acmlicensed††conference: Genetic and Evolutionary Computation Conference; July 14–18, 2024; Melbourne, VIC, Australia††booktitle: Genetic and Evolutionary Computation Conference (GECCO ’24), July 14–18, 2024, Melbourne, VIC, Australia††doi: 10.1145/3638529.3654210††isbn: 979-8-4007-0494-9/24/07††ccs: Theory of computation Evolutionary algorithms††ccs: Theory of computation Vector / streaming algorithms
1. Introduction
---------------

Neuroevolution has emerged as a distinct branch within the field of artificial intelligence (AI). Unlike the common approach in machine learning that uses stochastic gradient descent, neuroevolution employs evolutionary algorithms for network optimization. Beyond the traditional limit of parameter optimization, it also involves improving elements such as activation functions, hyperparameters, and the overall network structure. Moreover, compared to standard machine learning methods which often converge to a single solution, neuroevolution continually promotes a varied set of solutions throughout its exploration (Stanley et al., [2019](https://arxiv.org/html/2404.01817v3#bib.bib30)). These defining attributes not only give neuroevolution a strong sense of novelty and diversity but also empower it for certain open-ended challenges (Lehman and Stanley, [2011](https://arxiv.org/html/2404.01817v3#bib.bib16); Mouret and Clune, [2015](https://arxiv.org/html/2404.01817v3#bib.bib20)).

The NeuroEvolution of Augmenting Topologies (NEAT) (Stanley and Miikkulainen, [2002](https://arxiv.org/html/2404.01817v3#bib.bib32)) is well-recognized in the neuroevolution literature. Since its introduction in 2002, NEAT has been shown to be useful in various areas such as game AI (Stanley et al., [2006](https://arxiv.org/html/2404.01817v3#bib.bib29); Pham et al., [2018](https://arxiv.org/html/2404.01817v3#bib.bib23)), robotics (Silva et al., [2012](https://arxiv.org/html/2404.01817v3#bib.bib27); Auerbach and Bongard, [2011](https://arxiv.org/html/2404.01817v3#bib.bib3)) and self-driving systems (Yuksel, [2018](https://arxiv.org/html/2404.01817v3#bib.bib34)). The original NEAT algorithm provides a basic framework, and the subsequent works have kept exploiting its potential. For instance, offshoots like HyperNEAT (Stanley et al., [2009](https://arxiv.org/html/2404.01817v3#bib.bib31)) and ES-HyperNEAT (Risi et al., [2010](https://arxiv.org/html/2404.01817v3#bib.bib25)) adopted indirect encoding for expansive networks; DeepNEAT and CoDeepNEAT (Miikkulainen et al., [2019](https://arxiv.org/html/2404.01817v3#bib.bib19)) combined gradient descent methods to delve deeper into neural architecture. Recently, RankNEAT (Pinitas et al., [2022](https://arxiv.org/html/2404.01817v3#bib.bib24)) used neuroevolution for preference learning tasks, effectively optimizing network architectures in subjectively labeled data environments. Such ongoing development and flexibility highlight NEAT’s continued importance and appeal in the field.

During the past years, GPU acceleration has been a driving force behind the rapid advancements in AI, especially in the field of deep learning. The advancement in such hardware acceleration has facilitated the expansion of deep learning, with modern large language models incorporating hundreds of billions of parameters (Brown et al., [2020](https://arxiv.org/html/2404.01817v3#bib.bib6)). Using GPUs for tasks like inference and back-propagation has led to significant speed improvements. Meanwhile, within the field of neuroevolution, there is also a growing interest in leveraging GPUs for hardware acceleration. Building upon the JAX framework(Frostig et al., [2018](https://arxiv.org/html/2404.01817v3#bib.bib8)), some pioneering works such as EvoJAX (Tang et al., [2022](https://arxiv.org/html/2404.01817v3#bib.bib33)), evosax (Lange, [2023](https://arxiv.org/html/2404.01817v3#bib.bib15)), and EvoX (Huang et al., [2024](https://arxiv.org/html/2404.01817v3#bib.bib13)) represent this trend, seeking to tap into the robust computational capabilities of GPU to reduce the runtime of neuroevolution algorithms, especially when handling large population sizes or extensive problem scales.

However, despite the rapid emergence of these GPU-accelerated libraries, NEAT is left behind. Our analysis showed that though the NEAT algorithm has been implemented in various programming languages(McIntyre et al., [2023](https://arxiv.org/html/2404.01817v3#bib.bib18); peter ch, [2019](https://arxiv.org/html/2404.01817v3#bib.bib22); b2developer, [2022](https://arxiv.org/html/2404.01817v3#bib.bib4)), few of them harness GPUs to boost their performance. On those rare occasions when GPUs are used(Gajewsky, [2023](https://arxiv.org/html/2404.01817v3#bib.bib9)), the focus has primarily been on accelerating the network inference process of the NEAT algorithm, while other crucial parts, such as network search processes, are often overlooked. Moreover, they do not efficiently utilize GPU parallel computing, as key operations like fitness evaluation and mutation are executed sequentially in their frameworks. This limitation can be attributed to the unique nature of NEAT: It employs networks with _continuously evolving topologies_ during the algorithm’s execution. This characteristic poses a challenge for efficient GPU implementation.

To bridge this gap, we develop TensorNEAT, a tensorized NEAT library optimized for GPU acceleration. TensorNEAT employs a new tensorization method that transforms networks of varying topologies into tensors with uniform shape, ensuring that operations in the NEAT algorithm can be executed in parallel across the entire population. Implemented within the JAX framework, TensorNEAT enables automatic GPU acceleration without any specific configuration. In comparison with the existing popular open-source implementation of the NEAT algorithm, TensorNEAT achieves up to 500x speedups. Overall, our contributions are summarized as follows.

*   •We propose a tensorization method, which enables the transformation of networks with various topologies and their associated operations in the NEAT algorithm into uniformly structured tensors for tensor computation. This method allows operations within the NEAT algorithm to be executed in parallel across the entire population, thereby enhancing the efficiency of the process. 
*   •We develop TensorNEAT, a GPU-accelerated NEAT library based on JAX, characterized by high efficiency, flexible adaptability, and rich capabilities. TensorNEAT supports full GPU acceleration of representative NEAT algorithms, including the original NEAT algorithm, CPPN(Stanley, [2007](https://arxiv.org/html/2404.01817v3#bib.bib28)), and HyperNEAT(Stanley et al., [2009](https://arxiv.org/html/2404.01817v3#bib.bib31)). It also provides seamless interfaces with advanced control benchmarks, including Brax(Freeman et al., [2021](https://arxiv.org/html/2404.01817v3#bib.bib7)) and Gymnax(Lange, [2022](https://arxiv.org/html/2404.01817v3#bib.bib14)), featuring GPU-accelerated environments with various classical control and robotics control tasks. 
*   •We assessed TensorNEAT’s performance in a spectrum of complex robotics control tasks, benchmarked against the NEAT-Python library(McIntyre et al., [2023](https://arxiv.org/html/2404.01817v3#bib.bib18)). The results show that TensorNEAT significantly outperforms in terms of execution speed, especially under high computational demands and across various population sizes and network scales. 

2. Background
-------------

### 2.1. NeuroEvolution of Augmenting Topologies

Introduced by Kenneth O. Stanley and Risto Miikkulainen in 2002, the NeuroEvolution of Augmenting Topologies (NEAT) algorithm (Stanley and Miikkulainen, [2002](https://arxiv.org/html/2404.01817v3#bib.bib32)) represents a novel approach in neuroevolution. The NEAT algorithm manages a range of neural networks and simultaneously optimizes their topologies and weights to identify the most effective networks tailored for designated tasks. Algorithm[1](https://arxiv.org/html/2404.01817v3#alg1 "Algorithm 1 ‣ Appendix A NeuroEvolution of Augmenting Topologies ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") in appendix[A](https://arxiv.org/html/2404.01817v3#A1 "Appendix A NeuroEvolution of Augmenting Topologies ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") outlines NEAT’s core procedure. Starting with a population of simple neural networks, it iterates through evolutionary cycles of species formation, fitness evaluation, and genetic operations. This process dynamically refines the networks until it achieves desired fitness levels or reaches a generational cap, ultimately yielding the optimal network structure.

Setting itself apart from alternative neuroevolution algorithms, NEAT employs three distinctive techniques:

*   •Incremental Topological Expansion: The NEAT algorithm initializes its evolutionary search with the most basic networks, consisting of just a single hidden node bridging inputs to outputs. As evolution advances, the algorithm incorporates new nodes and connections, progressively sophisticating the network topology. This strategy pragmatically narrows down the search space, facilitating the resolution of complex challenges using more streamlined network structures. 
*   •Historical Markers for Nodes: Each node in a NEAT network is tagged with a unique historical marker. During the crossover process in the NEAT algorithm, only nodes with identical markers are combined. This method adeptly navigates the challenges of combining networks that possess different topological configurations. 
*   •Species-based Population Segmentation: NEAT categorizes its entire population into individual species. Conventional genetic procedures, such as selection, mutation, and crossover, are executed independently within each species. This approach serves dual purposes: it not only shields potentially advantageous network structures from premature extinction, but also amplifies the diversity of solutions within the evolutionary exploration. 

![Image 1: Refer to caption](https://arxiv.org/html/2404.01817v3/x1.png)

Figure 1. Illustration of the fundamental acceleration principle underlying our tensorization method. In traditional network computations, each network individually processes its inputs. By contrast, with our tensorization method, a single computation suffices to derive the batched output for all networks. 

Since its inception, the NEAT algorithm has undergone significant evolution, leading to the emergence of several innovative variants, each addressing unique challenges and applications. HyperNEAT(Stanley et al., [2009](https://arxiv.org/html/2404.01817v3#bib.bib31)) marked a pivotal advancement by employing an indirect encoding scheme. This approach allows efficient handling of large-scale neural networks, leveraging geometric regularities to create complex network topologies that can be applied to tasks requiring substantial representational capacity. Building upon this concept, ES-HyperNEAT(Risi et al., [2010](https://arxiv.org/html/2404.01817v3#bib.bib25)) further refined the approach, introducing enhanced techniques for evolving network structures with even greater scalability and adaptability.

Then the development of DeepNEAT and CoDeepNEAT(Miikkulainen et al., [2019](https://arxiv.org/html/2404.01817v3#bib.bib19)) represented a significant leap in integrating neuroevolution with deep learning principles. DeepNEAT extended the NEAT algorithm to the realm of deep neural networks. It leverages gradient descent to explore and optimize complex, layered neural architectures. CoDeepNEAT expanded this concept further, introducing a co-evolutionary approach that allowed for the simultaneous evolution of both the topology and components of deep neural networks. This fusion of gradient descent and evolutionary strategies enabled a more thorough and nuanced exploration of neural structures, opening new avenues in areas like feature learning and hierarchical network construction.

RankNEAT(Pinitas et al., [2022](https://arxiv.org/html/2404.01817v3#bib.bib24)) is another variant of the NEAT algorithm. It overcomes the limitations of Stochastic Gradient Descent by using neuroevolution to optimize network architectures, effectively reducing overfitting. RankNEAT has proven to be effective in affective computing, outshining traditional methods like RankNet. It is particularly successful in analyzing player arousal from game footage, highlighting its potential in handling subjective data.

Together, these variants of NEAT demonstrate the superior algorithm’s versatility and its continuous adaptation to address the evolving complexities of neural network design and application.

![Image 2: Refer to caption](https://arxiv.org/html/2404.01817v3/x2.png)

Figure 2. Illustration of the network encoding process. In our method, networks with varying topological structures are transformed into uniformly shaped tensors, enabling the representation of the entire network population as batched tensors. The orange cells symbolize attributes specific to the NEAT algorithm, such as historical markers and enabled flags. The yellow cells denote the attributes of the network’s nodes and connections, including biases, weights, and activation functions. The gray cells represent sections filled with NaN to ensure consistent tensor shapes.

### 2.2. NEAT Libraries

In the past two decades, the research community has witnessed the emergence of various NEAT libraries, including NEAT-Python(McIntyre et al., [2023](https://arxiv.org/html/2404.01817v3#bib.bib18)), MultiNEAT(peter ch, [2019](https://arxiv.org/html/2404.01817v3#bib.bib22)), and MonopolyNEAT(b2developer, [2022](https://arxiv.org/html/2404.01817v3#bib.bib4)). Among these libraries, the NEAT-Python library (McIntyre et al., [2023](https://arxiv.org/html/2404.01817v3#bib.bib18)) stands out as the foremost open-source implementation of NEAT at present. With over 1,200 GitHub stars, it has served as the foundational base for numerous academic research projects related to NEAT (Pinitas et al., [2022](https://arxiv.org/html/2404.01817v3#bib.bib24); Gao and Lan, [2021](https://arxiv.org/html/2404.01817v3#bib.bib10); Sarti and Ochoa, [2021](https://arxiv.org/html/2404.01817v3#bib.bib26)). These NEAT implementations use the object-oriented programming paradigm, involving the representation of core NEAT components such as populations, species, genomes, and genes as objects. This programming paradigm provides remarkable transparency, enhancing code readability and thereby enabling those new to the domain to rapidly grasp the complexities of the NEAT algorithm. However, maintaining objects also brings extra computational overheads in the running process, especially when encountered with large-scale problems or substantial population sizes.

Most existing NEAT implementations do not utilize GPUs to accelerate computation, with a few exceptions such as PyTorchNEAT(Gajewsky, [2023](https://arxiv.org/html/2404.01817v3#bib.bib9)). These libraries integrate tensor deep learning, such as PyTorch(Paszke et al., [2019](https://arxiv.org/html/2404.01817v3#bib.bib21)) and TensorFlow(Abadi et al., [2015](https://arxiv.org/html/2404.01817v3#bib.bib2)), enabling networks generated by NEAT to perform inference on GPUs. This approach also enables batch inference for multiple inputs, thereby enhancing network inference speed when dealing with a large number of input data. However, these libraries still rely on the object-oriented programming paradigm and have optimized only the inference aspect of the network. The acceleration of the neuroevolution process, which NEAT uses for network search, has not yet been achieved. Moreover, although individual network inferences can be batch-processed on GPUs, the inference process for the entire population in the NEAT algorithm remains sequential. Consequently, existing NEAT implementations cannot fully make use of the high parallel processing capabilities of GPUs.

3. Tensorization of NEAT
------------------------

To overcome the limitations of current NEAT implementations and fully leverage modern hardware to accelerate the NEAT algorithm’s efficiency, we introduce a new tensorization method. Figure[1](https://arxiv.org/html/2404.01817v3#S2.F1 "Figure 1 ‣ 2.1. NeuroEvolution of Augmenting Topologies ‣ 2. Background ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") depicts the key principle of acceleration supporting this approach. Our method facilitates the transformation of various network topologies and their related operations into uniformly structured tensors, suitable for tensor computation. By implementing function vectorization within network operations, our approach enables parallel processing across the entire population, which effectively harnesses the parallel computing power of GPUs. In the following sections, we detail the specific tensorization techniques employed in network encoding and operations.

![Image 3: Refer to caption](https://arxiv.org/html/2404.01817v3/x3.png)

Figure 3. An illustration of tensorized network operations. It demonstrates how traditional network operations are converted into equivalent tensor operations during tensorization. Changes from the original format are highlighted in red, underscoring the modifications made within the tensor.

![Image 4: Refer to caption](https://arxiv.org/html/2404.01817v3/x4.png)

![Image 5: Refer to caption](https://arxiv.org/html/2404.01817v3/x5.png)

Figure 4. Illustration of tensorized network inference process. It transforms a feedforward network’s topology into tensors and the subsequent calculation of node values for network inference. The network is first encoded into node and connection tensors, which are then ordered and expanded for processing. Finally, values are calculated through connection and node functions to produce the output.

### 3.1. Tensorized Encodings

In NEAT, a network 𝒩 𝒩\mathcal{N}caligraphic_N can be represented as:

𝒩=⟨N,C⟩,𝒩 𝑁 𝐶\mathcal{N}=\langle N,C\rangle,caligraphic_N = ⟨ italic_N , italic_C ⟩ ,

where N 𝑁 N italic_N, C 𝐶 C italic_C are nodes and connections in 𝒩 𝒩\mathcal{N}caligraphic_N, respectively. They can be represented as:

N={n 1,n 2,n 3,…}and C={c 1,c 2,c 3,…},formulae-sequence 𝑁 subscript 𝑛 1 subscript 𝑛 2 subscript 𝑛 3…and 𝐶 subscript 𝑐 1 subscript 𝑐 2 subscript 𝑐 3…N=\{n_{1},n_{2},n_{3},\ldots\}\quad\text{and}\quad C=\{c_{1},c_{2},c_{3},% \ldots\},italic_N = { italic_n start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_n start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , italic_n start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT , … } and italic_C = { italic_c start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_c start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , italic_c start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT , … } ,

with n i subscript 𝑛 𝑖 n_{i}italic_n start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT denoting the i 𝑖 i italic_i-th node in N 𝑁 N italic_N and c i subscript 𝑐 𝑖 c_{i}italic_c start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT denoting the i 𝑖 i italic_i-th connection in C 𝐶 C italic_C.

In the NEAT algorithm, each node n 𝑛 n italic_n can be represented as a tuple consisting of a historical marker and its attributes:

n=(k,attr 1,attr 2,…),𝑛 𝑘 subscript attr 1 subscript attr 2…n=(k,\text{attr}_{1},\text{attr}_{2},\ldots),italic_n = ( italic_k , attr start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , attr start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ) ,

where k∈ℕ 𝑘 ℕ k\in\mathbb{N}italic_k ∈ blackboard_N is the historical marking, and attr i subscript attr 𝑖\text{attr}_{i}attr start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is the i 𝑖 i italic_i-th attribute of n 𝑛 n italic_n. Similarly, a connection c 𝑐 c italic_c can be represented as:

c=(k i,k o,e,attr 1,attr 2,…),𝑐 subscript 𝑘 𝑖 subscript 𝑘 𝑜 𝑒 subscript attr 1 subscript attr 2…c=(k_{i},k_{o},e,\text{attr}_{1},\text{attr}_{2},\ldots),italic_c = ( italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT , italic_e , attr start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , attr start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ) ,

where k i∈ℕ subscript 𝑘 𝑖 ℕ k_{i}\in\mathbb{N}italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_N and k o∈ℕ subscript 𝑘 𝑜 ℕ k_{o}\in\mathbb{N}italic_k start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT ∈ blackboard_N are the historical markings of the input and output nodes of c 𝑐 c italic_c, respectively, e∈{𝚃𝚛𝚞𝚎,𝙵𝚊𝚕𝚜𝚎}𝑒 𝚃𝚛𝚞𝚎 𝙵𝚊𝚕𝚜𝚎 e\in\{\texttt{True},\texttt{False}\}italic_e ∈ { True , False } is the enabled flag. The attributes of a node n 𝑛 n italic_n can be bias b 𝑏 b italic_b, aggregation function f agg subscript 𝑓 agg f_{\text{agg}}italic_f start_POSTSUBSCRIPT agg end_POSTSUBSCRIPT like sum, and activation function f act subscript 𝑓 act f_{\text{act}}italic_f start_POSTSUBSCRIPT act end_POSTSUBSCRIPT like sigmoid and the attributes of a connection c 𝑐 c italic_c can be weight w 𝑤 w italic_w. Attributes that can be represented numerically, such as b 𝑏 b italic_b and w 𝑤 w italic_w, are directly stored in tensors. Those attributes that are not inherently numerical, such as f agg subscript 𝑓 agg f_{\text{agg}}italic_f start_POSTSUBSCRIPT agg end_POSTSUBSCRIPT and f act subscript 𝑓 act f_{\text{act}}italic_f start_POSTSUBSCRIPT act end_POSTSUBSCRIPT, are stored in tensors as specific integer values, for example, tanh is represented by 1, and sigmoid by 2.

We can use one-dimensional tensors to encode n 𝑛 n italic_n and c 𝑐 c italic_c:

𝒏 𝒏\displaystyle\bm{n}bold_italic_n=[k,attr 1,attr 2,…]∈ℝ 1+𝚗𝚘𝚊⁢(n),absent 𝑘 subscript attr 1 subscript attr 2…superscript ℝ 1 𝚗𝚘𝚊 𝑛\displaystyle=[k,\text{attr}_{1},\text{attr}_{2},\ldots]\in\mathbb{R}^{1+% \texttt{noa}(n)},= [ italic_k , attr start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , attr start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ] ∈ blackboard_R start_POSTSUPERSCRIPT 1 + noa ( italic_n ) end_POSTSUPERSCRIPT ,
𝒄 𝒄\displaystyle\bm{c}bold_italic_c=[k i,k o,e,attr 1,attr 2,…]∈ℝ 3+𝚗𝚘𝚊⁢(c),absent subscript 𝑘 𝑖 subscript 𝑘 𝑜 𝑒 subscript attr 1 subscript attr 2…superscript ℝ 3 𝚗𝚘𝚊 𝑐\displaystyle=[k_{i},k_{o},e,\text{attr}_{1},\text{attr}_{2},\ldots]\in\mathbb% {R}^{3+\texttt{noa}(c)},= [ italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT , italic_e , attr start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , attr start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ] ∈ blackboard_R start_POSTSUPERSCRIPT 3 + noa ( italic_c ) end_POSTSUPERSCRIPT ,

where 𝚗𝚘𝚊⁢(i)𝚗𝚘𝚊 𝑖\texttt{noa}(i)noa ( italic_i ) denotes the number of attributes of i 𝑖 i italic_i.

Then, the sets N 𝑁 N italic_N and C 𝐶 C italic_C can be represented as tensors 𝑵 𝑵\bm{N}bold_italic_N and 𝑪 𝑪\bm{C}bold_italic_C, respectively:

𝑵 𝑵\displaystyle\bm{N}bold_italic_N=[𝒏 1,𝒏 2,…]∈ℝ|N|×𝚕𝚎𝚗⁢(𝒏),absent subscript 𝒏 1 subscript 𝒏 2…superscript ℝ 𝑁 𝚕𝚎𝚗 𝒏\displaystyle=[\bm{n}_{1},\bm{n}_{2},\ldots]\in\mathbb{R}^{|N|\times\texttt{% len}(\bm{n})},= [ bold_italic_n start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , bold_italic_n start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ] ∈ blackboard_R start_POSTSUPERSCRIPT | italic_N | × len ( bold_italic_n ) end_POSTSUPERSCRIPT ,
𝑪 𝑪\displaystyle\bm{C}bold_italic_C=[𝒄 1,𝒄 2,…]∈ℝ|C|×𝚕𝚎𝚗⁢(𝒄),absent subscript 𝒄 1 subscript 𝒄 2…superscript ℝ 𝐶 𝚕𝚎𝚗 𝒄\displaystyle=[\bm{c}_{1},\bm{c}_{2},\ldots]\in\mathbb{R}^{|C|\times\texttt{% len}(\bm{c})},= [ bold_italic_c start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , bold_italic_c start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ] ∈ blackboard_R start_POSTSUPERSCRIPT | italic_C | × len ( bold_italic_c ) end_POSTSUPERSCRIPT ,

where |⋅||\cdot|| ⋅ | is the cardinality of a set, 𝚕𝚎𝚗⁢(⋅)𝚕𝚎𝚗⋅\texttt{len}(\cdot)len ( ⋅ ) is the length of a one-dimensional tensor, and 𝒏 i subscript 𝒏 𝑖\bm{n}_{i}bold_italic_n start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and 𝒄 j subscript 𝒄 𝑗\bm{c}_{j}bold_italic_c start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT are the tensor representations of the i 𝑖 i italic_i-th node and j 𝑗 j italic_j-th connection, respectively.

To address the issue where each network in the population possesses a varying number of nodes and connections, we employ tensor padding using NaN values for alignment. We set predefined maximum limits for the number of nodes and connections, represented as |N|max subscript 𝑁 max|N|_{\text{max}}| italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT and |C|max subscript 𝐶 max|C|_{\text{max}}| italic_C | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT, respectively. The resulting padded tensors, 𝑵^^𝑵\hat{\bm{N}}over^ start_ARG bold_italic_N end_ARG and 𝑪^^𝑪\hat{\bm{C}}over^ start_ARG bold_italic_C end_ARG, are formulated as:

𝑵^^𝑵\displaystyle\hat{\bm{N}}over^ start_ARG bold_italic_N end_ARG=[𝒏 1,𝒏 2,…,𝙽𝚊𝙽,…]∈ℝ|N|max×𝚕𝚎𝚗⁢(𝒏),absent subscript 𝒏 1 subscript 𝒏 2…𝙽𝚊𝙽…superscript ℝ subscript 𝑁 max 𝚕𝚎𝚗 𝒏\displaystyle=[\bm{n}_{1},\bm{n}_{2},\ldots,\texttt{NaN},\ldots]\in\mathbb{R}^% {|N|_{\text{max}}\times\texttt{len}(\bm{n})},= [ bold_italic_n start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , bold_italic_n start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , NaN , … ] ∈ blackboard_R start_POSTSUPERSCRIPT | italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT × len ( bold_italic_n ) end_POSTSUPERSCRIPT ,
𝑪^^𝑪\displaystyle\hat{\bm{C}}over^ start_ARG bold_italic_C end_ARG=[𝒄 1,𝒄 2,…,𝙽𝚊𝙽,…]∈ℝ|C|max×𝚕𝚎𝚗⁢(𝒄).absent subscript 𝒄 1 subscript 𝒄 2…𝙽𝚊𝙽…superscript ℝ subscript 𝐶 max 𝚕𝚎𝚗 𝒄\displaystyle=[\bm{c}_{1},\bm{c}_{2},\ldots,\texttt{NaN},\ldots]\in\mathbb{R}^% {|C|_{\text{max}}\times\texttt{len}(\bm{c})}.= [ bold_italic_c start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , bold_italic_c start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , NaN , … ] ∈ blackboard_R start_POSTSUPERSCRIPT | italic_C | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT × len ( bold_italic_c ) end_POSTSUPERSCRIPT .

Upon alignment, the tensors 𝑵^^𝑵\hat{\bm{N}}over^ start_ARG bold_italic_N end_ARG and 𝑪^^𝑪\hat{\bm{C}}over^ start_ARG bold_italic_C end_ARG of each network in the population can be concatenated, allowing us to express the entire population using two tensors: 𝑷 N subscript 𝑷 𝑁\bm{P}_{N}bold_italic_P start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT and 𝑷 C subscript 𝑷 𝐶\bm{P}_{C}bold_italic_P start_POSTSUBSCRIPT italic_C end_POSTSUBSCRIPT. These tensors encompass all nodes and connections within the population, respectively. Formally, the concatenated tensors, 𝑷 N subscript 𝑷 𝑁\bm{P}_{N}bold_italic_P start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT and 𝑷 C subscript 𝑷 𝐶\bm{P}_{C}bold_italic_P start_POSTSUBSCRIPT italic_C end_POSTSUBSCRIPT, are defined as:

𝑷 N subscript 𝑷 𝑁\displaystyle\bm{P}_{N}bold_italic_P start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT=[𝑵^1,𝑵^2,…]∈ℝ P×|N|max×𝚕𝚎𝚗⁢(𝒏),absent subscript^𝑵 1 subscript^𝑵 2…superscript ℝ 𝑃 subscript 𝑁 max 𝚕𝚎𝚗 𝒏\displaystyle=[\hat{\bm{N}}_{1},\hat{\bm{N}}_{2},\ldots]\in\mathbb{R}^{P\times% |N|_{\text{max}}\times\texttt{len}(\bm{n})},= [ over^ start_ARG bold_italic_N end_ARG start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , over^ start_ARG bold_italic_N end_ARG start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ] ∈ blackboard_R start_POSTSUPERSCRIPT italic_P × | italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT × len ( bold_italic_n ) end_POSTSUPERSCRIPT ,
𝑷 C subscript 𝑷 𝐶\displaystyle\bm{P}_{C}bold_italic_P start_POSTSUBSCRIPT italic_C end_POSTSUBSCRIPT=[𝑪^1,𝑪^2,…]∈ℝ P×|C|max×𝚕𝚎𝚗⁢(𝒄),absent subscript^𝑪 1 subscript^𝑪 2…superscript ℝ 𝑃 subscript 𝐶 max 𝚕𝚎𝚗 𝒄\displaystyle=[\hat{\bm{C}}_{1},\hat{\bm{C}}_{2},\ldots]\in\mathbb{R}^{P\times% |C|_{\text{max}}\times\texttt{len}(\bm{c})},= [ over^ start_ARG bold_italic_C end_ARG start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , over^ start_ARG bold_italic_C end_ARG start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ] ∈ blackboard_R start_POSTSUPERSCRIPT italic_P × | italic_C | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT × len ( bold_italic_c ) end_POSTSUPERSCRIPT ,

where P 𝑃 P italic_P denotes the population size, 𝑵^i subscript^𝑵 𝑖\hat{\bm{N}}_{i}over^ start_ARG bold_italic_N end_ARG start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and 𝑪^i subscript^𝑪 𝑖\hat{\bm{C}}_{i}over^ start_ARG bold_italic_C end_ARG start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT represent the tensor representations of the node and connection sets of the i 𝑖 i italic_i-th network, respectively. Fig.[2](https://arxiv.org/html/2404.01817v3#S2.F2 "Figure 2 ‣ 2.1. NeuroEvolution of Augmenting Topologies ‣ 2. Background ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") provides a graphical representation of the tensorized encoding process.

### 3.2. Tensorized Operations

Upon encoding networks as tensors, we subsequently express operations on NEAT networks as corresponding tensor operations. In this subsection, we detail the tensorized representations of three fundamental operations: node modification, connection modification, and attribute modification.

#### 3.2.1. Node Modification

Given a network 𝒩=⟨N,C⟩𝒩 𝑁 𝐶\mathcal{N}=\langle N,C\rangle caligraphic_N = ⟨ italic_N , italic_C ⟩, node set modifications in N 𝑁 N italic_N can involve either the addition of a new node n 𝑛 n italic_n or the removal of an existing node n 𝑛 n italic_n:

N′=N∪{n}or N′=N∖{n}.formulae-sequence superscript 𝑁′𝑁 𝑛 or superscript 𝑁′𝑁 𝑛\displaystyle N^{\prime}=N\cup\{n\}\quad\text{or}\quad N^{\prime}=N\setminus\{% n\}.italic_N start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = italic_N ∪ { italic_n } or italic_N start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = italic_N ∖ { italic_n } .

In the tensorized representation 𝑵^∈ℝ|N|max×𝚕𝚎𝚗⁢(𝒏)^𝑵 superscript ℝ subscript 𝑁 max 𝚕𝚎𝚗 𝒏\hat{\bm{N}}\in\mathbb{R}^{|N|_{\text{max}}\times\texttt{len}(\bm{n})}over^ start_ARG bold_italic_N end_ARG ∈ blackboard_R start_POSTSUPERSCRIPT | italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT × len ( bold_italic_n ) end_POSTSUPERSCRIPT of the node set N 𝑁 N italic_N, the tensorized operation for node addition can be represented as:

𝑵^⁢[r i]←𝒏 new,←^𝑵 delimited-[]subscript 𝑟 𝑖 subscript 𝒏 new\hat{\bm{N}}[r_{i}]\leftarrow\bm{n}_{\text{new}},over^ start_ARG bold_italic_N end_ARG [ italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] ← bold_italic_n start_POSTSUBSCRIPT new end_POSTSUBSCRIPT ,

where r i subscript 𝑟 𝑖 r_{i}italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT denotes the index of the first NaN row in 𝑵^^𝑵\hat{\bm{N}}over^ start_ARG bold_italic_N end_ARG, 𝒏 new subscript 𝒏 new\bm{n}_{\text{new}}bold_italic_n start_POSTSUBSCRIPT new end_POSTSUBSCRIPT stands for the tensor representation of the node being added, [⋅]delimited-[]⋅[\cdot][ ⋅ ] represents tensor slicing, and ←←\leftarrow← indicates the assignment operation.

Conversely, the tensorized operation for node removal can be depicted as:

𝑵^⁢[r j]←𝙽𝚊𝙽,←^𝑵 delimited-[]subscript 𝑟 𝑗 𝙽𝚊𝙽\hat{\bm{N}}[r_{j}]\leftarrow\texttt{NaN},over^ start_ARG bold_italic_N end_ARG [ italic_r start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ] ← NaN ,

with r j subscript 𝑟 𝑗 r_{j}italic_r start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT representing the index of the node for removal.

#### 3.2.2. Connection Modification

Given a network 𝒩=⟨N,C⟩𝒩 𝑁 𝐶\mathcal{N}=\langle N,C\rangle caligraphic_N = ⟨ italic_N , italic_C ⟩, modifications in the connection set C 𝐶 C italic_C can entail either adding a new connection or eliminating an existing connection c 𝑐 c italic_c:

C′=C∪{c}or C′=C∖{c}.formulae-sequence superscript 𝐶′𝐶 𝑐 or superscript 𝐶′𝐶 𝑐\displaystyle C^{\prime}=C\cup\{c\}\quad\text{or}\quad C^{\prime}=C\setminus\{% c\}.italic_C start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = italic_C ∪ { italic_c } or italic_C start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = italic_C ∖ { italic_c } .

In the tensorized representation, 𝑪^^𝑪\hat{\bm{C}}over^ start_ARG bold_italic_C end_ARG, of C 𝐶 C italic_C, the operations of connection addition or removal can be depicted as:

𝑪^⁢[r i]←𝒄 new or 𝑪^⁢[r j]←𝙽𝚊𝙽,formulae-sequence←^𝑪 delimited-[]subscript 𝑟 𝑖 subscript 𝒄 new or←^𝑪 delimited-[]subscript 𝑟 𝑗 𝙽𝚊𝙽\hat{\bm{C}}[r_{i}]\leftarrow\bm{c}_{\text{new}}\quad\text{or}\quad\hat{\bm{C}% }[r_{j}]\leftarrow\texttt{NaN},over^ start_ARG bold_italic_C end_ARG [ italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] ← bold_italic_c start_POSTSUBSCRIPT new end_POSTSUBSCRIPT or over^ start_ARG bold_italic_C end_ARG [ italic_r start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ] ← NaN ,

respectively. Here, r i subscript 𝑟 𝑖 r_{i}italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT indicates the index of the initial NaN row in 𝑪^^𝑪\hat{\bm{C}}over^ start_ARG bold_italic_C end_ARG, 𝒄 new subscript 𝒄 new\bm{c}_{\text{new}}bold_italic_c start_POSTSUBSCRIPT new end_POSTSUBSCRIPT represents the tensor form of the connection being introduced, and r j subscript 𝑟 𝑗 r_{j}italic_r start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT signifies the index of the connection for removal.

#### 3.2.3. Attribute Modification

NEAT is designed not only to modify the network structures but also the internal attributes, either in a node or a connection. Specifically, with node n=(k,attr 1,attr 2,…)𝑛 𝑘 subscript attr 1 subscript attr 2…n=(k,\text{attr}_{1},\text{attr}_{2},\ldots)italic_n = ( italic_k , attr start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , attr start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ), where k∈ℕ 𝑘 ℕ k\in\mathbb{N}italic_k ∈ blackboard_N is the historical marking, and attr i subscript attr 𝑖\text{attr}_{i}attr start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is the i 𝑖 i italic_i-th attribute of n 𝑛 n italic_n, when modifying the j 𝑗 j italic_j-th attribute in a node, the transformation can be represented as:

n′=(k,attr 1,attr 2,…,attr j′,…),superscript 𝑛′𝑘 subscript attr 1 subscript attr 2…subscript superscript attr′𝑗…n^{\prime}=(k,\text{attr}_{1},\text{attr}_{2},\ldots,\text{attr}^{\prime}_{j},% \ldots),italic_n start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = ( italic_k , attr start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , attr start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , attr start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , … ) ,

and the corresponding tensorized operation is:

𝒏⁢[1+j]←attr j′.←𝒏 delimited-[]1 𝑗 subscript superscript attr′𝑗\bm{n}[1+j]\leftarrow\text{attr}^{\prime}_{j}.bold_italic_n [ 1 + italic_j ] ← attr start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT .

Similarly, for a connection c=(k i,k o,e,attr 1,attr 2,…)𝑐 subscript 𝑘 𝑖 subscript 𝑘 𝑜 𝑒 subscript attr 1 subscript attr 2…c=(k_{i},k_{o},e,\text{attr}_{1},\text{attr}_{2},\ldots)italic_c = ( italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT , italic_e , attr start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , attr start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … ), where k i∈ℕ subscript 𝑘 𝑖 ℕ k_{i}\in\mathbb{N}italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_N and k o∈ℕ subscript 𝑘 𝑜 ℕ k_{o}\in\mathbb{N}italic_k start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT ∈ blackboard_N are the historical markings of the input and output nodes of c 𝑐 c italic_c, respectively, e∈{True,False}𝑒 True False e\in\{\text{True},\text{False}\}italic_e ∈ { True , False } is the enabled flag, when modifying the j 𝑗 j italic_j-th attribute in a connection, the transformation can represented as:

c′=(k i,k o,e,attr 1,attr 2,…,attr j′,…),superscript 𝑐′subscript 𝑘 𝑖 subscript 𝑘 𝑜 𝑒 subscript attr 1 subscript attr 2…subscript superscript attr′𝑗…c^{\prime}=(k_{i},k_{o},e,\text{attr}_{1},\text{attr}_{2},\ldots,\text{attr}^{% \prime}_{j},\ldots),italic_c start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = ( italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT , italic_e , attr start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , attr start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , attr start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , … ) ,

and the corresponding tensorized operation is:

𝒄⁢[3+j]←attr j′.←𝒄 delimited-[]3 𝑗 subscript superscript attr′𝑗\bm{c}[3+j]\leftarrow\text{attr}^{\prime}_{j}.bold_italic_c [ 3 + italic_j ] ← attr start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT .

By combining the aforementioned three operations, operations for searching networks in the NEAT algorithm including Mutation and Crossover can be transformed into operations on tensors. Fig.[3](https://arxiv.org/html/2404.01817v3#S3.F3 "Figure 3 ‣ 3. Tensorization of NEAT ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") provides a graphical representation of the tensorized network operations.

### 3.3. Tensorized Network Inference

Another crucial component in NEAT is the inference process, where a network receives inputs and generates corresponding outputs based on its topologies and weights. For a network 𝒩 𝒩\mathcal{N}caligraphic_N, given its node tensor 𝑵^^𝑵\hat{\bm{N}}over^ start_ARG bold_italic_N end_ARG and connection tensor 𝑪^^𝑪\hat{\bm{C}}over^ start_ARG bold_italic_C end_ARG, the inference process can be represented as:

𝑶=inference 𝑵^,𝑪^⁢(𝑰),𝑶 subscript inference^𝑵^𝑪 𝑰\bm{O}=\text{inference}_{\hat{\bm{N}},\hat{\bm{C}}}(\bm{I}),bold_italic_O = inference start_POSTSUBSCRIPT over^ start_ARG bold_italic_N end_ARG , over^ start_ARG bold_italic_C end_ARG end_POSTSUBSCRIPT ( bold_italic_I ) ,

where 𝑶 𝑶\bm{O}bold_italic_O and 𝑰 𝑰\bm{I}bold_italic_I denote the outputs and inputs in the inference process, respectively.

In our tensorization method, the inference process is divided into two stages: transformation and calculation. ‘Transformation’ involves converting the node tensor 𝑵^^𝑵\hat{\bm{N}}over^ start_ARG bold_italic_N end_ARG and connection tensor 𝑪^^𝑪\hat{\bm{C}}over^ start_ARG bold_italic_C end_ARG into formats more conducive to network inference. ‘Calculation’ refers to computing the output using the tensors produced in the transformation stage. When a network undergoes multiple inference operations, it only needs to be transformed once. Networks in the NEAT algorithm can be categorized as either feedforward or recurrent, based on the presence or absence of cycles in their topological structure. Here, we primarily focus on the transformation and calculation processes in feedforward networks.

In feedforward networks, the transformation process creates two new tensors: the topological order of nodes 𝑵 order∈ℝ|N|max subscript 𝑵 order superscript ℝ subscript 𝑁 max\bm{N}_{\text{order}}\in\mathbb{R}^{|N|_{\text{max}}}bold_italic_N start_POSTSUBSCRIPT order end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT | italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT end_POSTSUPERSCRIPT and the expanded connections 𝑪 exp∈ℝ|N|max×|N|max×𝚗𝚘𝚊⁢(c)subscript 𝑪 exp superscript ℝ subscript 𝑁 max subscript 𝑁 max 𝚗𝚘𝚊 𝑐\bm{C}_{\text{exp}}\in\mathbb{R}^{|N|_{\text{max}}\times|N|_{\text{max}}\times% \texttt{noa}(c)}bold_italic_C start_POSTSUBSCRIPT exp end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT | italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT × | italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT × noa ( italic_c ) end_POSTSUPERSCRIPT, where |N|max subscript 𝑁 max|N|_{\text{max}}| italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT represents the predefined maximum limit for the number of nodes and 𝚗𝚘𝚊⁢(c)𝚗𝚘𝚊 𝑐\texttt{noa}(c)noa ( italic_c ) denoting the number of attributes of connections in the network.

Given the absence of cycles in the network, topological sorting is employed to obtain 𝑵 order subscript 𝑵 order\bm{N}_{\text{order}}bold_italic_N start_POSTSUBSCRIPT order end_POSTSUBSCRIPT. For 𝑪 exp subscript 𝑪 exp\bm{C}_{\text{exp}}bold_italic_C start_POSTSUBSCRIPT exp end_POSTSUBSCRIPT, the generation rule can be represented as:

𝑪 exp⁢[𝑪^⁢[i]⁢[0,1]]←{𝙽𝚊𝙽,𝑪^⁢[i]⁢[2]=0 𝑪^[i][2:],𝑪^⁢[i]⁢[2]=1,\bm{C}_{\text{exp}}[\hat{\bm{C}}[i][0,1]]\leftarrow\begin{cases}\texttt{NaN},&% \hat{\bm{C}}[i][2]=0\\ \hat{\bm{C}}[i][2:],&\hat{\bm{C}}[i][2]=1\end{cases},bold_italic_C start_POSTSUBSCRIPT exp end_POSTSUBSCRIPT [ over^ start_ARG bold_italic_C end_ARG [ italic_i ] [ 0 , 1 ] ] ← { start_ROW start_CELL NaN , end_CELL start_CELL over^ start_ARG bold_italic_C end_ARG [ italic_i ] [ 2 ] = 0 end_CELL end_ROW start_ROW start_CELL over^ start_ARG bold_italic_C end_ARG [ italic_i ] [ 2 : ] , end_CELL start_CELL over^ start_ARG bold_italic_C end_ARG [ italic_i ] [ 2 ] = 1 end_CELL end_ROW ,

where i=0,1,2,…,|C|max 𝑖 0 1 2…subscript 𝐶 max i=0,1,2,\ldots,|C|_{\text{max}}italic_i = 0 , 1 , 2 , … , | italic_C | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT, and |C|max subscript 𝐶 max|C|_{\text{max}}| italic_C | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT is the predefined maximum limit for the number of connections. Recall that in each line c 𝑐 c italic_c of 𝑪^^𝑪\hat{\bm{C}}over^ start_ARG bold_italic_C end_ARG, the values are (k i,k o,e,attr⁢1,attr⁢2,…)subscript 𝑘 𝑖 subscript 𝑘 𝑜 𝑒 attr 1 attr 2…(k_{i},k_{o},e,\text{attr}1,\text{attr}2,\ldots)( italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT , italic_e , attr 1 , attr 2 , … ), with c⁢[0]=k i 𝑐 delimited-[]0 subscript 𝑘 𝑖 c[0]=k_{i}italic_c [ 0 ] = italic_k start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and c⁢[1]=k o 𝑐 delimited-[]1 subscript 𝑘 𝑜 c[1]=k_{o}italic_c [ 1 ] = italic_k start_POSTSUBSCRIPT italic_o end_POSTSUBSCRIPT indicating the indices of the input and output nodes of the connection, respectively, and c⁢[2]=e∈{𝚃𝚛𝚞𝚎,𝙵𝚊𝚕𝚜𝚎}𝑐 delimited-[]2 𝑒 𝚃𝚛𝚞𝚎 𝙵𝚊𝚕𝚜𝚎 c[2]=e\in\{\texttt{True},\texttt{False}\}italic_c [ 2 ] = italic_e ∈ { True , False } representing the enabled flag. Locations not updated by this rule default to the value NaN. The tensors 𝑵^^𝑵\hat{\bm{N}}over^ start_ARG bold_italic_N end_ARG, 𝑵 order subscript 𝑵 order\bm{N}_{\text{order}}bold_italic_N start_POSTSUBSCRIPT order end_POSTSUBSCRIPT, and 𝑪 exp subscript 𝑪 exp\bm{C}_{\text{exp}}bold_italic_C start_POSTSUBSCRIPT exp end_POSTSUBSCRIPT are then used as inputs for the calculation process.

In the forward process, we utilize the transformed tensors 𝑵^^𝑵\hat{\bm{N}}over^ start_ARG bold_italic_N end_ARG, 𝑵 order subscript 𝑵 order\bm{N}_{\text{order}}bold_italic_N start_POSTSUBSCRIPT order end_POSTSUBSCRIPT, 𝑪 exp subscript 𝑪 exp\bm{C}_{\text{exp}}bold_italic_C start_POSTSUBSCRIPT exp end_POSTSUBSCRIPT, and the input 𝑰 𝑰\bm{I}bold_italic_I to calculate the output 𝑶 𝑶\bm{O}bold_italic_O. We maintain a tensor 𝒗∈ℝ|N|max 𝒗 superscript ℝ subscript 𝑁 max\bm{v}\in\mathbb{R}^{|N|_{\text{max}}}bold_italic_v ∈ blackboard_R start_POSTSUPERSCRIPT | italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT end_POSTSUPERSCRIPT to store the values of nodes in the network. Initially, 𝒗 𝒗\bm{v}bold_italic_v is set to the default value NaN and then updated with:

𝒗⁢[k input]←𝑰,←𝒗 delimited-[]subscript 𝑘 input 𝑰\bm{v}[k_{\text{input}}]\leftarrow\bm{I},bold_italic_v [ italic_k start_POSTSUBSCRIPT input end_POSTSUBSCRIPT ] ← bold_italic_I ,

where k input subscript 𝑘 input k_{\text{input}}italic_k start_POSTSUBSCRIPT input end_POSTSUBSCRIPT denotes the indices of input nodes in the network. The value of nodes is iteratively calculated in the order specified by 𝑵 order subscript 𝑵 order\bm{N}_{\text{order}}bold_italic_N start_POSTSUBSCRIPT order end_POSTSUBSCRIPT. The rule to obtain the value 𝒗⁢[k]𝒗 delimited-[]𝑘\bm{v}[k]bold_italic_v [ italic_k ] of node 𝒏 k subscript 𝒏 𝑘\bm{n}_{k}bold_italic_n start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT can be expressed as:

𝒗[k]←f n(f c(𝒗|𝑪 exp[:][k])|𝑵^[k][1:]),\displaystyle\bm{v}[k]\leftarrow f_{n}(f_{c}(\bm{v}\ |\ \bm{C}_{\text{exp}}[:]% [k])\ |\ \hat{\bm{N}}[k][1:]),bold_italic_v [ italic_k ] ← italic_f start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT ( italic_f start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT ( bold_italic_v | bold_italic_C start_POSTSUBSCRIPT exp end_POSTSUBSCRIPT [ : ] [ italic_k ] ) | over^ start_ARG bold_italic_N end_ARG [ italic_k ] [ 1 : ] ) ,

where 𝑪 exp⁢[:]⁢[k]subscript 𝑪 exp delimited-[]:delimited-[]𝑘\bm{C}_{\text{exp}}[:][k]bold_italic_C start_POSTSUBSCRIPT exp end_POSTSUBSCRIPT [ : ] [ italic_k ] indicates the attributes of all connections to 𝒏 k subscript 𝒏 𝑘\bm{n}_{k}bold_italic_n start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT, and 𝑵^[k][1:]\hat{\bm{N}}[k][1:]over^ start_ARG bold_italic_N end_ARG [ italic_k ] [ 1 : ] denotes the attributes of 𝒏 k subscript 𝒏 𝑘\bm{n}_{k}bold_italic_n start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. f c subscript 𝑓 𝑐 f_{c}italic_f start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT and f n subscript 𝑓 𝑛 f_{n}italic_f start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT are the calculation functions for connections and nodes in the network, respectively. For a network with connection attribute weight w 𝑤 w italic_w, and node attributes bias b 𝑏 b italic_b, aggregation function f agg subscript 𝑓 agg f_{\text{agg}}italic_f start_POSTSUBSCRIPT agg end_POSTSUBSCRIPT and activation function f act subscript 𝑓 act f_{\text{act}}italic_f start_POSTSUBSCRIPT act end_POSTSUBSCRIPT, f c subscript 𝑓 𝑐 f_{c}italic_f start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT and f n subscript 𝑓 𝑛 f_{n}italic_f start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT can be defined as:

f c⁢(𝑰 conn|w)subscript 𝑓 𝑐 conditional subscript 𝑰 conn 𝑤\displaystyle f_{c}(\bm{I}_{\text{conn}}\ |\ w)italic_f start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT ( bold_italic_I start_POSTSUBSCRIPT conn end_POSTSUBSCRIPT | italic_w )=w⁢𝑰 conn,absent 𝑤 subscript 𝑰 conn\displaystyle=w\bm{I}_{\text{conn}},= italic_w bold_italic_I start_POSTSUBSCRIPT conn end_POSTSUBSCRIPT ,
f n⁢(𝑰 node|b,f agg,f act)subscript 𝑓 𝑛 conditional subscript 𝑰 node 𝑏 subscript 𝑓 agg subscript 𝑓 act\displaystyle f_{n}(\bm{I}_{\text{node}}\ |\ b,f_{\text{agg}},f_{\text{act}})italic_f start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT ( bold_italic_I start_POSTSUBSCRIPT node end_POSTSUBSCRIPT | italic_b , italic_f start_POSTSUBSCRIPT agg end_POSTSUBSCRIPT , italic_f start_POSTSUBSCRIPT act end_POSTSUBSCRIPT )=f act⁢(f agg⁢(𝑰 node)+b),absent subscript 𝑓 act subscript 𝑓 agg subscript 𝑰 node 𝑏\displaystyle=f_{\text{act}}(f_{\text{agg}}(\bm{I}_{\text{node}})+b),= italic_f start_POSTSUBSCRIPT act end_POSTSUBSCRIPT ( italic_f start_POSTSUBSCRIPT agg end_POSTSUBSCRIPT ( bold_italic_I start_POSTSUBSCRIPT node end_POSTSUBSCRIPT ) + italic_b ) ,

where 𝑰 conn subscript 𝑰 conn\bm{I}_{\text{conn}}bold_italic_I start_POSTSUBSCRIPT conn end_POSTSUBSCRIPT and 𝑰 nodes subscript 𝑰 nodes\bm{I}_{\text{nodes}}bold_italic_I start_POSTSUBSCRIPT nodes end_POSTSUBSCRIPT denote the inputs for connections and nodes, respectively.

After computing the values of all nodes, the tensor 𝒗⁢[k output]𝒗 delimited-[]subscript 𝑘 output\bm{v}[k_{\text{output}}]bold_italic_v [ italic_k start_POSTSUBSCRIPT output end_POSTSUBSCRIPT ] represents the network’s output, with k output subscript 𝑘 output k_{\text{output}}italic_k start_POSTSUBSCRIPT output end_POSTSUBSCRIPT indicating the indices of output nodes in the network. Fig.[4](https://arxiv.org/html/2404.01817v3#S3.F4 "Figure 4 ‣ 3. Tensorization of NEAT ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") graphically depicts the tensorized network inference process.

4. Implementation of TensorNEAT
-------------------------------

### 4.1. JAX-based Hardware Acceleration

JAX (Frostig et al., [2018](https://arxiv.org/html/2404.01817v3#bib.bib8)) is an open-source numerical computing library, offering APIs similar to NumPy (Harris et al., [2020](https://arxiv.org/html/2404.01817v3#bib.bib12)) and enabling efficient execution across various hardware platforms (CPU/GPU/TPU). Leveraging the capabilities of XLA, JAX facilitates the transformation of numerical code into optimized machine instructions. The optimization techniques provided by JAX have supported the development of numerous projects in evolutionary computation, as evidenced by (Tang et al., [2022](https://arxiv.org/html/2404.01817v3#bib.bib33)), (Lange, [2023](https://arxiv.org/html/2404.01817v3#bib.bib15)), (Lim et al., [2022](https://arxiv.org/html/2404.01817v3#bib.bib17)), and (Huang et al., [2024](https://arxiv.org/html/2404.01817v3#bib.bib13)). These advancements significantly contribute to JAX’s growing popularity in the scientific community.

TensorNEAT, integrating JAX, utilizes the functional programming paradigm to implement our proposed tensorization methods. This integration allows NEAT to effectively use hardware accelerators such as GPUs and TPUs. With uniform tensor shapes in network encoding, several NEAT operations such as mutation, crossover, and network inference can be vectorized across the population dimension. This vectorization, leveraging jax.vmap and jax.pmap functions, is suitable for both single and multi-device configurations.

### 4.2. User-friendly Interfaces

Designed with user-friendly interfaces, TensorNEAT provides mechanisms for adjusting algorithms to specific requirements. It offers an extensive set of hyperparameters, allowing users to fine-tune various computational elements of the NEAT algorithm. Additionally, its modular problem templates facilitate the integration of specialized problems. TensorNEAT also offers several open interfaces. By implementing a few functions, users can define the behavior of networks within the NEAT algorithm. A notable feature is the interface supporting the evolution of advanced network architectures, including Spiking Neural Networks (Ghosh-Dastidar and Adeli, [2009](https://arxiv.org/html/2404.01817v3#bib.bib11)) and Binary Neural Networks (hubara2016binarized). Details on the hyperparameters and the interfaces are elaborated in Appendix[B](https://arxiv.org/html/2404.01817v3#A2 "Appendix B Hyperparameters ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") and Appendix[C](https://arxiv.org/html/2404.01817v3#A3 "Appendix C Interfaces ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration").

### 4.3. Feature-rich Extensions

Extending beyond the conventional NEAT paradigm, TensorNEAT includes notable algorithmic extensions such as Compositional Pattern Producing Networks (CPPN) (Stanley, [2007](https://arxiv.org/html/2404.01817v3#bib.bib28)) and HyperNEAT (Stanley et al., [2009](https://arxiv.org/html/2404.01817v3#bib.bib31)), tailored for parallel processing on hardware accelerators. For evaluation, TensorNEAT provides a suite of standard test benchmarks, covering areas from numerical optimization to function approximation. Moreover, it integrates seamlessly with leading reinforcement learning environments like Gym (Brockman et al., [2016](https://arxiv.org/html/2404.01817v3#bib.bib5)), and hardware-optimized platforms such as gymnax (Lange, [2022](https://arxiv.org/html/2404.01817v3#bib.bib14)) and Brax (Freeman et al., [2021](https://arxiv.org/html/2404.01817v3#bib.bib7)), thus allowing users to evaluate the performance of the NEAT algorithm in various settings.

5. Experiment
-------------

This section presents an empirical comparison between TensorNEAT and NEAT-Python, focusing on their performance in three robotics control tasks: Swimmer, Hopper, and Halfcheetah, within the Brax environment. The experiments were conducted with both TensorNEAT and NEAT-Python configured using uniform parameter settings. Data presented are the average outcomes from ten independent trials, complete with 95% confidence intervals to ensure statistical robustness. Details on the experiment settings are shown in Appendix[D](https://arxiv.org/html/2404.01817v3#A4 "Appendix D Experiment Detail ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration").

Table 1. Runtimes over 100 100 100 100 generations on different hardware configurations, with a constant population size of 10,000 10 000 10,000 10 , 000.

Task Framework Hardware Time (s 𝑠 s italic_s)Speedup
Swimmer NEAT-Python EPYC 7543 (CPU)42279.63±3031.97 plus-or-minus 42279.63 3031.97 42279.63\pm 3031.97 42279.63 ± 3031.97 1.00 1.00 1.00 1.00
TensorNEAT RTX 4090 215.70±6.35 plus-or-minus 215.70 6.35\bm{215.70\pm 6.35}bold_215.70 bold_± bold_6.35 196.01 196.01\bm{196.01}bold_196.01
RTX 3090 292.22±20.19 plus-or-minus 292.22 20.19 292.22\pm 20.19 292.22 ± 20.19 144.68 144.68 144.68 144.68
RTX 2080Ti 434.94±12.49 plus-or-minus 434.94 12.49 434.94\pm 12.49 434.94 ± 12.49 97.21 97.21 97.21 97.21
EPYC 7543 (CPU)14678.10±616.85 plus-or-minus 14678.10 616.85 14678.10\pm 616.85 14678.10 ± 616.85 2.88 2.88 2.88 2.88
Hopper NEAT-Python EPYC 7543 (CPU)14438.13±900.68 plus-or-minus 14438.13 900.68 14438.13\pm 900.68 14438.13 ± 900.68 1.00 1.00 1.00 1.00
TensorNEAT RTX 4090 241.30±9.41 plus-or-minus 241.30 9.41\bm{241.30\pm 9.41}bold_241.30 bold_± bold_9.41 59.83 59.83\bm{59.83}bold_59.83
RTX 3090 336.08±26.96 plus-or-minus 336.08 26.96 336.08\pm 26.96 336.08 ± 26.96 42.96 42.96 42.96 42.96
RTX 2080Ti 518.40±7.22 plus-or-minus 518.40 7.22 518.40\pm 7.22 518.40 ± 7.22 27.85 27.85 27.85 27.85
EPYC 7543 (CPU)13473.01±544.07 plus-or-minus 13473.01 544.07 13473.01\pm 544.07 13473.01 ± 544.07 1.07 1.07 1.07 1.07
Halfcheetah NEAT-Python EPYC 7543 (CPU)149516.00±4817.90 plus-or-minus 149516.00 4817.90 149516.00\pm 4817.90 149516.00 ± 4817.90 1.00 1.00 1.00 1.00
TensorNEAT RTX 4090 274.74±14.21 plus-or-minus 274.74 14.21\bm{274.74\pm 14.21}bold_274.74 bold_± bold_14.21 544.21 544.21\bm{544.21}bold_544.21
RTX 3090 487.82±19.05 plus-or-minus 487.82 19.05 487.82\pm 19.05 487.82 ± 19.05 306.50 306.50 306.50 306.50
RTX 2080Ti 705.13±16.02 plus-or-minus 705.13 16.02 705.13\pm 16.02 705.13 ± 16.02 212.04 212.04 212.04 212.04
EPYC 7543 (CPU)15914.08±4005.08 plus-or-minus 15914.08 4005.08 15914.08\pm 4005.08 15914.08 ± 4005.08 9.40 9.40 9.40 9.40

First, we examined the evolution of average population fitness and the cumulative runtime of the algorithms across generations, with a constant population size of 10,000. As depicted in Fig.[5](https://arxiv.org/html/2404.01817v3#S5.F5 "Figure 5 ‣ 5. Experiment ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration"), TensorNEAT exhibits a more rapid improvement in population fitness over the course of the algorithm’s execution. This performance disparity between the two frameworks stems from the modification in the NEAT algorithm after tensorization, including network encoding and the computation of distances between networks. Furthermore, the analysis of execution times, illustrated in the final panel of Fig.[5](https://arxiv.org/html/2404.01817v3#S5.F5 "Figure 5 ‣ 5. Experiment ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration"), reveals a significant decrease in runtime with TensorNEAT compared to NEAT-Python.

![Image 6: Refer to caption](https://arxiv.org/html/2404.01817v3/x6.png)

![Image 7: Refer to caption](https://arxiv.org/html/2404.01817v3/x7.png)

![Image 8: Refer to caption](https://arxiv.org/html/2404.01817v3/x8.png)

![Image 9: Refer to caption](https://arxiv.org/html/2404.01817v3/x9.png)

![Image 10: Refer to caption](https://arxiv.org/html/2404.01817v3/x10.png)

![Image 11: Refer to caption](https://arxiv.org/html/2404.01817v3/x11.png)

Figure 5. Average fitness and wall-clock time against generation.

![Image 12: Refer to caption](https://arxiv.org/html/2404.01817v3/x12.png)

![Image 13: Refer to caption](https://arxiv.org/html/2404.01817v3/x13.png)

![Image 14: Refer to caption](https://arxiv.org/html/2404.01817v3/x14.png)

![Image 15: Refer to caption](https://arxiv.org/html/2404.01817v3/x15.png)

![Image 16: Refer to caption](https://arxiv.org/html/2404.01817v3/x16.png)

![Image 17: Refer to caption](https://arxiv.org/html/2404.01817v3/x17.png)

Figure 6. Wall-clock time against generation and population size. 

Given the iterative nature of NEAT’s process, there is an expected increase in per-iteration time as network structures become more complex. This phenomenon was investigated by comparing the per-generation runtimes of both algorithms. In the Swimmer and Hopper tasks, as shown in the left panels of Fig.[6](https://arxiv.org/html/2404.01817v3#S5.F6 "Figure 6 ‣ 5. Experiment ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration"), NEAT-Python exhibits a more marked increase in runtime, likely due to its less efficient object encoding mechanism, which becomes increasingly cumbersome with rising network complexity. By contrast, TensorNEAT, employing a tensorized encoding approach constrained by the pre-set maximum values of |N|max subscript 𝑁 max|N|_{\text{max}}| italic_N | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT and |C|max subscript 𝐶 max|C|_{\text{max}}| italic_C | start_POSTSUBSCRIPT max end_POSTSUBSCRIPT, achieves a consistent network encoding size, resulting in more stable iteration times.

Additionally, we investigated how runtime varies with changes in population size, ranging from 50 to 10,000. As illustrated in the right panels of Fig.[6](https://arxiv.org/html/2404.01817v3#S5.F6 "Figure 6 ‣ 5. Experiment ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration"), NEAT-Python’s runtime significantly increases with larger populations, whereas TensorNEAT only experiences a marginal increase in runtime. TensorNEAT demonstrates advantages across various population sizes, with its benefits being more pronounced in larger populations.

The adaptability of TensorNEAT was further validated by evaluating its performance across various hardware configurations, including the AMD EPYC 7543 CPU and a selection of mainstream GPU models. These evaluations were conducted in the Cart Pole environment, with a consistent population size of 10,000 10 000 10,000 10 , 000. The total wall-clock time was recorded over 100 100 100 100 generations of the algorithm, and the results are summarized in Table[1](https://arxiv.org/html/2404.01817v3#S5.T1 "Table 1 ‣ 5. Experiment ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration"). In every GPU configuration, TensorNEAT demonstrated a significant performance advantage over NEAT-Python. Notably, with the RTX 4090, TensorNEAT achieved a speedup surpassing 500×500\times 500 × in the Halfcheetah environment. It is also noteworthy that TensorNEAT realized a speedup on the same CPU device in comparison to NEAT-Python, especially in the more complex Halfcheetah environment.

Our experiments highlight TensorNEAT’s considerable superiority to NEAT-Python in terms of algorithmic execution speed. Moreover, as the computational demands increase, whether due to larger network structures or higher population sizes, TensorNEAT’s relative time consumption grows more slowly in contrast to NEAT-Python. These findings emphasize the significant acceleration and performance efficiency of the TensorNEAT algorithm, especially with GPU accelerations.

6. Conclusion
-------------

In this work, we address the scalability challenge of NeuroEvolution of Augmenting Topologies (NEAT) by introducing a tensorization method, which transforms the network topologies into uniformly shaped tensors for efficient parallel processing. Building upon the tensorization method, we develop TensorNEAT, which leverages JAX for automated function vectorization and hardware acceleration. Compatible with environments like Gym, Brax, and gymnax, TensorNEAT significantly outperforms traditional NEAT implementations, achieving over 500x speedups in various robotics control tasks.

Looking ahead, our roadmap for TensorNEAT includes expanding its reach to distributed computing environments, transcending the limitations of single-machine setups. Furthermore, we plan to augment TensorNEAT’s suite of functionalities by integrating advanced NEAT variants, such as DeepNEAT and CoDeepNEAT (Miikkulainen et al., [2019](https://arxiv.org/html/2404.01817v3#bib.bib19)), to further enhance its potential in solving complex neuroevolution challenges.

References
----------

*   (1)
*   Abadi et al. (2015) Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2015. _TensorFlow, Large-scale machine learning on heterogeneous systems_. [https://doi.org/10.5281/zenodo.4724125](https://doi.org/10.5281/zenodo.4724125)
*   Auerbach and Bongard (2011) Joshua E. Auerbach and Josh C. Bongard. 2011. Evolving complete robots with CPPN-NEAT: The utility of recurrent connections. In _Proceedings of the Annual Conference on Genetic and Evolutionary Computation_. 1475–1482. 
*   b2developer (2022) b2developer. 2022. MonopolyNEAT: NEAT implemented into Monopoly with a knockout tournament scheme. [https://github.com/b2developer/MonopolyNEAT](https://github.com/b2developer/MonopolyNEAT)Accessed: 2023-08-15. 
*   Brockman et al. (2016) Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. 2016. OpenAI Gym. arXiv:1606.01540 
*   Brown et al. (2020) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. _Advances in Neural Information Processing Systems_ 33 (2020), 1877–1901. 
*   Freeman et al. (2021) C.Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. 2021. Brax–A Differentiable Physics Engine for Large Scale Rigid Body Simulation. In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks_. 
*   Frostig et al. (2018) Roy Frostig, Matthew James Johnson, and Chris Leary. 2018. Compiling machine learning programs via high-level tracing. _Systems for Machine Learning_ 4, 9 (2018). 
*   Gajewsky (2023) Alex Gajewsky. 2023. PyTorch NEAT. [https://github.com/uber-research/PyTorch-NEAT](https://github.com/uber-research/PyTorch-NEAT)Accessed: 2023-08-07. 
*   Gao and Lan (2021) Zhenyu Gao and Gongjin Lan. 2021. A NEAT-Based Multiclass Classification Method with Class Binarization. In _Proceedings of the Genetic and Evolutionary Computation Conference Companion_. Association for Computing Machinery, New York, NY, USA, 277–278. [https://doi.org/10.1145/3449726.3459509](https://doi.org/10.1145/3449726.3459509)
*   Ghosh-Dastidar and Adeli (2009) Samanwoy Ghosh-Dastidar and Hojjat Adeli. 2009. Spiking neural networks. _International Journal of Neural Systems_ 19, 04 (2009), 295–308. 
*   Harris et al. (2020) Charles R. Harris, K.Jarrod Millman, Stéfan J. Van Der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, et al. 2020. Array programming with NumPy. _Nature_ 585, 7825 (2020), 357–362. 
*   Huang et al. (2024) Beichen Huang, Ran Cheng, Zhuozhao Li, Yaochu Jin, and Kay Chen Tan. 2024. EvoX: A Distributed GPU-accelerated Framework for Scalable Evolutionary Computation. _IEEE Transactions on Evolutionary Computation_ (2024). [https://doi.org/10.1109/TEVC.2024.3388550](https://doi.org/10.1109/TEVC.2024.3388550)
*   Lange (2022) Robert Tjarko Lange. 2022. _gymnax: A JAX-based Reinforcement Learning Environment Library_. [http://github.com/RobertTLange/gymnax](http://github.com/RobertTLange/gymnax)
*   Lange (2023) Robert Tjarko Lange. 2023. evosax: JAX-Based Evolution Strategies. In _Proceedings of the Companion Conference on Genetic and Evolutionary Computation_ (Lisbon, Portugal) _(GECCO ’23 Companion)_. Association for Computing Machinery, New York, NY, USA, 659–662. [https://doi.org/10.1145/3583133.3590733](https://doi.org/10.1145/3583133.3590733)
*   Lehman and Stanley (2011) Joel Lehman and Kenneth O. Stanley. 2011. Evolving a diversity of creatures through novelty search and local competition. In _Proceedings of the Annual Conference on Genetic and Evolutionary Computation_. 211–218. 
*   Lim et al. (2022) Bryan Lim, Maxime Allard, Luca Grillotti, and Antoine Cully. 2022. QDax: On the benefits of massive parallelization for quality-diversity. In _Proceedings of the Genetic and Evolutionary Computation Conference Companion_. Association for Computing Machinery, New York, NY, USA, 128–131. [https://doi.org/10.1145/3520304.3528927](https://doi.org/10.1145/3520304.3528927)
*   McIntyre et al. (2023) Alan McIntyre, Matt Kallada, Cesar G. Miguel, Carolina Feher de Silva, and Marcio Lobo Netto. 2023. Python implementation of the NEAT neuroevolution algorithm. [https://github.com/CodeReclaimers/neat-python](https://github.com/CodeReclaimers/neat-python)Accessed: 2023-08-07. 
*   Miikkulainen et al. (2019) Risto Miikkulainen, Jason Liang, Elliot Meyerson, Aditya Rawal, Daniel Fink, Olivier Francon, Bala Raju, Hormoz Shahrzad, Arshak Navruzyan, Nigel Duffy, et al. 2019. Evolving Deep Neural Networks. In _Artificial Intelligence in the Age of Neural Networks and Brain Computing_. Elsevier, 293–312. 
*   Mouret and Clune (2015) Jean-Baptiste Mouret and Jeff Clune. 2015. Illuminating search spaces by mapping elites. arXiv:1504.04909[cs.AI] 
*   Paszke et al. (2019) Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An imperative style, high-performance deep learning library. In _Advances in Neural Information Processing Systems_, H.Wallach, H.Larochelle, A.Beygelzimer, F.d’Alché Buc, E.Fox, and R.Garnett (Eds.). Curran Associates, Inc., 8024–8035. [http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)
*   peter ch (2019) peter ch. 2019. MultiNEAT: Portable NeuroEvolution library. [https://github.com/peter-ch/MultiNEAT](https://github.com/peter-ch/MultiNEAT)Accessed: 2023-08-15. 
*   Pham et al. (2018) Son Pham, Keyi Zhang, Tung Phan, Jasper Ding, and Christopher Dancy. 2018. Playing SNES games with neuroevolution of augmenting topologies. In _Proceedings of the AAAI Conference on Artificial Intelligence - Student Abstract Track_, Vol.32. 
*   Pinitas et al. (2022) Kosmas Pinitas, Konstantinos Makantasis, Antonios Liapis, and Georgios N. Yannakakis. 2022. RankNEAT: Outperforming stochastic gradient search in preference learning tasks. In _Proceedings of the Genetic and Evolutionary Computation Conference_. Association for Computing Machinery, New York, NY, USA, 1084–1092. [https://doi.org/10.1145/3512290.3528744](https://doi.org/10.1145/3512290.3528744)
*   Risi et al. (2010) Sebastian Risi, Joel Lehman, and Kenneth O. Stanley. 2010. Evolving the placement and density of neurons in the hyperneat substrate. In _Proceedings of the Annual Conference on Genetic and Evolutionary Computation_. 563–570. 
*   Sarti and Ochoa (2021) Stefano Sarti and Gabriela Ochoa. 2021. A NEAT visualisation of neuroevolution trajectories. In _Applications of Evolutionary Computation_. Springer, 714–728. 
*   Silva et al. (2012) Fernando Silva, Paulo Urbano, Sancho Oliveira, and Anders Lyhne Christensen. 2012. odNEAT: An algorithm for distributed online, onboard evolution of robot behaviours. In _The International Conference on the Synthesis and Simulation of Living Systems_. 251–258. [https://doi.org/10.1162/978-0-262-31050-5-ch034](https://doi.org/10.1162/978-0-262-31050-5-ch034)
*   Stanley (2007) Kenneth O. Stanley. 2007. Compositional pattern producing networks: A novel abstraction of development. _Genetic Programming and Evolvable Machines_ 8 (2007), 131–162. 
*   Stanley et al. (2006) Kenneth O. Stanley, Bobby D. Bryant, Igor Karpov, and Risto Miikkulainen. 2006. Real-time evolution of neural networks in the NERO video game. In _Proceedings of the 21st National Conference on Artificial Intelligence_. AAAI Press, 1671–1674. 
*   Stanley et al. (2019) Kenneth O. Stanley, Jeff Clune, Joel Lehman, and Risto Miikkulainen. 2019. Designing neural networks through neuroevolution. _Nature Machine Intelligence_ 1, 1 (2019), 24–35. 
*   Stanley et al. (2009) Kenneth O. Stanley, David B. D’Ambrosio, and Jason Gauci. 2009. A hypercube-based encoding for evolving large-scale neural networks. _Artificial Life_ 15, 2 (04 2009), 185–212. [https://doi.org/10.1162/artl.2009.15.2.15202](https://doi.org/10.1162/artl.2009.15.2.15202)
*   Stanley and Miikkulainen (2002) Kenneth O. Stanley and Risto Miikkulainen. 2002. Evolving neural networks through augmenting topologies. _Evolutionary Computation_ 10, 2 (2002), 99–127. 
*   Tang et al. (2022) Yujin Tang, Yingtao Tian, and David Ha. 2022. EvoJAX: Hardware-accelerated neuroevolution. In _Proceedings of the Genetic and Evolutionary Computation Conference Companion_. Association for Computing Machinery, New York, NY, USA, 308–311. [https://doi.org/10.1145/3520304.3528770](https://doi.org/10.1145/3520304.3528770)
*   Yuksel (2018) Mehmet Erkan Yuksel. 2018. Agent-based evacuation modeling with multiple exits using NeuroEvolution of Augmenting Topologies. _Advanced Engineering Informatics_ 35 (2018), 30–55. 

Appendix A NeuroEvolution of Augmenting Topologies
--------------------------------------------------

Algorithm 1 Main Process of the NEAT algorithm

0:

P 𝑃 P italic_P
(population size),

I 𝐼 I italic_I
(number of input nodes),

O 𝑂 O italic_O
(number of output nodes),

f target subscript 𝑓 target f_{\text{target}}italic_f start_POSTSUBSCRIPT target end_POSTSUBSCRIPT
(target fitness value),

G 𝐺 G italic_G
(maximum number of generations)

0:

b⁢e⁢s⁢t 𝑏 𝑒 𝑠 𝑡 best italic_b italic_e italic_s italic_t

p⁢o⁢p←←𝑝 𝑜 𝑝 absent pop\leftarrow italic_p italic_o italic_p ←
initialize

P 𝑃 P italic_P
networks with

I 𝐼 I italic_I
,

O 𝑂 O italic_O

for

g=1 𝑔 1 g=1 italic_g = 1
to

G 𝐺 G italic_G
do

f⁢i⁢t←←𝑓 𝑖 𝑡 absent fit\leftarrow italic_f italic_i italic_t ←
evaluate fitness values of

P⁢o⁢p 𝑃 𝑜 𝑝 Pop italic_P italic_o italic_p

if

max⁡(f⁢i⁢t)≥f target 𝑓 𝑖 𝑡 subscript 𝑓 target\max(fit)\geq f_{\text{target}}roman_max ( italic_f italic_i italic_t ) ≥ italic_f start_POSTSUBSCRIPT target end_POSTSUBSCRIPT
then

break

end if

s⁢p⁢e⁢c⁢i⁢e⁢s←←𝑠 𝑝 𝑒 𝑐 𝑖 𝑒 𝑠 absent species\leftarrow italic_s italic_p italic_e italic_c italic_i italic_e italic_s ←
divide

p⁢o⁢p 𝑝 𝑜 𝑝 pop italic_p italic_o italic_p
by distances between networks

p⁢o⁢p*←{}←𝑝 𝑜 superscript 𝑝 pop^{*}\leftarrow\{\}italic_p italic_o italic_p start_POSTSUPERSCRIPT * end_POSTSUPERSCRIPT ← { }

for

s 𝑠 s italic_s
in

s⁢p⁢e⁢c⁢i⁢e⁢s 𝑠 𝑝 𝑒 𝑐 𝑖 𝑒 𝑠 species italic_s italic_p italic_e italic_c italic_i italic_e italic_s
do

c←←𝑐 absent c\leftarrow italic_c ←
determine the number of new individuals by

f⁢i⁢t 𝑓 𝑖 𝑡 fit italic_f italic_i italic_t

s←←𝑠 absent s\leftarrow italic_s ←
generate

c 𝑐 c italic_c
networks using crossover and mutation

p⁢o⁢p*←p⁢o⁢p*∪s←𝑝 𝑜 superscript 𝑝 𝑝 𝑜 superscript 𝑝 𝑠 pop^{*}\leftarrow pop^{*}\cup s italic_p italic_o italic_p start_POSTSUPERSCRIPT * end_POSTSUPERSCRIPT ← italic_p italic_o italic_p start_POSTSUPERSCRIPT * end_POSTSUPERSCRIPT ∪ italic_s

end for

p⁢o⁢p←p⁢o⁢p*←𝑝 𝑜 𝑝 𝑝 𝑜 superscript 𝑝 pop\leftarrow pop^{*}italic_p italic_o italic_p ← italic_p italic_o italic_p start_POSTSUPERSCRIPT * end_POSTSUPERSCRIPT

end for

return

p⁢o⁢p⁢[arg⁡max⁡(f⁢i⁢t)]𝑝 𝑜 𝑝 delimited-[]𝑓 𝑖 𝑡 pop[\arg\max(fit)]italic_p italic_o italic_p [ roman_arg roman_max ( italic_f italic_i italic_t ) ]

Appendix B Hyperparameters
--------------------------

The hyperparameters in TensorNEAT consists of those that influence the algorithm’s dynamics and those who affect network behaviors:

*   •

Algorithmic Controls:

    *   –seed: Random seed (integer). 
    *   –fitness_target: Target fitness value for termintaion (float). 
    *   –generation_limit: Maximum number of generations for termintaion (float). 
    *   –pop_size: Population size (integer). 
    *   –network_type: Network type, either feedforward (no cycles) or recurrent (with cycles). 
    *   –inputs: Number of network inputs (integer). 
    *   –outputs: Number of network outputs (integer). 
    *   –max_nodes: Max nodes allowed in a network (integer). 
    *   –max_conns: Max connections allowed in a network (integer). 
    *   –max_species: Max species in the population (integer). 
    *   –compatibility_disjoint: Weight for disjoint genes in distance calculation between genomes (float). 
    *   –compatibility_homologous: Weight for homologous genes in distance calculation between genomes (float). 
    *   –node_add: Probability of a node addition during mutation (float). 
    *   –node_delete: Probability of a node deletion during mutation (float). 
    *   –conn_add: Probability of a connection addition during mutation (float). 
    *   –conn_delete: Probability of a connection deletion during mutation (float). 
    *   –compatibility_threshold: Distance threshold for genome speciation (float). 
    *   –species_elitism: Minimum species count to prevent all species are stagnated (integer). 
    *   –max_stagnation: Stagnation threshold for species (integer). If a species does not show improvement for max_stagnation consecutive generations, then this species will be stagnated. 
    *   –genome_elitism: Number of elite genomes preserved for next generation (integer). 
    *   –survival_threshold: Percentage of species survival for crossover (float). 
    *   –spawn_number_change_rate: Rate of change in species size over two consecutive generations (float). 

*   •

Network Behavior Controls:

    *   –bias_init_mean: Mean value for bias initialization (float). 
    *   –bias_init_std: Standard deviation for bias initialization (float). 
    *   –bias_mutate_power: Mutation strength for bias values (float). 
    *   –bias_mutate_rate: Probability of bias value mutation (float). 
    *   –bias_replace_rate: Probability to replace existing bias with a new value during mutation (float). 
    *   –response_init_mean: Mean value for response initialization (float). 
    *   –response_init_std: Standard deviation for response initialization (float). 
    *   –response_mutate_power: Mutation strength for response values (float). 
    *   –response_mutate_rate: Probability of response value mutation (float). 
    *   –response_replace_rate: Probability to replace existing response with a new value during mutation (float). 
    *   –weight_init_mean: Mean value for weight initialization (float). 
    *   –weight_init_std: Standard deviation for weight initialization (float). 
    *   –weight_mutate_power: Mutation strength for weight values (float). 
    *   –weight_mutate_rate: Probability of weight value mutation (float). 
    *   –weight_replace_rate: Probability to replace existing weight with a new value during mutation (float). 
    *   –activation_default: Default value for activation function. 
    *   –activation_options: Available activation functions. 
    *   –activation_replace_rate: Probability to change the activation function during mutation. 
    *   –aggregation_default: Default value for aggregation function. 
    *   –aggregation_options: Available aggregation functions. 
    *   –aggregation_replace_rate: Probability to change the aggregation function during mutation. 

Appendix C Interfaces
---------------------

### C.1. Network Interface

TensorNEAT offers users the flexibility to define a custom network that will be optimized by the NEAT algorithms by providing a network interface (as presented in Listing[1](https://arxiv.org/html/2404.01817v3#LST1 "Listing 1 ‣ C.1. Network Interface ‣ Appendix C Interfaces ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") and Listing[2](https://arxiv.org/html/2404.01817v3#LST2 "Listing 2 ‣ C.1. Network Interface ‣ Appendix C Interfaces ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration")). Users are required to define the specific behaviors associated with their network, including initialization, mutation, distance computation, and inference.

By implementing these interfaces, users can fit their specific requirements and leverage TensorNEAT’s power for a wide range of neural network architectures.

class BaseGene:

fixed_attrs=[]

custom_attrs=[]

def __init__ (self):

pass

def new_custom_attrs(self):

raise NotImplementedError

def mutate(self,randkey,gene):

raise NotImplementedError

def distance(self,gene1,gene2):

raise NotImplementedError

def forward(self,attrs,inputs):

raise NotImplementedError

@property

def length(self):

return len(self.fixed_attrs)+len(self.custom_attrs)

Listing 1: Gene interface.

class BaseGenome:

network_type=None

def __init__ (

self,

num_inputs:int,

num_outputs:int,

max_nodes:int,

max_conns:int,

node_gene:BaseNodeGene=DefaultNodeGene(),

conn_gene:BaseConnGene=DefaultConnGene(),

):

self.num_inputs=num_inputs

self.num_outputs=num_outputs

self.input_idx=jnp.arange(num_inputs)

self.output_idx=jnp.arange(num_inputs,num_inputs+num_outputs)

self.max_nodes=max_nodes

self.max_conns=max_conns

self.node_gene=node_gene

self.conn_gene=conn_gene

def transform(self,nodes,conns):

raise NotImplementedError

def forward(self,inputs,transformed):

raise NotImplementedError

Listing 2: Genome interface.

### C.2. Problem Template

In TensorNEAT, users also have the flexibility to define custom problems they wish to optimize using the NEAT algorithms. This can be done by implementing the Problem interface as illustrated in Alg.[3](https://arxiv.org/html/2404.01817v3#LST3 "Listing 3 ‣ C.2. Problem Template ‣ Appendix C Interfaces ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration"). Within this interface, users are required to provide the evaluation process for the problem, specify the input and output dimensions, and optionally implement a function to visualize the solution.

from typing import Callable

from utils import State

class BaseProblem:

jitable=None

def setup(self,randkey,state:State=State()):

"""initialize the state of the problem"""

raise NotImplementedError

def evaluate(self,randkey,state:State,act_func:Callable,params):

"""evaluate one individual"""

raise NotImplementedError

@property

def input_shape(self):

raise NotImplementedError

@property

def output_shape(self):

raise NotImplementedError

def show(self,randkey,state:State,act_func:Callable,params,*args,**kwargs):

raise NotImplementedError

Listing 3: Problem interface in TensorNEAT.

Appendix D Experiment Detail
----------------------------

In this section, we detail the environments and hyperparameters in experiments.

### D.1. Environments

We compare TensorNEAT to NEAT-Python in three reinforcement learning environments in Brax: Swimmer, Hopper and Halfcheetah. They are robotic control tasks and fig[7](https://arxiv.org/html/2404.01817v3#A4.F7 "Figure 7 ‣ D.1. Environments ‣ Appendix D Experiment Detail ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration") are screenshots of them. The primary objective of the robots is to advance as far as possible in a forward direction. These three environments have different action dimension and observation dimension, which are shown in table[2](https://arxiv.org/html/2404.01817v3#A4.T2 "Table 2 ‣ D.1. Environments ‣ Appendix D Experiment Detail ‣ Tensorized NeuroEvolution of Augmenting Topologies for GPU Acceleration").

![Image 18: Refer to caption](https://arxiv.org/html/2404.01817v3/extracted/5530572/graphs/swimmer.png)

(a)Swimmer

![Image 19: Refer to caption](https://arxiv.org/html/2404.01817v3/extracted/5530572/graphs/hopper.png)

(b)Hopper

![Image 20: Refer to caption](https://arxiv.org/html/2404.01817v3/extracted/5530572/graphs/halfcheetah.png)

(c)Halfcheetah

Figure 7. Robotics Control tasks in Brax.

Table 2. Action dimension and observation dimension of environments.

### D.2. Hyperparameters

Hyperparameters in TensorNEAT:

*   •

Algorithmic Controls:

    *   –seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 
    *   –fitness_target: Inf (to terminate at the fixed generation); 
    *   –generation_limit: 100; 
    *   –pop_size: [50, 100, 200, 500, 1000, 2000, 5000, 10000]; 
    *   –network_type: feedforward; 
    *   –inputs: the same as the observation dimension of the environment. 
    *   –outputs: the same as the action dimension of the environment. 
    *   –max_nodes: 50; 
    *   –max_conns: 100; 
    *   –max_species: 10; 
    *   –compatibility_disjoint: 1.0; 
    *   –compatibility_homologous: 0.5; 
    *   –node_add: 0.2; 
    *   –node_delete: 0; 
    *   –conn_add: 0.4; 
    *   –conn_delete: 0; 
    *   –compatibility_threshold: 3.5; 
    *   –species_elitism: 2; 
    *   –max_stagnation: 15; 
    *   –genome_elitism: 2; 
    *   –survival_threshold: 0.2; 
    *   –spawn_number_change_rate: 0.5; 

*   •

Network Behavior Controls:

    *   –bias_init_mean: 0; 
    *   –bias_init_std: 1.0; 
    *   –bias_mutate_power: 0.5; 
    *   –bias_mutate_rate: 0.7; 
    *   –bias_replace_rate: 0.1; 
    *   –response_init_mean: 1.0; 
    *   –response_init_std: 0; 
    *   –response_mutate_power: 0; 
    *   –response_mutate_rate: 0; 
    *   –response_replace_rate: 0; 
    *   –weight_init_mean: 0; 
    *   –weight_init_std: 1; 
    *   –weight_mutate_power: 0.5; 
    *   –weight_mutate_rate: 0.8; 
    *   –weight_replace_rate: 0.1; 
    *   –activation_default: tanh; 
    *   –activation_options: [tanh]; 
    *   –activation_replace_rate: 0; 
    *   –aggregation_default: sum; 
    *   –aggregation_options: [sum]; 
    *   –aggregation_replace_rate: 0; 

Hyperparameters in NEAT-Python:

*   •fitness_criterion: max 
*   •fitness_threshold: 999 
*   •pop_size: 10000 
*   •reset_on_extinction: False 
*   •

[DefaultGenome]

    *   –activation_default: tanh 
    *   –activation_mutate_rate: 0 
    *   –activation_options: tanh 
    *   –aggregation_default: sum 
    *   –aggregation_mutate_rate: 0.0 
    *   –aggregation_options: sum 
    *   –bias_init_mean: 0.0 
    *   –bias_init_stdev: 1.0 
    *   –bias_max_value: 30.0 
    *   –bias_min_value: -30.0 
    *   –bias_mutate_power: 0.5 
    *   –bias_mutate_rate: 0.7 
    *   –bias_replace_rate: 0.1 
    *   –compatibility_disjoint_coefficient: 1.0 
    *   –compatibility_weight_coefficient: 0.5 
    *   –conn_add_prob: 0.5 
    *   –conn_delete_prob: 0 
    *   –enabled_default: True 
    *   –enabled_mutate_rate: 0.01 
    *   –feed_forward: True 
    *   –initial_connection: full 
    *   –node_add_prob: 0.2 
    *   –node_delete_prob: 0 
    *   –num_hidden: 0 
    *   –num_inputs: the same as the observation dimension of the environment. 
    *   –num_outputs: the same as the action dimension of the environment. 
    *   –response_init_mean: 1.0 
    *   –response_init_stdev: 0.0 
    *   –response_max_value: 30.0 
    *   –response_min_value: -30.0 
    *   –response_mutate_power: 0.0 
    *   –response_mutate_rate: 0.0 
    *   –response_replace_rate: 0.0 
    *   –weight_init_mean: 0.0 
    *   –weight_init_stdev: 1.0 
    *   –weight_max_value: 30 
    *   –weight_min_value: -30 
    *   –weight_mutate_power: 0.5 
    *   –weight_mutate_rate: 0.8 
    *   –weight_replace_rate: 0.1 

*   •

[DefaultSpeciesSet]

    *   –compatibility_threshold: 3.0 

*   •

[DefaultStagnation]

    *   –species_fitness_func: max 
    *   –max_stagnation: 20 
    *   –species_elitism: 2 

*   •

[DefaultReproduction]

    *   –elitism: -9999 
    *   –survival_threshold: 0.2

