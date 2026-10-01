Title: Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases

URL Source: https://arxiv.org/html/2403.10112

Published Time: Tue, 26 Aug 2025 01:20:59 GMT

Markdown Content:
George Stamatelis,, Angelos-Nikolaos Kanatas,, 

Ioannis Asprogerakas, and George C. Alexandropoulos Part of this work has been presented in the IEEE International Conference on Communications (ICC), Denver, CO, USA, June 2024[[1](https://arxiv.org/html/2403.10112v2#bib.bib1)].G. Stamatelis and G. C. Alexandropoulos are with the Department of Informatics and Telecommunications, National and Kapodistrian University of Athens, Panepistimiopolis Ilissia, 15784 Athens, Greece. G. C. Alexandropoulos is also with the Department of Electrical and Computer Engineering, University of Illinois Chicago, IL 60601, USA (e-mails: {georgestamat, alexandg}@di.uoa.gr).A. -N. Kanatas and I. Asprogerakas are with the School of Electrical and Computer Engineering, National Technical University of Athens, Zografou Campus, 15780 Athens, Greece (e-mails: {el19169, el18942}@mail.ntua.gr).The research work has been supported by the Hellenic Foundation for Research and Innovation (HFRI) under the 5th Call for HFRI PhD Fellowships (Fellowship Number: 21080), and the Smart Networks and Services Joint Undertaking project 6G-DISAC under the European Union’s Horizon Europe research and innovation programme under Grant Agreement No 101139130.

###### Abstract

Active hypothesis testing is a thoroughly studied problem that finds numerous applications in wireless communications and sensor networks. In this paper, we focus on one centralized and one decentralized problem of active hypothesis testing in the presence of an eavesdropper. For the centralized problem including a single legitimate agent, we present a new framework based on deep NeuroEvolution (NE), whereas, for the decentralized problem, we develop a novel NE-based method for solving collaborative multi-agent tasks, which, interestingly, maintains all computational benefits of our single-agent NE-based scheme. To further reduce the computational complexity of the latter scheme, a novel multi-agent joint NE and pruning framework is also designed. The superiority of the proposed NE-based evasive active hypothesis testing schemes over conventional active hypothesis testing policies, as well as learning-based methods, is validated through extensive numerical investigations in an example use case of anomaly detection over wireless sensor networks. It is demonstrated that the proposed joint optimization and pruning framework achieves nearly identical performance with its unpruned counterpart, while removing a very large percentage of redundant deep neural network weights.

###### Index Terms:

Active hypothesis testing, sequential detection, privacy, neuroevolution, deep learning, multi-agent systems.

I Introduction
--------------

Active Hypothesis Testing (AHT) refers to the family of problems where one legitimate agent or decision maker, or a group of collaborating agents or decision makers, adaptively select(s) sensing actions and collect(s) observations in order to infer the underlying true hypothesis in a fast and reliable manner[[2](https://arxiv.org/html/2403.10112v2#bib.bib2), [3](https://arxiv.org/html/2403.10112v2#bib.bib3)]. AHT and related active sensing problems, such as active parameter estimation[[4](https://arxiv.org/html/2403.10112v2#bib.bib4)] and active change point detection[[5](https://arxiv.org/html/2403.10112v2#bib.bib5)], find numerous applications in wireless communications, including anomaly detection over sensor networks[[6](https://arxiv.org/html/2403.10112v2#bib.bib6), [7](https://arxiv.org/html/2403.10112v2#bib.bib7)], strong or weak radar models for target detection[[8](https://arxiv.org/html/2403.10112v2#bib.bib8)], camera object detection [[9](https://arxiv.org/html/2403.10112v2#bib.bib9)], cyber-intrusion detection, target search, and adaptive beamforming[[10](https://arxiv.org/html/2403.10112v2#bib.bib10)], as well as, very recently, localization [[11](https://arxiv.org/html/2403.10112v2#bib.bib11)] and channel estimation[[12](https://arxiv.org/html/2403.10112v2#bib.bib12)] enabled by reconfigurable intelligent surfaces.

The recent rise of distributed and edge machine learning approaches[[13](https://arxiv.org/html/2403.10112v2#bib.bib13), [14](https://arxiv.org/html/2403.10112v2#bib.bib14)], as well as Internet-of-Things (IoT) applications [[15](https://arxiv.org/html/2403.10112v2#bib.bib15)], is urging the development of efficient mechanisms for large-scale covert data collection. It has been shown in[[16](https://arxiv.org/html/2403.10112v2#bib.bib16)] that, even in encrypted IoT applications, eavesdroppers can accurately estimate sensitive information just by observing device interactions alone. The main focus of this paper is on decentralized collaboration mechanisms for active sensing that do not reveal information to third parties.

### I-A Background

##### Deep Reinforcement Learning (DRL)

Reinforcement Learning (RL) and especially DRL, which leverages the representation capabilities of Deep Neural Networks (DNNs), has emerged as a very powerful tool for complex decision making in modern wireless communication systems. The seminal paper on Deep Q Networks (DQN)[[17](https://arxiv.org/html/2403.10112v2#bib.bib17)] for video games and subsequent works on policy gradient methods, e.g. [[18](https://arxiv.org/html/2403.10112v2#bib.bib18), [19](https://arxiv.org/html/2403.10112v2#bib.bib19)] for robotics, laid the foundation for profound resource allocation performance in a wide range of communication systems. Although DRL can be used to solve traditional Markov Decision Processes (MDPs), its success is mainly attributed to its capabilities to find very good, near-optimal policies for Partially Observable MDPs (POMDPS), which are known to be NP-hard problems[[20](https://arxiv.org/html/2403.10112v2#bib.bib20)].

##### Multi-Agent Systems and DRL

When it comes to collaborative multi-agent MDPs and POMDPs, state-of-the-art DRL approaches [[21](https://arxiv.org/html/2403.10112v2#bib.bib21), [22](https://arxiv.org/html/2403.10112v2#bib.bib22)] are based on the idea of centralized learning and decentralized execution (CLDE) [[23](https://arxiv.org/html/2403.10112v2#bib.bib23)]. During training, the agents are provided with additional information that enhances the training process. The agent, however, must not depend on that information during deployment/testing. Most popular CLDE algorithms for tasks with heterogeneous agents are based on the Multi-Agent Deep Deterministic Policy Gradients (MADDPG) actor critic algorithm [[24](https://arxiv.org/html/2403.10112v2#bib.bib24)], where each agent is equipped with an individual actor and there is a global critic DNN. This algorithm finds numerous applications in wireless communication systems, including cognitive radio [[25](https://arxiv.org/html/2403.10112v2#bib.bib25)], power control [[26](https://arxiv.org/html/2403.10112v2#bib.bib26)], and edge caching [[27](https://arxiv.org/html/2403.10112v2#bib.bib27)]. Extensions of MADDPG based on Proximal Policy Optimization (PPO)[[18](https://arxiv.org/html/2403.10112v2#bib.bib18)] have been successfully applied to AHT [[28](https://arxiv.org/html/2403.10112v2#bib.bib28), [29](https://arxiv.org/html/2403.10112v2#bib.bib29)]. Furthermore, federated extensions of MADDPG have been recently discussed in [[30](https://arxiv.org/html/2403.10112v2#bib.bib30), [31](https://arxiv.org/html/2403.10112v2#bib.bib31)].

##### DNN Pruning

There is lately an increased demand for deploying pre-trained DNNs on lighter devices with memory and/or power constraints[[32](https://arxiv.org/html/2403.10112v2#bib.bib32), [33](https://arxiv.org/html/2403.10112v2#bib.bib33)], such as mobile phones, lightweight sensors, and various IoT. However, running large, over-parameterized DNNs on such devices is often impossible. DNN pruning algorithms[[33](https://arxiv.org/html/2403.10112v2#bib.bib33), [34](https://arxiv.org/html/2403.10112v2#bib.bib34)] remove unnecessary connections and/or neurons in order to get smaller neural networks with similar performance. Such algorithms are based on the lottery ticket hypothesis which states[[35](https://arxiv.org/html/2403.10112v2#bib.bib35)]: “Random dense feed-forward DNNs contain winning tickets, i.e., smaller subnetworks that can achieve almost identical performance to the initial network when trained alone.” To this end, pruning has been successfully applied to single-agent DLR problems[[36](https://arxiv.org/html/2403.10112v2#bib.bib36), [37](https://arxiv.org/html/2403.10112v2#bib.bib37)] and, more recently, to multi-agent DRL settings[[38](https://arxiv.org/html/2403.10112v2#bib.bib38)].

##### DRL for AHT

The problem of binary AHT was first studied by Chernoff in his pioneering work on sequential design of experiments[[2](https://arxiv.org/html/2403.10112v2#bib.bib2)]. This work proposed an asymptotically optimal heuristic, known as the Chernoff test, which remains popular even today. The Chernoff test was latter extended in the multi-hypothesis setting[[3](https://arxiv.org/html/2403.10112v2#bib.bib3)]. In[[39](https://arxiv.org/html/2403.10112v2#bib.bib39), [40](https://arxiv.org/html/2403.10112v2#bib.bib40), [41](https://arxiv.org/html/2403.10112v2#bib.bib41), [42](https://arxiv.org/html/2403.10112v2#bib.bib42)], AHT was modeled as a POMDP. The authors in[[39](https://arxiv.org/html/2403.10112v2#bib.bib39)] presented bounds based on dynamic programming, whereas[[40](https://arxiv.org/html/2403.10112v2#bib.bib40), [41](https://arxiv.org/html/2403.10112v2#bib.bib41), [42](https://arxiv.org/html/2403.10112v2#bib.bib42)] showcased the superiority of DRL strategies over conventional AHT heuristics. The recurrent DRL algorithm in[[40](https://arxiv.org/html/2403.10112v2#bib.bib40)] was shown to compete with classical model-based strategies without having knowledge of the environment dynamics. More complex AHT-based anomaly detection problems with sampling costs have recently attracted a lot of attention, e.g.,[[43](https://arxiv.org/html/2403.10112v2#bib.bib43), [44](https://arxiv.org/html/2403.10112v2#bib.bib44), [45](https://arxiv.org/html/2403.10112v2#bib.bib45)], and appropriate deep learning and DRL strategies that balance detection objectives with cost management were proposed. Collaborative multi-agent DRL for AHT was studied in[[28](https://arxiv.org/html/2403.10112v2#bib.bib28), [7](https://arxiv.org/html/2403.10112v2#bib.bib7), [29](https://arxiv.org/html/2403.10112v2#bib.bib29)]. Specifically, the authors in[[29](https://arxiv.org/html/2403.10112v2#bib.bib29)] discussed how sampling cost constraints can be managed in a multi-agent environment using Lagrange multipliers. Very recently in[[46](https://arxiv.org/html/2403.10112v2#bib.bib46), [47](https://arxiv.org/html/2403.10112v2#bib.bib47), [48](https://arxiv.org/html/2403.10112v2#bib.bib48)], AHT in the presence of adversaries that target to corrupt the observations of legitimate agents was studied. The first two works assumed no adaptive decision making from the adversary’s side and, in fact, in [[47](https://arxiv.org/html/2403.10112v2#bib.bib47)] the agent terminated when an adversary was detected. The last work focused on the case of adaptive and intelligent legitimate as well as adversarial agents with different information structures.

##### NeuroEvolution(NE)

The consideration of NE schemes for solving MDPs and POMDPs is an old idea, dating back, for example, to [[49](https://arxiv.org/html/2403.10112v2#bib.bib49), [50](https://arxiv.org/html/2403.10112v2#bib.bib50)], which has been left largely undeployed, mainly due to the recent impressive success of DRL approaches. However, in the past few years, it has been experimentally shown that, even simple NE schemes, can rival back-propagation algorithms, such as deep Q-learning and policy gradients, outperforming DRL approaches in various single-agent POMDPs [[51](https://arxiv.org/html/2403.10112v2#bib.bib51), [52](https://arxiv.org/html/2403.10112v2#bib.bib52), [53](https://arxiv.org/html/2403.10112v2#bib.bib53)]. Surprisingly, even very old NE methods can compete with popular state-of-the-art DRL algorithms, as shown in [[52](https://arxiv.org/html/2403.10112v2#bib.bib52)]. The main benefits of NE over DRL are summarized as follows:

*   •NE is easier to implement (replay buffers, advantage estimation, etc., are not needed) and to parallelize over multiple Central Processing Units (CPUs). Only scalar numbers indicating the fitness of an individual need to be shared between collaboratively computing nodes. 
*   •Reward reshaping and exploration techniques are not required in NE schemes. It is well known in RL and DRL literature that training algorithms with very sparse reward signals rarely produce good performance, and designing appropriate rewards can be a time-consuming trial and error task. On the other hand, NE only needs to specify a fitness function. This benefit comes extremely handy in decision-making problems that have multiple constraints besides their core objective, such as the ones studied in this paper for secure active sensing. 
*   •DRL schemes face instability problems, which are associated with back-propagation through time. This issue is totally absent in NE-based approaches. 

The core idea of NE is to directly search the space of policy DNNs via nature-inspired algorithms; note that, in NE, critic DNNs are not considered. In particular, each chromosome of an individual represents some parameters of a policy DNN [[51](https://arxiv.org/html/2403.10112v2#bib.bib51), [54](https://arxiv.org/html/2403.10112v2#bib.bib54)]. It is noted that, due to the large number of parameters in a DNN, it is typically infeasible to construct individuals representing all of the DNN parameters. To this end, techniques that take advantage of the DNN’s structure in order to construct smaller individuals are usually devised. Particularly, a generation of individuals is initialized randomly. Each individual is then evaluated, and its fitness function is stored. The individuals with the highest fitness function are selected for mating. During mating, the parameters of two or more individuals are merged by various methods (e.g., crossover operation). The new individuals then replace the ”weaker” individuals of the population. This procedure is repeated for multiple generations. It is noted that further genetic operations[[55](https://arxiv.org/html/2403.10112v2#bib.bib55)], such as mutation, can be utilized to increase exploration.

However, despite the recent impressive results of NE schemes for single-agent problems, to the best of our knowledge, there exist no works elaborating on how to extend them to multi-agent collaborative problems, which is the focus of this research work. Furthermore, this paper constitutes the first attempt at applying pruning methods on evolved policy DNNs.

##### Private Hypothesis Testing

Due to the growing concerns for data privacy, many works studied privacy in passive hypothesis testing problems, where there is no control over the sensing actions. For example, differentially private hypothesis testing was studied in[[56](https://arxiv.org/html/2403.10112v2#bib.bib56)], whereas[[57](https://arxiv.org/html/2403.10112v2#bib.bib57)] elaborated on how to perform remote estimation of the system state through sensor data while impairing the filtering ability of eavesdroppers. Secure distributed hypothesis testing was studied in[[58](https://arxiv.org/html/2403.10112v2#bib.bib58)]. A closely related problem is the active privacy utility trade-off in data sharing[[59](https://arxiv.org/html/2403.10112v2#bib.bib59), [60](https://arxiv.org/html/2403.10112v2#bib.bib60)]. In the problem studied in [[59](https://arxiv.org/html/2403.10112v2#bib.bib59)], there are two independent discrete variables S S and U U, and the observations generated depend on both. The DRL agent adaptively selects data release mechanisms and outputs observations to a service provider. The goal is to assist the service provider in determining the value of U U, while keeping S S hidden. In contrast, the authors in[[60](https://arxiv.org/html/2403.10112v2#bib.bib60)] investigated real-time data sharing methods for a Markov chain X t X_{t}, where the objective is to “hide” the true value of X t X_{t} at each time step, while ensuring that the distortion between the shared observations Y t Y_{t} and the actual X t X_{t} remains below a pre-defined threshold. These formulations differ from our work because our agent tries to both infer and hide the same variable. Besides that, the latter data sharing frameworks are only limited to a single-agent (centralized) scenario.

The problem of single-agent Evasive Active Hypothesis Testing (EAHT), where a passive eavesdropper (Eve) collects noisy estimates of the legit observations and tries to infer the underlying hypothesis, was studied in[[61](https://arxiv.org/html/2403.10112v2#bib.bib61)], focusing however explicitly on the asymptotic case. In that work, the authors formulated single-agent EAHT as a constrained optimization problem including the legitimate agent’s and Eve’s error exponent. However, near-optimal or optimal action selection policies were not presented. In this paper, motivated by the lack of explicit policies for EAHT, we present novel single- and multi-agent EAHT approaches for wireless sensor networks, which are both based on a deep NE framework. The contributions of this paper are summarized as follows:

1.   1.We formulate the single-agent EAHT problem studied in[[61](https://arxiv.org/html/2403.10112v2#bib.bib61)] as a constrained POMDP and present a NE-based method for solving it. Our numerical investigations showcase that our method satisfies the privacy constraint, while achieving similar accuracy to popular AHT methods that ignore the existence of any adversary. 
2.   2.A novel formulation of the decentralized multi-agent EAHT problem is presented, where a group of agents tries to infer the underlying hypothesis, while keeping it hidden from a passive eavesdropper. 
3.   3.We present a novel approach for solving decentralized POMDPs via deep NE, and apply it to the decentralized EAHT problem at hand. The proposed scheme is numerically compared with state-of-the-art multi-agent DRL algorithms. It is demonstrated that our NE-based method outperforms existing algorithms, while maintaining all computational benefits of our single-agent NE scheme. 
4.   4.A novel multi-agent joint NE and pruning scheme is devised, which is shown experimentally to achieve almost identical performance to the unpruned agents, despite removing over 90%90\% of the DNN’s weights. 

This paper extends its recent conference version[[1](https://arxiv.org/html/2403.10112v2#bib.bib1)] by including a novel multi-agent joint NE and pruning scheme, as well as more thorough experiments including more benchmark schemes, a second synthetic sensor model, additional wireless applications, and new experiments against sophisticated, learning-based eavesdroppers.

The remainder of this paper is organized as follows. Section II introduces the single-agent (centralized) EAHT problem and its multi-agent (decentralized) extension. Section III presents our NE-based solution methods, and Section IV includes our extensive experimental results demonstrating the superiority of the proposed EAHT schemes over various benchmarks. The paper is concluded in Section V.

### I-B Notations

Throughout this paper, calligraphic letters, e.g. 𝒳\mathcal{X}, are reserved for sets. Bold lower-case and upper-case letters denote vectors and matrices, respectively, e.g., 𝜽\boldsymbol{\theta} and 𝚯\boldsymbol{\Theta}. Notation [𝚯]l,m[\boldsymbol{\Theta}]_{l,m} denotes the element on the l l-th row and m m-th collumn of the matrix 𝚯\boldsymbol{\Theta}. Unless stated otherwise, the letter t t is reserved for time indices. Finally, E​[⋅]E[\cdot] represents the expectation operator, while E^​[⋅]\hat{E}[\cdot] denotes the sample average.

II EAHT Problem Formulations
----------------------------

Consider a security analyst monitoring a corporate network for signs of intrusion. The analyst (or an automated detection system) can deploy various tests, such as port scans, access log queries, or anomaly detection filters, to identify whether the system is under attack and, if so, determine the type and location of the threat. However, an adversary might be monitoring these tests as well to identify weakened components of the network with the goal to launch more tailored attacks. As another application, consider search and rescue missions, where a team of autonomous drones collaborates to locate survivors in a disaster zone (e.g., an earthquake-hit city), while avoiding detection by hostile actors (e.g., armed groups or adversarial drones). The swarm shares partial observations (e.g., thermal signatures and/or structural damage) to rapidly narrow down survivor locations, but carefully controls communication timing and searching actions to prevent eavesdroppers from inferring their progress. Some drones may even emit decoy signals or take deceptive patrol routes to distort the adversary’s belief distribution. While timely detection of survivors is the main goal, the swarm may also desire to ensure that the eavesdropper(s) cannot assign high confidence to a single hypothesis (e.g., that survivors exist in a specific building).

In this paper, we study methods to collect informative data in order to accurately classify the underlying state of a system, while keeping it hidden from eavesdropping third parties. Two EAHT problems are introduced in this section.A centralized one with a single agent, and a decentralized one with a group of agents having access to different sensing action sets. In the latter problem, the agents exchange information with each other and each one separately infers the hypothesis[[28](https://arxiv.org/html/2403.10112v2#bib.bib28)].

### II-A Centralized Problem

Let 𝒳≜{0,1,2,…,|𝒳|−1}\mathcal{X}\triangleq\{0,1,2,\ldots,|\mathcal{X}|-1\} be a finite set of hypotheses, while the true hypothesis x x is unknown. A legitimate agent has access to a finite set of sensing actions 𝒜\mathcal{A}, and at each time instance t t, it selects an action a t∈𝒜 a_{t}\in\mathcal{A}. In response to this action, the agent collects a noisy observation y t y_{t}. In parallel, an eavesdropper (Eve) being present in the system receives another noisy observation z t z_{t}. The conditional probability of y t y_{t} given x x and a t a_{t} is denoted as P​[y t|a t,x]P[y_{t}|a_{t},x], whereas the respective conditional probability of z t z_{t} is Q​[z t|a t,x]Q[z_{t}|a_{t},x].

We assume that the prior over all hypotheses π 0​(𝒳)\pi_{0}(\mathcal{X}) and the distributions P​[⋅]P[\cdot] and Q​[⋅]Q[\cdot] are either known a priori [[39](https://arxiv.org/html/2403.10112v2#bib.bib39), [61](https://arxiv.org/html/2403.10112v2#bib.bib61)], or can be reliably estimated from a large dataset. However, we will also experimentally verify the effectiveness of the proposed strategies in environments where the estimated probability kernels are incorrect approximations of the true dynamics.

The legitimate agent maintains an |𝒳||\mathcal{X}|-dimensional belief vector π t L​(𝒳)\pi^{L}_{t}(\mathcal{X}) over all possible hypotheses x∈𝒳 x\in\mathcal{X} at time instant t t, and Eve does the same via the belief vector π t E​(𝒳)\pi^{E}_{t}(\mathcal{X}). Each entry π t L​(x)\pi_{t}^{L}(x) is the posterior probability on the hypothesis x x, given the sequence of action and observations up to time t t. For the former, given an action observation pair (a t,y t)(a_{t},y_{t}), the legit belief on each hypothesis x x is updated as follows[[42](https://arxiv.org/html/2403.10112v2#bib.bib42)]:

π t L​(x)=π t−1 L​(x)​P​[y t|a t,x]∑x′∈𝒳 π t−1 L​(x′)​P​[y t|a t,x′].\pi^{L}_{t}(x)=\frac{\pi^{L}_{t-1}(x)P[y_{t}|a_{t},x]}{\sum_{x^{\prime}\in\mathcal{X}}\pi^{L}_{t-1}(x^{\prime})P[y_{t}|a_{t},x^{\prime}]}.(1)

Similarly, Eve’s belief given a pair (a t,z t)(a_{t},z_{t}) can be updated as:

π t E​(x)=π t−1 E​(x)​Q​[z t|a t,x]∑x′∈𝒳 π t−1 E​(x′)​Q​[z t|a t,x′].\pi^{E}_{t}(x)=\frac{\pi^{E}_{t-1}(x)Q[z_{t}|a_{t},x]}{\sum_{x^{\prime}\in\mathcal{X}}\pi^{E}_{t-1}(x^{\prime})Q[z_{t}|a_{t},x^{\prime}]}.(2)

By assuming that both agents deploy the optimal Maximum A Posteriori (MAP) decoding[[61](https://arxiv.org/html/2403.10112v2#bib.bib61), [48](https://arxiv.org/html/2403.10112v2#bib.bib48)] the error probabilities at each time instant t t can be expressed as follows:

γ t L=1−max x∈𝒳⁡π t L​(x),\displaystyle\gamma^{L}_{t}=1-\max_{x\in\mathcal{X}}\pi^{L}_{t}(x),(3)
γ t E=1−max x∈𝒳⁡π t E​(x).\displaystyle\gamma^{E}_{t}=1-\max_{x\in\mathcal{X}}\pi^{E}_{t}(x).(4)

We also assume that the legitimate agent controls the stopping time 1 1 1 In the context of AHT, the stopping time metric refers to the number of sensing actions performed before the final inference decision[[7](https://arxiv.org/html/2403.10112v2#bib.bib7), [40](https://arxiv.org/html/2403.10112v2#bib.bib40), [44](https://arxiv.org/html/2403.10112v2#bib.bib44)].τ\tau. To this end, once the episode terminates, both the agent and Eve guess the underlying hypothesis according to the maximum a posteriori decoding rule.

The goal of the legitimate agent is to reliably estimate the true hypothesis as quickly as possible while keeping Eve’s error probability above a certain application/agent-defined threshold. Let g t≜g​(a t|π t L​(𝒳))g_{t}\triangleq g(a_{t}|\pi^{L}_{t}(\mathcal{X})) represent the policy of the agent at each time instance t t. The policy is a probabilistic mapping from belief vectors to the action set. Hence, the total policy of the legitimate agent for a sensing horizon of τ\tau time slots is defined as follows:

g≜(g 1,g 2​⋯,g τ,τ).{\rm g}\triangleq(g_{1},g_{2}\cdots,g_{\tau},\tau).(5)

By using user defined scalars E E and L L, this problem can be formulated as a constrained POMDP problem as follows:

𝒪​𝒫 1\displaystyle\mathcal{OP}_{1}:min g​E​[τ]\displaystyle:\underset{{\rm g}}{\min}\,\,E[\tau]
s.t.​γ τ L≤L,γ t E≥E​∀t=1,2,…,τ.\displaystyle\quad\text{\text{s}.\text{t}.}\,\,\,\,\,\gamma^{L}_{\tau}\leq L,\,\,\gamma^{E}_{t}\geq E\,\,\,\,\forall t=1,2,\ldots,\tau.

The expectation in the latter objective is taken with respect to g{\rm g}, P​[⋅]P[\cdot], Q​[⋅]Q[\cdot], and π 0​(x)\pi_{0}(x). This indicates that the prior and probability kernels influence the belief updates in([1](https://arxiv.org/html/2403.10112v2#S2.E1 "In II-A Centralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")) and([2](https://arxiv.org/html/2403.10112v2#S2.E2 "In II-A Centralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")), which in turn influence future policies, beliefs, and decision rules. Note that, without the second constraint, 𝒪​𝒫 1\mathcal{OP}_{1} is essentially an AHT problem. It is also noted that, in practice, the error probabilities cannot be accurately revealed through a finite number of episodes, implying that there will always be a non-zero probability of leakage. To deal with this issue, we use sample averaging to simplify the constraints as follows:

E^​[1−max x∈𝒳⁡π τ L​(x)]≤L,E^​[1−max x∈𝒳⁡π t E​(x)]≥E​∀t.\hat{E}[1-\max_{x\in\mathcal{X}}\pi_{\tau}^{L}(x)]\leq L,\quad\hat{E}[1-\max_{x\in\mathcal{X}}\pi_{t}^{E}(x)]\geq E\,\,\forall t.(6)

Even without the instantaneous constrains, POMDPs are NP-hard [[20](https://arxiv.org/html/2403.10112v2#bib.bib20)], therefore, we do not expect to find exact solutions. In the next section, we will present near-optimal policies using deep policy optimization techniques.

### II-B Decentralized Problem

![Image 1: Refer to caption](https://arxiv.org/html/2403.10112v2/x1.png)

Figure 1: The decentralized EAHT problem under consideration. The agents monitor the environment through their sensing actions (thick red arrows) and, at the same time, they share information with each other (blue arrows). In their vicinity, there exists an eavesdropper monitoring their actions and collecting the corresponding observations through a noisy channel (dashed green lines).

In the decentralized problem depicted in Fig. [1](https://arxiv.org/html/2403.10112v2#S2.F1 "Figure 1 ‣ II-B Decentralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), a group of K K legitimate agents collaborates to infer the underlying hypothesis. We assume that each k k-th agent, with k=1,2,…,K k=1,2,\ldots,K, has access to a sensing action set 𝒜 k\mathcal{A}^{k}. By probing the environment with an action a t k a^{k}_{t} at time slot t t, the k k-th agent receives a noisy observation y t k y_{t}^{k} with conditional distribution P​[y t k|a t k,x]P[y_{t}^{k}|a_{t}^{k},x], while Eve observes the noisy quantity z t k z_{t}^{k} with conditional distribution Q​[z t k|a t k,x]Q[z_{t}^{k}|a_{t}^{k},x]. It is assumed that both y t k y_{t}^{k} and z t k z_{t}^{k} do not depend on the actions of other than the k k-th agent; such assumptions are common in the collaborative anomaly detection literature, e.g., [[7](https://arxiv.org/html/2403.10112v2#bib.bib7), [28](https://arxiv.org/html/2403.10112v2#bib.bib28)]. This agent is assumed to also broadcast the action and observation tuple (a t k,y t k)(a_{t}^{k},y_{t}^{k}) to a set 𝒪 k\mathcal{O}^{k} of neighboring legitimate agents, while receiving observations from another set ℐ k\mathcal{I}^{k} of agents.

Given a pair (a t,y t)({\rm a}_{t},{\rm y}_{t}) of actions and observations, with a t≜(a t 1,a t 2,…,a t K){\rm a}_{t}\triangleq(a_{t}^{1},a_{t}^{2},\ldots,a_{t}^{K}) and y t≜(y t 1,y t 2,…,y t K){\rm y}_{t}\triangleq(y_{t}^{1},y_{t}^{2},\ldots,y_{t}^{K}), each k k-th legitimate agent updates its belief according to the expression:

ρ t k​(x)=ρ t−1 k​(x)​∏(a t k,y t k)∈(a t,y t)P​[y t k|a t k,x]∑x′∈𝒳 ρ t−1 k​(x′)​∏(a t k,y t k)∈(a t,y t)P​[y t k|a t k,x′].\rho_{t}^{k}(x)=\frac{\rho_{t-1}^{k}(x)\prod_{(a_{t}^{k},y_{t}^{k})\in(\rm a_{t},\rm y_{t})}P[y_{t}^{k}|a_{t}^{k},x]}{\sum_{x^{\prime}\in\mathcal{X}}\rho_{t-1}^{k}(x^{\prime})\prod_{(a_{t}^{k},y_{t}^{k})\in(\rm a_{t},\rm y_{t})}P[y_{t}^{k}|a_{t}^{k},x^{\prime}]}.(7)

Similarly does Eve via the following belief update rule:

ρ t E​(x)=ρ t−1 E​(x)​∏(a t k,z t k)∈(a t,z t)Q​[z t k|a t k,x]∑x′∈𝒳 ρ t−1 E​(x′)​∏(a t k,z t k)∈(a t,z t)Q​[z t k|a t k,x′],\rho_{t}^{E}(x)=\frac{\rho_{t-1}^{E}(x)\prod_{(a_{t}^{k},z_{t}^{k})\in(\rm a_{t},\rm z_{t})}Q[z_{t}^{k}|a^{k}_{t},x]}{\sum_{x^{\prime}\in\mathcal{X}}\rho_{t-1}^{E}(x^{\prime})\prod_{(a_{t}^{k},z_{t}^{k})\in(\rm a_{t},\rm z_{t})}Q[z_{t}^{k}|a^{k}_{t},x^{\prime}]},(8)

where z t≜(z t 1,z t 2,…,z t K){\rm z}_{t}\triangleq(z_{t}^{1},z_{t}^{2},\ldots,z_{t}^{K}) denotes Eve’s observations.

In this multi-agent case, we will further assume that each of the action sets A k A^{k} also contains a “no sensing action” element, according to which the observations y t k y_{t}^{k} and z t k z_{t}^{k} are not generated. It is noted, however, that when a k k-th legitimate agent selects this option, it can still update its belief using information from the other K−1 K-1 agents. We also consider that each agent can exit independently the sensing process, therefore, their communication graph may vary with time. To treat this general case, we consider as stopping time the time instance that the last agent exits the sensing process, i.e., τ=max k⁡τ k\tau=\max_{k}\tau_{k}, with τ k\tau_{k} denoting the stopping time of each k k-th agent. We also use notation γ t,k L\gamma_{t,k}^{L} for the posterior error probability of each agent k k at time instant t t. Similar to 𝒪​𝒫 1\mathcal{OP}_{1}, our goal is to find a collective agent policy g C≜(g 1,g 2,…,g k){\rm g}^{C}\triangleq({\rm g}_{1},{\rm g}_{2},\ldots,{\rm g}_{k}), where each g k{\rm g}_{k} is obtained from ([5](https://arxiv.org/html/2403.10112v2#S2.E5 "In II-A Centralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")), that solves approximately the optimization problem:

𝒪​𝒫 2\displaystyle\mathcal{OP}_{2}:min g C​E​[τ]\displaystyle:\underset{{\rm g}^{C}}{\min}\,\,E[\tau]
s.t.​γ τ k,k L≤L​∀k,γ t E≥E​∀t=1,2,…,τ.\displaystyle\quad\text{\text{s}.\text{t}.}\,\,\,\,\,\gamma_{\tau_{k},k}^{L}\leq L\,\,\forall k,\,\,\gamma^{E}_{t}\geq E\,\,\forall t=1,2,\ldots,\tau.

Note that the error probability constraints can be relaxed using sample averaging as in([6](https://arxiv.org/html/2403.10112v2#S2.E6 "In II-A Centralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")) for 𝒪​𝒫 1\mathcal{OP}_{1}.

Remark (The Importance of Inter-Agent Communication): In our framework, we assume that each agent broadcasts its local sensing information to neighboring agents aiming to support more informed inference. While this broadcasting incurs some communication overhead, it can significantly enhance the detection capabilities of all agents. Inter-agent information exchange is a well-established concept with numerous applications in modern intelligent wireless communication systems. For example, in spectrum sharing environments[[62](https://arxiv.org/html/2403.10112v2#bib.bib62)], agents share local observations with nearby peers to construct more accurate beliefs about the underlying channel states. In anomaly detection tasks, sensors collaborate by exchanging information to improve anomaly localization [[7](https://arxiv.org/html/2403.10112v2#bib.bib7)]. Similarly, in edge caching systems, sharing learned representations of local observations and actions enables agents to refine popularity estimates, thereby increasing caching efficiency and network throughput [[63](https://arxiv.org/html/2403.10112v2#bib.bib63)]. As it will demonstrated later on in Section[IV](https://arxiv.org/html/2403.10112v2#S4 "IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), such communications can substantially reduce stopping times.

III Deep Neuroevolution Schemes for EAHT
----------------------------------------

In the section, we commence with the presentation of the application of NE to the considered centralized EAHT problem. Next, we present a novel NE-based method to deal with multi-agent POMDPs, which is then deployed to solve the considered decentralized EAHT problem. Finally, our NE-based method is extended to incorporate DNN pruning.

### III-A Centralized EAHT

The policy DNN of an individual, which is needed in the NE formulation, is a mapping from beliefs to actions. We will use the Cooperative Synapse NE (CoSyNE) method[[54](https://arxiv.org/html/2403.10112v2#bib.bib54)] to evolve a feed-forward policy DNN. The fitness function of a policy DNN θ\theta is defined as follows:

f​(θ)={−A E,A E≥1−E E^τ−1,otherwise,f(\theta)=\begin{cases}-A_{E},\quad A_{E}\geq 1-E\\ \hat{E}_{\tau}^{-1},\quad\,\,\,\text{otherwise}\end{cases},(9)

where A E≜E^​max t⁡max x⁡π t E​(x)A_{E}\triangleq\hat{E}\max_{t}\max_{x}\pi_{t}^{E}(x) represents the average of the maximum Eve’s belief value during an episode with E^\hat{E} being the sample average, and E^τ\hat{E}_{\tau} is the average horizon, both calculated from a large number N EP N_{\rm EP} of Monte Carlo episodes. It is noted that episodes in which Eve has large beliefs on some hypothesis are penalized. According to this definition, if a policy DNN cannot satisfy the privacy constraint, it is “encouraged” to do so; this is imposed from the first part of the fitness function f​(θ)f(\theta). Otherwise, the DNN is “encouraged” to minimize the expected stopping time. Finally, individuals that satisfy the privacy constraint with the shortest stopping time are selected for mating. Similar to recent works on deep learning for AHT and related problems, e.g., [[41](https://arxiv.org/html/2403.10112v2#bib.bib41), [29](https://arxiv.org/html/2403.10112v2#bib.bib29), [40](https://arxiv.org/html/2403.10112v2#bib.bib40), [44](https://arxiv.org/html/2403.10112v2#bib.bib44)], the policy DNN is only responsible for action selection. In this paper, we utilize a simple stopping rule, according to which termination takes place the first time t t for which holds γ t L<L\gamma_{t}^{L}<L. This stopping rule essentially handles the legitimate accuracy constraint, thus, it is unnecessary to include it in the fitness function calculation. A pseudocode describing our fitness function calculation is given in Algorithm[1](https://arxiv.org/html/2403.10112v2#algorithm1 "In III-A Centralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases").

Complexity Analysis: We will henceforth use the symbol 𝜽\boldsymbol{\theta} to denote the parameter vector concatenating all trainable weights N w N_{w} of a policy DNN θ\theta. To this end, the CoSyNE algorithm maintains a population of L pop L_{\rm pop}individuals in a matrix 𝚯∈ℝ L pop×N w\boldsymbol{\Theta}\in\mathbb{R}^{L_{\rm pop}\times N_{w}}, where each row corresponds to the weights of one individual with N w N_{w}chromosomes. Each m m-th column of 𝚯\boldsymbol{\Theta} (m=1,2,…,N w m=1,2,\ldots,N_{w}) corresponds to each m m-th subpopulation of individuals. This matrix is initialized randomly and the following steps are performed for each of the N gen N_{\rm gen} generations.

1.   1.Fitness evaluation: Foremost, the fitness of all individuals comprising the population is evaluated, and then, those individuals are shorted according to it. For each individual l l, each l l-th row of 𝚯\boldsymbol{\Theta} (l=1,2,…,L pop l=1,2,\ldots,L_{\rm pop}) is transformed to a DNN and provided to Algorithm[1](https://arxiv.org/html/2403.10112v2#algorithm1 "In III-A Centralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"). Assuming that the forward pass time for a feed-forward DNN is T FP T_{\rm FP}, then calculating the fitness for an individual requires O​(T FP​T​N EP)O(T_{\rm FP}TN_{\rm{EP}}) of complexity. Therefore, this step carries O​(L pop​T FP​T​N EP+log⁡(L pop))O(L_{\rm pop}T_{\rm FP}TN_{\rm{EP}}+\log(L_{\rm pop})) complexity. 
2.   2.Crossover and Mutation: The top ⌊L pop/4⌋\left\lfloor L_{\rm pop}/4\right\rfloor rows of 𝚯\boldsymbol{\Theta} are used as parents to construct ⌈3​L pop/4⌉\left\lceil 3L_{\rm pop}/4\right\rceil offsprings, denoted by θ ℓ o\mathbf{\theta}^{\rm o}_{\ell} for ℓ=⌈3​L pop/4⌉,⌈3​L pop/4⌉+1,…,L pop\ell=\left\lceil 3L_{\rm pop}/4\right\rceil,\left\lceil 3L_{\rm pop}/4\right\rceil+1,\ldots,L_{\rm pop}, through standard crossover and mutation mechanisms. Crossover combines the weights of two individuals and mutation adds Gaussian noise. The last ⌈3​L pop/4⌉\left\lceil 3L_{\rm pop}/4\right\rceil rows of 𝚯\boldsymbol{\Theta} are replaced by the offsprings. Mutation of the entire population requires O​(L pop​N w)O(L_{\rm pop}N_{w}) of complexity, whereas crossover requires O​(L pop 2​N w/16)O(L_{\rm pop}^{2}N_{w}/16) of complexity. 
3.   3.Permutation: Each chromosome [𝚯]l,m[\boldsymbol{\Theta}]_{l,m} (i.e., each (l,m)(l,m)-th element of 𝚯\boldsymbol{\Theta}) is assigned the following permutation probability:

p l,m perm=1−f l f max N w,p_{l,m}^{\rm perm}=1-\sqrt[N_{w}]{\frac{f_{l}}{f_{\rm max}}},

where f l f_{l} is the individual’s fitness function and f max f_{\rm max} is the best fitness of the population. Then, each chromosome is marked for permutation according to the above probability. For each m m-th subpopulation m=1,2,…,N w m=1,2,\ldots,N_{w}, the marked chromosomes are shuffled. The complexity of this step is O​(N w​L pop)O(N_{w}L_{\rm pop}). 

Putting all above together, the computational complexity of the CoSyNE algorithm incorporated within the proposed single-agent EAHT is N gen​O​((L pop​(T FP​T​N EP)+L pop 2​N w/16))N_{\rm gen}O((L_{\rm pop}(T_{\rm FP}TN_{\rm EP})+L^{2}_{\rm pop}N_{w}/16)). Note that, depending on implementation details (e.g., the type of data structures used), the exact complexity expression may differ.

Input:Individual DNN parameters

θ\theta
, thresholds

E E
and

L L
, prior

π 0​(𝒳)\pi_{0}(\mathcal{X})
, number of Monte Carlo episodes

N EP N_{\rm EP}
, and maximum horizon

T T
.

Set

A E←0 A_{E}\leftarrow 0
.

Set

τ←0\tau\leftarrow 0
.

for _e p=1,2,…,N EP e\_{p}=1,2,\ldots,N\_{\rm EP}_ do

Sample

x∼π 0​(𝒳)x\sim\pi_{0}(\mathcal{X})
.

Set

π 1 E​(x)=π 1 L​(x)=π 0​(x)\pi_{1}^{E}(x)=\pi_{1}^{L}(x)=\pi_{0}(x)∀x∈𝒳\forall x\in\mathcal{X}
.

for _t=1,2,…,T t=1,2,\ldots,T_ do

Choose action

a t a_{t}
using the individual

policy DNN

θ\theta
.

Sample

y t y_{t}
and

z t z_{t}
from

P​[y t|a t,x]P[y_{t}|a_{t},x]
and

Q​[z t|a t,x]Q[z_{t}|a_{t},x]
, respectively.

Update beliefs

π t L\pi_{t}^{L}
and

π t E\pi_{t}^{E}
using

respectively ([1](https://arxiv.org/html/2403.10112v2#S2.E1 "In II-A Centralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")) and ([2](https://arxiv.org/html/2403.10112v2#S2.E2 "In II-A Centralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")).

Set

γ t L←1−max x⁡π t L​(x)\gamma_{t}^{L}\leftarrow 1-\max_{x}\pi_{t}^{L}(x)
.

if _γ t L≤L\gamma\_{t}^{L}\leq L_ then

Break.

end if

end for

Set

τ←τ+t\tau\leftarrow\tau+t
.

Set

A E←A E+max t⁡max x⁡π t E​(x)A_{E}\leftarrow A_{E}+\max_{t}\max_{x}\pi_{t}^{E}(x)
.

end for

Set

τ←τ/N EP\tau\leftarrow\tau/N_{\rm EP}
.

Set

A E←A_{E}\leftarrow A E/N EP A_{E}/N_{\rm EP}
.

if _A E≥1−E A\_{E}\geq 1-E_ then

Output:

−A E-A_{E}
.

end if

Output:

1/τ 1/\tau
.

Algorithm 1 Fitness for Centralized EAHT

### III-B Decentralized EAHT

We now present a novel dual-component deep NE approach for multi-agent cooperative tasks, which builds upon existing single-agent NE algorithms; to this end, we will use the previously mentioned CoSyNE algorithm[[54](https://arxiv.org/html/2403.10112v2#bib.bib54)], but other algorithms can be used as well. The proposed approach maintains all the previously highlighted NE benefits, and can be applied to tasks with multiple heterogeneous agents. Its first component is a feature extractor neural network that is utilized by all agents, and its second component consists of K K individual branches, one for each agent. The idea is to deploy the feature extractor weights to learn functions that will be used by all agents. The individual branches are then used to learn specific policies for each agent. Recall that the agents are in general heterogeneous, hence, they might have vastly different beliefs and action sets of different sizes. Moreover, some agents may have to remain inactive more often because their actions could cause very significant information leakage. For the latter reasons, the proposed approach uses individual branches.

An individual with DNN parameters 𝜽\boldsymbol{\theta} can be split in the K+1 K+1 parts: f f and b 1,b 2,…,b K b_{1},b_{2},\ldots,b_{K}, where f f indicates the global extractor and each b k b_{k} represents each k k-th branch. The entire architecture is evolved as one network using CoSyNE[[54](https://arxiv.org/html/2403.10112v2#bib.bib54)]. Evolutionary operations, such as crossover, are performed by one algorithm on the genes of the entire individual 𝜽\boldsymbol{\theta} and not on the separate branches, allowing us to maintain all the previously mentioned benefits of single-agent NE. During the deployment/testing phase, each agent is provided with the common feature extractor and its individual branch. The proposed neural network architecture for NE-based multi-agent cooperative tasks is illustrated in Fig.[2](https://arxiv.org/html/2403.10112v2#S3.F2 "Figure 2 ‣ III-B Decentralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases").

![Image 2: Refer to caption](https://arxiv.org/html/2403.10112v2/x2.png)

Figure 2: The proposed neural network architecture for NE-based multi-agent cooperation. Each agent k k passes its belief to the global feature extractor f f, and then, to its individual branch b k b_{k} including a DNN for individual policy learning. The entire architecture can be evolved with typical NE algorithms, such as the deployed CoSyNE one here[[54](https://arxiv.org/html/2403.10112v2#bib.bib54)].

In the proposed decentralized EAHT scheme, the fitness function of an individual is evaluated as follows. For each evaluation episode, the hypothesis is randomly sampled from π 0​(⋅)\pi_{0}(\cdot) and the beliefs of all agents are initialized. At each time instance t t, each agent selects its action by passing the local belief through the feature extractor, and then, by forwarding the resulting output to its local branch. After N EP N_{\rm EP} Monte Carlo episodes take place, the fitness is computed according to([9](https://arxiv.org/html/2403.10112v2#S3.E9 "In III-A Centralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")), considering an appropriate adjustment to account for the decentralized stopping time, as defined in Section[II-B](https://arxiv.org/html/2403.10112v2#S2.SS2 "II-B Decentralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"). Each agent utilizes the stopping rule γ t L<L\gamma_{t}^{L}<L, as defined in the previous Section[III-A](https://arxiv.org/html/2403.10112v2#S3.SS1 "III-A Centralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"). A pseudocode describing the fitness function calculation for the decentralized EAHT case is included in Algorithm[2](https://arxiv.org/html/2403.10112v2#algorithm2 "In III-B Decentralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases").

Complexity Analysis: By denoting with T FP G T_{\rm FP}^{G} and T FP I T_{\rm FP}^{I} the forward pass times for the global feature extractor neural network and the DNNs at the individual branches, respectively, the computational complexity of the fitness calculation for one individual is O(K(T FP G+T FP I)T N EP))O(K(T_{\rm FP}^{G}+T_{\rm FP}^{I})TN_{\rm EP})). Since both the extractor and the individual branches are optimized as one joint structure, the time complexity required to construct a new generation of individual policy DNNs does not differ from that of the centralized method. Consequently, the total complexity of our decentralized NE-based optimization scheme is N gen O(L pop(T FP G+T FP I)T N EP+L pop 2 N w/16))N_{\rm gen}O(L_{\rm pop}(T_{\rm FP}^{G}+T_{\rm FP}^{I})TN_{\rm EP}+L^{2}_{\rm pop}N_{w}/16)).

Input:Individual DNN parameters

θ\theta
, thresholds

E E
and

L L
, prior

π 0​(𝒳)\pi_{0}(\mathcal{X})
, number of Monte Carlo episodes

N EP N_{\rm EP}
, maximum horizon

T T
.

Split

θ\theta
to

f,b 1,b 2,…,b K f,b_{1},b_{2},\ldots,b_{K}
.

Set

A E←0 A_{E}\leftarrow 0
.

Set

τ←0\tau\leftarrow 0
.

for _e p=1,2,…,N EP e\_{p}=1,2,\ldots,N\_{\rm EP}_ do

Sample

x∼π 0​(𝒳)x\sim\pi_{0}(\mathcal{X})
.

Set

ρ 1 E​(x)=ρ 1 k​(x)=π 0​(x)\rho_{1}^{E}(x)=\rho_{1}^{k}(x)=\pi_{0}(x)∀x∈𝒳\forall x\in\mathcal{X}
and

∀k=1,2,…,K\forall k=1,2,\ldots,K
.

for _t=1,2,…,T t=1,2,\ldots,T_ do

for _k=1,2,…,K k=1,2,\ldots,K_ do

Choose action

a t k a_{t}^{k}
using the policy DNN

b k b_{k}
and the extractor

f f
.

Sample

y t k y^{k}_{t}
and

z t k z^{k}_{t}
from

P​[y t k|a t k,x]P[y_{t}^{k}|a^{k}_{t},x]
and

Q​[z t k|a t k,x]Q[z_{t}^{k}|a^{k}_{t},x]
, respectively.

end for

for _k=1,2,…,K k=1,2,\ldots,K_ do

Update beliefs

ρ t k\rho_{t}^{k}
using ([7](https://arxiv.org/html/2403.10112v2#S2.E7 "In II-B Decentralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")).

Set

γ t k←1−max x⁡π t k​(x)\gamma_{t}^{k}\leftarrow 1-\max_{x}\pi_{t}^{k}(x)
.

if _γ t k≤L\gamma\_{t}^{k}\leq L_ then

Agent

k k
exits.

end if

end for

Update

ρ t E\rho^{E}_{t}
using ([8](https://arxiv.org/html/2403.10112v2#S2.E8 "In II-B Decentralized Problem ‣ II EAHT Problem Formulations ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")).

if _All agents have exited_ then

Break.

end if

end for

Set

τ←τ+t\tau\leftarrow\tau+t
.

Set

A E←A E+max t⁡max x⁡ρ t E​(x)A_{E}\leftarrow A_{E}+\max_{t}\max_{x}\rho_{t}^{E}(x)
.

end for

Set

τ←τ/N EP\tau\leftarrow\tau/N_{\rm EP}
.

Set

A E←A E/N EP A_{E}\leftarrow A_{E}/N_{\rm EP}
.

if _A E≥1−E A\_{E}\geq 1-E_ then

Output:

−A E-A_{E}
.

end if

Output:

1/τ 1/\tau
.

Algorithm 2 Fitness for Decentralized EAHT

Remark (The Role of the Global Extractor): While we use the term “global” to refer to the feature extractor f f, this operator actually processes only the local beliefs ρ t k\rho_{t}^{k} for each agent k k. To this end, copies of the same parameters of f f are shared by all agents. This is the intention behind the term “global,” which is used to learn common operations that will be used by all agents to improve efficiency. Parameter sharing is generally very successful in multi-agent DRL [[24](https://arxiv.org/html/2403.10112v2#bib.bib24), [64](https://arxiv.org/html/2403.10112v2#bib.bib64)] and is preferred over fully independent DNNs. This motivated us to adopt it herein in our NE framework.

#### III-B1 Joint NE and Pruning

In this section, we present a decentralized EAHT scheme that builds upon Algorithm[2](https://arxiv.org/html/2403.10112v2#algorithm2 "In III-B Decentralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") combining NE and pruning[[32](https://arxiv.org/html/2403.10112v2#bib.bib32), [38](https://arxiv.org/html/2403.10112v2#bib.bib38)]. In particular, the proposed scheme comprises two distinct steps: i) a joint optimization and pruning step; and ii) a fine-tuning step for the pruned solution, as shown in Algorithm[3](https://arxiv.org/html/2403.10112v2#algorithm3 "In III-B1 Joint NE and Pruning ‣ III-B Decentralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"). Each step employs a separate run of the CoSyNE algorithm to achieve its objectives.

In the first step, we initialize a population of dense DNNs, where each network comprises the global feature extractor f f and the K K individual branches b 1,b 2,…,b K b_{1},b_{2},\ldots,b_{K}, as described previously. During each fitness function evaluation, every layer of the candidate DNN undergoes unstructured weight-level pruning by a predefined pruning percentage p i p_{i}. To this end, redundant weights are set to zero, and the pruned network is subsequently evaluated following the procedure as the previous unpruned decentralized NE-based scheme. After the evaluation round, the top-performing individuals are selected and combined (mated) to produce the next generation of candidate solutions. This evolutionary process is repeated over multiple generations, and as pruning is applied iteratively, the networks are typically pruned beyond the initial pruning percentage p i p_{i}. The output of this step, denoted as θ∗\theta^{*}, is a sparsely structured network optimized for both performance and efficiency. This extensive pruning effect will be experimentally verified.

In the second step, the sparse structure of θ∗\theta^{*} is preserved, and the focus shifts to fine-tuning its nonzero weights. A population of networks is initialized, each retaining the structure of θ∗\theta^{*}, with unnecessary parameters masked to zero. For each individual, the nonzero weights of θ∗\theta^{*} are copied and perturbed with small-magnitude Gaussian noise to introduce diversity. The CoSyNE algorithm is then applied to this newly constructed population, enabling standard evolutionary refinement of the pruned solution. Individual evaluations in this step follow the same procedure outlined in the previous unpruned scheme.

The overall procedure for this decentralized EAHT scheme implementing joint NE and pruning, which is summarized in Algorithm[3](https://arxiv.org/html/2403.10112v2#algorithm3 "In III-B1 Joint NE and Pruning ‣ III-B Decentralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), leverages Algorithm[2](https://arxiv.org/html/2403.10112v2#algorithm2 "In III-B Decentralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") to evaluate candidates. It noted that our proposed joint NE-based optimization and pruning framework is general and can be applied to any genetic algorithm of choice besides the CoSyNE algorithm we are using in this paper.

Complexity Analysis: Pruning operations for a DNN with N w N_{w} learnable weights can be achieved with linear time complexity. Therefore, for the first step of our decentralized EAHT scheme with joint NE and pruning, the complexity expression becomes N gen O(L pop(T FP G+T FP I)N w T N EP+L pop 2 N w/16))N_{\rm gen}O(L_{\rm pop}(T_{\rm FP}^{G}+T_{\rm FP}^{I})N_{w}TN_{\rm EP}+L^{2}_{\rm pop}N_{w}/16)). For the second step, the complexity expression is N gen O(L pop(T FP G′+T FP I′)T N EP+L pop 2 N w′/16))N_{\rm gen}O(L_{\rm pop}(T_{\rm FP}^{G^{\prime}}+T_{\rm FP}^{I^{\prime}})TN_{\rm EP}+L^{2}_{\rm pop}N_{w}^{\prime}/16)), where N w′N_{w}^{\prime} represents the number of non-zero weights of θ∗\theta^{*}, whereas T FP G′T_{\rm FP}^{G^{\prime}} and T FP G′T_{\rm FP}^{G^{\prime}} denote the forward pass time of the pruned extractor and that of the individual branch DNNs, respectively. Since a significant number of weights will be pruned, we can safely assume that N w′≪N w N_{w}^{\prime}\ll N_{w}, T FP G′<T FP G T_{\rm FP}^{G^{\prime}}<T_{\rm FP}^{G}, and T FP l′<T FP l T_{\rm FP}^{l^{\prime}}<T_{\rm FP}^{l}, implying that the N gen O(L pop(T FP G′+T FP I′)T N EP+L pop 2 N w′/16))N_{\rm gen}O(L_{\rm pop}(T_{\rm FP}^{G^{\prime}}+T_{\rm FP}^{I^{\prime}})TN_{\rm EP}+L^{2}_{\rm pop}N_{w}^{\prime}/16)) term does not need to be included in the complexity expression.

Input:Pruning percentage

p i p_{i}
, number of generations

N gen N_{\rm gen}
, population size

L pop L_{\rm pop}
, and noise variance

σ 2\sigma^{2}
.

Step 1: Joint Optimization and Pruning

Initialize a population of

L pop L_{\rm pop}
dense DNNs.

for _generation g=1,2,…,N gen g=1,2,\ldots,N\_{\rm gen}_ do

for _each individual in the population_ do

Apply unstructured pruning with percentage

p i p_{i}
to each layer.

Evaluate the pruned network using

Algorithm[2](https://arxiv.org/html/2403.10112v2#algorithm2 "In III-B Decentralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases").

end for

Select top-performing individuals for mating.

Generate offsprings through genetic operations.

end for

Let

θ∗\theta^{*}
be the DNN of the best pruned individual

from the final generation.

Step 2: Fine-Tuning the Pruned Solution

Initialize a new population of

L pop L_{\rm pop}
DNNs

with the sparse structure of

θ∗\theta^{*}
.

for _each individual in the population_ do

Copy nonzero weights from

θ∗\theta^{*}
.

Add the Gaussian noise

𝒩​(0,σ 2)\mathcal{N}(0,\sigma^{2})
to the copied

weights.

end for

for _generation g=1,2,…,N gen g=1,2,\ldots,N\_{\rm gen}_ do

for _each individual in the population_ do

Evaluate the individual using Algorithm [2](https://arxiv.org/html/2403.10112v2#algorithm2 "In III-B Decentralized EAHT ‣ III Deep Neuroevolution Schemes for EAHT ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases").

end for

Select top-performing individuals for mating.

Generate offsprings through genetic operations.

end for

Output: Final fine-tuned sparse DNN

θ∗\theta^{*}
.

Algorithm 3 Joint NE and Pruning

IV Numerical Results and Discussion
-----------------------------------

In this section, we present performance evaluation results for our NE-based single- and multi-agent EAHT schemes, considering an anomaly detection scenario over wireless sensor networks. Different values for the thresholds L L and E E as well as for the number of sensors, S S, have been considered. Two different statistical sensor models commonly adopted in recent literature have been implemented[[6](https://arxiv.org/html/2403.10112v2#bib.bib6), [28](https://arxiv.org/html/2403.10112v2#bib.bib28)]. In addition, we considered an anomaly detection application where sensors transmit their data over Ricean fading channels, as well as a radar-assisted target detection application similar to[[8](https://arxiv.org/html/2403.10112v2#bib.bib8)].

TABLE I: Flipping probabilities for each sensor’s three distinct access actions.

### IV-A NE Implementation

The proposed single-agent NE scheme uses a feed-forward DNN with 2 2 hidden layers each comprising n h=200 n_{\rm h}=200 weights, whereas the proposed multi-agent algorithm utilizes a feature extractor with 2 2 hidden layers of n f=300 n_{\rm f}=300 hidden weights and K K branches (each corresponding to one of the K K agents/sensors), each including a 2 2-layer DNN with n b=300 n_{\rm b}=300 weights per layer. The mutation probability, p mut p_{\rm mut}, was set to the relatively high value 0.5 0.5 to ensure sufficient exploration, whereas the standard deviation of the mutation, σ mut\sigma_{\rm mut}, was given the value to 0.6 0.6. The population size was set to L pop=50 L_{\rm pop}=50 and we evolved the DNNs over N gen=50 N_{\rm gen}=50 generations. For each individual, the fitness function was evaluated over N EP=100 N_{\rm EP}=100 episodes, and, for the decentralized EAHT scheme with pruning, we have set p i=0.2 p_{i}=0.2. The obtained results of all learning agents were further averaged by running the respective algorithms for 20 20 different initialization seeds. All DNNs were trained on a GeForce RTX 3080 3080 GPU with 32 32 GB memory.

### IV-B Benchmark Schemes

For the single-agent problem, we have implemented two benchmark AHT strategies that ignore the existence of Eve: the Chernoff test[[2](https://arxiv.org/html/2403.10112v2#bib.bib2), [3](https://arxiv.org/html/2403.10112v2#bib.bib3)] and a myopic Extrinsic Jensen-Shannon (EJS) divergence maximization strategy[[65](https://arxiv.org/html/2403.10112v2#bib.bib65)]. In addition, since the majority of recent work on learning-based AHT uses deep actor critic algorithms, e.g., [[7](https://arxiv.org/html/2403.10112v2#bib.bib7), [28](https://arxiv.org/html/2403.10112v2#bib.bib28), [29](https://arxiv.org/html/2403.10112v2#bib.bib29), [44](https://arxiv.org/html/2403.10112v2#bib.bib44), [40](https://arxiv.org/html/2403.10112v2#bib.bib40)], we have considered one such algorithm with appropriately modified reward structures. We have particularly simulated the performance of two PPO DRL algorithms rewarded for error minimization, such as the one in[[29](https://arxiv.org/html/2403.10112v2#bib.bib29), [40](https://arxiv.org/html/2403.10112v2#bib.bib40)], and for confidence maximization, similar to the one in[[42](https://arxiv.org/html/2403.10112v2#bib.bib42)]. For these algorithms, if the privacy constraint failed, a large penalty was reached and, consequently, the episode was terminated. Similar penalty-based rewards have been used in related POMDPs, e.g., in[[59](https://arxiv.org/html/2403.10112v2#bib.bib59)]. For this DRL approach, larger DNNs than for the proposed NE-based schemes were used, in particular, an actor and a critic with 2 2 hidden layers each consisting of 300 300 learnable weights. Besides PPO, we also used an Advantage Actor Critic (A2C)[[19](https://arxiv.org/html/2403.10112v2#bib.bib19)] algorithm and a DQN [[17](https://arxiv.org/html/2403.10112v2#bib.bib17)] with the first reward structure.

Decentralized POMDPs are known to require at least exponential complexity[[66](https://arxiv.org/html/2403.10112v2#bib.bib66)], hence, it is extremely difficult to use mathematical methods for the considered multi-agent problem. For this reason, we have focused on baseline learning-based algorithms and used two DRL algorithms with individual actors and a global larger critic similar to[[28](https://arxiv.org/html/2403.10112v2#bib.bib28), [29](https://arxiv.org/html/2403.10112v2#bib.bib29), [24](https://arxiv.org/html/2403.10112v2#bib.bib24)]. More specifically, an Actor Critic (AC) algorithm and a PPO extension of the MADDPG structure [[24](https://arxiv.org/html/2403.10112v2#bib.bib24)] have been developed 2 2 2 Besides EAHT, similar algorithms have ben deployed in a wide variety of wireless applications, including caching [[27](https://arxiv.org/html/2403.10112v2#bib.bib27)], dynamic spectrum access[[67](https://arxiv.org/html/2403.10112v2#bib.bib67)], and power allocation [[26](https://arxiv.org/html/2403.10112v2#bib.bib26)], making them representative powerful benchmarks.. In the implementation of these benchmarks, when the privacy constraint was satisfied, they were rewarded for error minimization, otherwise, a large penalty was received. Apart from the penalty, the training was nearly identical to state-of-the-art DRL approaches for multi-agent active sensing[[28](https://arxiv.org/html/2403.10112v2#bib.bib28), [29](https://arxiv.org/html/2403.10112v2#bib.bib29)]. We also used two counterparts with gradual unstructured pruning, where the sparsity levels gradually increased according to a polynomial schedule [[33](https://arxiv.org/html/2403.10112v2#bib.bib33)]. More advanced pruning methods, like the recent one in[[67](https://arxiv.org/html/2403.10112v2#bib.bib67)], are left for future work.

### IV-C Results for Centralized EAHT

A number of S S independent and identical sensors were tasked to detect anomalies in their proximity[[6](https://arxiv.org/html/2403.10112v2#bib.bib6), [7](https://arxiv.org/html/2403.10112v2#bib.bib7)]. We have assumed that any number of sensors can be near an anomaly, hence, there were in total 2 S 2^{S} possible hypotheses. At each time instance t t, the single agent probed one sensor and received the following binary observation:

y t={s,with probability​1−P flip L 1−s,with probability​P flip L,y_{t}=\begin{cases}s,\quad\quad\,\,\,\,\text{with probability }1-P_{\rm flip}^{L}\\ 1-s,\quad\text{with probability}P_{\rm flip}^{L}\\ \end{cases},(10)

where s s is a binary number corresponding to the sensor’s state (whether it is near an anomaly or not) and P flip L P_{\rm flip}^{L} is the flipping probability. A similar expression held for Eve’s observation z t z_{t}, whose flipping probability is denoted by P flip E P_{\rm flip}^{E}. Note that binomial sensor models have been assumed in various relevant references, e.g., [[6](https://arxiv.org/html/2403.10112v2#bib.bib6), [7](https://arxiv.org/html/2403.10112v2#bib.bib7), [40](https://arxiv.org/html/2403.10112v2#bib.bib40), [41](https://arxiv.org/html/2403.10112v2#bib.bib41)]. We have further assumed that the single agent can access each sensor with three different actions, each corresponding to one of the three different flipping probability values. Therefore, the total actions available to the agent were 3​S 3S. The three different flipping probability values and the respective three distinct sensor access actions are listed in Table[I](https://arxiv.org/html/2403.10112v2#S4.T1 "TABLE I ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases").

![Image 3: Refer to caption](https://arxiv.org/html/2403.10112v2/x3.png)

(a)Legitimate error probability: E=0.3 E=0.3.

![Image 4: Refer to caption](https://arxiv.org/html/2403.10112v2/x4.png)

(b)Eve’s error probability: E=0.3 E=0.3.

![Image 5: Refer to caption](https://arxiv.org/html/2403.10112v2/x5.png)

(c)Average episode stopping time: E=0.3 E=0.3.

![Image 6: Refer to caption](https://arxiv.org/html/2403.10112v2/x6.png)

(d)Legitimate error probability: E=0.4 E=0.4.

![Image 7: Refer to caption](https://arxiv.org/html/2403.10112v2/x7.png)

(e)Eve’s error probability: E=0.4 E=0.4.

![Image 8: Refer to caption](https://arxiv.org/html/2403.10112v2/x8.png)

(f)Average episode stopping time: E=0.4 E=0.4.

Figure 3: Error probability performance for the centralized (single-agent) EAHT problem considering ([10](https://arxiv.org/html/2403.10112v2#S4.E10 "In IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"))’s binomial sensor model.

In the performance results illustrated in Fig. [3](https://arxiv.org/html/2403.10112v2#S4.F3 "Figure 3 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), we have set L=0.1 L=0.1 and considered two values for E E, namely E=0.4 E=0.4 and 0.3 0.3. In addition, the number of available sensors S S was varied between 2 2 to 6 6. It can be first observed that the legitimate error probability is lower than the threshold value L L for all approaches besides the EJS benchmark. Interestingly, the proposed NE-based EAHT approach and the PPO benchmarks lead to substantially higher error probability on Eve’s side. On the other hand, the conventional approaches cannot satisfy the privacy constraints, resulting in a large margin from the latter best schemes. It can also be seen that, for S<4 S<4, Eve’s error probability when running these benchmarks is always less than 0.3 0.3. In addition, for some experiments, Eve’s error probability was more than 50%50\% smaller than the desired threshold. It can be also seen that, as expected, there is a trade-off between the episode stopping time and the privacy objective, since the proposed NE-based and PPO methods need to perform a few more sensing actions. It is finally shown, that the CoSyNE algorithm achieves a shorter average stopping time than the PPO benchmarks in all simulated investigations. In fact, in some simulations’ settings, the stopping time of the CoSyNE algorithm was 20%20\% shorter than the stopping time with the PPO algorithms. Moreover, for S=6 S=6, CoSyNE terminates faster than the Chernoff test, which by design ignores the existence of the eavesdropper.

To investigate the robustness of the CoSyNE optimizer, we conducted a sensitivity study under deviations in the hyperparameters focusing on the larger environment with the S=6 S=6 sensors. For each hyperparameter, we varied its value, keeping the rest fixed. The training and testing procedures were repeated for each configuration in order to examine the effect that each hyperparameter has on the algorithm’s performance. We varied the mutation probability p mut p_{\rm mut} from 0.3 0.3 to 0.6 0.6, the standard deviation σ mut\sigma_{\rm mut} from 0.4 0.4 to 0.8 0.8, and the hidden size of the DNN n h n_{\rm h} from 150 150 to 300 300. The generated results demonstrated that, for each case, the CoSyNE optimizer can discover policies that satisfy the constraints, while reaching “good” stopping times. The Coefficients of Variation (CVs) for the stopping time are depicted in Fig.[4](https://arxiv.org/html/2403.10112v2#S4.F4 "Figure 4 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), clearly demonstrating the stability of our scheme. It is particularly shown that the CVs are consistently significantly smaller than 0.1 0.1, signifying very good robustness; the error probabilities are omitted from the presentation due to space constraints.

![Image 9: Refer to caption](https://arxiv.org/html/2403.10112v2/x9.png)

Figure 4: Coefficients of Variation (CVs) of the stopping time objective for the single-agent NE-based EAHT scheme, considering S=6 S=6 sensors and variations of hyperparameters.

![Image 10: Refer to caption](https://arxiv.org/html/2403.10112v2/x10.png)

Figure 5: The frequency of the first and third sensor access actions of the final evolved policies considering ([10](https://arxiv.org/html/2403.10112v2#S4.E10 "In IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"))’s binomial sensor model.

In Fig. [5](https://arxiv.org/html/2403.10112v2#S4.F5 "Figure 5 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), we plot the probabilities with which the optimized policies select the access action with the minimum privacy (both flipping probabilities are set to 0.125 0.125) and the maximum privacy (P flip L=0.25,P flip E=0.45 P^{L}_{\rm flip}=0.25,P_{\rm flip}^{E}=0.45) access modes. It is apparent that the solution is not trivial (e.g., select only maximum privacy, or quickly detect the hypothesis ignoring the E E values). In fact, a balance of all three access/protection levels is required to ensure secure and reliable inference. Interestingly, leakage to the eavesdropper is sometimes accepted in order to form initial estimates, and when quality beliefs are formed by agent L L, less informative actions can be selected. It can be also seen that, for larger E E values, the third action that maximizes privacy is taken a little more often.

To further validate the effectiveness of our approaches, we have simulated a second sensor model including Gaussian observations, similar to [[28](https://arxiv.org/html/2403.10112v2#bib.bib28), [44](https://arxiv.org/html/2403.10112v2#bib.bib44)]. According to this model, the observations returned by a probed sensor are given by:

y t∼{𝒩​(1,σ l 2),if the sensor is near an anomaly.𝒩​(0,σ l 2),otherwise.y_{t}\sim\begin{cases}\mathcal{N}\left(1,\sigma_{l}^{2}\right),\quad\text{if the sensor is near an anomaly.}\\ \mathcal{N}\left(0,\sigma_{l}^{2}\right),\quad\text{otherwise}.\par\end{cases}(11)

Again, a similar expression held for the Eve’s observations z t z_{t} with variance σ e 2\sigma_{e}^{2}. Like in the previous binomial model, we assumed three sensing modes for each sensor. The considered values of σ l 2\sigma_{l}^{2} and σ e 2\sigma_{e}^{2} for each mode are included in Table[II](https://arxiv.org/html/2403.10112v2#S4.T2 "TABLE II ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases").

By repeating the experiments of Fig.[3](https://arxiv.org/html/2403.10112v2#S4.F3 "Figure 3 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") for the Gaussian sensor model and E=0.3 E=0.3, it can be observed from the obtained results within Fig.[6](https://arxiv.org/html/2403.10112v2#S4.F6 "Figure 6 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") that again CoSyNE outperforms all benchmarks. Evidently, the EJS and the Chernoff algorithms cannot consistently satisfy the privacy constraint, and EJS misses the accuracy constraints. On the other hand, both the CoSyNE and the PPO algorithms satisfy the constraints, with the former achieving noticeably shorter stopping times.

All in all, it can be concluded from the results, for both sensor models in Figs.[3](https://arxiv.org/html/2403.10112v2#S4.F3 "Figure 3 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") and[6](https://arxiv.org/html/2403.10112v2#S4.F6 "Figure 6 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), that the classic benchmarks cannot satisfy the privacy and accuracy constraints in most settings. Interestingly, our proposed NE framework does meet those constraints in all settings, while achieving shorter stopping time than the considered modified DRL benchmarks with appropriate penalized reward signals.

TABLE II: Variances for each sensor’s three distinct access actions.

![Image 11: Refer to caption](https://arxiv.org/html/2403.10112v2/x11.png)

(a)Legitimate error probability: E=0.3 E=0.3.

![Image 12: Refer to caption](https://arxiv.org/html/2403.10112v2/x12.png)

(b)Eve’s error probability: E=0.3 E=0.3.

![Image 13: Refer to caption](https://arxiv.org/html/2403.10112v2/x13.png)

(c)Average episode stopping time: E=0.3 E=0.3.

Figure 6: Similar to Fig.[3](https://arxiv.org/html/2403.10112v2#S4.F3 "Figure 3 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), but for the Gaussian sensor model in ([11](https://arxiv.org/html/2403.10112v2#S4.E11 "In IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")).

#### IV-C1 Robustness Against DNN-Based Eavesdroppers

We now examine how our evolved policies can deal with eavesdroppers having learning capabilities. Focusing on the binomial observations of([10](https://arxiv.org/html/2403.10112v2#S4.E10 "In IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")) with E=0.3 E=0.3, we have collected a large training dataset of 80000 80000 episodes using our final policies, and another test set of 10000 10000 episodes. Each data point contains a sequence of actions a t a_{t}, observations z t z_{t}, and a label for the true hypothesis. We have trained different machine learning classifiers to verify that our method can trick a diverse set of adversaries. Due to the fact that trajectories have different lengths, we have considered two Recurrent Neural Networks (RNNs) with 3 3 hidden layers of 300 300 units, followed by a final softmax-activated output layer for classification. The RNNs were particularly a Long Short Term Memory Network (LSTM)[[68](https://arxiv.org/html/2403.10112v2#bib.bib68)] and a Gated Recurrent Unit (GRU)[[69](https://arxiv.org/html/2403.10112v2#bib.bib69)], with both being bidirectional enabling greater representation capabilities. These networks were developed using the pytorch framework[[70](https://arxiv.org/html/2403.10112v2#bib.bib70)] and optimized with the Adam optimizer[[71](https://arxiv.org/html/2403.10112v2#bib.bib71)] using a learning rate of 2.5×10−4 2.5\times 10^{-4}. We have also used a decision tree classifier which pads all sequences to the same length, i.e., max-padding.

It is important to note that an eavesdropper equipped with a large, high-fidelity dataset that closely matches the actual system represents a particularly powerful adversary; this effectively models a worst-case scenario. In many real-world deployments, such strong adversaries may not exist due to limited data access or mismatched system knowledge. Nonetheless, our proposed policies demonstrate strong resilience even under this pessimistic assumption. As illustrated in Fig.[7](https://arxiv.org/html/2403.10112v2#S4.F7 "Figure 7 ‣ IV-C1 Robustness Against DNN-Based Eavesdroppers ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), they are capable of misleading DNN-based eavesdroppers effectively across a wide range of settings.

![Image 14: Refer to caption](https://arxiv.org/html/2403.10112v2/x14.png)

Figure 7: Test set error probability estimates of various DNN-based eavesdroppers.

#### IV-C2 Incorrect Knowledge of Observation Models

So far, we have considered that both P​[⋅]P[\cdot] and Q​[⋅]Q[\cdot] are correctly estimated from a large dataset, as well as that the deployment conditions of our policies are identical to training. We will henceforth examine how our approach and the DRL benchmarks perform when P​[⋅]P[\cdot] and Q​[⋅]Q[\cdot] are only crude approximations of the true kernels P true​[⋅]P_{\rm true}[\cdot]Q true​[⋅]Q_{\rm true}[\cdot], respectively. We have particularly focused on the larger binomial sensor environment, testing the trained policies of Fig.[3](https://arxiv.org/html/2403.10112v2#S4.F3 "Figure 3 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") for S=6 S=6 sensors. We assumed that the agent updates its belief using the flipping probabilities of Table[I](https://arxiv.org/html/2403.10112v2#S4.T1 "TABLE I ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), however, the probabilities of the testing environment have been slightly perturbed being non-constant. More specifically, the legit flipping probability at each time instance t t was sampled from the uniform distribution in [0.85​P flip L,1.15​P flip L][0.85P_{\rm flip}^{L},1.15P_{\rm flip}^{L}]. It was also assumed that the agent underestimates the capabilities of the eavesdropper, whose flipping probability at each time instant t t was set to lie uniformly in [0.7​P flip E,0.9​P flip E][0.7P_{\rm flip}^{E},0.9P_{\rm flip}^{E}].

As shown in Table [III](https://arxiv.org/html/2403.10112v2#S4.T3 "TABLE III ‣ IV-C2 Incorrect Knowledge of Observation Models ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), our proposed NE-based approach, along with the PPO and DQN agents, successfully satisfies both the utility and privacy constraints. However, it can be seen that the A2C agent exhibits a slight violation of the privacy constraint. Notably, our method continues to achieve the shortest stopping times while maintaining robust performance under model mismatch. This highlights its strong generalization capability making it particularly well-suited for deployment in dynamic or uncertain environments.

TABLE III: Results for the case of incorrect knowledge of flipping probabilities and S=6 S=6 sensors.

#### IV-C3 Further Wireless System Applications

To further demonstrate the effectiveness of our approach, we have explored two additional realistic applications: an extension of the current sensor network scenario incorporating multipath fading conditions, and radar-based object detection.

##### Sensor Networks under Ricean Fading Channels

We have considered a system with S=3 S=3 sensors, each capable of detecting anomalies and broadcasting symbols x t,s x_{t,s} according to the following rule:

x t,s={1,if​s=1−1,otherwise.x_{t,s}=\begin{cases}1,&\text{if }s=1\\ -1,&\text{otherwise}\end{cases}.(12)

At each time step t t, the agent L L selects both a sensor and a transmit power level P t∈𝒫 P_{t}\in\mathcal{P}, where 𝒫\mathcal{P} is a discrete set of allowable power levels. The baseband received signal at agent L L is mathematically modeled as follows:

x^t,s L=P t​h t L​x t,s+n,\hat{x}_{t,s}^{L}=\sqrt{P_{t}}h^{L}_{t}x_{t,s}+n,(13)

where h t L h_{t}^{L} represents the complex Ricean fading channel gain coefficient between the selected sensor and agent L L, and n n is the Additive White Gaussian Noise (AWGN). The agent is then assumed to apply a hard decision rule according to which it decodes the received signal as y t=1 y_{t}=1 if Re⁡(x^t,s L)>0\operatorname{Re}(\hat{x}_{t,s}^{L})>0, and y t=0 y_{t}=0 otherwise. The flipping probabilities for each sensor and the power level have been empirically estimated using extensive Monte Carlo simulations.

A similar observation model has been considered for the eavesdropper, who receives the signal through an independent fading channel h t E h_{t}^{E}, which has been assumed to be characterized by a weaker Line-of-Sight (LoS) component. We have specifically used κ L=5​dB\kappa^{L}=5\,\mathrm{dB} as the Ricean factor for the legitimate agent and κ E=−2​dB\kappa^{E}=-2\,\mathrm{dB} as that for the eavesdropper. The set of the set of available power levels was 𝒫={−20​dB,−10​dB,0​dB}\mathcal{P}=\{-20\,\mathrm{dB},-10\,\mathrm{dB},0\,\mathrm{dB}\}.

The results for L=0.1 L=0.1 and E=0.3 E=0.3 are shown in Fig. [8](https://arxiv.org/html/2403.10112v2#S4.F8 "Figure 8 ‣ Sensor Networks under Ricean Fading Channels ‣ IV-C3 Further Wireless System Applications ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), where we varied the noise power from −90​dB-90\,\mathrm{dB} to −40​dB-40\,\mathrm{dB}. It can be observed that our approach can satisfy all constraints, unlike the DRL baselines which fail to satisfy privacy requirements under strong noise conditions. It is also shown that our approach achieves vastly shorter stopping times than the DRL benchmarks. Interestingly, it can also terminate quicker than the classic heuristics that ignore the existence of the eavesdropping agent E E.

![Image 15: Refer to caption](https://arxiv.org/html/2403.10112v2/x15.png)

(a)Legitimate error probability: E=0.3 E=0.3.

![Image 16: Refer to caption](https://arxiv.org/html/2403.10112v2/x16.png)

(b)Eve’s error probability: E=0.3 E=0.3.

![Image 17: Refer to caption](https://arxiv.org/html/2403.10112v2/x17.png)

(c)Average episode stopping time: E=0.3 E=0.3.

Figure 8: Similar metrics to Fig.[3](https://arxiv.org/html/2403.10112v2#S4.F3 "Figure 3 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") but for the case of a Ricean fading channel model with κ L=5​dB\kappa^{L}=5\,\mathrm{dB} and κ E=−2​dB\kappa^{E}=-2\,\mathrm{dB}.

##### Strong-or-weak radars for target detection

We now consider a radar target detection application based on the strong-or-weak return model[[8](https://arxiv.org/html/2403.10112v2#bib.bib8)], according to which the environment is either empty (i.e., no target is present), or exactly one of N targ N_{\rm targ} possible targets exists inside it. The legitimate agent L L has access to N targ N_{\rm targ} distinct waveforms, each designed to be optimal for detecting a specific target. At each time instant t t, the agent L L selects a waveform a t a_{t} for transmission and observes the reflected signal.

Under a Gaussian radar model, the received observation y t y_{t} follows a Gaussian distribution. If no target is present, y t∼𝒩​(0,σ L)y_{t}\sim\mathcal{N}(0,\sigma^{L}), i.e., zero-mean Gaussian with variance σ L\sigma^{L}. If the transmitted waveform matches the true target ν\nu, then y t∼𝒩​(m ν+,σ L 2)y_{t}\sim\mathcal{N}(m_{\nu}^{+},\sigma_{L}^{2}); otherwise, y t∼𝒩​(m ν−,σ L 2)y_{t}\sim\mathcal{N}(m_{\nu}^{-},\sigma_{L}^{2}), where m ν+m_{\nu}^{+} and m ν−m_{\nu}^{-} represent the strong and weak signal means, respectively. The eavesdropping agent E E observes the same waveform but through a degraded channel, resulting in a higher noise variance; this is denoted as σ E 2>σ L 2\sigma_{E}^{2}>\sigma_{L}^{2}.

In Fig.[9](https://arxiv.org/html/2403.10112v2#S4.F9 "Figure 9 ‣ Strong-or-weak radars for target detection ‣ IV-C3 Further Wireless System Applications ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), we have set N targ=5 N_{\rm targ}=5 and sampled m ν+m_{\nu}^{+} uniformly in [1,2][1,2] and similarly for m ν−m_{\nu}^{-} in [0.1,0.5][0.1,0.5] for each target ν\nu. The legitimate agent’s noise standard deviation was fixed at σ L 2=1\sigma_{L}^{2}=1, while the eavesdropper’s one varied from σ E 2=1.25\sigma_{E}^{2}=1.25 to 2 2. The constraint thresholds were set to L=0.1 L=0.1 and E=0.3 E=0.3. It can be demonstrated that the proposed CoSyNE-based approach outperforms all benchmarks satisfying the constraints, while also achieving the shortest stopping times. Notably, the EJS method performs very poorly in this setting.

![Image 18: Refer to caption](https://arxiv.org/html/2403.10112v2/x18.png)

(a)Legitimate error probability: E=0.3 E=0.3.

![Image 19: Refer to caption](https://arxiv.org/html/2403.10112v2/x19.png)

(b)Eve’s error probability: E=0.3 E=0.3.

![Image 20: Refer to caption](https://arxiv.org/html/2403.10112v2/x20.png)

(c)Average episode stopping time: E=0.3 E=0.3.

Figure 9: Similar metrics to Fig.[3](https://arxiv.org/html/2403.10112v2#S4.F3 "Figure 3 ‣ IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") but for the case of the strong-or-weak radar target detection application when the legitimate agent’s noise standard deviation is set to σ L 2=1\sigma_{L}^{2}=1.

### IV-D Results for Decentralized EAHT

In Fig.[10](https://arxiv.org/html/2403.10112v2#S4.F10 "Figure 10 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") and[11](https://arxiv.org/html/2403.10112v2#S4.F11 "Figure 11 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), we have considered the same observation models with those used in the single-agent case as well as K=4 K=4 fully connected agents. We have also set the thresholds to L=0.1 L=0.1 and E=0.3 E=0.3, and varied the number of sensors S S from 6 6 to 12 12, yielding at most 2 12 2^{12} possible hypotheses in total. We have assumed that the first two agents have access to the first half of the sensors, and the other two have access to the rest of the sensors. Since both our NE-based methods and the DRL benchmark for multi-agent EAHT satisfied the accuracy and privacy constraints, we include only the average stopping time for the binomial and Gaussian sensor models in Fig.[10](https://arxiv.org/html/2403.10112v2#S4.F10 "Figure 10 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") and Fig.[11](https://arxiv.org/html/2403.10112v2#S4.F11 "Figure 11 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), respectively. As shown, both the proposed unpruned and pruned (with only 10%10\% of the weights of the unpruned version) NE-based approaches achieve shorter stopping times than the designed DRL benchmarks. Interestingly, it is demonstrated that our pruned approach achieves better stopping times than the benchmarks, even if it had removed over 90%90\% of redundant weights in all simulated settings.

![Image 21: Refer to caption](https://arxiv.org/html/2403.10112v2/x21.png)

Figure 10: Average episode stopping time of decentralized EAHT for the threshold values L=0.1 L=0.1 and E=0.3 E=0.3, as well as for different values of S S, considering the binomial sensor model in([10](https://arxiv.org/html/2403.10112v2#S4.E10 "In IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")).

![Image 22: Refer to caption](https://arxiv.org/html/2403.10112v2/x22.png)

Figure 11: Similar to Fig.[10](https://arxiv.org/html/2403.10112v2#S4.F10 "Figure 10 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), but for ([11](https://arxiv.org/html/2403.10112v2#S4.E11 "In IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"))’s Gaussian sensor model. 

To evaluate the robustness of the proposed NE-based EAHT schemes under variations in the hyperparameters, we have performed a detailed sensitivity analysis. This analysis was conducted on both the unpruned and pruned agents, using the binomial sensor model with S=10 S=10 sensors. For each hyperparameter, we varied its value while keeping all other parameters fixed. The training and testing procedures were repeated for each configuration, allowing us to assess the impact of individual hyperparameter changes on the system’s performance. In particular, our sensitivity analysis included the following parameters:

1.   1.The probability p mut p_{\rm mut} was varied from 0.3 0.3 to 0.6 0.6. 
2.   2.The standard deviation σ mut\sigma_{\rm mut} was varied from 0.4 0.4 to 0.8 0.8. 
3.   3.The hidden size of the extractor n f n_{\rm f} was varied between 250 250 and 400 400. 
4.   4.The hidden size of each individual branch n b n_{\rm b} was varied between 250 250 and 400 400. 

Tables[IV](https://arxiv.org/html/2403.10112v2#S4.T4 "TABLE IV ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") and[V](https://arxiv.org/html/2403.10112v2#S4.T5 "TABLE V ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") list respectively the maximum legitimate and minimum eavesdropping error probabilities for each of the latter four sensitivity studies. It is clearly demonstrated that both proposed NE-based EAHT schemes always satisfy the accuracy and privacy constraints.

TABLE IV: The maximum (i.e., worst case) legitimate error probability for each sensitivity study. It can be seen that all values are below the threshold L=0.1 L=0.1.

TABLE V: The minimum (i.e., worst case) error probability at Eve for each sensitivity study. Clearly, all values are above the threshold E=0.3 E=0.3.

The CVs of the stopping time objective for each sensitivity study are illustrated in Fig.[12](https://arxiv.org/html/2403.10112v2#S4.F12 "Figure 12 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), verifying the robustness of both proposed schemes. As shown, the coefficients are always smaller than 0.06 0.06, implying that the stopping time objective is stable and robust to hyperparameter changes. All in all, the latter results underscore the stability and reliability of the proposed schemes, showcasing that their performance remains largely unaffected by hyperparameter variations. Such robustness is essential for practical implementations, ensuring consistent outcomes even under varying conditions.

![Image 23: Refer to caption](https://arxiv.org/html/2403.10112v2/x23.png)

Figure 12: Coefficients of variation of the stopping time objective for both the unpruned and pruned NE-based EAHT schemes, considering S=10 S=10 sensors and hyperparameters’ variation.

Finally, in Fig.[13](https://arxiv.org/html/2403.10112v2#S4.F13 "Figure 13 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), the evolution of the performance of our NE-based multi-agent EAHT schemes with respect to the generation number N gen N_{\rm gen} is illustrated. In particular, Fig.[13(a)](https://arxiv.org/html/2403.10112v2#S4.F13.sf1 "In Figure 13 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") depicts the fitness functions of our schemes in comparison with the fitness of the trained PPO-based agents. It can be observed that, initially, the fitness scores are negative, which signifies a failure to satisfy the privacy constraint. However, within just two generations, both algorithms identify candidate policies that meet this requirement. At first glance, the differences between the fitness functions of the different schemes may appear insignificant, but this happens because the fitness equals to 1/τ 1/\tau, when the privacy constraint is satisfied. However, a small increase in fitness indicates a decrease of a few time steps in the detection delay, which can be extremely important in applications of abnormal activity detection. For enhanced visualization, Fig.[13(b)](https://arxiv.org/html/2403.10112v2#S4.F13.sf2 "In Figure 13 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") presents the stopping times of all considered algorithms, starting from the third generation. As notably shown, within fewer than ten generations, both the pruned and unpruned agents surpass the performance of the multi-agent PPO-based benchmark.

![Image 24: Refer to caption](https://arxiv.org/html/2403.10112v2/x24.png)

(a)Fitness function.

![Image 25: Refer to caption](https://arxiv.org/html/2403.10112v2/x25.png)

(b)Stopping time.

Figure 13: Evolution of the EAHT performance versus the generation number N gen N_{\rm gen} for both the unpruned and pruned NE-based schemes, considering the same hyperparameter changes with Fig.[12](https://arxiv.org/html/2403.10112v2#S4.F12 "Figure 12 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") and S=8 S=8 sensors. Curves with our designed PPO-based scheme are also included for comparison purposes.

#### IV-D1 Non-Stationary Graphs

The previous experimentation considered only fully-connected agents. In this subsection, we investigate the generalization of the proposed NE-based decentralized agents when deployed in sparse time-varying communication graphs. To this end, we have assumed that each agent k k tries to broadcast the tuple (a t k,y t k)(a_{t}^{k},y_{t}^{k}) to all other agents, but the message is lost with probability l r l_{\rm r}, implying that the agents have different information sets. Hence, the structure of the agent connection graph is different at each time instant t t due to the message losses. We have evaluated the proposed policies in Figs.[10](https://arxiv.org/html/2403.10112v2#S4.F10 "Figure 10 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") and[11](https://arxiv.org/html/2403.10112v2#S4.F11 "Figure 11 ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases") on new testing environments, considering the same observation models but varying the loss rate from 0.1 0.1 to 0.25 0.25. as depicted in Fig.[14](https://arxiv.org/html/2403.10112v2#S4.F14 "Figure 14 ‣ IV-D1 Non-Stationary Graphs ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), although all methods can satisfy the privacy constraint, the proposed NE methods achieve noticeably shorter stopping times.

![Image 26: Refer to caption](https://arxiv.org/html/2403.10112v2/x26.png)

(a)Binomial sensor model.

![Image 27: Refer to caption](https://arxiv.org/html/2403.10112v2/x27.png)

(b)Gaussian sensor model.

Figure 14: Average episode stopping time of decentralized EAHT for threshold values L=0.1 L=0.1, E=0.3 E=0.3, S=6 S=6, as well as for different values of the message loss rate l r l_{\rm r}.

#### IV-D2 The Importance of Message Exchange

As previously noted, establishing secure and reliable communication channels between agents incurs costs. However, such communication can significantly enhance detection performance. To investigate the impact of message exchange, we have conducted a study on the binomial sensor model of([10](https://arxiv.org/html/2403.10112v2#S4.E10 "In IV-C Results for Centralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases")) involving a fully independent group of agents utilizing our proposed DNN architecture evolved with CoSyNE. In this setup, all agents had access to all sensors. Each agent independently probed one of the S S sensors and formed beliefs without message exchange. As illustrated in Fig. [15](https://arxiv.org/html/2403.10112v2#S4.F15 "Figure 15 ‣ IV-D2 The Importance of Message Exchange ‣ IV-D Results for Decentralized EAHT ‣ IV Numerical Results and Discussion ‣ Evasive Active Hypothesis Testing with Deep Neuroevolution: The Single- and Multi-Agent Cases"), this independence results in substantially increased stopping times. All other parameters (network structure, observation model, and algorithmic parameters) have been the same for both the independent and the networked CoSyNE agents.

![Image 28: Refer to caption](https://arxiv.org/html/2403.10112v2/x28.png)

Figure 15: Average stopping times for the decentralized EAHT with binomial observations considering networked versus independent CoSyNE agents. It is shown that lack of information exchange between agents more than doubles detection delay.

V Conclusions and Future Work
-----------------------------

In this paper, we studied both single- and multi-agent EAHT problems and presented NE-based solutions. Specifically for the decentralized multi-agent problem, we devised a novel NE method for dealing with collaborative multi-agent tasks, which maintains all computational benefits of single-agent NE. We also extended it by providing a second algorithm that jointly optimizes the multi-agent policy and removes the unnecessary DNN parameters. The robustness and superiority of the proposed NE-based EAHT approaches over benchmarks was demonstrated through extensive numerical simulations.

While the EAHT problem was introduced in[[61](https://arxiv.org/html/2403.10112v2#bib.bib61)] with asymptotic bounds on the eavesdropper’s error exponent, no concrete strategies with provable performance guarantees have been developed so far. In contrast, classical AHT benefits from theoretically grounded policies such as the Chernoff test or EJS maximization. Our NE-based approach fills this gap offering a practical, flexible alternative that performs competitively across a wide range of settings, and generalizes well, despite lacking theoretical guarantees. Besides the aforementioned bounds, devising novel theoretical strategies with privacy guarantees is a very important area for future research, even if these strategies do not work as well as NE.

One other research direction is to extend our experimental investigations and the theoretical analysis of[[61](https://arxiv.org/html/2403.10112v2#bib.bib61)] to other challenging active sensing tasks, like continuous high dimensional parameter estimation [[4](https://arxiv.org/html/2403.10112v2#bib.bib4)], change detection [[5](https://arxiv.org/html/2403.10112v2#bib.bib5)], and beam alignment [[10](https://arxiv.org/html/2403.10112v2#bib.bib10)]. It is also worthwhile to examine scenarios with multiple heterogeneous and active eavesdroppers. Finally, we intend to combine our data collection mechanism with active defense strategies[[72](https://arxiv.org/html/2403.10112v2#bib.bib72)] in order to handle adaptive and progressively improving eavesdroppers. These eavesdroppers can gradually infiltrate the network and acquire more accurate observation models[[73](https://arxiv.org/html/2403.10112v2#bib.bib73)].

References
----------

*   [1] G.Stamatelis et al., “Single- and multi-agent private active sensing: A deep neuroevolution approach,” in Proc. IEEE ICC, (Denver, USA), 2024. 
*   [2] H.Chernoff, “Sequential design of experiments,” Ann. Math. Stat., vol.30, no.3, pp.755–770, 1959. 
*   [3] S.Nitinawarat et al., “Controlled sensing for multihypothesis testing,” IEEE Trans. Autom. Control, vol.58, no.10, pp.2451–2464, 2013. 
*   [4] A.Mukherjee et al., “Active sampling of multiple sources for sequential estimation,” IEEE Trans. Signal Process, vol.70, pp.4571–4585, 2022. 
*   [5] A.Gopalan et al., “Bandit quickest changepoint detection,” in Proc. NeurIPS, 2021. 
*   [6] G.Joseph et al., “Anomaly detection via learning-based sequential controlled sensing,” IEEE Sensors J., vol.24, no.13, pp.21025–21037, 2024. 
*   [7] G.Joseph et al., “Scalable and decentralized algorithms for anomaly detection via learning-based controlled sensing,” IEEE Trans. Signal Inf. Process. Netw., vol.9, pp.640–654, 2023. 
*   [8] M.Franceschetti et al., “Chernoff test for strong-or-weak radar models,” IEEE Trans. Signal Process, vol.65, no.2, pp.289–302, 2017. 
*   [9] N.Atanasov et al., “Hypothesis testing framework for active object detection,” in Proc. IEEE ICRA, (Karlsruhe,Germany), 2013. 
*   [10] F.Sohrabi et al., “Deep active learning approach to adaptive beamforming for mmWave initial alignment,” in Proc. IEEE ICASSP, (Toronto, Canada), 2021. 
*   [11] Z.Zhang et al., “Active sensing for localization with reconfigurable intelligent surface,” arXiv preprint :2312.09002, 2023. 
*   [12] F.Sohrabi et al., “Active sensing for communications by learning,” IEEE J. Sel. Areas Commun, vol.40, no.6, pp.1780–1794, 2022. 
*   [13] G.C. Alexandropoulos et al., “Pervasive machine learning for smart radio environments enabled by reconfigurable intelligent surfaces,” Proc. IEEE, vol.110, no.9, pp.1494–1525, 2022. 
*   [14] M.G.S. Murshed et al., “Machine learning at the network edge: A survey,” ACM Comput. Surv., vol.54, Oct. 2021. 
*   [15] K.Rose et al., “The internet of things: An overview,” The internet society (ISOC), vol.80, no.15, pp.1–53, 2015. 
*   [16] J.Ren et al., “Information exposure from consumer IoT devices: A multidimensional, network-informed measurement approach,” in Proc. of the Internet Measur. Conf., (Amsterdam, Netherlands), 2019. 
*   [17] V.Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol.518, pp.529–533, Feb. 2015. 
*   [18] J.Schulman, F.Wolski, P.Dhariwal, A.Radford, and O.Klimov arXiv preprint: 1707.06347. 
*   [19] V.Mnih et al., “Asynchronous methods for deep reinforcement learning,” in Proc. ICML, (NY, USA), 20–22 Jun 2016. 
*   [20] C.H. Papadimitriou and J.N. Tsitsiklis, “The complexity of Markov decision processes,” Math. Oper. Res., vol.12, pp.441–450, 1987. 
*   [21] A.Wong et al., “Deep multiagent reinforcement learning: Challenges and directions,” Artificial Intelligence Review, 2022. 
*   [22] K.Zhang, Z.Yang, and T.Başar, Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms, pp.321–384. Cham: Springer International Publishing, 2021. 
*   [23] L.Kraemer and B.Banerjee, “Multi-agent reinforcement learning as a rehearsal for decentralized planning,” Neurocomputing, vol.190, pp.82–94, 2016. 
*   [24] R.Lowe et al., “Multi-agent actor-critic for mixed cooperative-competitive environments,” in Proc. NeurIPS, (Long Beach, California, USA), 2017. 
*   [25] A.Gao et al., “A cooperative spectrum sensing with multi-agent reinforcement learning approach in cognitive radio networks,” IEEE Commun. Lett., vol.25, no.8, pp.2604–2608, 2021. 
*   [26] Q.Fan et al., “MADDPG-based power allocation algorithm for network-assisted full-duplex cell-free mmwave massive mimo systems with dac quantization,” in Proc. IEEE WCSP, (Nanjing, China), 2022. 
*   [27] Y.Zhi et al., “Multi-agent reinforcement learning for cooperative edge caching in heterogeneous networks,” in Proc. IEEE WCSP, (Changsha, China), 2021. 
*   [28] H.Szostak and K.Cohen, “Decentralized anomaly detection via deep multi-agent reinforcement learning,” in Proc. Allerton, (Monticello, Illinois, USA), 2022. 
*   [29] G.Stamatelis and N.Kalouptsidis, “Deep reinforcement learning for active hypothesis testing with heterogeneous agents and cost constraints,” TechRxiv, May 2023. 
*   [30] S.Wu et al., “Distributed federated deep reinforcement learning based trajectory optimization for air-ground cooperative emergency networks,” IEEE Tran. Veh. Technol., vol.71, no.8, pp.9107–9112, 2022. 
*   [31] S.Xu et al., “Heterogeneous resource allocation in LEO networks: A federated multi-agent deep reinforcement learning method,” Proc. ACM ICCIP, (Hainan, China), 2025. 
*   [32] D.Blalock et al., “What is the state of neural network pruning?,” in Proc. MLSys, (Austin, USA), 2020. 
*   [33] S.Han et al., “Learning both weights and connections for efficient neural network.,” in Proc. NeurIPS, (Montreal, Canada), 2015. 
*   [34] Y.LeCun et al., “Optimal brain damage.,” in Proc. NeurIPS, (Denver, USA), 1990. 
*   [35] J.Frankle and M.Carbin, “The lottery ticket hypothesis: Finding sparse, trainable neural networks,” in Proc. ICLR, (New Orleans, USA), 2019. 
*   [36] L.Graesser et al., “The state of sparse training in deep reinforcement learning.,” in Proc. ICML, (Baltimore, USA), 2022. 
*   [37] G.Sohkar et al., “Dynamic sparse training for deep reinforcement learning,” in Proc. IJCAI, (Vienna, Austria), 2022. 
*   [38] W.Kim and Y.Sung, “Parameter sharing with network pruning for scalable multi-agent deep reinforcement learning,” arXiv preprint: 2303.00912, 2023. 
*   [39] M.Naghshvar and T.Javidi, “Active sequential hypothesis testing,” Ann. Stat., vol.41, Dec. 2013. 
*   [40] G.Stamatelis and N.Kalouptsidis, “Active hypothesis testing in unknown environments using recurrent neural networks and model free reinforcement learning,” in Proc. EUSIPCO, (Helsinki, Finland), 2023. 
*   [41] C.Zhong et al., “Deep actor-critic reinforcement learning for anomaly detection,” in Proc. IEEE GLOBECOM, (Hawaii, USA), 2019. 
*   [42] D.Kartik et al., “Policy design for active sequential hypothesis testing using deep learning,” in Proc. Allerton, (Monticello, Illinois, USA), 2018. 
*   [43] G.Joseph et al., “Anomaly detection under controlled sensing using actor-critic reinforcement learning,” in Proc. IEEE SPAWC, (Cannes, France), 2020. 
*   [44] C.Zhong et al., “Controlled sensing and anomaly detection via soft actor-critic reinforcement learning,” in Proc. IEEE ICASSP, (Singapore), 2022. 
*   [45] G.Joseph et al., “Temporal detection of anomalies via actor-critic based controlled sensing,” in Proc. IEEE GLOBECOM, (Madrid, Spain), 2021. 
*   [46] C.Zhong et al., “Learning-based robust anomaly detection in the presence of adversarial attacks,” in Proc. IEEE WCNC, (Austin, USA), 2022. 
*   [47] M.-C. Chang et al., “Controlled sensing with corrupted commands,” in Proc. Allerton, (Monticello, Illinois, USA), 2022. 
*   [48] N.Kalouptsidis and G.Stamatelis, “Neural predictor aided policy optimization for adversarial controlled sensing,” Elsevier SigPro, 2025. 
*   [49] D.E. Moriarty and R.Miikkulainen, “Efficient reinforcement learning through symbiotic evolution,” Machine Learning, no.AI94-224, pp.11–32, 1996. 
*   [50] X.Yao, “Evolving artificial neural networks,” Proc. IEEE, vol.87, no.9, pp.1423–1447, 1999. 
*   [51] T.Salimans et al., “Evolution strategies as a scalable alternative to reinforcement learning,” arXiv preprint :1703.03864, 2017. 
*   [52] P.Chrabaszcz et al., “Back to basics: Benchmarking canonical evolution strategies for playing Atari,” in Proc. IJCAI, (Stockholm, Sweden), 2018. 
*   [53] G.Stamatelis et al., “Evolving multi-branch attention convolutional neural networks for online RIS configuration,” IEEE Trans. Cogn. Commun. Netw., pp.1–1, 2025. 
*   [54] F.Gomez et al., “Accelerated neural evolution through cooperatively coevolved synapses,” JMLR, vol.9, no.31, pp.937–965, 2008. 
*   [55] J.H. Holland, Genetic Algorithms. Scientific American, 1992. 
*   [56] Z.Kazan et al., “The test of tests: a framework for differentially private hypothesis testing,” in Proc. ICML, (Honolulu, Hawaii, USA), 2023. 
*   [57] A.Tsiamis et al., “State estimation with secrecy against eavesdroppers,” in Proc. IFAC World Congress, (Toulouse, France), 2017. 
*   [58] M.Mhanna and P.Piantanida, “On secure distributed hypothesis testing,” in Proc. IEEE ISIT, (Hong Kong, China), 2015. 
*   [59] E.Erdemir et al., “Active privacy-utility trade-off against inference in time-series data sharing,” IEEE J. Sel. Areas Inf. Theory, vol.4, pp.159–173, 2023. 
*   [60] E.Erdemir et al., “Privacy-aware time-series data sharing with deep reinforcement learning,” IEEE Trans. Inf. Forensics Secur., vol.16, pp.389–401, 2021. 
*   [61] M.-C. Chang and M.R. Bloch, “Evasive active hypothesis testing,” IEEE J. Sel. Areas Inf. Theory, vol.2, no.2, pp.735–746, 2021. 
*   [62] J.Lundén et al., “Multiagent reinforcement learning based spectrum sensing policies for cognitive radio networks,” IEEE J. Sel. Topics Signal Process, vol.7, no.5, pp.858–868, 2013. 
*   [63] S.Chen et al., “Multi-agent deep reinforcement learning-based cooperative edge caching for ultra-dense next-generation networks,” IEEE Trans. Commun., vol.69, no.4, pp.2441–2456, 2021. 
*   [64] F.Christianos et al., “Scaling multi-agent reinforcement learning with selective parameter sharing,” Proc. ICML, 18–24 Jul 2021. 
*   [65] M.Naghshvar and T.Javidi, “Extrinsic Jensen-Shannon divergence with application in active hypothesis testing,” in Proc. IEEE ISIT, (Cambridge, USA), 2012. 
*   [66] S.D. Bernstein et al., “The complexity of decentralized control of markov decision processes,” Math. Oper. Res., vol.27, no.4, pp.819–840, 2002. 
*   [67] G.Stamatelis et al., “Multi-agent actor-critic with harmonic annealing pruning for dynamic spectrum access systems,” arXiv preprint: 2503.15172, 2025. 
*   [68] S.Hochreiter and J.Schmidhuber, “Long short-term memory,” Neural Computation, vol.9, no.8, pp.1735–1780, 1997. 
*   [69] J.Chung et al., “Empirical evaluation of gated recurrent neural networks on sequence modeling,” arXiv preprint: 1412.3555, 2014. 
*   [70] A.Paszke et al., “Automatic differentiation in pytorch,” in Proc. NeurIPS, (Long Beach, USA), 2017. 
*   [71] D.Kingma and J.Ba, “Adam: A method for stochastic optimization,” Proc. ICLR, Dec. 2014. 
*   [72] Y.Cao et al., “Deep-reinforcement-learning-based self-evolving moving target defense approach against unknown attacks,” IEEE Internet Things J, vol.11, no.20, pp.33027–33039, 2024. 
*   [73] E.Miehling et al., “A POMDP approach to the dynamic defense of large-scale cyber networks,” IEEE Trans. Inf. Forensics Secur., vol.13, no.10, pp.2490–2505, 2018. 

George Stamatelis  was born in Athens, Greece in the summer of 2000. He finished high school in 2018 2018 and in 2022 2022 he got his BSc in computer science from the department of informatics and telecommunications of the National and Kapodistirian University of Athens with highest honours. Since 2023 2023 he continuous his graduate studies in the same department. His research interests are machine learning, multiagent systems, anomaly detection and distributed signal processing.

Angelos-Nikolaos Kanatas studied electrical and computer engineering at the National and Technical University of Athens, and graduated with high honors. His research interests include deep learning, audio and speech processing, natural language processing, and reinforcement learning. He has worked as a research associate with the Institute for Language and Speech Processing at the Athena Research Center.

Ioannis Asprogerakas studied electrical and computer engineering at the National and Technical University of Athens, where he is currently conducting his thesis on diffusion models. His research interests include generative models, computer vision, and multi-modal learning.

George C. Alexandropoulos(Senior member, IEEE) received the Engineering Diploma (Integrated M.S.c), M.A.Sc., and Ph.D. degrees in Computer Engineering and Informatics from the School of Engineering, University of Patras, Greece in 2003, 2005, and 2010, respectively. He has held senior research positions at various Greek universities and research institutes, and he was a Senior Research Engineer and a Principal Researcher at the Mathematical and Algorithmic Sciences Lab, Paris Research Center, Huawei Technologies France, and at the Technology Innovation Institute, Abu Dhabi, United Arab Emirates, respectively. He is currently an Associate Professor with the Department of Informatics and Telecommunications, School of Sciences, National and Kapodistrian University of Athens (NKUA), Greece and with the Department of Electrical and Computer Engineering, University of Illinois Chicago, Chicago, IL, USA. His research interests span the general areas of algorithmic design and performance analysis for wireless networks with emphasis on multi-antenna transceiver hardware architectures, full duplex radios, active and passive Reconfigurable Intelligent Surfaces (RISs), Integrated Sensing And Communications (ISAC), millimeter wave and THz communications, as well as distributed machine learning algorithms.

