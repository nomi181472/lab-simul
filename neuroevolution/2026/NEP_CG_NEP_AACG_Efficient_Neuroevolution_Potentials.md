Title: NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials

URL Source: https://arxiv.org/html/2603.01234

Markdown Content:
Zheyong Fan [brucenju@gmail.com](https://arxiv.org/html/2603.01234v1/mailto:brucenju@gmail.com)College of Physical Science and Technology, Bohai University, Jinzhou, P. R. China Suzhou Laboratory, Suzhou, Jiangsu 215123, P. R. China Zhenhao Zhang Key Laboratory of Material Simulation Methods & Software of Ministry of Education, College of Physics, Jilin University, Changchun 130012, China Ke Xu [kickhsu@gmail.com](https://arxiv.org/html/2603.01234v1/mailto:kickhsu@gmail.com)College of Physical Science and Technology, Bohai University, Jinzhou, P. R. China Xuecheng Shao [shaoxc@jlu.edu.cn](https://arxiv.org/html/2603.01234v1/mailto:shaoxc@jlu.edu.cn)Key Laboratory of Material Simulation Methods & Software of Ministry of Education, College of Physics, Jilin University, Changchun 130012, China Haikuan Dong [donghaikuan@163.com](https://arxiv.org/html/2603.01234v1/mailto:donghaikuan@163.com)College of Physical Science and Technology, Bohai University, Jinzhou, P. R. China

###### Abstract

Machine-learned coarse-grained (CG) models often suffer from noisy training data, limiting their accuracy and transferability. We propose a method to generate low-noise training data based on the potential of mean force by constraining CG beads during atomistic simulations and accumulating time-averaged forces. Implemented within the neuroevolution potential (NEP) framework, our approach achieves training accuracy comparable to atomistic models trained on density functional theory data. For liquid water, the NEP-CG model accurately reproduces densities from 1 bar to 1 GPa, successfully extrapolating beyond the 0.5 GPa training limit, with a virial correction essential for the correct equation of state. For an anisotropic C 60 monolayer, distinguishing crystallographically distinct bead types reduces stress errors by an order of magnitude and captures directional thermal conductivity. We further introduce a multiscale NEP-AACG model integrating all-atom (AA) and CG degrees of freedom, demonstrated for gold nanowire fracture at an experimentally relevant strain rate. Computational speeds for NEP-CG models reach hundreds to thousands of ns/day using a single consumer-grade GPU. This work provides a robust framework for constructing accurate, transferable, and efficient CG models across diverse systems.

I Introduction
--------------

Molecular dynamics (MD) simulations have become indispensable in chemistry, biology, and materials science, providing atomistic insights into complex systems. However, a persistent challenge is the vast range of time and length scales governing molecular phenomena. Many processes occur on microsecond or longer timescales, far exceeding those accessible to conventional all-atom (AA) simulations. Coarse-grained (CG) modeling addresses this by grouping atoms into fewer interaction sites, or “beads”, reducing the effective degrees of freedom and enabling simulations of significantly larger systems and longer timescales Noid ([2013](https://arxiv.org/html/2603.01234#bib.bib1)). The central goal of CG models is to preserve the essential structural and thermodynamic properties of the original atomistic system.

The force-matching method, proposed by Izvekov and Voth Izvekov and Voth ([2005](https://arxiv.org/html/2603.01234#bib.bib2)) and placed on a rigorous statistical mechanics foundation by Noid et al.Noid _et al._ ([2008](https://arxiv.org/html/2603.01234#bib.bib3)), provides a systematic approach for optimizing CG model parameters. Subsequent developments have produced various tools and MD engines for constructing and deploying CG models Peng _et al._ ([2023](https://arxiv.org/html/2603.01234#bib.bib4)); Mirzoev, Nordenskiöld, and Lyubartsev ([2019](https://arxiv.org/html/2603.01234#bib.bib5)); Anderson, Glaser, and Glotzer ([2020](https://arxiv.org/html/2603.01234#bib.bib6)); Zhu _et al._ ([2013](https://arxiv.org/html/2603.01234#bib.bib7)); Xu _et al._ ([2025a](https://arxiv.org/html/2603.01234#bib.bib8)).

When a CG model is derived by formally integrating out degrees of freedom, multi-body interactions naturally emerge in the effective energy function. Including such multi-body terms has been shown to improve model accuracy Molinero and Moore ([2009](https://arxiv.org/html/2603.01234#bib.bib9)); Larini, Lu, and Voth ([2010](https://arxiv.org/html/2603.01234#bib.bib10)); Wang _et al._ ([2021](https://arxiv.org/html/2603.01234#bib.bib11)). The advent of machine-learned potentials (MLPs)Unke _et al._ ([2021](https://arxiv.org/html/2603.01234#bib.bib12)) offers new opportunities, as MLPs excel at capturing many-body effects in complex systems.

Consequently, there has been growing interest in using MLPs to learn CG models from AA MD trajectories. John and Csanyi John and Csányi ([2017](https://arxiv.org/html/2603.01234#bib.bib13)) developed a CG model based on the Gaussian approximation potential framework Bartók _et al._ ([2010](https://arxiv.org/html/2603.01234#bib.bib14)), demonstrating significantly higher accuracy than pair-potential models while remaining faster than AA simulations for solvent-free biomolecular systems. Zhang et al.Zhang _et al._ ([2018a](https://arxiv.org/html/2603.01234#bib.bib15)) constructed a DP-CG water model using the force-matching method and the deep potential approach Zhang _et al._ ([2018b](https://arxiv.org/html/2603.01234#bib.bib16)), accurately reproducing oxygen structural characteristics. Wang et al.Wang _et al._ ([2019](https://arxiv.org/html/2603.01234#bib.bib17)) introduced CGnets, showing that MLPs can capture explicit-solvent free energy surfaces with only a few CG beads, unlike classical methods. MLP-based CG methods have since been extended to proteins Majewski _et al._ ([2023](https://arxiv.org/html/2603.01234#bib.bib18)); Charron _et al._ ([2025](https://arxiv.org/html/2603.01234#bib.bib19)) and other complex systems. Beyond local invariant-feature-based MLPs, approaches using graph neural networks Husic _et al._ ([2020](https://arxiv.org/html/2603.01234#bib.bib20)); Ruza _et al._ ([2020](https://arxiv.org/html/2603.01234#bib.bib21)); Thaler, Stupp, and Zavadlav ([2022](https://arxiv.org/html/2603.01234#bib.bib22)) and equivariant features Loose _et al._ ([2023](https://arxiv.org/html/2603.01234#bib.bib23)) have also been explored for constructing CG models.

Despite these advances, current MLP-based CG methods often rely on training data with substantial noise. Reported root-mean-square error (RMSE) values for typical water models Loose _et al._ ([2023](https://arxiv.org/html/2603.01234#bib.bib23)) range from 3.49 to 11.8 kcal/mol/Å(0.151–0.512 eV/Å), significantly higher than typical AA MLP errors for water (e.g., ∼\sim 0.05 eV/Å) Zhang _et al._ ([2018b](https://arxiv.org/html/2603.01234#bib.bib16)). This noise causes several problems: (1) large training errors obscure convergence assessment; (2) noise complicates hyperparameter selection; (3) many training structures are required even for a single state point; and (4) resulting models are often applicable only to a single density and lack transferability across different pressures.

To address these challenges, we propose a method for generating low-noise training data based on the potential of mean force definition. After thermal equilibration, we constrain the bead degrees of freedom during AA MD simulations and accumulate time-averaged forces. These averages correspond to the true mean forces on the beads and are inherently smooth, making them easy for an MLP to learn.

Our implementation uses the neuroevolution potential (NEP) framework Fan _et al._ ([2021](https://arxiv.org/html/2603.01234#bib.bib24)) within the GPUMD package Xu _et al._ ([2025b](https://arxiv.org/html/2603.01234#bib.bib25)), one of the most computationally efficient MLP architectures available. NEP has found widespread application in AA simulations Ying _et al._ ([2025](https://arxiv.org/html/2603.01234#bib.bib26)) and recently in CG contexts Argun and Statt ([2025](https://arxiv.org/html/2603.01234#bib.bib27)); Li, Wang, and Zheng ([2025](https://arxiv.org/html/2603.01234#bib.bib28)). By reducing training data noise, we achieve accuracy comparable to AA models trained on density-functional theory (DFT) data, resolving the issues above: (1) high accuracy enables clear convergence assessment; (2) optimal hyperparameters can be selected based on training accuracy; (3) only one or a few structures suffice for a robust model at a single state point; and (4) accurate virial data enable description of various strained states, conferring transferability across strain conditions. Notably, we find that CG models require larger cutoff radii than their AA counterparts but do not need many trainable parameters.

Beyond pure CG simulations, we introduce a multiscale NEP-AACG model that combines AA and CG degrees of freedom within a single framework. This integration is natural for NEP, which handles many-component systems efficiently Song _et al._ ([2024](https://arxiv.org/html/2603.01234#bib.bib29)); Liang _et al._ ([2025](https://arxiv.org/html/2603.01234#bib.bib30)).

II Methods
----------

### II.1 Neuroevolution potentials

Our CG and AACG models are based on the NEP approach Fan _et al._ ([2021](https://arxiv.org/html/2603.01234#bib.bib24)), specifically its fourth-generation version (NEP4)Song _et al._ ([2024](https://arxiv.org/html/2603.01234#bib.bib29)) implemented in the GPUMD package Xu _et al._ ([2025b](https://arxiv.org/html/2603.01234#bib.bib25)). NEP4 is particularly suitable for multi-component systems.

NEP uses a feedforward neural network to represent the site energy U i U_{i} of atom i i as a function of a descriptor vector 𝐪\mathbf{q} with N des N_{\mathrm{des}} components:

U i​(𝐪)=U i​({q ν i}ν=1 N des).U_{i}(\mathbf{q})=U_{i}\left(\{q^{i}_{\nu}\}_{\nu=1}^{N_{\mathrm{des}}}\right).(1)

The network has a single hidden layer with N neu N_{\mathrm{neu}} neurons and tanh\tanh activation:

U i=∑μ=1 N neu w μ(1)​tanh⁡(∑ν=1 N des w μ​ν(0)​q ν i−b μ(0))−b(1),U_{i}=\sum_{\mu=1}^{N_{\mathrm{neu}}}w^{(1)}_{\mu}\tanh\left(\sum_{\nu=1}^{N_{\mathrm{des}}}w^{(0)}_{\mu\nu}q^{i}_{\nu}-b^{(0)}_{\mu}\right)-b^{(1)},(2)

where 𝐰(0)\mathbf{w}^{(0)} and 𝐰(1)\mathbf{w}^{(1)} are connection weights, and 𝐛(0)\mathbf{b}^{(0)} and b(1)b^{(1)} are biases.

The descriptor for atom i i comprises radial and angular components constructed from the local atomic environment within a cutoff distance. The radial components are:

q n i=∑j≠i g n​(r i​j),0≤n≤n max R,q^{i}_{n}=\sum_{j\neq i}g_{n}(r_{ij}),\quad 0\leq n\leq n_{\mathrm{max}}^{\mathrm{R}},(3)

giving n max R+1 n_{\mathrm{max}}^{\mathrm{R}}+1 radial components. Angular components include up to five-body terms. For three-body terms:

q n​l i=∑m=−l l(−1)m​A n​l​m i​A n​l​(−m)i,q^{i}_{nl}=\sum_{m=-l}^{l}(-1)^{m}A^{i}_{nlm}A^{i}_{nl(-m)},(4)

with

A n​l​m i=∑j≠i g n​(r i​j)​Y l​m​(θ i​j,ϕ i​j),A^{i}_{nlm}=\sum_{j\neq i}g_{n}(r_{ij})Y_{lm}(\theta_{ij},\phi_{ij}),(5)

where Y l​m Y_{lm} are spherical harmonics, 0≤n≤n max A 0\leq n\leq n_{\mathrm{max}}^{\mathrm{A}}, and 1≤l≤l max 3​b 1\leq l\leq l_{\mathrm{max}}^{\mathrm{3b}}. Four-body and five-body terms follow Ref.Fan _et al._ ([2022](https://arxiv.org/html/2603.01234#bib.bib31)).

The radial functions g n​(r i​j)g_{n}(r_{ij}) appear in both radial and angular descriptors, expanded as:

g n​(r i​j)=∑k=0 N bas R c n​k i​j​f k​(r i​j),g_{n}(r_{ij})=\sum_{k=0}^{N_{\mathrm{bas}}^{\mathrm{R}}}c^{ij}_{nk}f_{k}(r_{ij}),(6)

where

f k​(r i​j)=1 2​[T k​(2​(r i​j/r c R−1)2−1)+1]​f c​(r i​j),f_{k}(r_{ij})=\frac{1}{2}\left[T_{k}\left(2\left(r_{ij}/r_{\mathrm{c}}^{\mathrm{R}}-1\right)^{2}-1\right)+1\right]f_{\mathrm{c}}(r_{ij}),(7)

with T k T_{k} Chebyshev polynomials and f c​(r i​j)f_{\mathrm{c}}(r_{ij}) a cosine cutoff:

f c​(r i​j)={1 2​[1+cos⁡(π​r i​j r c R)],r i​j≤r c R;0,r i​j>r c R.f_{\mathrm{c}}(r_{ij})=\begin{cases}\frac{1}{2}\left[1+\cos\left(\pi\frac{r_{ij}}{r_{\mathrm{c}}^{\mathrm{R}}}\right)\right],&r_{ij}\leq r_{\mathrm{c}}^{\mathrm{R}};\\ 0,&r_{ij}>r_{\mathrm{c}}^{\mathrm{R}}.\end{cases}(8)

For angular descriptors, N bas R N_{\mathrm{bas}}^{\mathrm{R}} and r c R r_{\mathrm{c}}^{\mathrm{R}} are replaced by N bas A N_{\mathrm{bas}}^{\mathrm{A}} and r c A r_{\mathrm{c}}^{\mathrm{A}}. Importantly, atom-type information is encoded directly in the expansion coefficients c n​k i​j c_{nk}^{ij}, enabling efficient handling of multi-component systems without separate descriptor branches.

Parameters are optimized using the separable natural evolution strategy (SNES)Schaul, Glasmachers, and Schmidhuber ([2011](https://arxiv.org/html/2603.01234#bib.bib32)). The loss function combines RMSE for energies, forces, and virials with ℒ 1\mathcal{L}_{1} and ℒ 2\mathcal{L}_{2} regularization, weighted by tunable hyperparameters λ e\lambda_{\rm e}, λ f\lambda_{\rm f}, λ v\lambda_{\rm v}, λ 1\lambda_{1}, and λ 2\lambda_{2}. Energies and virials are in eV/atom, forces in eV/Å.

### II.2 Force-matching for coarse-graining

The force-matching method, originally developed for extracting classical effective forces from ab initio simulations Ercolessi and Adams ([1994](https://arxiv.org/html/2603.01234#bib.bib33)), was adapted for CG models Izvekov and Voth ([2005](https://arxiv.org/html/2603.01234#bib.bib2)) and given a rigorous statistical foundation by Noid et al.Noid _et al._ ([2008](https://arxiv.org/html/2603.01234#bib.bib3)). The idea is to derive a CG potential such that forces on CG beads approximate the ensemble-averaged forces on corresponding atomistic groups.

Consider an atomistic system with N AA N_{\rm AA} atoms with coordinates 𝐫\mathbf{r} and potential energy U​(𝐫)U(\mathbf{r}). The system is mapped onto N CG N_{\mathrm{CG}}CG beads defined by a linear mapping 𝐑=𝐂⋅𝐫\mathbf{R}=\mathbf{C}\cdot\mathbf{r}. Each bead I I represents a group of atoms, with position typically given by the center of mass:

𝐑 I=∑i∈I C I​i​𝐫 i,C I​i=m i∑i′∈I m i′.\mathbf{R}_{I}=\sum_{i\in I}C_{Ii}\mathbf{r}_{i},\quad C_{Ii}=\frac{m_{i}}{\sum_{i^{\prime}\in I}m_{i^{\prime}}}.(9)

In the canonical ensemble, the equilibrium distribution of CG coordinates is:

P​(𝐑)=⟨δ​(𝐑−𝐂⋅𝐫)⟩∝∫𝑑 𝐫​δ​(𝐑−𝐂⋅𝐫)​e−β​U​(𝐫).P(\mathbf{R})=\left\langle\delta(\mathbf{R}-\mathbf{C}\cdot\mathbf{r})\right\rangle\propto\int d\mathbf{r}\,\delta(\mathbf{R}-\mathbf{C}\cdot\mathbf{r})e^{-\beta U(\mathbf{r})}.(10)

This distribution can be generated by a CG model with potential U CG​(𝐑)U_{\mathrm{CG}}(\mathbf{R}) satisfying P​(𝐑)∝e−β​U CG​(𝐑)P(\mathbf{R})\propto e^{-\beta U_{\mathrm{CG}}(\mathbf{R})}, identifying U CG​(𝐑)U_{\mathrm{CG}}(\mathbf{R}) as the potential of mean force (PMF):

U CG​(𝐑)=−k B​T​ln⁡[∫𝑑 𝐫​δ​(𝐑−𝐂⋅𝐫)​e−β​U​(𝐫)].U_{\mathrm{CG}}(\mathbf{R})=-k_{\mathrm{B}}T\ln\left[\int d\mathbf{r}\,\delta(\mathbf{R}-\mathbf{C}\cdot\mathbf{r})e^{-\beta U(\mathbf{r})}\right].(11)

The PMF depends on the thermodynamic state point, reflecting integrated-out degrees of freedom.

From the PMF, the mean force on bead I I at CG configuration 𝐑\mathbf{R} is:

𝐅 I CG​(𝐑)=−∇𝐑 I U CG​(𝐑)=⟨∑i∈I 𝐟 i⟩𝐑,\mathbf{F}_{I}^{\mathrm{CG}}(\mathbf{R})=-\nabla_{\mathbf{R}_{I}}U_{\mathrm{CG}}(\mathbf{R})=\left\langle\sum_{i\in I}\mathbf{f}_{i}\right\rangle_{\mathbf{R}},(12)

where 𝐟 i\mathbf{f}_{i} are atomic forces and ⟨⋯⟩𝐑\langle\cdots\rangle_{\mathbf{R}} denotes an ensemble average constrained by 𝐂⋅𝐫=𝐑\mathbf{C}\cdot\mathbf{r}=\mathbf{R}. Thus, target forces for a CG model are constrained ensemble averages of total forces on each bead’s atoms.

Traditional force-matching minimizes the following loss function Izvekov and Voth ([2005](https://arxiv.org/html/2603.01234#bib.bib2)):

χ 2=⟨1 3​N CG​∑I=1 N CG‖∑i∈I 𝐟 i​(𝐫)−𝐅 I CG​(𝐂⋅𝐫)‖2⟩,\chi^{2}=\left\langle\frac{1}{3N_{\mathrm{CG}}}\sum_{I=1}^{N_{\mathrm{CG}}}\left\|\sum_{i\in I}\mathbf{f}_{i}(\mathbf{r})-\mathbf{F}_{I}^{\mathrm{CG}}(\mathbf{C}\cdot\mathbf{r})\right\|^{2}\right\rangle,(13)

where 𝐅 I CG​(𝐑)\mathbf{F}_{I}^{\mathrm{CG}}(\mathbf{R}) are predicted CG forces. A crucial distinction exists between Eq.([12](https://arxiv.org/html/2603.01234#S2.E12 "In II.2 Force-matching for coarse-graining ‣ II Methods ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")) and Eq.([13](https://arxiv.org/html/2603.01234#S2.E13 "In II.2 Force-matching for coarse-graining ‣ II Methods ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")): the former defines the ensemble-averaged forces, while the latter optimizes the CG model against instantaneous forces. This approximation in the traditional force-matching method works well for models with limited flexibility. However, flexible MLPs could be prone to overfitting the substantial noise in instantaneous forces, especially with limited data, a challenge that is only partially mitigated by standard regularization techniques.

### II.3 NEP-CG and NEP-AACG methods

#### II.3.1 NEP-CG method

Building on atomistic NEP and force-matching, we develop NEP-CG and NEP-AACG approaches. The key innovation is generating low-noise training data that directly correspond to the PMF via constrained MD simulations, rather than fitting to instantaneous forces.

For each thermodynamic state, we first quilibrate the atomistic system at target temperature and pressure/density using atomistic NEP. Then we apply constraints to fix the CG bead positions and continue the simulation in the NVE ensemble, accumulating instantaneous forces (Eq.([12](https://arxiv.org/html/2603.01234#S2.E12 "In II.2 Force-matching for coarse-graining ‣ II Methods ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials"))) and virial on each bead I I. Time-averaged quantities converge to true mean values as ∼1/T\sim 1/\sqrt{T} (with T T production time). In practice, we found that a production time of 1–10 ns is sufficient to achieve a level of noise notably lower than the attainable accuracy for the NEP approach.

A notable feature in our approach is that we also use virial stresses as training targets, which is crucial for obtaining CG models transferrable to different pressure states. For a CG bead I I, the instantaneous virial is the sum of constituent atom virials and the ensemble average is:

𝐖 I=⟨∑i∈I 𝐖 i⟩𝐑,\mathbf{W}_{I}=\left\langle\sum_{i\in I}\mathbf{W}_{i}\right\rangle_{\mathbf{R}},(14)

with the atomistic virial contribution from atom i i given by Fan _et al._ ([2021](https://arxiv.org/html/2603.01234#bib.bib24))

𝐖 i=∑j≠i 𝐫 i​j⊗∂U j∂𝐫 j​i.\mathbf{W}_{i}=\sum_{j\neq i}\mathbf{r}_{ij}\otimes\frac{\partial U_{j}}{\partial\mathbf{r}_{ji}}.(15)

Here 𝐫 i​j≡𝐫 j−𝐫 i\mathbf{r}_{ij}\equiv\mathbf{r}_{j}-\mathbf{r}_{i}. The total virial in the CG system is

𝐖=∑I 𝐖 I.\mathbf{W}=\sum_{I}\mathbf{W}_{I}.(16)

We found that training directly with this virial definition systematically underpredicts pressure and overpredicts density, because coarse-graining eliminates kinetic (ideal gas) contributions from integrated-out atoms. To remedy this, we introduce a virial correction:

𝐖→𝐖+(N AA−N CG)​k B​T​𝐈,\mathbf{W}\to\mathbf{W}+(N_{\mathrm{AA}}-N_{\mathrm{CG}})k_{\mathrm{B}}T\mathbf{I},(17)

where 𝐈\mathbf{I} is the identity tensor. This correction compensates for lost degrees of freedom, ensuring correct pressure response.

For pure CG simulations, NEP-CG models are trained to reproduce ensemble-averaged quantities from constrained simulations. The NEP architecture (Section[II.1](https://arxiv.org/html/2603.01234#S2.SS1 "II.1 Neuroevolution potentials ‣ II Methods ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")) is unchanged, but site energies U i U_{i} represent free energy contributions of CG beads. Training data include bead forces (Eq.[12](https://arxiv.org/html/2603.01234#S2.E12 "In II.2 Force-matching for coarse-graining ‣ II Methods ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")) and virial tensor. We will show that minimizing the NEP loss function with these targets yields accuracy comparable to atomistic NEP models trained on DFT data. The method is also data efficient. For one thermodynamic state, only one or a few configurations suffice for a robust NEP-CG model, unlike conventional approaches requiring thousands of snapshots. Consequently, we still have a relative small dataset for training a CG model applicable to a wide range of pressures states.

#### II.3.2 NEP-AACG method

We further introduce NEP-AACG, which integrates atomistic and CG degrees of freedom within a single model for multiscale simulations. The NEP architecture handles mixed-resolution systems by treating atomistic sites and CG beads as distinct species. Descriptors are constructed from local environments that may include both types, with the same radial and angular framework applying uniformly. Different particle pairs may require different cutoff radii to optimally capture local environments.

Training data are generated by extending constrained dynamics: for each configuration, all sites (atomistic and CG) are constrained, and ensemble-averaged quantities on every site are accumulated. The NEP-AACG model learns a consistent free energy surface spanning both resolutions by predicting all quantities simultaneously.

Coupling between atomistic and CG regions emerges naturally from training data, without ad hoc schemes. The unified NEP framework enables smooth transitions between regions and supports systems with spatial resolution variation. NEP-AACG thus provides a unified, data-driven approach to multiscale modeling.

III Results and discussion
--------------------------

We demonstrate the capabilities of the NEP-CG and NEP-AACG methods through three representative examples. First, we use liquid water, one of the most extensively studied systems in both traditional CG and MLP-CG approaches, to illustrate the key features of our NEP-CG method and introduce a virial correction scheme essential for accurate density prediction.

Second, we consider a solid-state system: a monolayer of covalently bonded C 60 molecules in the quasi-hexagonal phase. Here we examine the effects of coarse-graining on vibrational properties and demonstrate how the NEP-CG model captures the essential dynamics while significantly reducing computational cost.

Third, we showcase the NEP-AACG method through a study of gold metal. We construct a mixed-resolution model and apply it to investigate the deformation of gold nanowires under tension, where the central region is modeled atomistically while surrounding regions are coarse-grained to extend accessible length and time scales.

### III.1 NEP-CG model for liquid water

To construct NEP-CG models for liquid water, we select a NEP-AA model from our previous work Xu _et al._ ([2025c](https://arxiv.org/html/2603.01234#bib.bib34)). This atomistic model was trained against CCSD(T)-level MB-pol reference data and accurately predicts structural, thermodynamic, and transport properties across a wide range of conditions.

#### III.1.1 Training data generation

We first perform atomistic MD simulations using the NEP-AA model in the NPT ensemble to achieve thermal equilibrium at 300 K and pressures from 1 bar to 0.5 GPa. Specifically, we select discrete pressure points: 1 bar, 10 bar, 0.1 GPa, 0.2 GPa, and 0.5 GPa. For each pressure, we conduct a 1 ns NPT simulation for equilibration, then switch off the thermostat and barostat while applying constraints to the CG beads. Each CG bead corresponds to the center of mass of the three atoms in a water molecule. We perform a 10 ns NVE simulation to accumulate force and virial data. The atomistic system contains 1536 atoms, mapping to 512 CG beads. The timestep for integration is 0.5 fs. The complete training dataset comprises just 5 structures (2560 water beads in total), confirming the high data efficiency of our approach.

#### III.1.2 Model hyperparameters

The potential energy surface of a CG model is expected to be smoother than its atomistic counterpart due to integrated-out degrees of freedom. Consequently, the CG model requires substantially fewer trainable parameters. In the NEP approach, the number of trainable parameters is primarily controlled by the number of hidden-layer neurons N neu N_{\rm neu}. For the NEP-AA model Xu _et al._ ([2025c](https://arxiv.org/html/2603.01234#bib.bib34)), N neu=60 N_{\rm neu}=60; through extensive testing, we found N neu=10 N_{\rm neu}=10 sufficient for the NEP-CG model, confirming the expected smoothness.

Cutoff radii for radial and angular descriptors influence both accuracy and performance. Since the water molecule center of mass is comparable in size to an oxygen atom, we adopt the same cutoffs as the NEP-AA model: radial cutoff 6 Å, angular cutoff 4 Å.

#### III.1.3 Model accuracy

![Image 1: Refer to caption](https://arxiv.org/html/2603.01234v1/fig1-water-parity.png)

Figure 1: Parity plots comparing forces and stresses predicted by NEP-CG models against NEP-AA reference data for liquid water at 300 K. (a,b) Results from a NEP-CG model trained against instantaneous forces and stresses sampled at 1 bar, showing systematic deviation from the y=x y=x line and significant scatter. (c,d) Results from a NEP-CG model trained against ensemble-averaged forces and stresses using our constrained dynamics approach, incorporating data sampled across target pressures from 1 bar to 0.5 GPa. The ensemble-based model demonstrates excellent agreement and substantially reduced errors across the entire pressure range. Different colors correspond to different force and virial components.

Figure[1](https://arxiv.org/html/2603.01234#S3.F1 "Figure 1 ‣ III.1.3 Model accuracy ‣ III.1 NEP-CG model for liquid water ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials") shows parity plots for forces and stresses. For comparison, we also trained a NEP-CG model using the conventional instantaneous-force method on a NEP-AA trajectory at 300 K and 1 bar. The instantaneous-force approach yields significant deviation from the y=x y=x line with an under-predicted slope (Fig.[1](https://arxiv.org/html/2603.01234#S3.F1 "Figure 1 ‣ III.1.3 Model accuracy ‣ III.1 NEP-CG model for liquid water ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")a,b), because instantaneous forces are not the correct targets and the true mean forces are buried in noise. This issue is more pronounced for stresses. The resulting RMSEs are 0.15 eV/Å for forces and 0.14 GPa for stresses. The force RMSE is consistent with values reported by Loose et al.Loose _et al._ ([2023](https://arxiv.org/html/2603.01234#bib.bib23)) for different MLPs trained against empirical water models.

In contrast, our ensemble-force training scheme produces parity plots that closely follow the y=x y=x line (Fig.[1](https://arxiv.org/html/2603.01234#S3.F1 "Figure 1 ‣ III.1.3 Model accuracy ‣ III.1 NEP-CG model for liquid water ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")c,d), confirming that ensemble-averaged forces are the correct targets. The RMSEs for forces and stresses are reduced to 0.080 eV/Å and 0.0084 GPa, respectively. Previous MLP-CG studies typically consider only a single pressure state point, whereas our model accurately predicts stresses across 1 bar to 0.5 GPa, demonstrating transferability across pressure conditions.

#### III.1.4 Structural properties and equation of state

![Image 2: Refer to caption](https://arxiv.org/html/2603.01234v1/fig2-water-rdf-density.png)

Figure 2: Structural properties and equation of state of liquid water predicted by the NEP-CG model compared to NEP-AA reference data at 300 K. (a) Radial distribution functions g​(r)g(r) for CG beads at 1 bar and 0.5 GPa, showing excellent structural agreement. (b) Density as a function of pressure from 1 bar to 1 GPa. Results are shown for the NEP-AA reference, the NEP-CG model with virial correction (Eq.[17](https://arxiv.org/html/2603.01234#S2.E17 "In II.3.1 NEP-CG method ‣ II.3 NEP-CG and NEP-AACG methods ‣ II Methods ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")), and the NEP-CG model without virial correction. The virial-corrected model accurately reproduces NEP-AA densities across the entire pressure range.

A canonical validation of a CG model is its ability to reproduce the radial distribution function (RDF) of the beads from the underlying AA model. Figure[2](https://arxiv.org/html/2603.01234#S3.F2 "Figure 2 ‣ III.1.4 Structural properties and equation of state ‣ III.1 NEP-CG model for liquid water ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")a compares RDF s calculated using the NEP-CG and NEP-AA models at 1 bar and 0.5 GPa. The excellent agreement demonstrates transferability across different pressure conditions.

Figure[2](https://arxiv.org/html/2603.01234#S3.F2 "Figure 2 ‣ III.1.4 Structural properties and equation of state ‣ III.1 NEP-CG model for liquid water ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")b compares densities from 1 bar to 1 GPa. Training data only extended to 0.5 GPa; the inclusion of 1 GPa tests extrapolation capability. The NEP-CG model shows good agreement with the NEP-AA model across all pressures, with density increasing from approximately 1 g/cm 3 at 1 bar to 1.2 g/cm 3 at 1 GPa. Successful extrapolation beyond the training range underscores the robustness of our ensemble-based approach.

Figure[2](https://arxiv.org/html/2603.01234#S3.F2 "Figure 2 ‣ III.1.4 Structural properties and equation of state ‣ III.1 NEP-CG model for liquid water ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")b also shows results from a NEP-CG model trained without the virial correction (Eq.[17](https://arxiv.org/html/2603.01234#S2.E17 "In II.3.1 NEP-CG method ‣ II.3 NEP-CG and NEP-AACG methods ‣ II Methods ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")). Without this correction, density is significantly overestimated across the entire pressure range, confirming that the virial correction is essential for accurately reproducing the equation of state.

### III.2 NEP-CG model for fullerene monolayer network

Our second example applies the NEP-CG framework to a monolayer of covalently bonded C 60 molecules in the quasi-hexagonal phase (QHP) Hou _et al._ ([2022](https://arxiv.org/html/2603.01234#bib.bib35)), which exhibits anisotropic mechanical and thermal properties due to directional bonding. An atomistic NEP-AA model for this system was recently developed to study its thermal transport Dong _et al._ ([2023](https://arxiv.org/html/2603.01234#bib.bib36)). Here we construct a CG model where each C 60 molecule is represented by a single bead.

A key feature of the QHP-C 60 monolayer is anisotropic bonding: each C 60 molecule is covalently linked to six neighbors, with bond types differing by direction: [2+2][2+2] cycloaddition bonds along the [010][010] direction (y y-axis), and C-C single bonds along the [110][110] and [1​1¯​0][1\bar{1}0] directions. To capture this anisotropy, we introduce two distinct bead types corresponding to the two C 60 molecules in the primitive cell. Using a single bead type would not distinguish the x x and y y directions and would fail to capture the anisotropy. Indeed, correctly encoding species has been demonstrated to be crucial, as neglecting it may introduce unphysical symmetries Görlich and Zavadlav ([2026](https://arxiv.org/html/2603.01234#bib.bib37)).

#### III.2.1 Training data and hyperparameters

Training data were generated from NEP-AA simulations of a 7200-atom system, mapping onto 120 CG beads. For each training structure, we performed NPT simulations at 300 K under target in-plane pressures ranging from −1-1 to 1 1 GPa in both directions, using a 1 fs timestep. After 1 ns equilibration, the C 60 center-of-mass positions were constrained, and ensemble-averaged forces and virials were accumulated over a 10 ns production run. In total, 23 training structures were generated, capturing the anisotropic response under various in-plane strain states.

Due to the large size of each C 60 molecule and the extended range of intermolecular interactions, a cutoff radius of 25 Å is optimal for both radial and angular descriptors. This relatively large cutoff is necessary to adequately capture interactions between the large C 60 molecules and ensure accurate representation of the anisotropic bonding environment. Because coarse-graining integrates out intramolecular degrees of freedom, the resulting potential energy surface is substantially smoother, allowing a small neural network with N neu=5 N_{\mathrm{neu}}=5 to achieve high training accuracy.

#### III.2.2 Model accuracy and importance of bead-type distinction

![Image 3: Refer to caption](https://arxiv.org/html/2603.01234v1/fig3-parity-c60.png)

Figure 3: Parity plots comparing forces and stresses predicted by NEP-CG models against NEP-AA reference data for the QHP-C 60 monolayer at 300 K under various in-plane strain states. (a,b) Results from a one-type NEP-CG model treating all C 60 molecules as identical beads, exhibiting systematic errors. (c,d) Results from a two-type NEP-CG model distinguishing the two crystallographically distinct C 60 molecules, accounting for directional dependence of covalent linkages. The two-type model demonstrates substantially improved agreement, particularly for stress components.

Figure[3](https://arxiv.org/html/2603.01234#S3.F3 "Figure 3 ‣ III.2.2 Model accuracy and importance of bead-type distinction ‣ III.2 NEP-CG model for fullerene monolayer network ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials") presents parity plots comparing forces and stresses predicted by two NEP-CG models. The one-type model, treating all C 60 molecules as identical beads, exhibits systematic errors (Fig.[3](https://arxiv.org/html/2603.01234#S3.F3 "Figure 3 ‣ III.2.2 Model accuracy and importance of bead-type distinction ‣ III.2 NEP-CG model for fullerene monolayer network ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")a,b), with RMSE values of 0.11 eV/Å for forces and 0.083 GPa for stresses. These deviations indicate failure to capture the directional dependence of covalent linkages.

In contrast, the two-type model, distinguishing crystallographically distinct molecules, demonstrates substantially improved agreement (Fig.[3](https://arxiv.org/html/2603.01234#S3.F3 "Figure 3 ‣ III.2.2 Model accuracy and importance of bead-type distinction ‣ III.2 NEP-CG model for fullerene monolayer network ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")c,d). Force RMSE decreases to 0.090 eV/Å, while stress RMSE drops by more than an order of magnitude to 0.0025 GPa. These accuracy gains are made observable through our low-noise ensemble-average training scheme.

The superior performance of the two-type model underscores a key principle for coarse-graining anisotropic systems: when atomistic interactions exhibit strong directional dependence that cannot be captured by spherically symmetric beads alone, introducing distinct bead types based on crystallographic equivalence provides a natural way to encode this anisotropy. However, this strategy has limitations, necessitating genuinely anisotropic CG models in more general cases Nguyen and Huang ([2022](https://arxiv.org/html/2603.01234#bib.bib38)); Wilson and Huang ([2023](https://arxiv.org/html/2603.01234#bib.bib39)).

#### III.2.3 Thermal transport

![Image 4: Refer to caption](https://arxiv.org/html/2603.01234v1/x1.png)

Figure 4: Lattice thermal conductivity of the QHP-C 60 monolayer as a function of production time in HNEMD simulations. (a) Thermal conductivity along the x x-direction (average of [110][110] and [1​1¯​0][1\bar{1}0] directions). (b) Thermal conductivity along the y y-direction ([010][010] direction). Lighter lines show five independent simulations, darker lines their average, and shaded areas the standard error bounds. CG results are scaled by a factor of 60 to account for reduced degrees of freedom.

The underlying NEP-AA model was originally developed to study thermal transport in the QHP-C 60 monolayer Dong _et al._ ([2023](https://arxiv.org/html/2603.01234#bib.bib36)). We further validate the two-type NEP-CG model by examining its ability to reproduce thermal conductivity using the homogeneous non-equilibrium molecular dynamics (HNEMD) method Fan _et al._ ([2019](https://arxiv.org/html/2603.01234#bib.bib40)). The effective thickness of the monolayer is chosen as 8.785 Å to be consistent with the previous NEP-AA calculations Dong _et al._ ([2023](https://arxiv.org/html/2603.01234#bib.bib36)).

The NEP-CG model successfully reproduces the anisotropic heat conduction characteristic of the QHP-C 60 monolayer, with thermal conductivity κ\kappa larger in the y y-direction than in the x x-direction. For quantitative comparison, we note that the CG model has 60 times fewer degrees of freedom than the atomistic system (one bead per 60 atoms), hence 60 times smaller volumetric heat capacity. Multiplying the raw NEP-CG thermal conductivity values by this factor yields results of the same order as the NEP-AA reference: scaled NEP-CG values are κ x=64±12\kappa_{x}=64\pm 12 W/mK and κ y=95±9\kappa_{y}=95\pm 9 W/mK, compared to NEP-AA results of κ x=102±3\kappa_{x}=102\pm 3 W/mK and κ y=137±7\kappa_{y}=137\pm 7 W/mK. The CG model thus captures both the qualitative anisotropy and approximate magnitude of thermal conductivity despite the significant reduction in degrees of freedom.

### III.3 NEP-AACG model for gold

![Image 5: Refer to caption](https://arxiv.org/html/2603.01234v1/fig5-gold_parity.png)

Figure 5: Parity plots for (a) forces and (b) stresses predicted by the unified NEP-AACG model compared to reference data. Squares represent the pure AA dataset (DFT references Song _et al._ ([2024](https://arxiv.org/html/2603.01234#bib.bib29))). Circles represent both pure CG and mixed AACG datasets (references from constrained NEP-AA simulations). All predictions are from the same NEP-AACG model, demonstrating its ability to simultaneously describe atomistic, coarse-grained, and mixed-resolution configurations.

Our third case study demonstrates the NEP-AACG framework for gold systems. The underlying NEP-AA model is taken from the UENP-v1 potential Song _et al._ ([2024](https://arxiv.org/html/2603.01234#bib.bib29)), developed for 16 metals and their alloys, which outperforms traditional embedded-atom method potentials across a wide range of physical properties. We focus on 300 K while considering a comprehensive set of strain conditions.

#### III.3.1 Training data generation

To generate training data, we perform NVT simulations using the NEP-AA model at 300 K under various strain states: isotropic strains −10%-10\% to 5%5\%, biaxial strains −10%-10\% to 5%5\%, uniaxial strains −10%-10\% to 10%10\%, and shear strains −5%-5\% to 5%5\%. Simulations use 2048 gold atoms with a 5 fs timestep: 1 ns NVT equilibration followed by 10 ns constrained simulation to accumulate reference data.

For each strained configuration, we design two mapping schemes. In pure CG mapping, each CG bead represents four gold atoms (one FCC unit cell), reducing 2048 atoms to 512 beads. For mixed-resolution AACG mapping, half the atoms are coarse-grained using the same mapping, while the remaining atoms retain atomistic resolution, enabling multiscale simulations.

#### III.3.2 Model hyperparameters

We train a NEP-AACG model by combining three datasets: the original Au dataset from UENP-v1, the pure CG dataset, and the AACG dataset. Within the NEP framework, we define two particle types: atomistic Au and CG beads, which have different effective sizes. Based on systematic testing, using different cutoff radii for different particle types proves beneficial. For the original UENP-v1 model, radial and angular cutoffs are 6 Å and 5 Å, respectively. For CG beads, larger cutoffs of 8 Å and 7 Å are optimal, reflecting their larger effective size. For mixed atom-bead interactions, we define cutoffs as arithmetic means of pure-type cutoffs: 7 Å (radial) and 6 Å (angular). With training data encompassing both pure AA and mixed-resolution configurations, we select an intermediate hidden-layer size N neu=30 N_{\rm neu}=30, balancing model capacity and computational efficiency.

#### III.3.3 Model accuracy

Figure[5](https://arxiv.org/html/2603.01234#S3.F5 "Figure 5 ‣ III.3 NEP-AACG model for gold ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials") presents parity plots for forces and stresses predicted by the unified NEP-AACG model. The model demonstrates excellent accuracy across all three datasets. For the pure AA dataset (DFT references Song _et al._ ([2024](https://arxiv.org/html/2603.01234#bib.bib29))), the NEP-AACG model achieves RMSEs of 0.10 eV/Å for forces and 0.50 GPa for stresses, comparable to the original UENP-v1 model Song _et al._ ([2024](https://arxiv.org/html/2603.01234#bib.bib29)), confirming that adding CG and mixed-resolution data does not compromise atomistic accuracy.

For pure CG and mixed AACG configurations (references from constrained NEP-AA simulations), force RMSE is 0.077 eV/Å and stress RMSE is 0.086 GPa. The substantially lower stress error reflects both the smoother CG potential energy surface and the narrower range of forces and stresses in CG-related datasets.

Crucially, all predictions are generated by a single unified NEP-AACG model with two particle types and interaction-specific cutoffs. Simultaneous accuracy across purely atomistic, purely coarse-grained, and mixed-resolution configurations demonstrates that the model has learned a consistent free energy surface spanning multiple resolution levels, enabling seamless multiscale simulations without interface artifacts.

#### III.3.4 Bulk mechanical validation

![Image 6: Refer to caption](https://arxiv.org/html/2603.01234v1/fig6-stress-strain.png)

Figure 6: Validation of the NEP-AACG model for uniaxial tensile deformation of bulk gold. (a) Axial stress σ x\sigma_{x} versus applied engineering strain ε x\varepsilon_{x} for fully atomistic NEP-AA and pure CG NEP-AACG models. (b) Transverse strain ε y\varepsilon_{y} versus axial strain ε x\varepsilon_{x}, illustrating the Poisson effect. Excellent agreement across the strain range up to 5%5\% confirms that the coarse-grained model faithfully reproduces the mechanical response. Simulations at 300 K.

To validate predictive capability, we compare the NEP-AACG model against the reference NEP-AA model in uniaxial tensile deformation of bulk face-centered cubic gold along the x x-direction, with engineering strain ε x\varepsilon_{x} up to 5%. Two simulations are conducted: fully atomistic simulation with NEP-AA and 16384 atoms, and pure CG simulation with NEP-AACG and 4096 beads.

Figure[6](https://arxiv.org/html/2603.01234#S3.F6 "Figure 6 ‣ III.3.4 Bulk mechanical validation ‣ III.3 NEP-AACG model for gold ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")a shows the stress-strain behavior. Axial stress σ x\sigma_{x} increases linearly with ε x\varepsilon_{x} in the elastic regime. Transverse strain ε y\varepsilon_{y} exhibits the expected Poisson contraction (Fig.[6](https://arxiv.org/html/2603.01234#S3.F6 "Figure 6 ‣ III.3.4 Bulk mechanical validation ‣ III.3 NEP-AACG model for gold ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")b). From the linear region, we estimate a Poisson ratio of approximately ν≈0.4\nu\approx 0.4, consistent with typical values for gold. The NEP-AACG results agree quantitatively with the NEP-AA reference across the entire strain range, confirming that the coarse-grained model, trained on ensemble-averaged data from strained configurations, successfully captures the mechanical response.

#### III.3.5 Multiscale nanowire fracture simulation

![Image 7: Refer to caption](https://arxiv.org/html/2603.01234v1/x2.png)

Figure 7: Tensile deformation of a gold nanowire using the NEP-AACG model. (a) Schematic of the tensile-test setup: CG bulk reservoir, atomistic AA transition zone, and atomistic nanowire specimen. (b) Engineering stress–strain response. Insets show representative atomic configurations at selected strains for the area highlighted by the black rectangle.

After validating the NEP-AACG model for bulk simulations, we demonstrate its capabilities in a proof-of-concept multiscale application: simulating fracture of gold nanowires under tension, which has attracted extensive interest Bro-Jørgensen _et al._ ([2025](https://arxiv.org/html/2603.01234#bib.bib41)). This problem exemplifies the multiscale nature of materials deformation, where atomistic resolution is required in the fracture region while extended length scales are necessary to avoid boundary condition artifacts.

We construct a gold nanowire tensile-test model with a total length of approximately 80 nm, comprising a few regions: a central atomistic nanowire region of about 37 nm long and 3 nm thick where fracture is expected, two end regions modeled as CG beads, and two AA transition regions. The CG regions extend to the boundaries, serving as reservoirs that apply realistic mechanical constraints to the atomistic zone (Fig.[7](https://arxiv.org/html/2603.01234#S3.F7 "Figure 7 ‣ III.3.5 Multiscale nanowire fracture simulation ‣ III.3 NEP-AACG model for gold ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")a).

We apply uniaxial tension by deforming the system at a constant engineering strain rate of 10 7 10^{7} s-1, which is within the range achievable in high-speed loading experiments Sun _et al._ ([2022](https://arxiv.org/html/2603.01234#bib.bib42)). The simulated stress-strain relation is shown in Fig.[7](https://arxiv.org/html/2603.01234#S3.F7 "Figure 7 ‣ III.3.5 Multiscale nanowire fracture simulation ‣ III.3 NEP-AACG model for gold ‣ III Results and discussion ‣ NEP-CG and NEP-AACG: Efficient coarse-grained and multiscale all-atom-coarse-grained neuroevolution potentials")b, along with snapshots at different strain levels.

The maximum stresses fluctuate about 30 MPa at strains between 0.01 and 0.05, during which the nanowire maintains its overall structural integrity. Beyond a strain of 0.07, the stress drops to below 10 MPa as the nanowire begins to thin in the middle. Finally, at strains exceeding 0.1, the nanowire undergoes complete fracture.

### III.4 Computational performance gains

A key advantage of coarse-grained models is computational efficiency, enabling access to extended time and length scales. We benchmark NEP-CG models against their NEP-AA counterparts using two metrics: particle throughput (particle⋅\cdot step/s) and effective simulation speed (ns/day) for equivalent spatial sizes. Benchmarks were performed on an NVIDIA RTX 5090 GPU using GPUMD Xu _et al._ ([2025b](https://arxiv.org/html/2603.01234#bib.bib25)), with each simulation running 1000 steps in the NVT ensemble across various system sizes.

Particle throughput isolates intrinsic computational complexity, primarily influenced by neural network size and average number of neighbors per particle. For water, the NEP-CG model achieves significantly higher throughput due to its smaller network (N neu=10 N_{\mathrm{neu}}=10 vs. 60 60) and fewer neighbors per bead from reduced particle density. For C 60, the advantage is even more pronounced: the NEP-CG model uses N neu=5 N_{\mathrm{neu}}=5 versus N neu=50 N_{\mathrm{neu}}=50 used for the NEP-AA model, and the 25 Å cutoff, while large in absolute value, still yields fewer neighbors per bead due to reduced interaction sites.

More practically relevant is effective simulation speed in ns/day for the same spatial size, incorporating three factors favoring CG models: higher particle throughput, larger allowable timesteps due to smoother CG energy surfaces and elimination of high-frequency vibrations, and fewer beads than atoms for the same volume.

For water, the NEP-AA model uses a 0.5 fs timestep; the NEP-CG model safely accommodates 2 fs. Combined with higher throughput and 3:1 atom-to-bead reduction, the NEP-CG model achieves approximately 50-fold speedup in ns/day for equivalent spatial sizes.

For the C 60 monolayer, gains are even more dramatic. The NEP-AA model requires a 1 fs timestep; the NEP-CG model allows 20 fs. This, combined with 60:1 atom-to-bead reduction and substantially higher throughput, yields approximately 1000-fold speedup in ns/day for equivalent spatial sizes.

![Image 8: Refer to caption](https://arxiv.org/html/2603.01234v1/fig8-speed.png)

Figure 8: Computational performance comparison between NEP-CG and NEP-AA models. (a) Particle throughput (particle⋅\cdot step/s) versus number of particles, demonstrating higher intrinsic efficiency of CG models due to smaller neural networks and fewer neighbors per particle. (b) Effective simulation speed (ns/day) versus equivalent atom count, accounting for larger timesteps and reduced particle count. Benchmarks on NVIDIA RTX 5090 GPU with GPUMD.

IV Summary and Conclusions
--------------------------

We have developed and demonstrated two complementary methods for constructing coarse-grained models within the NEP framework: NEP-CG for pure coarse-grained simulations and NEP-AACG for multiscale simulations that seamlessly integrate atomistic and coarse-grained degrees of freedom. The central innovation of our approach lies in generating low-noise training data through constrained molecular dynamics simulations, which directly yield ensemble-averaged forces corresponding to the potential of mean force. This contrasts with conventional force-matching methods that fit to noisy instantaneous forces, often leading to large training errors and low data efficiency.

We illustrated the capabilities of our methods through three representative examples. For liquid water, the NEP-CG model trained with ensemble-averaged forces achieved substantially lower errors than models trained on instantaneous forces, with force and stress RMSE reductions of nearly 50% and an order of magnitude, respectively. A virial correction scheme compensating for lost degrees of freedom proved essential for accurately reproducing the density-pressure equation of state across pressures from 1 bar to 1 GPa, including successful extrapolation beyond the training range.

For the quasi-hexagonal phase C 60 monolayer, we demonstrated the importance of capturing anisotropic bonding through distinct bead types corresponding to crystallographically inequivalent molecules. The two-type NEP-CG model dramatically improved stress prediction (RMSE reduction from 0.083 GPa to 0.0025 GPa) compared to a one-type model and successfully reproduced the anisotropic thermal conductivity. After appropriate scaling for reduced degrees of freedom, the CG model yielded thermal conductivities of the same order as the atomistic reference.

Our third example showcased the NEP-AACG framework for gold, demonstrating its ability to simultaneously describe purely atomistic, purely coarse-grained, and mixed-resolution configurations within a single unified model. The model accurately reproduced the stress-strain behavior of bulk gold under uniaxial tension and enabled multiscale simulations of nanowire fracture at an experimentally relevant strain rate of 10 7 10^{7} s-1.

Finally, we quantified the computational gains enabled by our NEP-CG models. For water, the combination of higher particle throughput, a fourfold timestep increase (0.5 fs to 2 fs), and a 3:1 particle reduction yielded approximately 50×\times speedup in ns/day. For the C 60 monolayer, a 20-fold timestep increase (1 fs to 20 fs) combined with a 60:1 particle reduction and higher throughput resulted in a dramatic 1000×\times speedup. Overall, the computational speeds for NEP-CG models reach hundreds to thousands of ns/day.

Despite these successes, our work has several limitations that point to important future directions. First, all training data were generated at a single temperature (300 K). The potential of mean force is inherently temperature-dependent, and extending our approach to incorporate multiple temperatures Ruza _et al._ ([2020](https://arxiv.org/html/2603.01234#bib.bib21)) represents a natural step toward thermodynamically consistent CG models.

Second, our models employ only isotropic beads. While the two-type approach for the C 60 monolayer successfully captured mechanical and thermal anisotropy, this strategy cannot represent situations where orientational degrees of freedom are essential Nguyen and Huang ([2022](https://arxiv.org/html/2603.01234#bib.bib38)); Wilson and Huang ([2023](https://arxiv.org/html/2603.01234#bib.bib39)); Campos-Villalobos _et al._ ([2024](https://arxiv.org/html/2603.01234#bib.bib43)); Argun and Statt ([2025](https://arxiv.org/html/2603.01234#bib.bib27)). Extending the NEP framework to accommodate anisotropic particles would significantly broaden applicability to soft matter systems such as polymers and liquid crystals.

Third, the partitioning into AA and CG regions in our NEP-AACG models is predefined and fixed during simulations, without the capability for dynamic resolution adaptation. This limits applications where the required level of detail changes over time, such as in systems where important events may occur in initially coarse-grained regions.

Future work will explore these directions, aiming to develop temperature-transferable CG models and anisotropic bead representations within the NEP-CG and NEP-AACG frameworks. Applications to more complex systems, including biomolecules and soft materials, will further demonstrate the versatility and power of our ensemble-based training methodology.

Data availability: All the training datasets and trained machine-learned potential models generated in this work are freely available in the nep-data repository ([https://gitlab.com/brucefan1983/nep-data](https://gitlab.com/brucefan1983/nep-data)).

###### Acknowledgements.

This work was supported by the Advanced Materials-National Science and Technology Major Project (No. 2024ZD0606900). ZF was supported by the Science Foundation from Education Department of Liaoning Province (No. LJ232510167001). KX acknowledges support from the Department of Science and Technology of Liaoning Province (No. JYTMS20231613) and Bohai University On-campus Doctoral Start-up Project Funding.

Declaration of Conflict of Interest
-----------------------------------

The authors have no conflicts to disclose.

References
----------

*   Noid (2013)W.G. Noid, “Perspective: Coarse-grained models for biomolecular systems,” [The Journal of Chemical Physics 139, 090901 (2013)](http://dx.doi.org/10.1063/1.4818908). 
*   Izvekov and Voth (2005)S.Izvekov and G.A. Voth, “A multiscale coarse-graining method for biomolecular systems,” [The Journal of Physical Chemistry B 109, 2469–2473 (2005)](http://dx.doi.org/10.1021/jp044629q). 
*   Noid _et al._ (2008)W.G. Noid, J.-W. Chu, G.S. Ayton, V.Krishna, S.Izvekov, G.A. Voth, A.Das, and H.C. Andersen, “The multiscale coarse-graining method. I. A rigorous bridge between atomistic and coarse-grained models,” [The Journal of Chemical Physics 128, 244114 (2008)](http://dx.doi.org/10.1063/1.2938860). 
*   Peng _et al._ (2023)Y.Peng, A.J. Pak, A.E.P. Durumeric, P.G. Sahrmann, S.Mani, J.Jin, T.D. Loose, J.Beiter, and G.A. Voth, “OpenMSCG: A Software Tool for Bottom-Up Coarse-Graining,” [The Journal of Physical Chemistry B 127, 8537–8550 (2023)](http://dx.doi.org/10.1021/acs.jpcb.3c04473). 
*   Mirzoev, Nordenskiöld, and Lyubartsev (2019)A.Mirzoev, L.Nordenskiöld, and A.Lyubartsev, “Magic v.3: An integrated software package for systematic structure-based coarse-graining,” [Computer Physics Communications 237, 263–273 (2019)](http://dx.doi.org/https://doi.org/10.1016/j.cpc.2018.11.018). 
*   Anderson, Glaser, and Glotzer (2020)J.A. Anderson, J.Glaser, and S.C. Glotzer, “HOOMD-blue: A Python package for high-performance molecular dynamics and hard particle Monte Carlo simulations,” [Computational Materials Science 173, 109363 (2020)](http://dx.doi.org/https://doi.org/10.1016/j.commatsci.2019.109363). 
*   Zhu _et al._ (2013)Y.-L. Zhu, H.Liu, Z.-W. Li, H.-J. Qian, G.Milano, and Z.-Y. Lu, “GALAMOST: GPU-accelerated large-scale molecular simulation toolkit,” [Journal of Computational Chemistry 34, 2197–2211 (2013)](http://dx.doi.org/https://doi.org/10.1002/jcc.23365). 
*   Xu _et al._ (2025a)J.Xu, S.Guo, M.Zhen, Z.Yu, Y.Zhu, G.Milano, and Z.Lu, “PyGAMD: Python graphics processing unit-accelerated molecular dynamics software,” [Materials Genome Engineering Advances 3, e70019 (2025a)](http://dx.doi.org/https://doi.org/10.1002/mgea.70019). 
*   Molinero and Moore (2009)V.Molinero and E.B. Moore, “Water modeled as an intermediate element between carbon and silicon,” [The Journal of Physical Chemistry B 113, 4008–4016 (2009)](http://dx.doi.org/10.1021/jp805227c). 
*   Larini, Lu, and Voth (2010)L.Larini, L.Lu, and G.A. Voth, “The multiscale coarse-graining method. VI. Implementation of three-body coarse-grained potentials,” [The Journal of Chemical Physics 132, 164107 (2010)](http://dx.doi.org/10.1063/1.3394863). 
*   Wang _et al._ (2021)J.Wang, N.Charron, B.Husic, S.Olsson, F.Noé, and C.Clementi, “Multi-body effects in a coarse-grained protein force field,” [The Journal of Chemical Physics 154, 164113 (2021)](http://dx.doi.org/10.1063/5.0041022). 
*   Unke _et al._ (2021)O.T. Unke, S.Chmiela, H.E. Sauceda, M.Gastegger, I.Poltavsky, K.T. Schütt, A.Tkatchenko, and K.-R. Müller, “Machine learning force fields,” [Chemical Reviews 121, 10142–10186 (2021)](http://dx.doi.org/10.1021/acs.chemrev.0c01111). 
*   John and Csányi (2017)S.T. John and G.Csányi, “Many-Body Coarse-Grained Interactions Using Gaussian Approximation Potentials,” [The Journal of Physical Chemistry B 121, 10934–10949 (2017)](http://dx.doi.org/10.1021/acs.jpcb.7b09636). 
*   Bartók _et al._ (2010)A.P. Bartók, M.C. Payne, R.Kondor, and G.Csányi, “Gaussian approximation potentials: the accuracy of quantum mechanics, without the electrons,” [Phys. Rev. Lett. 104, 136403 (2010)](http://dx.doi.org/10.1103/PhysRevLett.104.136403). 
*   Zhang _et al._ (2018a)L.Zhang, J.Han, H.Wang, R.Car, and W.E, “DeePCG: Constructing coarse-grained models via deep neural networks,” [The Journal of Chemical Physics 149, 034101 (2018a)](http://dx.doi.org/10.1063/1.5027645). 
*   Zhang _et al._ (2018b)L.Zhang, J.Han, H.Wang, R.Car, and W.E, “Deep Potential Molecular Dynamics: A Scalable Model with the Accuracy of Quantum Mechanics,” [Phys. Rev. Lett. 120, 143001 (2018b)](http://dx.doi.org/10.1103/PhysRevLett.120.143001). 
*   Wang _et al._ (2019)J.Wang, S.Olsson, C.Wehmeyer, A.Pérez, N.E. Charron, G.de Fabritiis, F.Noé, and C.Clementi, “Machine learning of coarse-grained molecular dynamics force fields,” [ACS Central Science 5, 755–767 (2019)](http://dx.doi.org/10.1021/acscentsci.8b00913). 
*   Majewski _et al._ (2023)M.Majewski, A.Pérez, P.Thölke, S.Doerr, N.E. Charron, T.Giorgino, B.E. Husic, C.Clementi, F.Noé, and G.De Fabritiis, “Machine learning coarse-grained potentials of protein thermodynamics,” [Nature communications 14, 5739 (2023)](http://dx.doi.org/10.1038/s41467-023-41343-1). 
*   Charron _et al._ (2025)N.E. Charron, K.Bonneau, A.S. Pasos-Trejo, A.Guljas, Y.Chen, F.Musil, J.Venturin, D.Gusew, I.Zaporozhets, A.Krämer, _et al._, “Navigating protein landscapes with a machine-learned transferable coarse-grained model,” [Nature chemistry 17, 1284–1292 (2025)](http://dx.doi.org/10.1038/s41557-025-01874-0). 
*   Husic _et al._ (2020)B.E. Husic, N.E. Charron, D.Lemm, J.Wang, A.Pérez, M.Majewski, A.Krämer, Y.Chen, S.Olsson, G.de Fabritiis, F.Noé, and C.Clementi, “Coarse graining molecular dynamics with graph neural networks,” [The Journal of Chemical Physics 153, 194101 (2020)](http://dx.doi.org/10.1063/5.0026133). 
*   Ruza _et al._ (2020)J.Ruza, W.Wang, D.Schwalbe-Koda, S.Axelrod, W.H. Harris, and R.Gómez-Bombarelli, “Temperature-transferable coarse-graining of ionic liquids with dual graph convolutional neural networks,” [The Journal of Chemical Physics 153, 164501 (2020)](http://dx.doi.org/10.1063/5.0022431). 
*   Thaler, Stupp, and Zavadlav (2022)S.Thaler, M.Stupp, and J.Zavadlav, “Deep coarse-grained potentials via relative entropy minimization,” [The Journal of Chemical Physics 157, 244103 (2022)](http://dx.doi.org/10.1063/5.0124538). 
*   Loose _et al._ (2023)T.D. Loose, P.G. Sahrmann, T.S. Qu, and G.A. Voth, “Coarse-Graining with Equivariant Neural Networks: A Path Toward Accurate and Data-Efficient Models,” [The Journal of Physical Chemistry B 127, 10564–10572 (2023)](http://dx.doi.org/10.1021/acs.jpcb.3c05928). 
*   Fan _et al._ (2021)Z.Fan, Z.Zeng, C.Zhang, Y.Wang, K.Song, H.Dong, Y.Chen, and T.Ala-Nissila, “Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport,” [Physical Review B 104, 104309 (2021)](http://dx.doi.org/10.1103/PhysRevB.104.104309). 
*   Xu _et al._ (2025b)K.Xu, H.Bu, S.Pan, E.Lindgren, Y.Wu, Y.Wang, J.Liu, K.Song, B.Xu, Y.Li, T.Hainer, L.Svensson, J.Wiktor, R.Zhao, H.Huang, C.Qian, S.Zhang, Z.Zeng, B.Zhang, B.Tang, Y.Xiao, Z.Yan, J.Shi, Z.Liang, J.Wang, T.Liang, S.Cao, Y.Wang, P.Ying, N.Xu, C.Chen, Y.Zhang, Z.Chen, X.Wu, W.Jiang, E.Berger, Y.Li, S.Chen, A.J. Gabourie, H.Dong, S.Xiong, N.Wei, Y.Chen, J.Xu, F.Ding, Z.Sun, T.Ala-Nissila, A.Harju, J.Zheng, P.Guan, P.Erhart, J.Sun, W.Ouyang, Y.Su, and Z.Fan, “GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials,” [Materials Genome Engineering Advances 3, e70028 (2025b)](http://dx.doi.org/https://doi.org/10.1002/mgea.70028). 
*   Ying _et al._ (2025)P.Ying, C.Qian, R.Zhao, Y.Wang, K.Xu, F.Ding, S.Chen, and Z.Fan, “Advances in modeling complex materials: The rise of neuroevolution potentials,” [Chemical Physics Reviews 6, 011310 (2025)](http://dx.doi.org/10.1063/5.0259061). 
*   Argun and Statt (2025)B.R. Argun and A.Statt, “Machine-learning potentials for efficient simulations of anisotropic colloids,” [The Journal of Chemical Physics 163, 234120 (2025)](http://dx.doi.org/10.1063/5.0303706). 
*   Li, Wang, and Zheng (2025)M.Li, L.Wang, and Z.Zheng, “Coarse-grained machine learning potential for mesoscale multilayered graphene,” [npj Computational Materials 11, 374 (2025)](http://dx.doi.org/10.1038/s41524-025-01849-2). 
*   Song _et al._ (2024)K.Song, R.Zhao, J.Liu, Y.Wang, E.Lindgren, Y.Wang, S.Chen, K.Xu, T.Liang, P.Ying, N.Xu, Z.Zhao, J.Shi, J.Wang, S.Lyu, Z.Zeng, S.Liang, H.Dong, L.Sun, Y.Chen, Z.Zhang, W.Guo, P.Qian, J.Sun, P.Erhart, T.Ala-Nissila, Y.Su, and Z.Fan, “General-purpose machine-learned potential for 16 elemental metals and their alloys,” [Nature Communications 15, 10208 (2024)](http://dx.doi.org/10.1038/s41467-024-54554-x). 
*   Liang _et al._ (2025)T.Liang, K.Xu, E.Lindgren, Z.Chen, R.Zhao, J.Liu, E.Berger, B.Tang, B.Zhang, Y.Wang, K.Song, P.Ying, N.Xu, H.Dong, S.Chen, P.Erhart, Z.Fan, T.Ala-Nissila, and J.Xu, “NEP89: Universal neuroevolution potential for inorganic and organic materials across 89 elements,” [arXiv:2504.21286 (2025), 10.48550/arXiv.2504.21286](http://dx.doi.org/10.48550/arXiv.2504.21286). 
*   Fan _et al._ (2022)Z.Fan, Y.Wang, P.Ying, K.Song, J.Wang, Y.Wang, Z.Zeng, K.Xu, E.Lindgren, J.M. Rahm, _et al._, “GPUMD: A package for constructing accurate machine-learned potentials and performing highly efficient atomistic simulations,” [The Journal of Chemical Physics 157, 114801 (2022)](http://dx.doi.org/10.1063/5.0106617). 
*   Schaul, Glasmachers, and Schmidhuber (2011)T.Schaul, T.Glasmachers, and J.Schmidhuber, “High dimensions and heavy tails for natural evolution strategies,” in [_Proceedings of the 13th Annual Conference on Genetic and Evolutionary Computation_](http://dx.doi.org/10.1145/2001576.2001692), GECCO ’11 (Association for Computing Machinery, New York, NY, USA, 2011) p. 845–852. 
*   Ercolessi and Adams (1994)F.Ercolessi and J.B. Adams, “Interatomic Potentials from First-Principles Calculations: The Force-Matching Method,” [Europhysics Letters 26, 583 (1994)](http://dx.doi.org/10.1209/0295-5075/26/8/005). 
*   Xu _et al._ (2025c)K.Xu, T.Liang, N.Xu, P.Ying, S.Chen, N.Wei, J.Xu, and Z.Fan, “NEP-MB-pol: a unified machine-learned framework for fast and accurate prediction of water’s thermodynamic and transport properties,” [npj Computational Materials 11, 279 (2025c)](http://dx.doi.org/10.1038/s41524-025-01777-1). 
*   Hou _et al._ (2022)L.Hou, X.Cui, B.Guan, S.Wang, R.Li, Y.Liu, D.Zhu, and J.Zheng, “Synthesis of a monolayer fullerene network,” [Nature 606, 507–510 (2022)](http://dx.doi.org/10.1038/s41586-022-04771-5). 
*   Dong _et al._ (2023)H.Dong, C.Cao, P.Ying, Z.Fan, P.Qian, and Y.Su, “Anisotropic and high thermal conductivity in monolayer quasi-hexagonal fullerene: A comparative study against bulk phase fullerene,” [International Journal of Heat and Mass Transfer 206, 123943 (2023)](http://dx.doi.org/https://doi.org/10.1016/j.ijheatmasstransfer.2023.123943). 
*   Görlich and Zavadlav (2026)F.Görlich and J.Zavadlav, “Mapping Still Matters: Coarse-Graining with Machine Learning Potentials,” [Journal of Chemical Information and Modeling 66, 2166–2176 (2026)](http://dx.doi.org/10.1021/acs.jcim.5c03035). 
*   Nguyen and Huang (2022)H.T.L. Nguyen and D.M. Huang, “Systematic bottom-up molecular coarse-graining via force and torque matching using anisotropic particles,” [The Journal of Chemical Physics 156, 184118 (2022)](http://dx.doi.org/10.1063/5.0085006). 
*   Wilson and Huang (2023)M.O. Wilson and D.M. Huang, “Anisotropic molecular coarse-graining by force and torque matching with neural networks,” [The Journal of Chemical Physics 159, 024110 (2023)](http://dx.doi.org/10.1063/5.0143724). 
*   Fan _et al._ (2019)Z.Fan, H.Dong, A.Harju, and T.Ala-Nissila, “Homogeneous nonequilibrium molecular dynamics method for heat transport and spectral decomposition with many-body potentials,” [Phys. Rev. B 99, 064308 (2019)](http://dx.doi.org/10.1103/PhysRevB.99.064308). 
*   Bro-Jørgensen _et al._ (2025)W.Bro-Jørgensen, J.M. Hamill, D.Donadio, and G.C. Solomon, “Bridging the Gap: Using Machine Learning Force Fields to Simulate Gold Break Junctions at Pulling Speeds Closer to Experiments,” [ACS Nano 19, 39735–39746 (2025)](http://dx.doi.org/10.1021/acsnano.5c11887). 
*   Sun _et al._ (2022)X.Sun, P.Yao, S.Qu, S.Yu, X.Zhang, W.Wang, C.Huang, and D.Chu, “Material properties and machining characteristics under high strain rate in ultra-precision and ultra-high-speed machining process: a review,” [The International Journal of Advanced Manufacturing Technology 120, 7011–7042 (2022)](http://dx.doi.org/10.1007/s00170-022-09111-5). 
*   Campos-Villalobos _et al._ (2024)G.Campos-Villalobos, R.Subert, G.Giunta, and M.Dijkstra, “Machine-learned coarse-grained potentials for particles with anisotropic shapes and interactions,” [npj Computational Materials 10, 228 (2024)](http://dx.doi.org/10.1038/s41524-024-01405-4).

