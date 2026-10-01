# 2607.02954

Source: https://arxiv.org/html/2607.02954



Microcosmos: Reimagining Artificial Life for the GPU Era

Report GitHub Issue
×

Title:

Content selection saved. Describe the issue below:
Description:

Submit without GitHub
Submit in GitHub

arXiv is now an independent nonprofit!
Learn more
×

Back to arXiv

Why HTML?

Report Issue

Back to Abstract

Download PDF

Abstract
Introduction
Related Work

Artificial Life Simulators
Bio-Inspiration
Computer Graphics
Design Optimization in Fluids

Methods

Filaments

Bending Pass
Position Pass

Fields
Fluid-filament Interaction
Encoding Schemes and Search Methods

Experiments

Hand-designed Locomotion
Filament Folding
Neuroevolution of Controller Circuit
QD Search for Locomotion Strategies
Scalability

Discussion and Conclusion
Acknowledgments
References

 License: CC BY 4.0
 

arXiv:2607.02954v1 [cs.NE] 03 Jul 2026

Microcosmos: Reimagining Artificial Life for the GPU Era

Mark Tensen

Affiliation: Artificial Life Institute, Japan

Email: *mark.tensen@alife.institute

  
Ciaran Regan

Affiliation: Artificial Life Institute, Japan

Affiliation: Sakana AI, Japan

Affiliation: University of Tsukuba, Japan

  
Bert Wang-Chak Chan

Affiliation: Artificial Life Institute, Japan

  
Mizuki Oka,
Kenneth O. Stanley, and Grisha Szep

Affiliation: Artificial Life Institute, Japan

Affiliation: Chiba Institute of Technology, Japan

Affiliation: Lila Sciences, USA

Abstract
Most artificial life simulators either operate on abstract substrates disconnected from physical reality, or simulate physically grounded worlds that do not scale to the population sizes required for open-ended evolution. We present Microcosmos, a simulation engine in which artificial lifeforms are modeled as elastic filament chains inhabiting a two-dimensional viscous fluid world, designed from the ground up for modern GPU hardware and end-to-end differentiable simulation. We validate the engine through four experiments. Hand-designed locomotion strategies confirm that the fluid coupling respects known physical constraints. Gradient-based optimization of filament folding demonstrates both the full differentiability of the simulator and the expressivity of the filament encodings. Neuroevolution and quality-diversity search produce a wide range of swimming and chemotaxis behaviors automatically. Linear scaling with particle count confirms the engine supports large-scale simulation. Microcosmos is released as an open platform with the long-term goal of supporting large-scale open-ended evolutionary simulations, designed to be physically plausible and computationally scalable.

Submission type: Full Paper

Code available at: https://github.com/alife-institute/microcosmos
Supplementary videos available at: https://alife.institute/microcosmos-supp
††
 
 
 
 
 
 
 ©2026 [Mark Tensen, Ciaran Regan, Bert Wang-Chak Chan, Mizuki Oka, Kenneth O. Stanley, and Grisha Szep]. Published under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

(a) Filaments bend, interact and flow in a fluid field.

(b) Diverse swimming gaits discovered via QD search.

Figure 1: Microcosmos: a scalable simulator for artificial life. Individuals are modeled as flexible filaments in a simulated fluid environment. (a) The three core physics components: (i) Filaments have preferred resting shapes, able to bend and deform elastically. (ii) Self-avoidance and inter-body repulsion mediated by scalar fields. (iii) Two-way coupling between individuals and the surrounding fluid. (b) Diverse swimming strategies discovered by quality-diversity search, illustrating the behavioral richness the simulator supports. See supplementary materials Figure S1a
for animations.

Introduction

What is preventing artificial life (ALife) from achieving the kind of open-ended evolutionary complexity we observe in nature? Despite decades of progress, many of the open problems identified by Bedau et al. (2000) remain unsolved. We argue that the field is held back by both a lack of physical grounding in our simulation substrates and insufficient computational scalability to support evolutionary search at scale. Abstract rule-based substrates (Langton, 1986; Chan, 2019; Ray, 1991; Alakuijala et al., 2024) are efficient to evolve but disconnected from the physical principles underlying life, as studied in fields such as biophysics and fluid dynamics. Conversely, more physically grounded simulators tend to be computationally intractable for large scale evolutionary search (Faure et al., 2012; Nedelec and Foethke, 2007), operate at too high a level of abstraction (Heinemann, 2024; Alakuijala et al., 2024), or are designed for reinforcement learning rather than the open-ended dynamics of life (Bhatia et al., 2021; Matthews et al., 2025; Lagemann et al., 2025). Progress demands simulators that are physically grounded enough to be credible, efficient enough to support evolutionary search at scale, and behaviorally
 rich enough to inspire.

Looking to the real world for guidance, we draw inspiration from two domains. The first is locomotion at the microscale, where organisms inhabit a world governed by viscous fluid dynamics and where inertia is negligible (Purcell, 1977). The diversity of swimming gaits, body plans, and collective behaviors observed in microorganisms arises from the interplay between elastic body mechanics and the surrounding fluid (Lauga and Powers, 2009). The second is the diversity of shape and function that emerges from protein folding, where linear chains of amino acids collapse into three-dimensional structures that determine biological activity. In both cases, the underlying building blocks are filaments: chains of connected units with simple nearest-neighbor topology. This shared structure is not a coincidence. Filamentous organization, from bacterial flagella to cytoskeletal networks to nematode body plans, is among the most ancient and ubiquitous forms of biological complexity (Elgeti et al., 2015). Furthermore, nearest-neighbor connectivity makes the simulation of these filaments highly scalable.

Here we present Microcosmos, a scalable simulation engine that takes these filaments as its core abstraction for artificial life. Artificial lifeforms are modeled as elastic filament chains inhabiting a two-dimensional viscous fluid world, with genetic encodings that specify both form and behavior. Within this world, creatures swim, fold into precise shapes, and forage across chemical gradients. We validate these capabilities using a broad range of experiments. We demonstrate that the fluid simulation supports lifelike locomotion strategies, including jellyfish, cilia, and tadpole gaits, consistent with known physical constraints. We demonstrate the full differentiability of Microcosmos by training filaments to fold into target shapes via gradient descent. Through neuroevolution and quality-diversity (QD) search, we show that a wide range of swimming strategies can be discovered automatically. Finally, we highlight the scalability of our simulation, showing that runtime increases linearly with the number of particles. To our knowledge, this is the first system to combine differentiable flexible filament simulation with a resolved fluid solver for both morphology and controller optimization. We release Microcosmos as an open platform, in the hope that it serves as a foundation for the wider artificial life community to build upon. Figure 1 gives an overview of the engine physics and showcases some of the gaits discovered via QD search.

Related Work

Artificial Life Simulators

Artificial life research has explored a diverse range of computational substrates in pursuit of open-ended complexification. Abstract computational worlds such as cellular automata, Lenia (Chan, 2019), Tierra (Ray, 1991), Avida (Ofria and Wilke, 2004), BFF (Alakuijala et al., 2024), and Chromaria (Soros and Stanley, 2014) show that rich evolutionary dynamics can emerge from minimal rules, while particle-based systems such as Boids (Reynolds, 1987), Particle Life (Ventrella, 2017; Mohr, 2023), and Particle Lenia (Mordvintsev et al., 2022) demonstrate that self-organizing collective behaviors can likewise emerge from simple local interactions. However, all of these substrates remain disconnected from physical reality, limiting the embodied behaviors they can support. Physically grounded, embodied simulators (Sims, 1994; Miconi, 2008; Cheney et al., 2014; Kriegman et al., 2020) demonstrate that collision-based worlds can produce diverse behaviors; however such approaches are often computationally intractable. Other systems achieve tractability through GPU-accelerated agent-based frameworks (Richmond et al., 2010; Richmond et al., 2023) or architectures built for indefinite scalability (Ackley and Small, 2014),
 but operate at higher levels of abstraction (Suarez et al., 2019; Lu et al., 2024; Heinemann, 2024; Whidden, 2025). Microcosmos addresses these limitations by grounding artificial life in viscous fluid dynamics, producing embodied creatures whose behaviors are physically plausible, visually compelling and computationally tractable at scale.

Bio-Inspiration

The microscopic world offers a compelling substrate for artificial life research. Filamentous structures (chains of cells, polymers, flagella, and cytoskeletal networks) are among the most ancient forms of biological organization. From bacterial flagella to nematode body plans, elongated elastic bodies immersed in viscous fluids represent a minimal yet expressive design space in which evolution has produced remarkable functional diversity (Elgeti et al., 2015).

The physics of this regime imposes fundamental constraints on what strategies can succeed. At low Reynolds numbers, the Navier-Stokes equations become time-reversible, and Purcell’s scallop theorem dictates that reciprocal motion cannot produce net locomotion in Stokes flow (Purcell, 1977). Breaking this symmetry requires time-irreversible deformation strategies, such as the undulatory waves of C. elegans (Boyle et al., 2012), propagating transverse waves analyzed by Taylor (1951), or the asymmetric ciliary strokes described in Purcell (1977). These constraints make low-Reynolds-number locomotion a natural testbed for studying how physics shapes the evolution of morphology and behavior.

Computer Graphics

Computer graphics (CG) research is fundamentally concerned with the efficient synthesis of believable virtual environments, and breakthroughs in CG have historically driven breakthroughs in ALife. Boids (Reynolds, 1987), originally a CG technique for animating flocks, became a foundational model for emergent collective behavior. Sims’ virtual creatures (Sims, 1994) applied rigid-body physics from computer animation to synthetic biological evolution. This trend continues with differentiable graphics tools such as DiffTaichi (Hu et al., 2019) and DiffPD (Du et al., 2021), which enable gradient-based optimization of soft bodies and fluid-structure interactions, and with voxel-based physics used to design soft robots (Cheney et al., 2014; Kriegman et al., 2020). More recently, advances in real-time physics simulations have enabled RL benchmark frameworks such as HydroGym (Lagemann et al., 2025) and FluidGym (Becktepe et al., 2026). Microcosmos continues this lineage, drawing on GPU-efficient fluid and soft-body simulations from CG to ground artificial life in the mechanics that govern life at the microscale.

Design Optimization in Fluids

Microcosmos sits at the intersection of three research areas: flexible filament simulation, fluid dynamics, and design optimization. Each pairwise combination has been explored. Filaments have been simulated in resolved fluids without optimization (Tekinalp et al., 2025; Tian et al., 2011), but these systems are not differentiable, so any search over morphology must rely on gradient-free methods. Filament mechanics have been made differentiable (Bergou et al., 2008; Stuyck and Chen, 2023), enabling gradient-based optimization of rest shapes and stiffness, but without hydrodynamic coupling the optimized morphologies have no notion of locomotion or drag. Differentiable fluid solvers have been paired with design optimization (Ma et al., 2021; Lee et al., 2023; Fan et al., 2026), but these systems use rigid or beam-like bodies and target engineering inverse problems rather than morphological evolution.

The closest prior systems each miss one of the three ingredients. SophT (Tekinalp et al., 2025) has resolved fluid coupling with Cosserat rods but is not differentiable. PyElastica (Zhang et al., 2019) provides Cosserat rods with optimization via reinforcement learning (Naughton et al., 2021) but uses local drag approximations rather than a resolved fluid solver. Diff-FlowFSI (Fan et al., 2026) achieves full differentiability with resolved flow, but its beam elements lack the topological flexibility needed for morphological evolution. Microcosmos occupies this remaining gap: gradient-based optimization of flexible filament morphology through a fully coupled lattice Boltzmann fluid simulation, designed from the ground up for evolutionary search over body plans and behaviors.

Methods

The Microcosmos engine consists of two coupled components: filaments, which represent artificial lifeforms, and fields which represent the environment. Filaments are implemented as graphs, where each node’s state describes its position in continuous space. Fields, on the other hand, are modeled as discrete grids which store information about the world, such as the fluid state 11
1
 
 
 
 
 
 
 
 In fluid-solid-interaction research this is a common paradigm, where grid-based fields and continuous valued graphs are called the Eulerian and Lagrangian components respectively (Peskin, 2002).. These two elements are overlaid, interacting as a dynamical system. This formulation could be considered a middle ground between grid-based and particle-based paradigms, such as Cellular Automata and Particle Life, respectively.22
2
 
 
 
 
 
 
 
 It should be noted that analogs to this paradigm already exist within the field of Artificial Life, for instance in the modeling of stigmergy-based systems like slime molds. Here, agents operate in continuous space, but deposit pheromones onto a discrete environmental grid. These pheromones are subsequently diffused across the grid, where they are read by other agents to influence their behavior (Jones, 2010). The entire simulator is implemented in JAX Bradbury et al. (2018), enabling GPU acceleration and end-to-end automatic differentiation.

Beyond biological inspirations, this filament-field formulation is largely motivated by scalability. The engine must support large populations of interacting filaments over evolutionary timescales. As such, every algorithmic choice avoids quadratic scaling with the number of nodes. For instance, the fluid solver requires only local updates. These design choices make the simulation scalable and fully parallelizable.

Filaments

Filamentous structures span the micro- and nanoscale, from bacterial flagella and cytoskeletal networks that drive low-Reynolds-number locomotion, to proteins that fold one-dimensional sequences into three-dimensional volumes, to contractile muscle filaments.
 We model filaments as discrete elastic rods that resist both stretching and bending (Bergou et al., 2008). These can be extended to have loops and branches with a graph representation. Intersections between filaments are avoided via a grid-based steric repulsion field (see Fields below) rather than per-particle collision handling. The steric field does not guarantee
 exact collision dynamics; however, the benefits of scaling are worth the loss in accuracy.

Each segment carries a per-edge orientation angle θ\theta, following the Cosserat rod formulation (Antman, 2005). In this framework, θ\theta defines the material frame of the edge.
To enable local control, the physics parameters are set on the edge level instead of globally: rest length LrestL_{\text{rest}}, bending rest angle θrest\theta_{\text{rest}}, and optionally stiffness coefficients (kbendk_{\text{bend}}, ksheark_{\text{shear}}, kstretchk_{\text{stretch}}) controlling how strongly each constraint is enforced. Together, they define the creature’s morphology and serve as the targets of both gradient-based and evolutionary optimization.

The dynamics are solved using Position-Based Dynamics (PBD) (Müller et al., 2007), an iterative constraint projection method that directly modifies node positions and edge orientations to satisfy geometric constraints. PBD was chosen for two reasons: it is unconditionally stable regardless of timestep, and its operations are simple arithmetic that JAX can differentiate through directly, unlike implicit solvers that would require differentiating through a linear solve.

Enforcing Cosserat constraints within a PBD framework follows established methods (Umetani et al., 2014), which we adapt here for the planar (2D) case. Each iteration decouples the solver into two phases. First, a bending pass updates θ\theta toward θrest\theta_{\text{rest}}. Second, a position pass enforces nodal connectivity by resolving stretch and shear errors based on the updated orientations and rest length LrestL_{\text{rest}}.

Bending Pass

For adjacent edges with material frame angles θ1\theta_{1} and θ2\theta_{2}, we define the bending constraint Cbend=(θ2−θ1)−θrestC_{\text{bend}}=(\theta_{2}-\theta_{1})-\theta_{\text{rest}}. Applying a bending stiffness kbendk_{\text{bend}}, the material frames are updated symmetrically:

θ1←θ1+12​kbend​Cbend,θ2←θ2−12​kbend​Cbend\theta_{1}\leftarrow\theta_{1}+\frac{1}{2}k_{\text{bend}}\,C_{\text{bend}},\qquad\theta_{2}\leftarrow\theta_{2}-\frac{1}{2}k_{\text{bend}}\,C_{\text{bend}}

(1)

Position Pass

For an edge connecting nodes 𝐱1\mathbf{x}_{1} and 𝐱2\mathbf{x}_{2}, let 𝐯=𝐱2−𝐱1\mathbf{v}=\mathbf{x}_{2}-\mathbf{x}_{1} be the edge vector and 𝐝⁡(θ)=[cos⁡θ,sin⁡θ]T\mathbf{d}(\theta)=[\cos\theta,\sin\theta]^{T} be the material director. The stretch constraint along the geometric tangent is:

𝐂stretch=(‖𝐯‖−Lrest)​𝐯‖𝐯‖\mathbf{C}_{\text{stretch}}=\left(\|\mathbf{v}\|-L_{\text{rest}}\right)\frac{\mathbf{v}}{\|\mathbf{v}\|}

(2)

To decouple stretch and shear, the total positional correction Δ​𝐱\Delta\mathbf{x} blends this stretch constraint with the full Kirchhoff-Love error, (𝐯−Lrest​𝐝)(\mathbf{v}-L_{\text{rest}}\mathbf{d}), using a shear stiffness parameter kshear∈[0,1]k_{\text{shear}}\in[0,1]:

Δ​𝐱=12​kstretch​[𝐂stretch+kshear​(𝐯−Lrest​𝐝−𝐂stretch)]\Delta\mathbf{x}=\frac{1}{2}k_{\text{stretch}}\left[\mathbf{C}_{\text{stretch}}+k_{\text{shear}}\big(\mathbf{v}-L_{\text{rest}}\mathbf{d}-\mathbf{C}_{\text{stretch}}\big)\right]

(3)

The node positions are then updated symmetrically:

𝐱1←𝐱1+Δ​𝐱,𝐱2←𝐱2−Δ​𝐱\mathbf{x}_{1}\leftarrow\mathbf{x}_{1}+\Delta\mathbf{x},\qquad\mathbf{x}_{2}\leftarrow\mathbf{x}_{2}-\Delta\mathbf{x}

(4)

Fields

The primary field is the fluid velocity, a persistent vector field simulated using the Lattice Boltzmann Method (LBM) with the D2Q9 model (Qian et al., 1992) (two dimensions, nine discrete velocities). LBM was chosen because each lattice site updates independently during the streaming step, making it naturally parallel on GPU (𝒪⁡(n)\mathcal{O}(n)) and compatible with spatial domain decomposition. Operating at the mesoscopic scale between molecular dynamics and direct Navier-Stokes solvers, LBM handles moving filament boundaries without remeshing (bending the grid to follow the swimmer), and its viscosity is controlled by a single relaxation parameter τ\tau, making it straightforward to sweep across physical regimes. Self-avoidance between filament nodes is handled through a grid-based steric repulsion field rather than pairwise distance checks, avoiding 𝒪⁡(n2)\mathcal{O}(n^{2}) scaling: node positions are deposited onto a grid, diffused with a Gaussian filter via FFT convolution to produce a density field, and the gradient of this density field generates repulsive forces.

In neuroevolution experiments, we additionally introduce a persistent scalar energy field consisting of Gaussian packets placed at random positions, providing a chemotaxis target for evolving creatures.

Fluid-filament Interaction

Fluid and filaments interact through the Immersed Boundary Method (IBM) Peskin (2002). IBM enables a two-way information exchange: first, the filaments’ nodes sample the local fluid velocity to determine their own drag, and second, the agent broadcasts its own velocity back onto the grid. By smoothing these interactions over a small neighborhood (a diffuse interface), the method converts single node positions
into a continuous, approximated solid body.

The resulting diffuse (as opposed to a solid) fluid-solid boundary comes with the challenge of fluid leaking through the filaments, which is especially the case with a zero-thickness filament.
To address this issue, we used a number of
 LBM/IBM best-practices:
Multi-Direct Forcing (MDF) (Wang et al., 2008), Two-Relaxation-Time (TRT) (Ginzburg et al., 2008), and broadcasting the single filament into a two-layer version of itself to artificially amplify a zero-thickness filament into a solid object (Mittal and Iaccarino, 2005).

Encoding Schemes and Search Methods

Microcosmos is highly flexible in terms of the representation of filaments, supporting both direct and indirect encoding schemes (Fekiač et al., 2011; Clune et al., 2011; Miikkulainen and Forrest, 2021) that map from searchable parameters (the genotype) to the physical form and behavior (the phenotype).

Direct encoding uses one-to-one mappings, where every individual parameter is specified and potentially optimized. For Microcosmos, that means the control signals and parameters (θrest,Lrest,kbend,kshear,kstretch\theta_{\text{rest}},L_{\text{rest}},k_{\text{bend}},k_{\text{shear}},k_{\text{stretch}}, etc) are individually specified for each node.

In contrast, indirect encoding uses a generative or developmental model, with its own searchable parameters, to generate the physical traits for each node based on spatial-temporal or sensory inputs. Compositional Pattern Producing Network (CPPN) (Stanley, 2007), L-systems (De Campos et al., 2011) and cellular automata (Gutiérrez et al., 2005; Najarro et al., 2022) are prominent examples.

The Microcosmos engine is search-method agnostic and end-to-end differentiable. In our experiments, we use gradient-based optimization for filament folding experiments (Filament Folding) and QD search for neuroevolution experiments (Neuroevolution of Controller Circuit, QD Search for Locomotion Strategies). We also have locomotion experiments (Hand-designed Locomotion) that use hand-selected parameters to validate the fluid-filament coupling.

Figure 2: Realistic fluid dynamics of the Microcosmos engine as confirmed by five creatures with hand-designed geometry and locomotion (left) and viscosity sweep (right). Consistent with Purcell’s scallop theorem, time-reversible strategies (e.g. ray) produce zero net displacement at high viscosity, while non-reciprocal strategies (e.g. cilia) maintain viable locomotion. See supplementary materials Figure S2 for animations. 

Figure 3: Differentiability of the Microcosmos engine as demonstrated by MNIST digit folding via stochastic gradient descent (SGD). Each column shows a different target digit (0–9). The first three rows show the filament folding through a single 250-step simulation run (top to bottom: early, mid, and late), with the bottom row showing the corresponding target digit. See supplementary materials Figure S3 for animations.

(a) Fitness grid of elite creatures, using the same behavioral space as in Figure 1 (b). High-fitness swimmers are widely distributed inside the grid, confirming the effectiveness of the BDs in use.

(b) Best fitness and archive coverage through evolution progress.

Figure 4: Evolvability of diverse locomotion strategies as demonstrated by QD algorithm MAP-Elites. See Figure 1 (b) for the MAP-Elites archive of elite creatures, snapshot at simulation
step 500. 

Experiments

We validate the Microcosmos engine through experiments targeting physical correctness, differentiability, evolvability, and scalability.

Hand-designed Locomotion

We demonstrate five hand-designed geometries and locomotion strategies that validate the physics engine’s fluid coupling (Figure 2). Viscosity is swept from 10−310^{-3} to 22; this spans approximately Reynolds number R​e∼103Re\sim 10^{3} to 11 depending on the swimmer, covering the transition from inertia-dominated to viscosity-dominated swimming. Although these values already cover a useful and diverse scale, we acknowledge that outside this range our fluid-filament coupling may give rise to artifacts.

Each creature exhibits distinct behaviors: The worm and the tadpole use a similar sinusoidal wave propagation along a soft filament analogous to the undulatory waves of C. elegans discussed above, with the tadpole reaching lower speeds because of the drag caused by its head; the ray uses beating via a single time-dependent angle constraint; the cilia uses asymmetrical movement of short filaments attached around a stiff spherical body; the jellyfish uses contractile movement via uniform distance and bending constraints on an open circle.

The peak displacement observed in Figure 2 from cilia at high viscosity is a hallmark of asymmetrical swimming strokes. In these regimes, propulsive efficiency relies on the increased fluid traction generated by higher viscous resistance. Unlike the worm’s undulatory waves, cilia exploit drag asymmetry more directly: a stiff stroke sweeps fluid broadside while the recovery stroke curls tightly, making them better matched to vanishingly low Reynolds numbers. Ultimately, a very high viscosity brings all movement close to a stop.

Filament Folding

To demonstrate the differentiability and expressivity of the simulation and filament parameterization, we optimize filaments to fold into target shapes, offering a simplified 2D analog of the protein folding problem. The angles and lengths of a 1000-node linear filament are optimized directly using Adam (learning rate 0.040.04, 300 parameter update steps), targeting MNIST digit shapes. Starting from a random initial configuration, we simulate the filament for 250 simulation steps and compute the folding loss as the mean squared error between the Gaussian-splatted particle density at the final timestep and the target digit image at 128×128128\times 128 resolution. Gradients of this loss flow end-to-end through the full simulation via JAX automatic differentiation, including through the PBD constraint solver and fluid computation, directly updating the morphological parameters.

Results for each MNIST digit class are shown in Figure 3. Gradient descent reliably recovers parameters that fold the filament into recognizable digit shapes across all ten classes. Although the reproductions are imperfect, the results confirm that the Cosserat rod formulation provides a sufficiently expressive and differentiable substrate for 2D morphogenesis.

Neuroevolution of Controller Circuit

We experimented with evolving a Compositional Pattern Producing Network (CPPN) (Stanley, 2007) controller for the filaments. We used one shared CPPN to perform sensorimotor control for each node. The CPPN inputs include the node’s position along the filament (ranging from 0.0 to 1.0) and a global timer, such that each node has a basic sense of space and time. The CPPN outputs the rest angle θrest\theta_{\text{rest}} and the rest length LrestL_{\text{rest}}, such that the node can control its own movement.

Neuroevolution of Augmenting Topologies (NEAT) (Stanley and Miikkulainen, 2002) is used to evolve the CPPN, where the algorithm mutates the network architecture, activation functions, and connection weights. Both are implemented using TensorNEAT (Wang et al., 2025). The fitness function is defined as the total displacement traveled. We found that the highest fitness swimmers predominantly perform sinusoidal locomotion.

We also added an “energy” field, a scalar field with scattered packets representing local energy akin to food. The CPPN takes the energy value at the node’s position as an additional input, and fitness becomes the total displacement traveled plus the total energy collected, rewarding both movement and chemotaxis. See supplementary materials Figure S4 for animations.

QD Search for Locomotion Strategies

To discover more diverse types of locomotion, we apply QD search (Pugh et al., 2016). In particular, we use Multi-dimensional Archive of Phenotypic Elites (MAP-Elites) (Mouret and Clune, 2015) to search for diverse locomotion strategies. Here we show the results using behavioral descriptors (BDs)
 defined
 as the bending effort EbendE_{\text{bend}} and stretching effort EposE_{\text{pos}}, where θrestt\theta_{\text{rest}}^{t} and LresttL_{\text{rest}}^{t} are rest angles and rest lengths at simulation time tt, respectively.

Ebend\displaystyle E_{\text{bend}}
=∑|θrestt−1−θrestt|2\displaystyle=\sum|\theta_{\text{rest}}^{t-1}-\theta_{\text{rest}}^{t}|^{2}

(5)

Epos\displaystyle E_{\text{pos}}
=∑|Lrestt−1−Lrestt|2\displaystyle=\sum|L_{\text{rest}}^{t-1}-L_{\text{rest}}^{t}|^{2}

(6)

A diverse set of swimming strategies emerged from the CPPN-QD search (Figure 1 (b)).
The strategies can roughly be clustered into a few categories, including sinusoidal movements (i, iii, x), directional turning (v, vi, vii), complex motions (ii, viii, ix), circular structure paddling with “flippers” (iv). See supplementary materials Figure S1a and Figure S1b for animations.

Scalability

To demonstrate scalability, we measure wall clock time as we increase the number of particles at a fixed grid resolution (256×256256\times 256), running 1000 steps with up to 500k particles33
3
 
 
 
 
 
 
 
 Ran with 1 NVIDIA L40S GPU, JAX version 0.8.10.8.1.
. As shown in fig. 5, Microcosmos scales linearly, unlike simulators that rely on pairwise interactions such as Particle Life (Mohr, 2023), which scale as 𝒪⁡(n2)\mathcal{O}(n^{2}). Tiling strategies (Green, 2010) help but still degrade to quadratic scaling under non-uniform particle distributions. Every design choice in Microcosmos, from grid-based steric repulsion to the local updates of LBM, targets linear scaling.

Figure 5: Wall clock time scales linearly with the number of particles in Microcosmos, in contrast to the 𝒪⁡(n2)\mathcal{O}(n^{2}) scaling of simulators that compute pairwise particle interactions.

Discussion and Conclusion

Microcosmos was built from a simple conviction that artificial life has been held back not by a lack of imagination, but by a lack of the right tools. The simulation substrates available to the field have forced an uncomfortable choice between physical credibility and computational tractability. We set out to narrow that gap by drawing on advances in fluid dynamics, biophysics, and computer graphics that the ALife community has largely not yet absorbed. Like any simulation, it abstracts away much of the physical world, but it occupies a useful and under-explored region between credibility and scalability.

The result is a simulation engine in which artificial lifeforms are modeled as elastic filament chains inhabiting a two-dimensional viscous fluid world. Filaments are among the most ancient and ubiquitous structures in biology, and their nearest-neighbor topology makes them naturally scalable to simulate. The fluid environment, solved via the Lattice Boltzmann Method, imposes real physical constraints on locomotion, grounding the evolutionary dynamics in physical reality. The entire pipeline is implemented in JAX and is end-to-end differentiable, supporting both gradient-based optimization and evolutionary search.

Our experiments validate the physical correctness of the fluid coupling, the differentiability of the full simulation pipeline, and the ability to automatically discover diverse locomotion and chemotaxis behaviors, all while scaling linearly with particle count. The simulation is not without limitations. For instance, the topologies of the filaments are currently restricted to cactus graphs and the physics engine has a number of hyperparameters that may require adjustment.

We offer Microcosmos as an open platform and a call to the wider ALife community to build upon it. Our future work will focus on multi-agent interactions, not only simulating larger ecosystems but also allowing for unpredictable interactions between individuals to emerge. Our bet is that this will require some notion of birth, death and self-assembly. This would open studies of competition and symbiosis as well as the emergence of individuality and function. Further ahead, we hope to see large-scale compute runs applied to open-ended evolution in physically grounded virtual worlds. Just as scaling compute transformed modern AI, we believe scaling physically grounded simulations may do the same for artificial life.

Acknowledgments

This work was supported by JSPS KAKENHI Grant Number 24H02200 and a post-doc fellowship (PE25014). We thank Takashi Ikegami for his invaluable support.

References

Ackley and Small (2014)
D. Ackley and T. Small

Indefinitely scalable computing= artificial life engineering.

In Artificial Life conference proceedings,

pp. 606–613.

Cited by: Artificial Life Simulators.

Alakuijala et al. (2024)
J. Alakuijala, J. Evans, B. Laurie, A. Mordvintsev, E. Niklasson, E. Randazzo, L. Versari, et al.

Computational life: how well-formed, self-replicating programs emerge from simple interaction.

arXiv preprint arXiv:2406.19108.

Cited by: Introduction,
Artificial Life Simulators.

Antman (2005)
S. S. Antman

Nonlinear problems of elasticity.

 Springer.

Cited by: Filaments.

Becktepe et al. (2026)
J. Becktepe, A. Franz, N. Thuerey, and S. Peitz

Plug-and-play benchmarking of reinforcement learning algorithms for large-scale flow control.

arXiv preprint arXiv:2601.15015.

Cited by: Computer Graphics.

Bedau et al. (2000)
M. A. Bedau, J. S. McCaskill, N. H. Packard, S. Rasmussen, C. Adami, D. G. Green, T. Ikegami, K. Kaneko, and T. S. Ray

Open problems in artificial life.

Artificial life 6 (4), pp. 363–376.

Cited by: Introduction.

Bergou et al. (2008)
M. Bergou, M. Wardetzky, S. Robinson, B. Audoly, and E. Grinspun

Discrete elastic rods.

In ACM SIGGRAPH 2008 Papers,

pp. 1–12.

Cited by: Design Optimization in Fluids,
Filaments.

Bhatia et al. (2021)
J. Bhatia, H. Jackson, Y. Tian, J. Xu, and W. Matusik

Evolution gym: a large-scale benchmark for evolving soft robots.

Advances in Neural Information Processing Systems 34, pp. 2201–2214.

Cited by: Introduction.

Boyle et al. (2012)
J. H. Boyle, S. Berri, and N. Cohen

Gait modulation in c. elegans: an integrated neuromechanical model.

Frontiers in computational neuroscience 6, pp. 10.

Cited by: Bio-Inspiration.

Bradbury et al. (2018)
JAX: composable transformations of Python+NumPy programs

External Links: Link

Cited by: Methods.

Chan (2019)
B. W. Chan

Lenia – biology of artificial life.

Complex Systems 28 (3), pp. 251–286.

Cited by: Introduction,
Artificial Life Simulators.

Cheney et al. (2014)
N. Cheney, R. MacCurdy, J. Clune, and H. Lipson

Unshackling evolution: evolving soft robots with multiple materials and a powerful generative encoding.

ACM SIGEVOlution 7 (1), pp. 11–23.

Cited by: Artificial Life Simulators,
Computer Graphics.

Clune et al. (2011)
J. Clune, K. O. Stanley, R. T. Pennock, and C. Ofria

On the performance of indirect encoding across the continuum of regularity.

IEEE Transactions on Evolutionary Computation 15 (3), pp. 346–367.

Cited by: Encoding Schemes and Search Methods.

De Campos et al. (2011)
L. M. L. De Campos, M. Roisenberg, and R. C. L. de Oliveira

Automatic design of neural networks with l-systems and genetic algorithms-a biologically inspired methodology.

In The 2011 international joint conference on neural networks,

pp. 1199–1206.

Cited by: Encoding Schemes and Search Methods.

Du et al. (2021)
T. Du, K. Wu, P. Ma, S. Wah, A. Spielberg, D. Rus, and W. Matusik

Diffpd: differentiable projective dynamics.

ACM Transactions on Graphics (ToG) 41 (2), pp. 1–21.

Cited by: Computer Graphics.

Elgeti et al. (2015)
J. Elgeti, R. G. Winkler, and G. Gompper

Physics of microswimmers—single particle motion and collective behavior: a review.

Reports on progress in physics 78 (5), pp. 056601.

Cited by: Introduction,
Bio-Inspiration.

Fan et al. (2026)
X. Fan, X. Liu, M. Wang, and J. Wang

Diff-flowfsi: a gpu-optimized differentiable cfd platform for high-fidelity turbulence and fsi simulations.

Computer Methods in Applied Mechanics and Engineering 448, pp. 118455.

Cited by: Design Optimization in Fluids,
Design Optimization in Fluids.

Faure et al. (2012)
F. Faure, C. Duriez, H. Delingette, J. Allard, B. Gilles, S. Marchesseau, H. Talbot, H. Courtecuisse, G. Bousquet, I. Peterlik, et al.

Sofa: a multi-model framework for interactive physical simulation.

In Soft tissue biomechanical modeling for computer assisted surgery,

pp. 283–321.

Cited by: Introduction.

Fekiač et al. (2011)
J. Fekiač, I. Zelinka, and J. C. Burguillo

A review of methods for encoding neural network topologies in evolutionary computation.

In Proceedings of 25th European conference on modeling and simulation ECMS,

pp. 410–416.

Cited by: Encoding Schemes and Search Methods.

Ginzburg et al. (2008)
I. Ginzburg, F. Verhaeghe, and D. d’Humieres

Two-relaxation-time lattice boltzmann scheme: about parametrization, velocity, pressure and mixed boundary conditions.

Communications in computational physics 3 (2), pp. 427–478.

Cited by: Fluid-filament Interaction.

Green (2010)
S. Green

Particle simulation using cuda.

NVIDIA whitepaper 6, pp. 121–128.

Cited by: Scalability.

Gutiérrez et al. (2005)
G. Gutiérrez, A. Sanchis, P. Isasi, J. M. Molina, and I. M. Galván

Non-direct encoding method based on cellular automata to design neural network architectures.

Computing and Informatics 24 (3), pp. 225–247.

Cited by: Encoding Schemes and Search Methods.

Heinemann (2024)
C. Heinemann

ALIEN: artificial life environment.

Note: https://www.alien-project.orgAccessed: 2026-04-12

Cited by: Introduction,
Artificial Life Simulators.

Hu et al. (2019)
Y. Hu, L. Anderson, T. Li, Q. Sun, N. Carr, J. Ragan-Kelley, and F. Durand

Difftaichi: differentiable programming for physical simulation.

arXiv preprint arXiv:1910.00935.

Cited by: Computer Graphics.

Jones (2010)
J. Jones

Characteristics of pattern formation and evolution in approximations of physarum transport networks.

Artificial life 16 (2), pp. 127–153.

Cited by: footnote 2.

Kriegman et al. (2020)
S. Kriegman, D. Blackiston, M. Levin, and J. Bongard

A scalable pipeline for designing reconfigurable organisms.

Proceedings of the National Academy of Sciences 117 (4), pp. 1853–1859.

Cited by: Artificial Life Simulators,
Computer Graphics.

Lagemann et al. (2025)
C. Lagemann, S. Mokbel, M. Gondrum, M. Rüttgers, J. Callaham, L. Paehler, S. Ahnert, N. Zolman, K. Lagemann, N. Adams, et al.

Hydrogym: a reinforcement learning platform for fluid dynamics.

arXiv preprint arXiv:2512.17534.

Cited by: Introduction,
Computer Graphics.

Langton (1986)
C. G. Langton

Studying artificial life with cellular automata.

Physica D: nonlinear phenomena 22 (1-3), pp. 120–149.

Cited by: Introduction.

Lauga and Powers (2009)
E. Lauga and T. R. Powers

The hydrodynamics of swimming microorganisms.

Reports on progress in physics 72 (9), pp. 096601.

Cited by: Introduction.

Lee et al. (2023)
J. H. Lee, M. Y. Michelis, R. Katzschmann, and Z. Manchester

Aquarium: a fully differentiable fluid-structure interaction solver for robotics applications.

arXiv preprint arXiv:2301.07028.

Cited by: Design Optimization in Fluids.

Lu et al. (2024)
C. Lu, M. Beukman, M. Matthews, and J. Foerster

Jaxlife: an open-ended agentic simulator.

In Artificial Life Conference Proceedings 36,

Vol. 2024, pp. 47.

Cited by: Artificial Life Simulators.

Ma et al. (2021)
P. Ma, T. Du, J. Z. Zhang, K. Wu, A. Spielberg, R. K. Katzschmann, and W. Matusik

Diffaqua: a differentiable computational design pipeline for soft underwater swimmers with shape interpolation.

ACM Transactions on Graphics (TOG) 40 (4), pp. 1–14.

Cited by: Design Optimization in Fluids.

Matthews et al. (2025)
M. Matthews, M. Beukman, C. Lu, and J. Foerster

Kinetix: investigating the training of general agents through open-ended physics-based control tasks.

In International Conference on Learning Representations,

Cited by: Introduction.

Miconi (2008)
T. Miconi

Evosphere: evolutionary dynamics in a population of fighting virtual creatures.

In 2008 IEEE Congress on evolutionary computation (IEEE World congress on computational intelligence),

pp. 3066–3073.

Cited by: Artificial Life Simulators.

Miikkulainen and Forrest (2021)
R. Miikkulainen and S. Forrest

A biological perspective on evolutionary computation.

Nature Machine Intelligence 3 (1), pp. 9–15.

Cited by: Encoding Schemes and Search Methods.

Mittal and Iaccarino (2005)
R. Mittal and G. Iaccarino

Immersed boundary methods.

Annu. Rev. Fluid Mech. 37 (1), pp. 239–261.

Cited by: Fluid-filament Interaction.

Mohr (2023)
T. Mohr

How particle life emerges from simplicity.

Note: https://www.youtube.com/watch?v=p4YirERTVF0Accessed: 2026-04-10

Cited by: Artificial Life Simulators,
Scalability.

Mordvintsev et al. (2022)
A. Mordvintsev, E. Niklasson, and E. Randazzo

Particle lenia and the energy-based formulation.

Note: https://google-research.github.io/self-organising-systems/particle-lenia/

Cited by: Artificial Life Simulators.

Mouret and Clune (2015)
J. Mouret and J. Clune

Illuminating search spaces by mapping elites.

arXiv preprint arXiv:1504.04909.

Cited by: QD Search for Locomotion Strategies.

Müller et al. (2007)
M. Müller, B. Heidelberger, M. Hennix, and J. Ratcliff

Position based dynamics.

Journal of Visual Communication and Image Representation 18 (2), pp. 109–118.

Cited by: Filaments.

Najarro et al. (2022)
E. Najarro, S. Sudhakaran, C. Glanois, and S. Risi

HyperNCA: growing developmental networks with neural cellular automata.

arXiv preprint arXiv:2204.11674.

Cited by: Encoding Schemes and Search Methods.

Naughton et al. (2021)
N. Naughton, J. Sun, A. Tekinalp, T. Parthasarathy, G. Chowdhary, and M. Gazzola

Elastica: a compliant mechanics environment for soft robotic control.

IEEE Robotics and Automation Letters 6 (2), pp. 3389–3396.

External Links: Document

Cited by: Design Optimization in Fluids.

Nedelec and Foethke (2007)
F. Nedelec and D. Foethke

Collective langevin dynamics of flexible cytoskeletal fibers.

New Journal of Physics 9 (11), pp. 427–427.

Cited by: Introduction.

Ofria and Wilke (2004)
C. Ofria and C. O. Wilke

Avida: a software platform for research in computational evolutionary biology.

Artificial Life 10 (2), pp. 191–229.

External Links: Document

Cited by: Artificial Life Simulators.

Peskin (2002)
C. S. Peskin

The immersed boundary method.

Acta numerica 11, pp. 479–517.

Cited by: Fluid-filament Interaction,
footnote 1.

Pugh et al. (2016)
J. K. Pugh, L. B. Soros, and K. O. Stanley

Quality diversity: a new frontier for evolutionary computation.

Frontiers in Robotics and AI 3, pp. 40.

External Links: Document

Cited by: QD Search for Locomotion Strategies.

Purcell (1977)
E. Purcell

Life at low reynolds number.

American Journal of Physics 45 (1).

Cited by: Introduction,
Bio-Inspiration.

Qian et al. (1992)
Y. H. Qian, D. d’Humières, and P. Lallemand

Lattice BGK models for Navier-Stokes equation.

Europhysics Letters 17 (6), pp. 479–484.

Cited by: Fields.

Ray (1991)
T. S. Ray

Evolution and optimization of digital organisms.

Scientific Excellence in Supercomputing.

Cited by: Introduction,
Artificial Life Simulators.

Reynolds (1987)
C. W. Reynolds

Flocks, herds and schools: a distributed behavioral model.

In Proceedings of the 14th annual conference on Computer graphics and interactive techniques,

pp. 25–34.

Cited by: Artificial Life Simulators,
Computer Graphics.

Richmond et al. (2023)
P. Richmond, R. Chisholm, P. Heywood, M. K. Chimeh, and M. Leach

FLAME gpu 2: a framework for flexible and performant agent based simulation on gpus.

Software: Practice and Experience 53 (8), pp. 1659–1680.

Cited by: Artificial Life Simulators.

Richmond et al. (2010)
P. Richmond, D. Walker, S. Coakley, and D. Romano

High performance cellular level agent-based simulation with flame for the gpu.

Briefings in bioinformatics 11 (3), pp. 334–347.

Cited by: Artificial Life Simulators.

Sims (1994)
K. Sims

Evolved virtual creatures.

In ACM SIGGRAPH,

Cited by: Artificial Life Simulators,
Computer Graphics.

Soros and Stanley (2014)
L. Soros and K. Stanley

Identifying necessary conditions for open-ended evolution through the artificial life world of chromaria.

In Artificial Life Conference Proceedings,

pp. 793–800.

Cited by: Artificial Life Simulators.

Stanley and Miikkulainen (2002)
K. O. Stanley and R. Miikkulainen

Evolving neural networks through augmenting topologies.

Evolutionary Computation 10 (2), pp. 99–127.

External Links: Document

Cited by: Neuroevolution of Controller Circuit.

Stanley (2007)
K. O. Stanley

Compositional pattern producing networks: a novel abstraction of development.

Genetic Programming and Evolvable Machines 8 (2), pp. 131–162.

External Links: Document

Cited by: Encoding Schemes and Search Methods,
Neuroevolution of Controller Circuit.

Stuyck and Chen (2023)
T. Stuyck and H. Chen

DiffXPBD: differentiable position-based simulation of compliant constraint dynamics.

Proceedings of the ACM on Computer Graphics and Interactive Techniques 6 (3).

External Links: Document

Cited by: Design Optimization in Fluids.

Suarez et al. (2019)
J. Suarez, Y. Du, P. Isola, and I. Mordatch

Neural mmo: a massively multiagent game environment for training and evaluating intelligent agents.

arXiv preprint arXiv:1903.00784.

Cited by: Artificial Life Simulators.

Taylor (1951)
G. I. Taylor

Analysis of the swimming of microscopic organisms.

Proceedings of the Royal Society of London. Series A 209 (1099), pp. 447–461.

Cited by: Bio-Inspiration.

Tekinalp et al. (2025)
A. Tekinalp, Y. Bhosale, S. Cui, F. K. Chan, and M. Gazzola

Self-propelling, soft, and slender structures in fluids: cosserat rods immersed in the velocity–vorticity formulation of the incompressible navier–stokes equations.

Computer Methods in Applied Mechanics and Engineering 440, pp. 117910.

Cited by: Design Optimization in Fluids,
Design Optimization in Fluids.

Tian et al. (2011)
F. Tian, H. p. Luo, L. Zhu, J. C. Liao, and X. Lu

An efficient immersed boundary-lattice Boltzmann method for the hydrodynamic interaction of elastic filaments.

Journal of Computational Physics 230 (19), pp. 7266–7283.

External Links: Document

Cited by: Design Optimization in Fluids.

Umetani et al. (2014)
N. Umetani, R. Schmidt, and J. Stam

Position-based elastic rods.

In ACM SIGGRAPH 2014 Talks,

pp. 1–1.

Cited by: Filaments.

Ventrella (2017)
J. Ventrella

Clusters (lifelike particle systems).

Note: https://www.ventrella.com/Clusters/Accessed: 2026-04-10

Cited by: Artificial Life Simulators.

Wang et al. (2025)
L. Wang, M. Zhao, E. Liu, K. Sun, and R. Cheng

TensorNEAT: a gpu-accelerated library for neuroevolution of augmenting topologies.

arXiv preprint arXiv:2504.08339.

Cited by: Neuroevolution of Controller Circuit.

Wang et al. (2008)
Z. Wang, J. Fan, and K. Luo

Combined multi-direct forcing and immersed boundary method for simulating flows with moving particles.

International Journal of Multiphase Flow 34 (3), pp. 283–302.

Cited by: Fluid-filament Interaction.

Whidden (2025)
P. Whidden

Mote: an interactive ecosystem simulation.

Note: https://www.youtube.com/watch?v=Hju0H3NHxVIAccessed: 2026-03-24

Cited by: Artificial Life Simulators.

Zhang et al. (2019)
X. Zhang, F. K. Chan, T. Parthasarathy, and M. Gazzola

Modeling and simulation of complex dynamic musculoskeletal architectures.

Nature Communications 10 (1), pp. 4825.

External Links: Document

Cited by: Design Optimization in Fluids.

 Experimental support, please
 view the build logs
 for errors. Generated by
 

 L
 A
 T
 E

xml

.
 

Instructions for reporting errors
We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
 support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
 methods listed below:

Click the "Report Issue" (

) button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.
Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we
 may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
 should not be a barrier to accessing research. Thank you for your continued support in championing open access for
 all.
Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

 We gratefully acknowledge support from
 our major funders,
 member institutions, ,
 and all contributors.
 

About
·
Help
·
Contact
·
Subscribe
·
Copyright
·
Privacy
·
Accessibility
·
Operational Status (opens in new tab)

Major funding support from

