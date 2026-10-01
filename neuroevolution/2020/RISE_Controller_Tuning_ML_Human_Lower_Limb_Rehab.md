# RISE Controller Tuning and System Identification Through Machine Learning for Human Lower Limb Rehabilitation via Neuromuscular Electrical Stimulation

Héber H. Arcolezi<sup>a,b,\*</sup>, Willian R. B. M. Nunes<sup>c</sup>, Rafael A. de Araujo<sup>b</sup>, Selene Cerna<sup>a</sup>, Marcelo A. A. Sanches<sup>b</sup>, Marcelo C. M. Teixeira<sup>b</sup>, Aparecido A. de Carvalho<sup>b</sup>

<sup>a</sup>*Femto-ST Institute, Univ. Bourgogne Franche-Comté, UBFC, CNRS, Belfort, 90000, France*

<sup>b</sup>*Department of Electrical Engineering, São Paulo State University, UNESP, Ilha Solteira, 15385-000, Brazil*

<sup>c</sup>*Department of Electrical Engineering, Federal University of Technology - Paraná, UTFPR, Apucarana, 86812-460, Brazil*

---

## Abstract

Neuromuscular electrical stimulation (NMES) has been effectively applied in many rehabilitation treatments of individuals with spinal cord injury (SCI). In this context, we introduce a novel, robust, and intelligent control-based methodology to closed-loop NMES systems. Our approach utilizes a robust control law to guarantee system stability and machine learning tools to optimize both the controller parameters and system identification. Regarding the latter, we introduce the use of past rehabilitation data to build more realistic data-driven identified models. Furthermore, we apply the proposed methodology for the rehabilitation of lower limbs using a control technique named the robust integral of the sign of the error (RISE), an offline improved genetic algorithm optimizer, and neural network models. Although in the literature, the RISE controller presented good results on healthy subjects, without any fine-tuning method, a

---

\*Corresponding author

*Email addresses:* `heber.hwang_arcolezi@univ-fcomte.fr` (Héber H. Arcolezi), `willianr@utfpr.edu.br` (Willian R. B. M. Nunes), `rafael.araujo@unesp.br` (Rafael A. de Araujo), `selene_leya.cerna_nahuis@univ-fcomte.fr` (Selene Cerna), `marcelo.sanches@unesp.br` (Marcelo A. A. Sanches), `marcelo.minhoto@unesp.br` (Marcelo C. M. Teixeira), `aa.carvalho@unesp.br` (Aparecido A. de Carvalho)trial and error approach would quickly lead to muscle fatigue for individuals with SCI. In this paper, for the first time, the RISE controller is evaluated with two paraplegic subjects in one stimulation session and with seven healthy individuals in at least two and at most five sessions. The results showed that the proposed approach provided a better control performance than empirical tuning, which can avoid premature fatigue on NMES-based clinical procedures.

*Keywords:* Neuromuscular electrical stimulation, Spinal cord injury, RISE controller, Knee joint, Machine learning.

---

## 1. Introduction

Neuromuscular electrical stimulation (NMES) and functional electrical stimulation (FES) have been effectively applied in many rehabilitation treatments for people with spinal cord injury (SCI) in the past years. Damages in the spinal cord may be engendered by traumatic causes such as road accidents, sports injuries, and violence, or nontraumatic ones such as diseases and tumors. Spinal cord injury is commonly a permanent cause, which can generate issues such as loss of bodily perception, difficulties related to sexual functions, partial or total paralysis, and severe pain (Ho et al. (2014); Lynch & Popovic (2008); Popović (2014); Wagner et al. (2018)). However, the main consequences depend on several factors, such as the patient's personal condition, the level of the lesion and its damages, the availability of time and resources, and socioeconomic factors. For instance, in low-income countries, SCI normally leads to death, whereas in high-income countries, people with SCI enjoy a better and more productive life (Bickenbach (2013)).

The application of NMES/FES for SCI rehabilitation is one of the most frequently used methods (Marquez-Chin & Popovic (2020); Kapadia et al. (2020)). It provides many health and social benefits to patients; for example, it helps to preserve and recover muscle strength and prevent flaccidity and hypotrophy, which are evidence of muscle inactivity; it also offers higher expectation and quality of life, and facilitates social reinsertion (Peckham & Knutson(2005); Marquez-Chin & Popovic (2020); Lynch & Popovic (2008)). Moreover, NMES/FES are techniques based on the use of equipment that generates electrical signals for muscle stimulation at the motor level. More specifically, the aim is generating a muscle contraction via electrodes placed superficially or intramuscularly. The electrical stimulation consists of applying a pulsed current or voltage signal that can depolarize neurons above the activation threshold. The amplitude, pulse width (PW), frequency, and shape of the pulse determine which neurons are recruited. Muscle control can be realized by amplitude, PW, or frequency modulation (Lynch & Popovic (2012); Popović (2014)).

Even though there are several investigations on the closed-loop control of NMES/FES systems for lower limb rehabilitation (*cf.* Ferrarin et al. (2001); Previdi & Carpanzano (2003); Jezernik et al. (2004); Cheng et al. (2016); Mohammed et al. (2012); Wu et al. (2017); Hmed et al. (2017); Sharma et al. (2012); Gaino et al. (2017); dos Santos et al. (2015); Nunes et al. (2019); Teodoro et al. (2020); Müller et al. (2017); Page & Freeman (2020); Gaino et al. (2020) and the references within), these systems are hardly put into production. Alternatively, there exist commercial stimulators normally available on open-loop designs and with pre-programmed electrical stimulation, which are not adequate to deal with the nonlinear and time-varying nature of muscles (Page & Freeman (2020); Lynch & Popovic (2012)). Hence, given the numerous challenges in the design of automatic stimulation strategies, further investigation is needed in this field. For example, control strategies are needed to compensate for modeling errors on the plant, system faults, individual's muscles behavior, and inter/intra-subject variability in muscle properties (Sharma et al. (2009, 2012); Yu et al. (2013, 2015)). The variability in muscle properties leads to the difficulty of predicting the exact contraction force exerted by the muscle, which results in unknown mapping between the stimulation parameters and the muscle force.

In this sense, the design and evaluation of the robust integral of the sign of the error (RISE) control (Xian et al. (2003); Xian et al. (2004)) for tracking the nonlinear dynamics of electrically stimulated lower limbs are presented in this paper. Despite several control laws investigated in the literature, this studyconsiders RISE control law by some fundamental characteristics, such as the consideration of unmodeled disturbances and uncertainties in the plant. Nevertheless, adjusting the controller parameters is the main component to guarantee high-quality control performance; that is, the method can only guarantee good responses (semi-global asymptotic stability), appropriately selecting the gain constants.

Stegath et al. (2007, 2008) and Sharma et al. (2009) are pioneers authors on RISE controller development for the lower limb tracking control. Afterward, Sharma et al. (2012) presented an improvement of RISE control method for the same application using a feedforward neural network (NN) term. Downey et al. (2013) and Downey et al. (2015) developed an RISE controller for the asynchronous stimulation to the lower limb. Kawai et al. (2014) simulated the tracking control performance of an RISE-based controller to model the co-contraction control of the human lower limb. Kushima et al. (2015) modeled an FES knee bending and stretching system, and developed an RISE-based controller to stimulate the quadriceps and hamstring muscle groups. In the similar context of NMES, but for upper limbs, Lew et al. (2016) implemented RISE controller for the rehabilitation of post-stroke individuals.

Even though previous investigations for this problem with RISE controller presented good results without any fine-tuning method, the motivation of this paper is the absence of clever algorithms to properly select the gain constants of RISE controller. In the aforementioned studies, the authors did not show the controller tuning method or empirical approach (pretrial tests) for defining gain parameters before the real experiments are conducted. In addition, experiment validations were made only on healthy individuals; however, the muscles of people with SCI are not as strong as healthy muscles (Mohammed et al. (2012); Lynch & Popovic (2012)).

More specifically, Stegath et al. (2008), Sharma et al. (2009), and Downey et al. (2015) present four inequalities to gain constants, which are sufficient conditions to guarantee semi-global asymptotic stability for an uncertain nonlinear muscle model. There are infinite combinations of gains in  $\mathbb{R}^+$  that satisfy theseinequalities; yet, as presented in the aforementioned works, a “trial and error” methodology might be feasible to set gain constants to the controller for healthy subjects. However, this procedure must be reconsidered when treating people with SCI to avoid some common problems. For instance, for SCI rehabilitation via NMES/FES, there might exist rapid muscle fatigue, muscle tremors due to incomplete tetanus, and harsh muscle spasms (Ho et al. (2014); Lynch & Popovic (2012); Popović (2014); Peckham & Knutson (2005)).

Therefore, to overcome the aforementioned problems, this paper proposes a novel robust and intelligent control-based methodology for NMES/FES systems. More precisely, we aim to overcome the empirical tuning technique for clinical procedures using RISE controller, as observed in the literature. Moreover, this study proposes to extend the analysis of RISE controller to individuals with SCI that do not present ideal conditions as healthy individuals. The proposed methodology includes an identification step based on machine learning (ML) black-box models with the novelty of using past identification and control data for each patient, a robust control law (e.g., RISE technique) to guarantee the semi-global asymptotic stability, and an ML-based offline controller optimizer.

In Arcolezi et al. (2019), our group proposed an offline improved genetic algorithm (IGA) optimizer to RISE controller. Simulations were performed using a nonlinear mathematical model of the knee joint for three paraplegics and one healthy individual. In this study, our proposed methodology is implemented and evaluated with seven healthy and two paraplegic individuals using RISE control law, the aforementioned IGA optimizer, and NN black-box models.

The first hypothesis in this paper is that using an empirical approach to clinical procedures would present a large number of poor performances, while a more adequate tuning with a more representative identified model can provide better tracking control of the lower limb. That is, we assume no background knowledge with the RISE controller for clinicians intending to design and apply it to real-life scenarios. The second hypothesis is that by using past rehabilitation data for identifying an individual, this model will improve the description of the relationship between the angular position and the delivered electrical stim-ulation, whereby fatigue and other problems as tremors are already implicit in the data.

The remaining sections of this paper are organized as follows: Section 2 presents the theoretical background; Section 3 introduces the proposed control-based methodology and the materials and methods used in the experiments; Section 4 presents the results and its analysis; and finally, Section 5 provides the conclusions of this paper and future works.

## 2. Theoretical Background

In this section, we briefly present the musculoskeletal dynamics about the knee joint (Subsection 2.1) and RISE control method (Subsection 2.2). We summarize the IGA for the optimization procedure (Subsection 2.3), and we discuss nonlinear system identification via NN models (Subsection 2.4).

### 2.1. System dynamics

The musculoskeletal dynamics based on electrical stimulation is given as (Ferrarin et al. (2001); Sharma et al. (2009))

$$J\ddot{\theta}(t) = \Lambda_g(\theta(t)) + \Lambda_e(\theta(t)) + \Lambda_v(\dot{\theta}(t)) + \Lambda_d(t) + \Lambda_{es}(t), \quad (1)$$

where  $J \in \mathbb{R}$  is the unknown inertia of the combined shank and foot;  $\theta(t)$ ,  $\dot{\theta}(t)$ ,  $\ddot{\theta}(t) \in \mathbb{R}$  is the angular position, velocity and acceleration, respectively.

The gravitational component  $\Lambda_g(\theta(t)) \in \mathbb{R}$  is expressed as

$$\Lambda_g(\theta(t)) = -mgl \sin(\theta(t)), \quad (2)$$

where  $m \in \mathbb{R}$  denotes the unknown combined mass of the shank and foot;  $l \in \mathbb{R}$  is the unknown length between the knee-joint and center of mass of the shank and foot; and  $g \in \mathbb{R}$  is the gravitational acceleration.

The elastic effects due to joint stiffness  $\Lambda_e(\theta(t)) \in \mathbb{R}$  can be modeled as

$$\Lambda_e(\theta(t)) = -(\psi_1\theta(t) - \psi_1\psi_3) \left( e^{-\psi_2\theta(t)} \right), \quad (3)$$where  $\psi_1, \psi_2, \psi_3 \in \mathbb{R}$  are unknown positive coefficients.

The viscous effects due to damping  $\Lambda_v(\dot{\theta}(t)) \in \mathbb{R}$  is defined as

$$\Lambda_v(\dot{\theta}(t)) = -\kappa_1 \tanh(-\kappa_2 \dot{\theta}(t)) + \kappa_3 \dot{\theta}(t), \quad (4)$$

where  $\kappa_1, \kappa_2, \kappa_3 \in \mathbb{R}$  are unknown positive constants.

The torque produced at the knee joint by the electrical stimulation  $\Lambda_{es}(t) \in \mathbb{R}$  is related to the positive moment  $\varsigma(\theta(t)) \in \mathbb{R}$  from the extension and flexion of the leg, the unknown nonlinear function  $\nu(\theta, \dot{\theta}) \in \mathbb{R}$  corresponding to muscle tendon force, and the electrical potential  $u(t) \in \mathbb{R}$  applied to the quadriceps muscle:

$$\Lambda_{es}(t) = \varsigma(\theta(t))\nu(\theta(t), \dot{\theta}(t))u(t). \quad (5)$$

Finally,  $\Lambda_d(t) \in \mathbb{R}$  is the unmodeled bounded disturbances (e.g., fatigue, spasms, tremor, and delay).

## 2.2. RISE-based control

RISE control method proposed by Xian et al. (2003); Xian et al. (2004) utilizes a continuous and high gain control signal, which guarantees semi-global asymptotic stability considering bounded smooth external disturbances and bounded modeling uncertainties. The use of the integral of the sign of the error in RISE technique minimizes the commonly chattering problem seen in sliding-mode controllers. To achieve the stated control objective, i.e., to enable the lower limb to track a desired angular trajectory despite external disturbances and modeling uncertainties, a position tracking error denoted by  $e_1(t) \in \mathbb{R}$ , is defined as

$$e_1(t) = \theta_d(t) - \theta(t), \quad (6)$$

where  $\theta_d(t)$  is the angular trajectory to be tracked with the premise of having bounded continuous-time derivatives, and  $\theta(t)$  is the real angular position. Furthermore, to facilitate the control design, filtered tracking errors  $e_2(t) \in \mathbb{R}$  and$r(t) \in \mathbb{R}$  are defined as

$$e_2(t) = \dot{e}_1(t) + \alpha_1 e_1(t), \quad (7)$$

$$r(t) = \dot{e}_2(t) + \alpha_2 e_2(t), \quad (8)$$

where  $\alpha_1, \alpha_2 \in \mathbb{R}$  are positive and selectable control gains.

Multiplying (8) by  $J$ , and considering (1)-(7),  $\dot{e}_2 = \ddot{\theta}_d(t) + \alpha_1 \dot{e}_1 - \ddot{\theta}(t)$ , one obtains

$$Jr = \Upsilon(\dot{\theta}_d, \dot{\theta}, \theta, \dot{e}_1, e_2) - \Psi(\dot{\theta}, \theta)u - \Lambda_d, \quad (9)$$

where  $\Upsilon(\dot{\theta}_d, \dot{\theta}, \theta, \dot{e}_1, e_2) \in \mathbb{R}$  defined as

$$\Upsilon(\dot{\theta}_d, \dot{\theta}, \theta, \dot{e}_1, e_2) = \ddot{\theta}_d + \alpha_1 \dot{e}_1 + \alpha_2 e_2 - \Lambda_g(\theta) - \Lambda_e(\theta) - \Lambda_v(\dot{\theta}),$$

and  $\Psi(\theta, \dot{\theta}) \in \mathbb{R}$  a function monotonic and bounded, expressed as

$$\Psi(\dot{\theta}, \theta) = \varsigma(\theta)\nu(\theta, \dot{\theta}).$$

For stability analysis, from (9) can be determined the open-loop error system

$$\mathcal{J}_\Psi r = \mathcal{Y}_\Psi - u - \mathcal{L}_\Psi,$$

where  $\mathcal{J}_\Psi = \Psi^{-1}J$ ,  $\mathcal{Y}_\Psi = \Psi^{-1}\Upsilon$ , and  $\mathcal{L}_\Psi = \Psi^{-1}\Lambda_d$ , and consequently one obtains

$$\mathcal{J}_\Psi \dot{r} = -\dot{u} - e_2 + \tilde{\mathcal{W}} + \mathcal{W}_d,$$

where  $\tilde{\mathcal{W}} = \mathcal{W} - \mathcal{W}_d$ ,  $\tilde{\mathcal{W}}(e_1, e_2, r, t) \in \mathbb{R}$ ,  $\mathcal{W} \in \mathbb{R}$  corresponds to the term

$$\mathcal{W} = -\frac{1}{2}\dot{\mathcal{J}}_\Psi r + \dot{\mathcal{Y}}_\Psi - \dot{\mathcal{L}}_\Psi + e_2,$$

and  $\mathcal{W}_d \in \mathbb{R}$  expressed as

$$\mathcal{W}_d = \dot{\mathcal{J}}_\Psi \ddot{\theta}_d + \mathcal{J}_\Psi \ddot{\ddot{\theta}}_d - \dot{\Lambda}_e - \dot{\Lambda}_g - \dot{\Lambda}_v - \dot{\Lambda}_d.$$Based on the mean value theorem applied to upper bound  $\|\tilde{\mathcal{W}}\| \leq \varsigma(\|\zeta\|)\|\zeta\|$ , where  $\zeta \in \mathbb{R}^3$ ,  $\zeta = [r^T \ e_1^T \ e_2^T]^T$ ,  $\varsigma(\|\zeta\|) \in \mathbb{R}$  is a positive globally invertible nondecreasing function, and considering that  $\theta_d$ , and its derivatives  $\theta_d^{(k)} \in \mathcal{L}_\infty, \forall k \in \mathbb{I} = \{1, 2, 3, 4\}$ , the following constraints can be established  $\|\mathcal{W}_d\| \leq \mathcal{E}_{\mathcal{W}_d}$ ,  $\|\dot{\mathcal{W}}_d\| \leq \mathcal{E}_{\dot{\mathcal{W}}_d}$ , such as  $\mathcal{E}_{\mathcal{W}_d}, \mathcal{E}_{\dot{\mathcal{W}}_d} \in \mathbb{R}$  are positive constants (Utkin (2013)).

Note that the system error equations obtained to nonlinear dynamic model are similar to other studies with the RISE controller in (Sharma et al. (2009); Stegath et al. (2008); Patre et al. (2008); Makkar et al. (2007); Xian et al. (2004); Xian et al. (2003)). Based on the open-loop error system, the control input  $u(t) \in \mathbb{R}$ , is designed as

$$u(t) = (k_s + 1)e_2(t) - (k_s + 1)e_2(0) + \int_0^t [(k_s + 1)\alpha_2 e_2(\tau) + \beta sgn(e_2(\tau))]d\tau, \quad (10)$$

where  $k_s, \beta \in \mathbb{R}$  also represents positive and adjustable control gains,  $u(t)$  is the control signal, and  $sgn(\cdot)$  is the known signum function.

The RISE controller, given in (10), ensures that all system signals are bounded under closed-loop operation and the position tracking error is regulated in sense that

$$\lim_{t \rightarrow \infty} \|e_1(t)\| \rightarrow 0,$$

yields semi-global asymptotic stability provided the control gain  $k_s$  sufficiently large, and  $\beta$  satisfying the following sufficient condition

$$\beta > \mathcal{E}_{\mathcal{W}_d} + \frac{1}{\alpha_2} \mathcal{E}_{\dot{\mathcal{W}}_d}, \quad (11)$$

where  $\mathcal{E}_{\mathcal{W}_d}, \mathcal{E}_{\dot{\mathcal{W}}_d} \in \mathbb{R}$  are known positive constants. More details about the stability analysis of the RISE method can be found in (Patre et al. (2008); Makkar et al. (2007); Xian et al. (2004)).

The ideal first derivative of the error  $H(s) = \frac{Y(s)}{U(s)} = \frac{sE(s)}{E(s)} = s$  is an improper function, that is,  $H(s) = \frac{\sum_{j=0}^m b_j s^j}{\sum_{i=0}^n a_i s^i}$ ,  $a_i, b_j \in \mathbb{R}, \forall i = 1, 2, \dots, n, \forall j = 1, 2, \dots, m, m > n, |H(\infty)| = \infty$ . The unfeasibility of practical implementation using the ideal derivative is solved by a filtered derivative (Khadraet al. (2016)). Thus, the filtered tracking error is calculated by

$$H(s) = \frac{Y(s)}{U(s)} = \frac{s}{\tau s + 1}, \quad (12)$$

where  $\tau$  is the time constant between the signal and its derivative. Note that (12) is a low pass filter (LPF) that attenuates high-frequency noises.

### 2.3. Improved genetic algorithm

The IGA was introduced in Arcolezi et al. (2019) to optimize the gains parameters of RISE controller for a representative model of an individual. This algorithm is summarized in this paper. First, there is a pre-processing stage for bounding the gain limits to efficiently initiate (i.e., the random initial population within the constraints of stability) and maintain the search (i.e., genetic operators such as recombination, mutation, and replacement operator). Second, a simple fast genetic algorithm (FGA) is used in the construction phase to generate a good initial population. Thereafter, a complete genetic algorithm (CGA) is applied to improve the quality of this population and hence achieve a global (or local) minimum.

Figure 1 describes the FGA with a flow chart. In the chart,  $N_p$  is the size of the initial population (small),  $M_r$  is the mutation rate, and the stopping criterion is the number of generations  $N_g$ . More specifically,  $N_g$  represents the size of the real initial population (RIP) to initiate the local search phase. The CGA is similar to the FGA, with a more stringent test to the replacement operator. We recommend that readers refer to (Arcolezi et al. (2019)) for a more descriptive version of the algorithm.

### 2.4. System identification via neural networks

Nonlinear systems identification and modeling have been applied in most areas of science to predict the future behavior of dynamic systems. System identification has been an active field in control theory, and it is an important approach to explore, study, and understand the world by a formal description of events as a model. The use of NNs to identify nonlinear systems has been a```

graph TD
    Start([Start]) --> GenInd[Randomly generate Np individuals]
    GenInd --> AssignVal[Assign objective value to each individual]
    AssignVal --> SelectInd[Select two individuals via tournament selection]
    SelectInd --> Recomb[Recombination via single-point crossover]
    Recomb --> RandCheck{rand > Mr}
    RandCheck -- No --> Replace[Apply replacement operator]
    RandCheck -- Yes --> Mutate[Apply mutation operator]
    Mutate --> Replace
    Replace --> StopCheck{Stopping criterion}
    StopCheck -- No --> GenInd
    StopCheck -- Yes --> End([End])
  
```

Figure 1: Flow chart of the fast genetic algorithm.

prospective direction since previous research presented in (Hornik et al. (1989); Chen et al. (1990); Narendra & Parthasarathy (1990); Chu et al. (1990)), for example. In the following, the use of NNs for the identification of discrete dynamic system is briefly described.

The construction of black-box models is essentially based on the quality of measured data about the system. The fundamental concept of this approach is to model the direct input-output relationship, i.e., identifying and modeling just with data, in which the main objective is to find the weights and other coefficients (known as hyperparameters) of the NN. Moreover, NNs are based on a collection of inter-connected units named neurons. These neurons are structured into three or more layers, input, hidden(s), and output. Neuralnetworks are in the core of deep learning (several neurons and hidden layers) and have become a progressively popular research topic. Generally, NNs can be divided into two large classes: feedforward and recurrent NNs.

Fundamentally, an operator  $F$  from an input space  $\mathbb{U}$  to an output space  $\mathbb{Y}$  expresses the model of the system to be identified, where the goal is to find a function  $\hat{F}$  that approximates  $F$  to a specific requirement. By the Stone-Weierstrass theorem, there exists a continuous and bounded function  $F$ , that can be uniformly approximated as closely as desired by a polynomial function  $\hat{F}$ . Furthermore, according to the universal approximation theorem, there exists a combination of hyperparameters of an NN that allows it to identify and learn any continuous nonlinear function defined on a closed interval (Hornik et al. (1989)).

Consider a single-input and single-output discrete system structure with only the input and output data available:

$$y(k) = F[y(k-1), \dots, y(k-n); u(k-1), \dots, u(k-m)], \quad (13)$$

where  $F(\cdot)$  is an unknown nonlinear difference equation that represents the plant dynamics;  $u$  and  $y$  are measurable scalar input and output, respectively; and  $m$  and  $n$  are the maximum lags for the system output and input; that is, they are the last values of the input and output respectively. In short, the next value of the dependent output signal  $y(k)$  is regressed on previous values of the output and input signals.

The identification for the discrete-time system in (13) can be performed by the following two major types of identification structures presented in the literature: the parallel and the series-parallel identification model (Narendra & Parthasarathy (1990)). The first structure depends on past inputs of the system and the outputs of the NN model. The second structure uses both past inputs and system's outputs. Mathematically, these models are respectively describedas

$$\hat{y}(k) = \hat{F}[\hat{y}(k-1), \dots, \hat{y}(k-n); u(k-1), \dots, u(k-m)], \quad (14)$$

$$\hat{y}(k) = \hat{F}[y(k-1), \dots, y(k-n); u(k-1), \dots, u(k-m)], \quad (15)$$

where  $\hat{y}$  is the model output;  $y$  is the real system output;  $\hat{F}$  is the model structure; and  $m$  and  $n$  are the regression orders for the input and output, respectively. These last two parameters are chosen before the identification process, where  $n$  is the output memory to indicate how many past steps of output will be used in the system identification, and  $m$  refers to the time-step of input values and it is the longest memory that a model can store. In this paper, we used a feedforward NN (multilayer perceptron - MLP) to approximate the nonlinear mapping function  $F(\cdot)$  in (13) using the series parallel structure in (15).

### 3. Materials and Methods

In this section, we first present an overview of our proposed methodology (Subsection 3.1). Next, we provide information on the volunteering participants (Subsection 3.2) and on the instrumentation used for real experiments (Subsection 3.3). Lastly, we describe how we applied the proposed methodology in this study for both, data acquisition and experimental procedures (Subsection 3.4).

#### 3.1. Proposed methodology

Fig. 2 illustrates an overview of the proposed methodology. In the first session of a new patient (no previous data), a stimulation test is performed to acquire information on the relationship between delivered electrical stimulation and the achieved angular position. The acquired data are appropriately treated to pass through an identification step via NN black-box models. Once this relationship is efficiently mapped as a model, a simulation process is initiated using clever algorithms. The aim is to minimize a well-defined objective functionto adequately set-up the gains of RISE controller for the patient. Therefore, to finalize the first session, the rehabilitation procedure is retaken with fine-tuned gains for a better control-stimulation session. This could prevent premature fatigue and other unwanted factors that would be present for people with SCI by not choosing an appropriate gain combination.

In future sessions, all data (system identification and control evaluation) from previous rehabilitation sessions are used for training an NN model in an offline scheme. That is, before each (next) session, all data from a patient are combined to a single dataset and used to map the relationship between angular position and electrical stimulation. Thus, the same optimization process using the trained model provides fine-tuned gain parameters to be afterward applied to the rehabilitation procedure. The gains of the controller are found using only past rehabilitation data, which is motivated by the belief that preliminary electrical stimulation could lead to quick muscle fatigue during the real clinical procedure. Moreover, between stimulation sessions, there exist factors such as fatigue, hydration, evolution/gain of strength, rest, and therapeutic sessions, which might influence one's response to NMES/FES and make the control-stimulation inefficiently.

The use of NNs is motivated by the advantages of these methods for the nonlinear system identification problem and by the high power for computation and storage of data encountered nowadays. Regarding the identification step, the novelty of the proposed methodology is the use of past rehabilitation data. The primary purpose is to build up a dataset for each patient, where the number of data will increase during rehabilitation sessions, and the identified model will improve with more data and details about the nonlinear muscular behavior. As highlighted in the literature, muscular behavior is susceptible to parametric variation between one day and another, for instance, the evolution and gain of strength due to previous rehabilitation sessions.

Moreover, one of the primary advantages of performing simulations for an NMES-based knee extension is the liberty of studying this problem from different perspectives and divergent levels of abstraction with the acquired data.The diagram illustrates a three-stage methodology for robust and intelligent control-based rehabilitation:

- **1) SYSTEM IDENTIFICATION:** A neuromuscular electrical stimulation setup with surface electrodes provides data to a neural network. The network consists of an input layer, hidden layer(s), and an output layer. The output is used to increment data through sessions.
- **2) OFF-LINE CONTROLLER OPTIMIZER (SIMULATION LOOP):** A simulation loop where a RISE Controller and an Identified System interact. The error  $e_t$  is calculated from the desired output  $\theta_d^*$  and the actual output  $\theta_{out}$ . The output  $\theta_{out}$  is used to minimize a cost function  $J$ .
- **3) CONTINUE REHABILITATION PROCEDURE:** The fine-tuned parameters from the optimization loop are used in the rehabilitation procedure with surface electrodes. The resulting data is then used to increment data through sessions.

Figure 2: The proposed robust and intelligent control-based methodology.

While the application of NMES to humans presents limitations due to muscle fatigue, which restricts the number of experiments, simulation provides numerous executions to better study the feasibility and practicality of the designed system. Moreover, simulation supplies continuous feedback to continuously improve the system (Jezernik et al. (2004)).

### 3.2. Analyzed individuals

The study with volunteers was authorized through a research ethics committee involving human beings (CAAE: 79219317.2.1001.5402) at São Paulo State University (UNESP). Written informed consent was obtained from all participants before their participation. In this study, seven healthy individuals (male, aged 22-28) labeled as H1-H7 and two male individuals with SCI, labeled as P1 and P2, participated in the experiments. Table 1 presents information on the two SCI individuals, including age, injury data, and ASIA (American Spinal Injury Association) Impairment Scale (AIS).Table 1: Specific data on individuals with SCI.

<table border="1">
<thead>
<tr>
<th>Individual</th>
<th>Age (years)</th>
<th>Injury level</th>
<th>Injury time</th>
<th>AIS</th>
</tr>
</thead>
<tbody>
<tr>
<td>P1</td>
<td>32</td>
<td>L4, L5</td>
<td>9 years</td>
<td>B</td>
</tr>
<tr>
<td>P2</td>
<td>43</td>
<td>C5, C6</td>
<td>17 years</td>
<td>C</td>
</tr>
</tbody>
</table>

### 3.3. Instrumentation

Fig. 3 illustrates the test platform used for conducting the experiments at the Instrumentation and Biomedical Engineering Laboratory (“Laboratório de Instrumentação e Engenharia Biomédica - LIEB”) at UNESP - Ilha Solteira. The platform was composed of an NI (National Instruments<sup>®</sup>, USA) myRIO controller to operate in real time; a current-based neuromuscular electrical stimulator; an instrumented chair composed of an electrogoniometer NIP 01517.0001 (Lynx<sup>®</sup>, São Paulo, Brazil), a gyroscope LPR510AL (ST Microelectronics<sup>®</sup>, Switzerland), two triaxial accelerometers MMA7341 (Freescale<sup>®</sup>, USA); and two user interfaces developed in LabVIEW<sup>®</sup>, one for identification and the other for controlling.

The neuromuscular electrical stimulator delivers rectangular, biphasic, symmetrical pulses to the individual’s muscle, allowing a control adjustment of the PW in a range of 0 – 400 $\mu$ s. We controlled the stimulation intensity by setting the pulse amplitude to the quadriceps and controlling the PW. In this study, we fixed the following parameters: stimulation frequency at 25 Hz (constant frequency train - CFT technique) and pulse amplitude at 80 mA for healthy individuals and 120 mA for paraplegic ones. The difference in pulse amplitude occurred due to insufficient contractions using amplitude below 120 mA for the paraplegic individuals and their respective muscular atrophy conditions. Lastly, we used surface electrodes with rectangular self-adhesive CARCI 50 mm x 90 mm settings.Figure 3: Test platform for electrical stimulation experiments.

### 3.4. Data acquisition and experimental procedure

The chair backrest and the knee joint position were adjusted to ensure the volunteers' comfort. Each individual had a different knee angular position in the resting condition. The angular position in this condition was measured and taken as an offset during the experimental protocol. A muscle analysis was conducted to determine the motor point and guarantee the proper positioning of the surface electrodes. More precisely, the electrophysiological procedurefor identifying the motor point consists of mapping the muscle surface using a stimulation electrode to identify the skin area above the muscle, where the motor threshold is the lowest for a given electrical current; this skin area is the most responsive to electrical stimulation (Gobbo et al. (2014)). After this procedure, the electrodes were properly positioned allowing the neuromuscular electrical stimulation to maximize the effectiveness of the evoked voltage, minimizing the intensity of the injected current and the level of discomfort to the volunteer.

After the motor-point identification, a few open-loop tests were performed by applying a step input during four seconds. It is worth highlighting the definition of the electrical current level of the stimulator, as well as evaluating the PW values for different operating points of the lower limb extension. If the value  $\rho_{max}$  tends to the saturation value of the stimulator, the electrical current amplitude must be increased so that the control system adequately compensates for disturbances and uncertainties in the process. Moreover, the  $\rho_{min}$  is related to the minimum joint extension value from the resting position. In this study, the tests were performed to obtain  $\rho_{max}$  and  $\rho_{min}$  corresponding to  $\theta_{max} = 40^\circ$  and  $\theta_{min} = 10^\circ$ , respectively. Note that we could adopt other values of lower limb extension, but we consider that it was a suitable value for gait control application (Nunes et al. (2019)). Lastly,  $\rho_{min}$  and  $\rho_{max}$  were also useful to select the initial PW and an upper bound to the control signal, respectively. If  $\rho_{max}$  does not approach the saturation value of the stimulator ( $400\mu s$ ), with the consent of each individual, we select an adequate upper bound to the control signal for each stimulation session, aiming to minimize the discomfort level to volunteers.

During the experiments, healthy individuals were instructed to relax, to not influence the leg motion voluntarily, and allow the stimulation to control it. During electrical stimulation sessions, the individuals could deactivate the stimulation pulses using a stop button under any uncomfortable situation (as shown in Fig. 3).

In the following two subsections, the experimental setup is detailed. First, the case of an individual using the proposed methodology for the first time, i.e.,without any previous data, is considered (Subsection 3.4.1). Next, the case of individuals who participate in more than one rehabilitation session is considered (Subsection 3.4.2).

### 3.4.1. *First session*

In the first session of a new patient, a one-minute stimulation test was conducted. In this stimulation, the experimental system identification procedure was performed by randomly applying PW values belonging to the set of values mapped to each individual. The electrical PW random value was constant for a random time between four and seven seconds. Consequently, a new test has randomness in the domain of the PW of the electrical stimulation as well as in the time of each stimulation. In this work, the power of muscle activation by electrical stimulation in paraplegic individuals was greater. Before the tests were performed, these individuals were not admitted to a rehabilitation research program involving daily electrically stimulated exercise of their lower limbs. Consequently, under high stimulation intensity, there was only partial recruitment of synergistic motor units and there was the co-activation of antagonists (Doucet et al. (2012)). Unfortunately, this is a disadvantage of conventional single-electrode stimulation, whose increased stimulation intensity will lead to increased muscle fatigue (Laubacher et al. (2017); Maffiuletti (2010)). To minimize early fatigue in paraplegic individuals (Gregory et al. (2007)), the total test time was reduced to 40 s.

The motivation to adopt this methodology is to map a tracking situation and recognize the completely nonlinear and time-varying nature of muscles under long electrical stimulation time. The PW ( $\mu s$ ) and angular position (rad) data were automatically recorded with a sampling period of 20 ms, i.e.,  $T_s = 0.02$  (s), resulting in datasets with approximately 3000 samples (60 s) at most.

Afterward, the identification data were read and manipulated for feeding up a shallow MLP with one hidden layer. In the literature, one hidden layer has been proved to be sufficient to approximate any continuous function on a compact domain (Hornik et al. (1989); Previdi (2002)). We tuned the numberof neurons via a random search procedure (Bergstra & Bengio (2012)), in which a combination of hyperparameters is randomly selected to find the best solution for the built model. This process was only done for individual H1, which was the first volunteer for this study, and it took less than 30 min to find an appropriate architecture to be used for all other individuals. The number of neurons was selected as 250; hyperbolic tangent activation was used in each neuron from the hidden layer, and the output layer was composed of one neuron with linear activation, which gives the estimated output  $\hat{y}(k)$ .

We experimented with several  $m$  and  $n$  values, and the one with the best time-utility trade-off was  $m = n = 1$ . This resulted in datasets containing the last input value “Pulse\_Width( $k - 1$ )” and the last output value “Angular\_Position ( $k - 1$ )” as features, and the actual output value “Angular\_Position ( $k$ )” as target. The MLP NN model requires a normal input arranged as  $[samples, features]$ , where the observations at previous time-steps are inputted as features to the model. In general, the training time of each NN model in the first session did not exceed 5 min as the number of samples was small ( $\sim 3,000$  for healthy individuals and  $\sim 2,000$  for SCI ones).

Therefore, using the estimated model, we performed an optimization procedure based on the proposed IGA to find the best gains combination for two reference trajectories. The first trajectory is a sinusoidal wave ranging from  $10^\circ$  to  $40^\circ$  and the second trajectory is a  $40^\circ$  step wave ( $30^\circ$  for individuals with SCI); the first and second trajectories simulate isotonic and isometric contractions, respectively. A smooth range of motion at  $40^\circ$  and a small-time period (sine wave) was used to avoid premature fatigue by diminishing muscle effort.

Considering a real-world application of the proposed methodology and by assuming a limited time for a rehabilitation session, we used the following as the initial parameters of the IGA simulations: population size  $N_p = 8$ , mutation rate  $M_r = 0.5$ , number of generations  $N_g = 6$  (size of RIP), and  $k = 1$  iteration. The algorithm ran only once providing  $N_g$  combination of RISE controller gains. Generally, the running time did not exceed 10 min of execution.

Notice that the proposed IGA in Arcolezi et al. (2019) has a pre-processingstep (step 1 of the algorithm), which tries to bound the gain values when applying the genetic operators (crossover, mutation) to the required stability conditions presented in Subsection 2.2. However, there is still a possibility that given an identified model and the optimization procedure that gain values deviate from the required conditions. Yet, as genetic algorithms are population-based, one can compare and select the most appropriate combination of gains for a given individual that satisfies the gain's condition. Before the real experiment, previous simulations of both trajectories were made to visually inspect the system response.

Lastly, using empirical gains and the ones encountered by the IGA, the controlling procedure was implemented for both trajectories. Data were recorded with a sampling period  $Ts = 0.005$  (s), generally resulting in a dataset with approximately 12,000 samples (60 s) at most.

The programming language used in this research was Matlab<sup>®</sup>, both for developing the optimization algorithm and for the system identification procedure via NNs. The simulation system was developed using the Matlab/Simulink<sup>®</sup> platform, which contains both sine and step trajectories, a saturation block to bound the control signal from 0  $\mu$ s to  $\rho_{max}$   $\mu$ s for each individual, the RISE controller block, and the identified NN block for each individual.

#### 3.4.2. *More than one session*

For individuals who participated in more sessions, with at least 48 hours of difference between two consecutive sessions, the one-minute stimulation test (identification step) was not considered, as it was performed when an individual participated for the first time. The data from previous rehabilitation sessions were used to train an NN model in an offline scheme. Before a new session, all data from an individual were combined to a single dataset and used to better map the relationship between angular position and electrical stimulation. In this study, we only used the control data resulting from the control-stimulation sessions with fine-tuned IGA gains, as it would be in real life rather than empirical gains.Thus, using each trained identified model, we performed an optimization procedure based on the IGA to find the best gain combination for both sine and step reference trajectories. As this optimization was performed in an offline scheme and before the next session, time and computational costs were not too strict as they were for the first session. Therefore, the initial parameters of IGA used for simulations were as follows: population size  $N_p = 10$ , mutation rate  $M_r = 0.3$ , number of generations  $N_g = 30$  (size of RIP) and  $k = 1$  iteration. The algorithm ran only once, and several gain combinations from the set of solutions were simulated to check the system response and select the best gain combination for both trajectories. Generally, the total time for both system identification and RISE gains optimization procedures took about 1  $h$  for each individual/session.

For the experimental part, the electrodes were positioned at the motor-point identified in the first session, and similarly, a few open-loop tests, applying a step input during four seconds, were performed, to determine a bounded PW band related to  $\theta_{min} = 10^\circ$  and  $\theta_{max} = 40^\circ$ . Afterward, a small-time interval for muscle rest was provided.

Therefore, knowing the fine-tuned gain parameters for each individual, we applied the controlling procedure for both references, and then employed an empirical gain combination for comparing results.

#### 4. Results and Discussion

In this section, we report the results obtained by applying our proposed methodology in real experiments. During this study, individuals H1-H4 participated in five sessions, H5 in three sessions, and H6-H7 in two sessions. Individuals with SCI participated in only one session due to displacement difficulties. For all individuals, the first session took more time and one additional stimulation than the subsequent ones. This was due to the one-minute stimulation test, and the training/optimization time during the session to find the best gain combination. Before the start of any control-stimulation test, five combina-tions of empirical gains  $(\alpha_1; \alpha_2; k_s; \beta)$  were chosen as  $(1; 2; 30; 5)$ ,  $(0.5; 1; 30; 1.5)$ ,  $(0.8; 1.2; 20.5; 2.5)$ ,  $(5; 2; 15; 3)$ ,  $(4; 7; 25; 8)$  for sessions one to five, respectively. As the system responses to any combination of gains were unknown, they were all chosen at random. Subsections 4.1 and 4.2 present our nonlinear control and system identification results and analysis, respectively. Lastly, we provide a general discussion in Subsection 4.3.

#### 4.1. Control-based NMES results

Figures 4 and 5 illustrate the tracking results on both trajectories and their delivered PWs (Deliv. PWs) for individuals P1 and P2, respectively. Additionally, Table 2 presents control results for the sine wave, comparing the proposed methodology with an empirical tuning for all individuals (*Ind.*) in each session (*Sess.*). The metrics in this table are the root mean square error (RMSE) between the desired and actual knee angles considering the whole period of control-stimulation; and the time of effective control (TEC), which represents how much time in seconds the lower limb was control-stimulated to track the reference angle. When the lower limb did not track the reference angle, the RMSE metric is represented by NC, meaning “not calculated”. More precisely, the TEC metric is the time between the initial control-stimulation until the leg stops tracking the reference angle ( $\pm 5^\circ$  error) for 5 s. In the worst-case, if the leg never tracks the reference angle, NC is assigned.

Similarly, Table 3 presents control results for the step wave, comparing the proposed methodology with an empirical tuning for all individuals in each session. The metrics in this table are the RMSE, TEC, and the averaged and standard deviation (std) values of the knee angular position around the operating point (AvStd. OP) in degrees. The *AvStd. OP* metric will be regarded as an indicator to evaluate the oscillatory behavior during regulation around an operation point ( $40^\circ$  for healthy individuals and  $30^\circ$  for individuals with SCI). For individuals who participated in more than one session, Tables 2 and 3 present the averaged (Avg.) and the std values for both RMSE and TEC metrics, which are calculated considering all sessions of each individual. The symbol (\*) indi-cates there is no std value, as there are more “NC” than real values. Lastly, similar to Figs. 4 and 5, Appendix A provides supplementary illustrations for the tracking results of individuals H1 (session *v*), H2 (session *ii*), and H4 (session *v*), respectively, as well as the fine-tuned gains ( $\alpha_1$ ;  $\alpha_2$ ;  $ks$ ;  $\beta$ ) used for each RISE-based control-stimulation session, in Table A.5.

Figure 4: Experimental results for individual P1 comparing empirical gains and the proposed methodology. The first and second rows illustrate the tracking results for the sine wave and the corresponding delivered PWs (with zoom during five seconds on the third row), respectively. Similarly, the fourth and fifth rows illustrate the tracking results for the step wave and the corresponding delivered PWs (with zoom during five seconds on the last row), respectively.

As shown in Tables 2 and 3 and Figs. 4 and 5, the proposed methodology could be effectively applied to clinical procedures for treating people with SCIFigure 5: Experimental results for individual P2 comparing empirical gains and the proposed methodology. The first and second rows illustrate the tracking results for the sine wave and the corresponding delivered PWs (with zoom during five seconds on the third row), respectively. Similarly, the fourth and fifth rows illustrate the tracking results for the step wave and the corresponding delivered PWs (with zoom during five seconds on the last row), respectively.

via NMES/FES. In general, tremors (mainly for P1) and fatigue were detected for both individuals with SCI at the end of each trajectory (sine and wave). This was because neither of them had been admitted to a rehabilitation research program involving daily electrical stimulation exercise of their lower limbs. In all experiments, P1 had no perception of the stimulation, while P2 experienced small discomfort due to the electrical stimulation intensity. Results from P1 validate and substantiate the first hypothesis presenting very good trackingTable 2: Performance results for the sine wave on control experiments using the proposed methodology and empirical tuning for all individuals in their respective sessions.

<table border="1">
<thead>
<tr>
<th rowspan="2">Ind.</th>
<th rowspan="2">Sess.</th>
<th colspan="2">Empirical</th>
<th colspan="2">Proposed methodology</th>
</tr>
<tr>
<th>RMSE</th>
<th>TEC</th>
<th>RMSE</th>
<th>TEC</th>
</tr>
</thead>
<tbody>
<tr>
<td>P1</td>
<td>i</td>
<td>9.147<math>^{\circ}</math></td>
<td>30 s</td>
<td>2.984<math>^{\circ}</math></td>
<td>30 s</td>
</tr>
<tr>
<td>P2</td>
<td>i</td>
<td>11.296<math>^{\circ}</math></td>
<td>30 s</td>
<td>10.730<math>^{\circ}</math></td>
<td>30 s</td>
</tr>
<tr>
<td rowspan="5">H1</td>
<td>i</td>
<td>7.494<math>^{\circ}</math></td>
<td>60 s</td>
<td>5.830<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>ii</td>
<td>8.752<math>^{\circ}</math></td>
<td>60 s</td>
<td>5.933<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>iii</td>
<td>14.092<math>^{\circ}</math></td>
<td>60 s</td>
<td>7.337<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>iv</td>
<td>6.377<math>^{\circ}</math></td>
<td>60 s</td>
<td>3.629<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>v</td>
<td>6.383<math>^{\circ}</math></td>
<td>60 s</td>
<td>3.562<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>8.62(2.87)<math>^{\circ}</math></td>
<td>60(0) s</td>
<td>5.26(1.46)<math>^{\circ}</math></td>
<td>60(0) s</td>
</tr>
<tr>
<td rowspan="5">H2</td>
<td>i</td>
<td>5.212<math>^{\circ}</math></td>
<td>60 s</td>
<td>5.055<math>^{\circ}</math></td>
<td>45 s</td>
</tr>
<tr>
<td>ii</td>
<td>8.317<math>^{\circ}</math></td>
<td>60 s</td>
<td>3.885<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>iii</td>
<td>11.741<math>^{\circ}</math></td>
<td>35 s</td>
<td>3.633<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>iv</td>
<td>4.887<math>^{\circ}</math></td>
<td>40 s</td>
<td>3.562<math>^{\circ}</math></td>
<td>40 s</td>
</tr>
<tr>
<td>v</td>
<td>10.713<math>^{\circ}</math></td>
<td>33 s</td>
<td>4.858<math>^{\circ}</math></td>
<td>23 s</td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>8.17(2.78)<math>^{\circ}</math></td>
<td>46(12) s</td>
<td>4.20(0.63)<math>^{\circ}</math></td>
<td>46(14) s</td>
</tr>
<tr>
<td rowspan="5">H3</td>
<td>i</td>
<td>NC</td>
<td>NC</td>
<td>6.019<math>^{\circ}</math></td>
<td>30 s</td>
</tr>
<tr>
<td>ii</td>
<td>9.221<math>^{\circ}</math></td>
<td>50 s</td>
<td>7.615<math>^{\circ}</math></td>
<td>50 s</td>
</tr>
<tr>
<td>iii</td>
<td>NC</td>
<td>NC</td>
<td>4.616<math>^{\circ}</math></td>
<td>33 s</td>
</tr>
<tr>
<td>iv</td>
<td>3.775<math>^{\circ}</math></td>
<td>33 s</td>
<td>6.688<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>v</td>
<td>19.096<math>^{\circ}</math></td>
<td>60 s</td>
<td>6.516<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>10.70(6.3)<math>^{\circ}</math></td>
<td>48(11) s</td>
<td>6.29(0.98)<math>^{\circ}</math></td>
<td>47(13) s</td>
</tr>
<tr>
<td rowspan="5">H4</td>
<td>i</td>
<td>NC</td>
<td>NC</td>
<td>9.382<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>ii</td>
<td>12.794<math>^{\circ}</math></td>
<td>60 s</td>
<td>4.823<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>iii</td>
<td>8.246<math>^{\circ}</math></td>
<td>60 s</td>
<td>4.640<math>^{\circ}</math></td>
<td>30 s</td>
</tr>
<tr>
<td>iv</td>
<td>3.534<math>^{\circ}</math></td>
<td>31 s</td>
<td>4.561<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>v</td>
<td>16.483<math>^{\circ}</math></td>
<td>60 s</td>
<td>3.717<math>^{\circ}</math></td>
<td>42 s</td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>10.3(4.86)<math>^{\circ}</math></td>
<td>53(13) s</td>
<td>5.42(2.01)<math>^{\circ}</math></td>
<td>50(12) s</td>
</tr>
<tr>
<td rowspan="3">H5</td>
<td>i</td>
<td>NC</td>
<td>NC</td>
<td>6.006<math>^{\circ}</math></td>
<td>20 s</td>
</tr>
<tr>
<td>ii</td>
<td>8.070<math>^{\circ}</math></td>
<td>50 s</td>
<td>3.017<math>^{\circ}</math></td>
<td>21 s</td>
</tr>
<tr>
<td>iii</td>
<td>NC</td>
<td>NC</td>
<td>3.872<math>^{\circ}</math></td>
<td>52 s</td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>8.070(*)<math>^{\circ}</math></td>
<td>50(*) s</td>
<td>4.30(1.26)<math>^{\circ}</math></td>
<td>31(15) s</td>
</tr>
<tr>
<td rowspan="2">H6</td>
<td>i</td>
<td>NC</td>
<td>NC</td>
<td>10.128<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>ii</td>
<td>9.105<math>^{\circ}</math></td>
<td>60 s</td>
<td>6.553<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>9.105(*)<math>^{\circ}</math></td>
<td>60(*) s</td>
<td>8.34(1.79)<math>^{\circ}</math></td>
<td>60(0) s</td>
</tr>
<tr>
<td rowspan="2">H7</td>
<td>i</td>
<td>NC</td>
<td>NC</td>
<td>8.500<math>^{\circ}</math></td>
<td>60 s</td>
</tr>
<tr>
<td>ii</td>
<td>NC</td>
<td>NC</td>
<td>6.630<math>^{\circ}</math></td>
<td>50 s</td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>NC</td>
<td>NC</td>
<td>7.56(0.94)<math>^{\circ}</math></td>
<td>55(5) s</td>
</tr>
</tbody>
</table>

results using the proposed methodology. When empirical gains were used, the lower limb tracked the sine wave with a lag and presented a slow response toTable 3: Performance results for the step wave on control experiments using the proposed methodology and empirical tuning for all individuals in their respective sessions.

<table border="1">
<thead>
<tr>
<th rowspan="2">Ind.</th>
<th rowspan="2">Sess.</th>
<th colspan="3">Empirical</th>
<th colspan="3">Proposed methodology</th>
</tr>
<tr>
<th>RMSE</th>
<th>TEC</th>
<th>AvStd. OP</th>
<th>RMSE</th>
<th>TEC</th>
<th>AvStd. OP</th>
</tr>
</thead>
<tbody>
<tr>
<td>P1</td>
<td>i</td>
<td>10.995<math>^{\circ}</math></td>
<td>30 s</td>
<td>29.44(6.03)<math>^{\circ}</math></td>
<td>5.978<math>^{\circ}</math></td>
<td>25 s</td>
<td>29.88(3.25)<math>^{\circ}</math></td>
</tr>
<tr>
<td>P2</td>
<td>i</td>
<td>10.106<math>^{\circ}</math></td>
<td>23 s</td>
<td>24.15(0.59)<math>^{\circ}</math></td>
<td>6.613<math>^{\circ}</math></td>
<td>21 s</td>
<td>28.51(0.99)<math>^{\circ}</math></td>
</tr>
<tr>
<td rowspan="5">H1</td>
<td>i</td>
<td>5.920<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.35(2.50)<math>^{\circ}</math></td>
<td>6.167<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.54(1.63)<math>^{\circ}</math></td>
</tr>
<tr>
<td>ii</td>
<td>12.291<math>^{\circ}</math></td>
<td>60 s</td>
<td>36.19(6.92)<math>^{\circ}</math></td>
<td>8.201<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.54(4.61)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iii</td>
<td>7.266<math>^{\circ}</math></td>
<td>60 s</td>
<td>37.61(3.76)<math>^{\circ}</math></td>
<td>4.164<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.98(1.48)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iv</td>
<td>6.741<math>^{\circ}</math></td>
<td>60 s</td>
<td>38.70(4.59)<math>^{\circ}</math></td>
<td>4.404<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.89(1.64)<math>^{\circ}</math></td>
</tr>
<tr>
<td>v</td>
<td>6.887<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.94(5.88)<math>^{\circ}</math></td>
<td>4.425<math>^{\circ}</math></td>
<td>60 s</td>
<td>40.01(1.34)<math>^{\circ}</math></td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>7.82(2.28)<math>^{\circ}</math></td>
<td>60(0) s</td>
<td>-</td>
<td>5.47(1.54)<math>^{\circ}</math></td>
<td>60(0) s</td>
<td>-</td>
</tr>
<tr>
<td rowspan="5">H2</td>
<td>i</td>
<td>9.764<math>^{\circ}</math></td>
<td>35 s</td>
<td>38.45(2.00)<math>^{\circ}</math></td>
<td>6.212<math>^{\circ}</math></td>
<td>37 s</td>
<td>39.80(2.50)<math>^{\circ}</math></td>
</tr>
<tr>
<td>ii</td>
<td>NC</td>
<td>NC</td>
<td>NC</td>
<td>7.856<math>^{\circ}</math></td>
<td>25 s</td>
<td>38.05(1.53)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iii</td>
<td>11.822<math>^{\circ}</math></td>
<td>57 s</td>
<td>33.50(4.14)<math>^{\circ}</math></td>
<td>5.457<math>^{\circ}</math></td>
<td>37 s</td>
<td>39.88(3.36)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iv</td>
<td>6.424<math>^{\circ}</math></td>
<td>34 s</td>
<td>38.94(1.92)<math>^{\circ}</math></td>
<td>4.890<math>^{\circ}</math></td>
<td>45 s</td>
<td>39.42(1.33)<math>^{\circ}</math></td>
</tr>
<tr>
<td>v</td>
<td>6.226<math>^{\circ}</math></td>
<td>35 s</td>
<td>39.83(4.11)<math>^{\circ}</math></td>
<td>7.233<math>^{\circ}</math></td>
<td>38 s</td>
<td>40.19(3.34)<math>^{\circ}</math></td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>8.56(2.35)<math>^{\circ}</math></td>
<td>46(11) s</td>
<td>-</td>
<td>6.33(1.09)<math>^{\circ}</math></td>
<td>37(6) s</td>
<td>-</td>
</tr>
<tr>
<td rowspan="5">H3</td>
<td>i</td>
<td>15.359<math>^{\circ}</math></td>
<td>48 s</td>
<td>32.47(7.59)<math>^{\circ}</math></td>
<td>8.176<math>^{\circ}</math></td>
<td>32 s</td>
<td>39.54(1.34)<math>^{\circ}</math></td>
</tr>
<tr>
<td>ii</td>
<td>8.230<math>^{\circ}</math></td>
<td>45 s</td>
<td>38.53(2.19)<math>^{\circ}</math></td>
<td>5.598<math>^{\circ}</math></td>
<td>28 s</td>
<td>39.71(0.63)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iii</td>
<td>14.233<math>^{\circ}</math></td>
<td>38 s</td>
<td>33.06(6.26)<math>^{\circ}</math></td>
<td>6.258<math>^{\circ}</math></td>
<td>30 s</td>
<td>39.66(0.95)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iv</td>
<td>5.472<math>^{\circ}</math></td>
<td>40 s</td>
<td>39.54(0.71)<math>^{\circ}</math></td>
<td>6.357<math>^{\circ}</math></td>
<td>55 s</td>
<td>39.64(5.21)<math>^{\circ}</math></td>
</tr>
<tr>
<td>v</td>
<td>7.102<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.88(6.52)<math>^{\circ}</math></td>
<td>4.491<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.84(2.23)<math>^{\circ}</math></td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>10.08(3.97)<math>^{\circ}</math></td>
<td>46(8) s</td>
<td>-</td>
<td>6.18(1.2)<math>^{\circ}</math></td>
<td>41(14) s</td>
<td>-</td>
</tr>
<tr>
<td rowspan="5">H4</td>
<td>i</td>
<td>13.914<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.08(9.81)<math>^{\circ}</math></td>
<td>5.943<math>^{\circ}</math></td>
<td>60 s</td>
<td>40.02(2.97)<math>^{\circ}</math></td>
</tr>
<tr>
<td>ii</td>
<td>8.354<math>^{\circ}</math></td>
<td>60 s</td>
<td>40.49(2.82)<math>^{\circ}</math></td>
<td>4.694<math>^{\circ}</math></td>
<td>60 s</td>
<td>40.00(0.87)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iii</td>
<td>8.830<math>^{\circ}</math></td>
<td>60 s</td>
<td>42.26(1.86)<math>^{\circ}</math></td>
<td>7.286<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.92(2.35)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iv</td>
<td>4.551<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.89(1.47)<math>^{\circ}</math></td>
<td>6.777<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.82(5.33)<math>^{\circ}</math></td>
</tr>
<tr>
<td>v</td>
<td>7.871<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.92(7.29)<math>^{\circ}</math></td>
<td>4.895<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.88(2.52)<math>^{\circ}</math></td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>8.70(3.01)<math>^{\circ}</math></td>
<td>60(0) s</td>
<td>-</td>
<td>5.92(1.02)<math>^{\circ}</math></td>
<td>60(0) s</td>
<td>-</td>
</tr>
<tr>
<td rowspan="3">H5</td>
<td>i</td>
<td>NC</td>
<td>NC</td>
<td>NC</td>
<td>5.719<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.83(3.74)<math>^{\circ}</math></td>
</tr>
<tr>
<td>ii</td>
<td>8.076<math>^{\circ}</math></td>
<td>52 s</td>
<td>39.13(2.43)<math>^{\circ}</math></td>
<td>5.481<math>^{\circ}</math></td>
<td>50 s</td>
<td>39.58(1.11)<math>^{\circ}</math></td>
</tr>
<tr>
<td>iii</td>
<td>13.032<math>^{\circ}</math></td>
<td>45 s</td>
<td>33.66(5.39)<math>^{\circ}</math></td>
<td>6.351<math>^{\circ}</math></td>
<td>50 s</td>
<td>39.42(1.99)<math>^{\circ}</math></td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>10.55(2.48)<math>^{\circ}</math></td>
<td>47(3) s</td>
<td>-</td>
<td>5.85(0.37)<math>^{\circ}</math></td>
<td>57(5) s</td>
<td>-</td>
</tr>
<tr>
<td rowspan="2">H6</td>
<td>i</td>
<td>12.789<math>^{\circ}</math></td>
<td>60 s</td>
<td>31.64(5.37)<math>^{\circ}</math></td>
<td>6.578<math>^{\circ}</math></td>
<td>60 s</td>
<td>40.01(2.80)<math>^{\circ}</math></td>
</tr>
<tr>
<td>ii</td>
<td>7.506<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.6(2.89)<math>^{\circ}</math></td>
<td>4.040<math>^{\circ}</math></td>
<td>60 s</td>
<td>39.62(1.49)<math>^{\circ}</math></td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>10.15(2.64)<math>^{\circ}</math></td>
<td>60(0) s</td>
<td>-</td>
<td>5.31(1.27)<math>^{\circ}</math></td>
<td>60(0) s</td>
<td>-</td>
</tr>
<tr>
<td rowspan="2">H7</td>
<td>i</td>
<td>9.554<math>^{\circ}</math></td>
<td>40 s</td>
<td>38.82(4.25)<math>^{\circ}</math></td>
<td>7.044<math>^{\circ}</math></td>
<td>21 s</td>
<td>39.63(2.49)<math>^{\circ}</math></td>
</tr>
<tr>
<td>ii</td>
<td>13.135<math>^{\circ}</math></td>
<td>60 s</td>
<td>36.38(10.51)<math>^{\circ}</math></td>
<td>5.212<math>^{\circ}</math></td>
<td>60 s</td>
<td>40.04(1.68)<math>^{\circ}</math></td>
</tr>
<tr>
<td colspan="2"><b>Avg.(std)</b></td>
<td>11.34(1.79)<math>^{\circ}</math></td>
<td>50(10) s</td>
<td>-</td>
<td>6.13(0.92)<math>^{\circ}</math></td>
<td>40(19) s</td>
<td>-</td>
</tr>
</tbody>
</table>

the step trajectory. Moreover, the RMSE of 2.9842 $^{\circ}$  for the IGA sine wave from P1 was the best result during all experiments in this research, which isa third of the RMSE obtained using empirical gains  $9.1471^\circ$ . However, in the final seconds (about 28 s), the lower limb would start to have more tremors due to the fatigue factor; this is also noticed after about 15 s to the step wave for both our proposed method and empirical tuning.

Furthermore, the tracking result for the sine wave of individual P2 was not as satisfactory such as for P1. However, as seen in the Deliv. PWs curve (Fig. 5), this poor sine wave tracking could be due to an underestimation for the upper bound to the control signal value (as this individual experienced discomfort under NMES); selecting a higher value may have resulted in good tracking. This inference is substantiated by the good results achieved in the step wave after a 3 min interval for muscle rest and by having consent to increase the upper bound value to the PW. A good regulation around the operation point was achieved for approximately 21 s, with 50% less RMSE than that obtained using empirical tuning. More specifically, using empirical tuning led to poor performances for both sine and step trajectories, as the leg did not track the sine wave, and the regulation around the operation point featured a stationary error.

For healthy individuals, as seen in Tables 2 and 3 and in the figures of Appendix A, using empirical gains led to several poor performances. In many tests, the control-stimulated lower limb did not track the reference angle (“NC”) or presented high oscillatory comportment. This problem is, for example, demonstrated in Figs. A.10 and A.12 and in Table 3 regarding the *AvStd. OP* metric, as using empirical gains resulted in average values (knee angular position) below the operation point with high std values. When the proposed methodology was used, for all individuals, satisfactory and suitable tracking results were acquired for both the tracking of sine wave via isotonic contraction and the regulation around an operation point (step wave) as isometric contraction. On average, for each individual, our proposed methodology presented much lower RMSE while still achieving high TEC (Tables 2 and 3).

Finally, for most healthy individuals, when our proposed solution was used and RISE controller was not tuned with empirical gains, the lower limb robustlytried to track the reference angle for 60 s. This could not be possible if we had performed pretrial tests, which could generate muscle fatigue due to prior stimulations. In contrast, RISE controller presented by Stegath et al. (2007) and Stegath et al. (2008) demonstrated tracking control for 8 s for a step trajectory and 20 s for a sine wave; Sharma et al. (2009) and Sharma et al. (2012) presented tracking control for 30 s for a step- and a sine-type signal; Kushima et al. (2015) presented tracking control for 30 s for a sine wave, and Downey et al. (2015) presented tracking control for 45 s (for conventional stimulation) for a sine trajectory. On the other hand, in some of our experiments, significant “chattering” was noticed in the control input. Yet, except for individual P2, none of the other voluntary participants reported discomfort due to NMES while presenting satisfactory tracking of the lower limb with high TEC. Further improvements to RISE controller tuning (i.e., IGA) may help to smooth this “chattering” problem, which is undesirable and may lead to poor controller performance (Lynch & Popovic (2012)).

#### 4.2. Nonlinear system identification results

Table 4 presents the following metrics for all individuals (*Ind.*) in each session (*Sess.*): (i) the Pearson correlation coefficient (*Corr.*) between the input (PW) and output (angular position) data using past control data as sessions progress; (ii) the Coefficient of determination ( $R^2$ ); and (iii) the mean squared error (*MSE*). These metrics are explained in the following: First, the *Corr.* between the input and output data indicates the correlation between both data, which clarifies how “difficult” it is to identify the system dynamics. More specifically, *Corr.* measures the linear correlation between two variables  $x$  and  $y$ . The *Corr.* value ranges from -1 to 1. The higher the value, the stronger the correlation. A negative value indicates an inverse correlation, while a positive value indicates a regular correlation. Second, the coefficient of determination ( $R^2$ ) is the proportion of the variance in the dependent variable that is predictable from the independent variable. The larger  $R^2$  is, the more the variability is indicatedby the linear regression model<sup>1</sup>. Third, the MSE is the average squared error between the NN outputs and the real ones.

Additionally, Figs. 6 and 7 compare the results from simulation and real experiments using either empirical or fine-tuned IGA gains. These figures were selected for illustration purposes only, as the objective is to highlight the benefits of using past data for the nonlinear system identification step. In Appendix A, we provide more illustrations comparing simulation versus the real experiment results.

Table 4: Identification results for all individuals in their respective sessions.

<table border="1">
<thead>
<tr>
<th>Ind.</th>
<th>Sess.</th>
<th>Corr.</th>
<th><math>R^2</math></th>
<th>MSE</th>
<th>Ind.</th>
<th>Sess.</th>
<th>Corr.</th>
<th><math>R^2</math></th>
<th>MSE</th>
</tr>
</thead>
<tbody>
<tr>
<td>P1</td>
<td>i</td>
<td>0.4153</td>
<td>0.836</td>
<td>0.001</td>
<td>P2</td>
<td>i</td>
<td>0.1035</td>
<td>0.796</td>
<td>0.003</td>
</tr>
<tr>
<td rowspan="5">H1</td>
<td>i</td>
<td>0.5908</td>
<td>0.726</td>
<td>0.002</td>
<td rowspan="5">H2</td>
<td>i</td>
<td>0.7789</td>
<td>0.869</td>
<td>0.006</td>
</tr>
<tr>
<td>ii</td>
<td>0.1738</td>
<td>0.157</td>
<td>0.038</td>
<td>ii</td>
<td>0.2594</td>
<td>0.416</td>
<td>0.039</td>
</tr>
<tr>
<td>iii</td>
<td>0.0469</td>
<td>0.101</td>
<td>0.042</td>
<td>iii</td>
<td>0.2640</td>
<td>0.308</td>
<td>0.039</td>
</tr>
<tr>
<td>iv</td>
<td>-0.1109</td>
<td>0.159</td>
<td>0.041</td>
<td>iv</td>
<td>0.2606</td>
<td>0.282</td>
<td>0.037</td>
</tr>
<tr>
<td>v</td>
<td>-0.0916</td>
<td>0.113</td>
<td>0.040</td>
<td>v</td>
<td>0.2769</td>
<td>0.292</td>
<td>0.038</td>
</tr>
<tr>
<td rowspan="5">H3</td>
<td>i</td>
<td>0.8333</td>
<td>0.820</td>
<td>0.003</td>
<td rowspan="5">H4</td>
<td>i</td>
<td>0.7339</td>
<td>0.974</td>
<td>0.001</td>
</tr>
<tr>
<td>ii</td>
<td>0.4325</td>
<td>0.498</td>
<td>0.023</td>
<td>ii</td>
<td>0.0476</td>
<td>0.377</td>
<td>0.054</td>
</tr>
<tr>
<td>iii</td>
<td>0.3083</td>
<td>0.506</td>
<td>0.023</td>
<td>iii</td>
<td>-0.1050</td>
<td>0.323</td>
<td>0.054</td>
</tr>
<tr>
<td>iv</td>
<td>0.3523</td>
<td>0.510</td>
<td>0.024</td>
<td>iv</td>
<td>-0.0502</td>
<td>0.294</td>
<td>0.052</td>
</tr>
<tr>
<td>v</td>
<td>0.2650</td>
<td>0.492</td>
<td>0.026</td>
<td>v</td>
<td>-0.1017</td>
<td>0.281</td>
<td>0.049</td>
</tr>
<tr>
<td rowspan="3">H5</td>
<td>i</td>
<td>0.6738</td>
<td>0.881</td>
<td>0.001</td>
<td rowspan="3">H6</td>
<td>i</td>
<td>0.7182</td>
<td>0.815</td>
<td>0.001</td>
</tr>
<tr>
<td>ii</td>
<td>0.3774</td>
<td>0.682</td>
<td>0.030</td>
<td>ii</td>
<td>0.3078</td>
<td>0.476</td>
<td>0.028</td>
</tr>
<tr>
<td>iii</td>
<td>0.3458</td>
<td>0.599</td>
<td>0.035</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td rowspan="2">H7</td>
<td>i</td>
<td>0.5201</td>
<td>0.767</td>
<td>0.004</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>ii</td>
<td>0.5520</td>
<td>0.476</td>
<td>0.017</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

As presented in Table 4, there are considerable decrements in the *Corr.* between the input and output data as sessions progress (given the addition of control data from every new session). Moreover, while the data from healthy individuals in the first session are highly correlated ( $0.5201 \leq Corr. \leq 0.8333$ ), the ones from individuals with SCI are poorly correlated, as their muscles do not respond to NMES/FES as well as the muscles of the healthy individuals. Moreover, due to less correlation between data, the generalization and learning

<sup>1</sup><https://fr.mathworks.com/help/stats/coefficient-of-determination-r-squared.html>

