# GPT-in-the-Loop: Adaptive Decision-Making for Multiagent Systems

Nathalia Nascimento, Paulo Alencar, Donald Cowan

David R. Cheriton School of Computer Science  
University of Waterloo (UW)  
Waterloo, Canada  
{nmoraesd, palencar, dcowan}@uwaterloo.ca

## Abstract

This paper introduces the “GPT-in-the-loop” approach, a novel method combining the advanced reasoning capabilities of Large Language Models (LLMs) like Generative Pre-trained Transformers (GPT) with multiagent (MAS) systems. Venturing beyond traditional adaptive approaches that generally require long training processes, our framework employs GPT-4 for enhanced problem-solving and explanation skills. Our experimental backdrop is the smart streetlight Internet of Things (IoT) application. Here, agents use sensors, actuators, and neural networks to create an energy-efficient lighting system. By integrating GPT-4, these agents achieve superior decision-making and adaptability without the need for extensive training. We compare this approach with both traditional neuroevolutionary methods and solutions provided by software engineers, underlining the potential of GPT-driven multiagent systems in IoT. Structurally, the paper outlines the incorporation of GPT into the agent-driven Framework for the Internet of Things (FloT), introduces our proposed GPT-in-the-loop approach, presents comparative results in the IoT context, and concludes with insights and future directions.

**Keywords:** GPT-in-the-loop, LLM-in-the-loop, Multiagent system (MAS), self-adaptation, Generative pre-trained transformer (GPT).

## Introduction

Exploratory investigations are currently underway to harness the reasoning capabilities of Generative Pre-trained Transformers (GPT) for practical applications. Recent studies (Richardson and Sabharwal 2022) (Webb, Holyoak, and Lu 2023) (Wei et al. 2022) (Huang and Chang 2023) indicate that large language models, especially those exceeding 100 billion parameters, are showcasing emergent reasoning abilities. Webb et al. (Webb, Holyoak, and Lu 2023) demonstrated that models like GPT-3 might match or even outdo human reasoning in certain tasks—a trajectory GPT-4 seems set to follow. Further supporting this, (Wei et al. 2022) reveals that a “chain of thought” approach can significantly enhance reasoning in these models, suggesting new methods to utilize their reasoning prowess in real-world scenarios.

Conversely, in the multiagent domain, developing autonomous systems, especially agents that autonomously develop their skills through environment interactions, is an

ambitious scientific endeavor (Nolfi 2022). These agents aim to expand their behavioral repertoire in an open-ended manner. A major thrust is enabling them to employ world models, using common sense knowledge akin to humans, to enhance their performance (Nolfi 2022). Such knowledge can be gleaned via self-supervised learning, allowing agents to mentally plan and reason. While neuroevolutionary approaches offer potential solutions (Almansoori, Alkilabi, and Tuci 2023; do Nascimento and de Lucena 2017; Lan, Chen, and Eiben 2019), refining neural networks for performance enhancement is time-intensive, costly, and complex, especially in real-time settings with physical agents. In addition to problem-solving skills, the agents should also offer an explanation for their decisions (Sado et al. 2023).

Bridging these two domains, the concept of “GPT-in-the-loop” emerges as a promising approach. By leveraging the advanced reasoning capabilities of GPT models within the loop of agent decision-making, there’s potential to address the challenges in multiagent systems more efficiently. This fusion could harness GPT’s inherent adaptability and reasoning prowess, potentially reducing reliance on long training processes that are usual to adaptive approaches (Nolfi 2022). Inspired by human-in-the-loop approaches (Mosqueira-Rey et al. 2023), our proposal defines novel GPT and multiagent system interactions.

Building upon the FloT framework for adaptable Internet of Things (IoT) applications (do Nascimento and de Lucena 2017), we incorporate the “GPT-in-the-loop” methodology. To create self-adaptive IoT agents, FloT supports the use of different decision-making engines, like neural networks, state machines, and if-else statement; as the use of different adaptive processes, like evolutionary algorithms, backpropagation, and reinforcement learning. FloT’s flexibility in decision-making engines and adaptive processes make it conducive for GPT integration. This flexibility paves the way for GPT to augment reasoning or adaptive functions. For instance, within an interactive MAS setup, GPT can amplify decision-making, aiding agents in outputs and interactions. In MAS teaching, GPT might guide the adaptive process or even dictate the decision-making engine entirely, adjusting agent behaviors based on environmental feedback.

Furthermore, we have applied the GPT-in-the-loop model to smart streetlights, a benchmark IoT application (Nascimento et al. 2018). In this scenario, agents, equipped withsensors, actuators, and a neural network, evolve to develop a communication system and behavior that optimizes energy while ensuring adequate lighting. As this study (Nascimento et al. 2018) also assessed 14 software engineers' solutions to the same challenge, it allows us to perform a direct comparison between the neuroevolutionary approach, the engineers' solutions, and our GPT-in-the-loop method.

The paper is structured as follows: Section 2 delves into the GPT and FiOT background. Section 3 details our primary contribution, the GPT-in-the-loop approach. Section 4 offers performance results and comparisons within the IoT scenario. We conclude in Section 5.

## Background

### LLM and GPT

Large Language Models (LLMs) and Generative Pre-trained Transformers (GPT) are integral parts of AI's Natural Language Processing (NLP) realm. While LLM is a broad category encompassing models that predict word sequences and can be used for various tasks such as text generation and translation, GPT, developed by OpenAI (OpenAI 2023), is a specific LLM type. GPT, renowned for generating text akin to human writing, undergoes extensive pre-training before fine-tuning for specialized tasks. In essence, GPT is a sub-class of LLMs, but not all LLMs are GPT models. Other prominent LLM examples include BERT, RoBERTa, and XLNet.

GPT (Generative Pre-trained Transformer) is rooted in the Transformer neural network design (Vaswani et al. 2017; Brown et al. 2020). Representing breakthroughs in natural language processing, GPT, especially in its advanced iterations like GPT-4, utilizes a deep architecture of many layers of these transformers. A GPT solution comprises several key components, such as a pre-trained neural network model, a fine-tuning component to improve the model for specific tasks, an inference engine that uses the fine-tuned GPT model to generate responses or predictions (i.e. the inference engine feeds input data into the model and processes the model's output), and data pipeline that handles the flow of data in and out of the model (Brown et al. 2020).

### FiOT: Framework for Self-Adaptive IoT Multiagent Systems

The Framework for the Internet of Things (FiOT) (do Nascimento and de Lucena 2017; Nascimento 2023) is a software framework designed for building control systems for self-operating agents through learning or rule-based methods. Utilizing FiOT results in a Java software element pre-loaded with features for recognizing autonomous entities, assigning control, developing software agents, collecting device data, and ensuring agent-device interactions.

FiOT's features can be customized based on the application's needs. These include: 1) a control unit, ranging from basic if-else conditions to neural networks or preset state machines; 2) a controller adaptation method using techniques like reinforcement learning or genetic algorithms; and 3) a mechanism to evaluate decision-making processes in controlled devices.

There are two primary agents in FiOT: AdaptiveAgent and ObserverAgent. The former oversees IoT devices and uses the controller for decision-making. Its foundation is the MAPE-K loop (Redbooks and Organization 2004), an esteemed model for enhancing system autonomy. It perceives, acts, and reasons, tailoring outputs based on the chosen decision system. Meanwhile, the ObserverAgent gauges overall agent activity and can refine the control system adopted by IoT agents.

### Approach: GPT-in-the-loop

Drawing inspiration from human-in-the-loop methodologies (Mosqueira-Rey et al. 2023), our proposition delineates novel interactions between GPT and multiagent systems. We propose three main interaction modes:

- • **Active MAS:** Traditional algorithms drive agents while GPT clarifies outcomes.
- • **Interactive MAS:** This encourages a more integrated collaboration between the GPT's reasoning and the MAS, which has been our primary focus in this work as depicted in Figure 1.
- • **MAS Teaching:** Here, the GPT directs the MAS adaptation.

In the interactive MAS model, GPT shapes the decision-making engine of the agent. This engine processes inputs, generates outcomes, and influences the manner in which the agent engages with its environment, which in turn impacts application performance. Feedback from these engagements can re-engage the GPT, leading to refinements in agent behaviors.

Figure 1: GPT-in-the-loop: GPT crafts the decision-making engine for the agent, drawing from environmental feedback.

While the interactive MAS mode stands at the heart of our research, we chose to integrate GPT with FiOT. This framework paves the way for probing diverse interaction forms. It permits a complete overhaul of the IoT agents' decision-making engine or, alternatively, steers the evolution/training process orchestrated by the ObserverAgent.

Figures 2 and 3 illustrate the seamless extension of FiOT to accommodate the GPT-in-the-loop model, tappinginto both the AdaptiveAgent’s controller and the Observer-Agent’s adaptation process. Notably, both the decision-making controller and the adaptive procedure are flexible points at the framework. This allows for varied runtime implementations, so long as class signatures (parameters, inputs, and outputs) remain consistent. For instance, environmental feedback can prompt GPT to craft a new controller for agents.

```

classDiagram
    class FIoTAgent {
        <<kernel>>
        -world : AgentList
        -controllerList : ControllerList
    }
    class ControllerList {
        <<kernel>>
        +getInstance() : void
        +addController(typeAgent : String, configuration : File) : void
        +editController(typeAgent : String, configuration : File) : void
        +getController(typeAgent : String) : void
    }
    class Controller {
        <<interface>>
        <<hotspot>>
        +getOutput(input[] : float[]) : float[]
        +createController(controlConfiguration : File) : void
        +change(configuration[] : float[]) : void
    }
    class NeuralNetwork {
        <<alternative>>
        <<hotspot>>
        +setWeight(weight[]) : float[] : void
    }
    class StateMachine {
        <<alternative>>
        <<hotspot>>
    }
    FIoTAgent --> ControllerList
    ControllerList "0..*" --> Controller
    Controller <|-- NeuralNetwork
    Controller <|-- StateMachine
    Controller -.-> GPTGeneratedCodeV1 : GPT-generated code V1
    Controller -.-> GPTGeneratedCodeV2 : GPT-generated code V2
    
```

Figure 2: Augmenting FIoT to empower agents with decision-making abilities using GPT-crafted code.

## Application Scenario: Smart Streetlights

In our experiment, we replicated the streetlight scenario from (Nascimento et al. 2018) using the FIoT framework. The goal was to create autonomous streetlights balancing energy conservation with effective illumination, ensuring individuals could navigate their paths seamlessly. These streetlights, equipped with sensors and communicative tools, had three core functions: data collection, decision-making, and action execution. The focus of this experiment was on the decision-making aspect.

The original study utilized a three-layer neural network, evolved through a genetic algorithm, to automate the streetlights’ decision rules. Software engineers also tackled the challenge, developing decision-making solutions. They were presented with the same simulated scenario, facilitating a comparison of human-devised solutions with the automated neural network method. Subsequently, these solutions were tested in an expanded environment. This second phase aimed to assess whether the decision-making module, originally designed for the first scenario, could be effectively reused in a different environment.

Incorporating the GPT-in-the-loop methodology, and paralleling the strategy in (Nascimento et al. 2018), GPT engaged with the primary scenario until it derived a solution surpassing a fitness score of 62 (we set it based on the best fitness value presented in (Nascimento et al. 2018)). This

derived decision mechanism was then trialed in the expansive environment. Conclusively, we set the GPT-in-the-loop results as a benchmark, juxtaposing them against the top solutions from the neuroevolutionary algorithm, the best software engineer participant, and GPT’s own solution.

To facilitate a clear comparison between the two methods, Table 1 showcases the application of the Streetlight Control case study using a neuroevolutionary approach, highlighting the flexible points of the FIoT framework. Conversely, Table 2 delineates the implementation of the Streetlight Control application through the GPT-in-the-loop-based approach, capitalizing on the adaptability of the FIoT framework. Both tables aim to provide a foundation for evaluating the efficacy of each solution within the same application context.

Table 1: Implementing FIoT flexible points to synthesize streetlight controllers using an ML-based approach (Nascimento et al. 2018).

<table border="1">
<thead>
<tr>
<th>FIoT Framework</th>
<th>Light Control Application</th>
</tr>
</thead>
<tbody>
<tr>
<td>Controller</td>
<td>Three Layer Neural Network</td>
</tr>
<tr>
<td>Making Evaluation</td>
<td>Collective Fitness Evaluation: the solution is evaluated based on the energy consumption, the number of people that finished their routes after the simulation ends, and the total time spent by people to move during their trip</td>
</tr>
<tr>
<td>Controller Adaptation</td>
<td>Evolutionary Algorithm: Generate a pool of candidates to represent the neural network parameters</td>
</tr>
</tbody>
</table>

## Results and Discussion

The GPT-in-the-loop approach required three iterations to reach a fitness score of 62 in the first scenario. Comparatively, the original evolutionary approach underwent 200 generations, with each generation undergoing 50 interactions with the environment. This section details the solutions GPT proposed for the streetlight controllers in each iteration, elucidating the adaptive changes influenced by GPT’s explanations. Ultimately, we compared the outcomes of this innovative method against those of neuroevolution and human-generated solutions.

In the initial iteration, the problem description presented to the software engineers in (Nascimento et al. 2018) served as the prompt. For every iteration, GPT generated an ‘if-else statement’ controller accompanied by a rationale for the decision.

The code, as shown in Figure 4, emerged from the first iteration. The foundational idea behind this solution was: 1) Ambient Light Detection: It’s the sole strategy highlighting that poles will perpetually be in “listening” mode, regardless of external lighting conditions; 2) Wireless Communication: Stresses the significance of interaction with the pre-Figure 3: Elevating FloT to incorporate GPT as a potential adaptive strategy for the Observer Agent.

Table 2: Detailed implementation of FloT flexible points to synthesize streetlight controllers using the GPT-in-the-loop approach.

<table border="1">
<thead>
<tr>
<th>FloT Framework</th>
<th>Light Control Application</th>
</tr>
</thead>
<tbody>
<tr>
<td>Controller</td>
<td>GPT-based Decision Engine: Use if-else statement controllers optimized for the given scenario's constraints and goals</td>
</tr>
<tr>
<td>Making Evaluation</td>
<td>Iterative Fitness Evaluation: the solution iterates until it exceeds a fitness score of 62, evaluating based on energy consumption, the number of people that complete routes, and the cumulative time of people's journeys</td>
</tr>
<tr>
<td>Controller Adaptation</td>
<td>GPT-in-the-loop: GPT engages in interactive loops, refining its if-else controllers based on environment feedback until the desired fitness level is reached</td>
</tr>
</tbody>
</table>

ceding pole and adjusting illumination based on that communication; 3) Energy Conservation: Sets a distinct condition whereby, in the absence of motion detection for a pre-determined cycle count, the light's intensity diminishes.

The first iteration resulted in an average fitness of 29.49. Given its inadequacy, GPT was re-engaged to produce an enhanced solution. For the second iteration, the environment's outputs, such as energy consumption and trip time, were in-

Figure 4: GPT's decision-making engine from the first iteration.

```

@Override
public double[] getOutput(double[] input) {
    // Input variables
    double lightingSensor = this.getLightingSensor();
    double motionSensor = this.getMotionSensor();
    double wirelessReceiver = (this.
        getPreviousListeningDecision() == 1.0) ? this.
        getWirelessReceiver() : 0.0; // If not listening,
        ignore the value.
    // Decision variables
    double lightDecision = 0.0; // By default, we keep it off
    double wirelessTransmit = 0.0; // By default, we don't
        transmit anything
    double listeningDecision = 1.0; // By default, we always
        listen to the surrounding posts
    // If it's day, we keep the light off
    if (lightingSensor == 1.0) {
        lightDecision = 0.0;
    } else {
        // If we detect movement
        if (motionSensor == 1.0) {
            lightDecision = 1.0;
            wirelessTransmit = 1.0; // We alert the
                surrounding posts about the detected movement
        }
        // If the neighboring post detected movement and we are
        listening
        else if (wirelessReceiver > 0.0) {
            lightDecision = 0.5; // We partially light up to
                prepare the path
        }
    }
    // Action decisions
    this.setLightDecision(lightDecision);
    this.setWirelessTransmitter(wirelessTransmit);
    this.setListeningDecision(listeningDecision);
    return this.actuator;
}

```

corporated into the GPT prompt. The code from the second iteration is showcased in Figure 5. Contrary to the first solution, which emphasized persistent ambient light listening, previous pole communication, and energy preservation, the second solution accentuates anticipatory lighting influencedby adjacent movements and subdued illumination during extreme darkness.

Figure 5: GPT’s decision-making engine from the second iteration (input and output settings excluded).

```

@Override
public double[] getOutput(double[] input) {
    /*...*/
    double lightDecision = 0.0;
    double wirelessTransmit = 0.0;
    double listeningDecision = 1.0; // Always
        listening to optimize the person’s path
    // If motion is detected
    if (motionSensor == 1.0) {
        lightDecision = 1.0; // Turn on fully for
            the passerby
        wirelessTransmit = 1.0; // Notify the nearby
            posts to prepare the path
    // If a neighboring post reports movement
} else if (wirelessReceiver == 1.0) {
    lightDecision = 0.5; // Partially light up
        to anticipate the passerby’s arrival
    // Energy saving without compromising safety
} else if (lightingSensor == 0.0) {
    lightDecision = 0.5; // Keep partially lit
        if it’s too dark, to ensure some level
        of safety
}
    /*...*/
    return this.actuator;
}

```

The second iteration notably improved the fitness value, achieving 61.2. However, since this score remained below the target, GPT produced another controller, depicted in Figure 6. Unlike the solution from the second iteration, this new approach concentrates on curtailing data transmissions by sending signals exclusively upon initial motion detection, determining listening necessity based on present light status, and integrating adaptive dimming.

The integration of environmental feedback into successive GPT iterations consistently led to performance improvements. Table 3 displays the fitness outcomes across three iterations for scenario 1, whereas Table 4 presents the outcomes when the decision algorithms were implemented in a more intricate environment for scenario 2. In both scenarios, the third solution proposed by GPT outperformed its predecessors. When juxtaposed with the optimal outcome derived from the neuroevolution method, the solution offered by GPT achieved a superior fitness score in both scenarios. In scenario 1, one participant managed to devise a solution with a slightly better fitness score than that of GPT’s. However, this solution faltered in the second scenario. Evaluating GPT’s performance against the most successful participant-driven solutions in the second scenario, GPT’s solution was unrivaled.

In the quest for optimized streetlight controllers, GPT’s iterative approach showcased notable adaptability and improvement. The model’s ability to integrate environment feedback between iterations culminated in solutions compet-

Figure 6: GPT’s decision-making engine from the third iteration (input and output settings excluded).

```

@Override
public double[] getOutput(double[] input) {
    /*...*/
    if (motionSensor == 1.0) {
        lightDecision = 1.0;
        if (this.getPreviousListeningDecision()
            != 1.0) {
            wirelessTransmit = 1.0; // Transmit
                motion detection once
            listeningDecision = 0.0; // No need
                to listen when we already detect
                motion
        }
    } else if (wirelessReceiver == 1.0) {
        lightDecision = 0.5;
        if (this.getPreviousListeningDecision()
            != 1.0) {
            wirelessTransmit = 0.5; // Inform
                neighbors about anticipatory
                lighting
        }
    } else if (lightingSensor == 0.0) {
        lightDecision = 0.5;
    }
    /*...*/
    return this.actuator;
}

```

itive with human and neuroevolution strategies. Moreover, GPT’s intrinsic explainability, as evident in its generated ‘if-else statements’ and accompanying rationale, offers valuable insights for users, bridging the gap between automation and human understanding.

## Conclusion and Future Horizons

The synergy between Large Language Models (LLMs) like GPT-4 and multiagent systems promises to redefine the boundaries of autonomous interactions and adaptability. Our research underscores the compelling advantages of this integration. The GPT-in-the-loop methodology exemplifies how problem-solving abilities can be significantly enhanced in a dynamic setting. When LLMs are incorporated into agents, it catalyzes a two-fold benefit: a supercharged reasoning mechanism for each agent and a more efficient communication process across a diverse multi-agent landscape.

Furthermore, GPT’s unique ability to elucidate its decision-making process brings a new dimension of transparency. This clarity not only strengthens confidence in the system’s actions but also paves the way for a deeper understanding of intricate decisions.

Nonetheless, this promising integration is met with inherent challenges. From the substantial computational needs of LLMs to the subtleties surrounding their decisions and looming ethical considerations, there’s a clear call for meticulous evaluation. The forward-looking vision of agents dynamically leveraging a cloud-hosted GPT to optimize their actions in real-time is undeniably ambitious. To fully mate-Table 3: Performance comparison of GPT iterations, best neuroevolution solution, and best participant’s solution in the first scenario.

<table border="1">
<thead>
<tr>
<th>Solution</th>
<th>Energy</th>
<th>People</th>
<th>TotalFTrip</th>
<th>Fitness</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPT (iteration 1)</td>
<td>4.03</td>
<td>66.66</td>
<td>59.25</td>
<td>29.49</td>
</tr>
<tr>
<td>GPT (iteration 2)</td>
<td>15.02</td>
<td>100</td>
<td>54.62</td>
<td>61.2</td>
</tr>
<tr>
<td>GPT (iteration 3)</td>
<td>11.92</td>
<td>100</td>
<td>54.62</td>
<td>62.44</td>
</tr>
<tr>
<td>Best neuroevolution’s solution</td>
<td>8.1</td>
<td>100</td>
<td>62.03</td>
<td>59.53</td>
</tr>
<tr>
<td>Best participant’s solution</td>
<td>9.46</td>
<td>100</td>
<td>55.55</td>
<td>62.88</td>
</tr>
</tbody>
</table>

Table 4: Performance comparison of GPT iterations, best neuroevolution solution, and best participant’s solution in the second scenario.

<table border="1">
<thead>
<tr>
<th>Solution</th>
<th>Energy</th>
<th>People</th>
<th>TotalFTrip</th>
<th>Fitness</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPT (iteration 1)</td>
<td>2.08</td>
<td>66.66</td>
<td>48.51</td>
<td>36.72</td>
</tr>
<tr>
<td>GPT (iteration 2)</td>
<td>11.29</td>
<td>100</td>
<td>41.10</td>
<td>70.81</td>
</tr>
<tr>
<td>GPT (iteration 3)</td>
<td>9.76</td>
<td>100</td>
<td>41.10</td>
<td>71.42</td>
</tr>
<tr>
<td>Best neuroevolution’s solution</td>
<td>8.46</td>
<td>100</td>
<td>46.29</td>
<td>68.83</td>
</tr>
<tr>
<td>Best participant’s solution</td>
<td>50.52</td>
<td>100</td>
<td>38.14</td>
<td>56.9</td>
</tr>
</tbody>
</table>

rialize this vision, further research and exploration are essential, especially in leveraging GPT-in-the-loop to enhance diverse GPT-Multiagent interactions.

### Exploring Further Configurations for the GPT-in-the-loop

The premise of “GPT-in-the-loop” holds tremendous potential in the realm of multiagent systems, leveraging the sophisticated reasoning capabilities of GPT models directly into agent decision-making processes. Given the inspiration drawn from the human-in-the-loop approaches (Mosqueira-Rey et al. 2023), our roadmap defines diverse GPT and multiagent system interactions, which can be expanded in several directions:

1. 1. Active MAS Involvement: A scenario wherein traditional algorithms guide the multiagent systems, and GPT steps in to provide clarity and interpretation of results. This interaction mode mainly draws on GPT’s unparalleled explainability prowess, making complex decisions more transparent and comprehensible.
   1. (a) GPT as a Decentralized Decision Engine: A promising direction is to use GPT as the primary decision-maker for each agent. Instead of one general reasoning mechanism for all agents, envision each agent having its personalized GPT. This approach allows agents to make context-specific decisions in real-time, drawing from GPT’s vast knowledge to address their unique situations.
2. 2. Interactive MAS Integration: This model envisions a more intimate alliance between GPT’s reasoning faculties and the multiagent system. Here, there’s a bidirectional flow of information and decisions, ensuring that both GPT and MAS evolve and adapt symbiotically.
3. 3. MAS Teaching: GPT’s role as a tutor or mentor to multiagent systems. GPT could oversee, instruct, and guide the adaptation process of MAS.

### Enhancing Human Engagement in the Loop

While the human element remains foundational, especially in shaping the initial system prompt or documentation, the potential for a more intertwined human-machine partnership exists.

1. 1. Direct Influence: Encouraging humans to directly shape agent behaviors is key. An intuitive interface could enable users to propose behaviors, pinpoint overarching goals, or lay out specific parameters. This merges human intuition with technological prowess, targeting the best results for agents.
2. 2. Feedback Mechanism: It’s beneficial when agents offer clear summaries of their decisions, from data analysis to behavioral tweaks. Such transparency strengthens trust, offers clarity, and provides avenues for system enhancements based on human feedback.
3. 3. Making Sense of Complexity: Even though adaptive systems are complex by design, demystifying their workings is essential. Translating intricate operations into comprehensible language paves the way for enhanced human-machine interactions.

### Diversifying Application Scenarios:

Venturing beyond our preliminary framework, our ambition is to validate the GPT-in-the-loop approach in a spectrum of applications, especially when integrated with realistic robotics frameworks like Evorobot (Nolfi 2020) and Webots (Michel 2004). Such platforms enable the deployment of neural networks sculpted by evolutionary techniques.

The domain of evolutionary robotics unravels complex challenges, a notable one being the food foraging task (Pontes-Filho et al. 2022). Here, agents are tasked with distinguishing nourishing food sources from harmful ones, adeptly navigating environmental intricacies for optimal survival. In this setup, agents traverse a dynamic landscape, reliant on a singular light sensor, to ascertain the edibility ofproximate food. Represented in alternating colors of black and white, the safety of the food keeps shifting, mandating constant adaptability. Agents face a binary choice: to consume or avoid the food, within a given time frame.

Figure 7 depicts our conceptualization of GPT-in-the-loop within a distinct application setting, accounting for an alternative MAS interaction paradigm. Here, the graphic portrays a MAS teaching interaction: while agents predominantly adhere to a conventional evolutionary path, GPT plays a supportive role in their evolution.

Figure 7: GPT-in-the-loop: GPT supporting the evolutionary process.

**Evolutionary GPT Engagement:** Embedding GPT within the evolutionary paradigm offers captivating prospects. GPT, transcending its observational role, can proactively shape the evolutionary trajectory. This encompasses guiding individual selection, fine-tuning genetic algorithms, and pinpointing ideal neural network configurations. Incorporating GPT’s analytical prowess with evolutionary strategies could potentially evolve solutions that are not only optimal but also explainable.

Integrating LLMs into such narratives exhibits significant potential. With the GPT-in-the-loop approach, we’re amplifying agent adaptability and delving deep into the multifaceted GPT-MAS interactions delineated in subsection . This synergy might herald a transformative shift in the adaptability and prowess of future robotic agents.

### Diversifying LLM Choices:

While we centered on GPT-4, many other LLMs exist with unique capabilities. Exploring these options and creating clear evaluation standards might lead to even more effective multiagent strategies.

### Addressing the Black-Box Concern:

GPT-4 remains proprietary and opaque despite its explanatory capabilities. To ensure trust and safety, there’s imperative to decode its operational logic, facilitating rigorous testing and risk-mitigation strategies.

### Acknowledgment

This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC), and the Centre for Community Mapping (COMAP).

### References

Almansoori, A.; Alkilabi, M.; and Tuci, E. 2023. On the evolution of mechanisms for three-option collective decision-making in a swarm of simulated robots. In *Proceedings of the Genetic and Evolutionary Computation Conference*, 4–12.

Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. 2020. Language models are few-shot learners. *Advances in neural information processing systems*, 33: 1877–1901.

do Nascimento, N. M.; and de Lucena, C. J. P. 2017. Fiot: An agent-based framework for self-adaptive and self-organizing applications based on the internet of things. *Information Sciences*, 378: 161–176.

Huang, J.; and Chang, K. C.-C. 2023. Towards Reasoning in Large Language Models: A Survey. arXiv:2212.10403.

Lan, G.; Chen, J.; and Eiben, A. 2019. Evolutionary predator-prey robot systems: From simulation to real world. In *Proceedings of the genetic and evolutionary computation conference companion*, 123–124.

Michel, O. 2004. Cyberbotics ltd. webots™: professional mobile robot simulation. *International Journal of Advanced Robotic Systems*, 1(1): 5.

Mosqueira-Rey, E.; Hernández-Pereira, E.; Alonso-Ríos, D.; Bobes-Bascarán, J.; and Fernández-Leal, Á. 2023. Human-in-the-loop machine learning: A state of the art. *Artificial Intelligence Review*, 56(4): 3005–3054.

Nascimento, N. 2023. A software framework for developing self-adaptive IoT agents. <https://github.com/nathyecompl/fiot>.

Nascimento, N.; Alencar, P.; Lucena, C.; and Cowan, D. 2018. Toward human-in-the-loop collaboration between software engineers and machine learning algorithms. In *2018 IEEE International Conference on Big Data (Big Data)*, 3534–3540. IEEE.

Nolfi, S. 2020. A tool for training robots through evolutionary and reinforcement learning methods. <https://github.com/snolfi/evorobotpy2>.

Nolfi, S. 2022. Progress and challenges in adaptive robotics. *Frontiers in Robotics and AI*, 9: 1020462.

OpenAI. 2023. GPT Models API. Accessed: 2023-08-07.

Pontes-Filho, S.; Olsen, K.; Yazidi, A.; Riegler, M. A.; Halvorsen, P.; and Nichole, S. 2022. Towards the Neuroevolution of Low-level artificial general intelligence. *Frontiers in Robotics and AI*, 9: 1007547.

Redbooks, I.; and Organization, I. B. M. C. I. T. S. 2004. *A Practical Guide to the IBM Autonomic Computing Toolkit*. IBM redbooks. IBM, International Support Organization. ISBN 9780738498058.Richardson, K.; and Sabharwal, A. 2022. Pushing the limits of rule reasoning in transformers through natural language satisfiability. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, 11209–11219.

Sado, F.; Loo, C. K.; Liew, W. S.; Kerzel, M.; and Wermter, S. 2023. Explainable Goal-Driven Agents and Robots - A Comprehensive Review. *ACM Comput. Surv.*, 55(10).

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. *Advances in neural information processing systems*, 30.

Webb, T.; Holyoak, K. J.; and Lu, H. 2023. Emergent analogical reasoning in large language models. *Nature Human Behaviour*, 1–16.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35: 24824–24837.

