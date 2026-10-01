Title: Multi-objective Portfolio Optimization Via Gradient Descent

URL Source: https://arxiv.org/html/2507.16717

Markdown Content:
###### Abstract

Traditional approaches to portfolio optimization, often rooted in Modern Portfolio Theory and solved via quadratic programming or evolutionary algorithms, struggle with scalability or flexibility, especially in scenarios involving complex constraints, large datasets and/or multiple conflicting objectives. To address these challenges, we introduce a benchmark framework for multi-objective portfolio optimization (MPO) using gradient descent with automatic differentiation. Our method supports any optimization objective, such as minimizing risk measures (e.g., CVaR) or maximizing Sharpe ratio, along with realistic constraints, such as tracking error limits, UCITS regulations, or asset group restrictions. We have evaluated our framework across six experimental scenarios, from single-objective setups to complex multi-objective cases, and have compared its performance against standard solvers like CVXPY and SKFOLIO. Our results show that our method achieves competitive performance while offering enhanced flexibility for modeling multiple objectives and constraints. We aim to provide a practical and extensible tool for researchers and practitioners exploring advanced portfolio optimization problems in real-world conditions.

###### keywords:

Automatic differentiation , UCITS , Portfolio Optimization under Constraints

††journal: Expert Systems With Applications

\affiliation

[label1]organization=Grupo de Neurocomputación Biológica, Departamento de Ingeniería Informática, Escuela Politécnica Superior, Universidad Autónoma de Madrid,addressline=c/ Francisco Tomás y Valiente, 11, city=Madrid, postcode=28049, state=Madrid, country=Spain

\affiliation

[label2]organization=March Asset Management, S.G.I.I.C., S.A.U., addressline=Castello, 74, city=Madrid, postcode=28006, state=Madrid, country=Spain

{highlights}

Gradient descent framework for multi-objective portfolio optimization

Supports any objective, such as risk minimization or Sharpe ratio maximization

Constraints are handled through regularization-inspired terms

Easily extensible to more complex MPO problems just by adding new constraints

Fully reproducible and accessible for researchers and practitioners

1 Introduction
--------------

Portfolio optimization (PO) is a crucial aspect of financial management and investment planning with the aim of developing the best combination of having less risk and obtaining more profit in an investment. The most common investment strategy is to build a portfolio considering different securities to diversify risk. However, traditional portfolio analysis requires evaluating the return and risk conditions of individual securities, which may not be successful. This concept is rooted in the Modern Portfolio Theory and the Efficient Frontier, introduced by Markowitz ([1952](https://arxiv.org/html/2507.16717v1#bib.bib22)). According to this theory, an investor attempts to maximize its portfolio’s return for a given amount of risk, or vice-versa, minimize its risk for a given level of expected return. However, the Markowitz model has some drawbacks. It relies on historical price series and the covariance matrix, making it sensitive to input data and computationally challenging when dealing with a large number of assets. Over the years, this concept evolved into the Capital Asset Pricing Model (CAPM) of Sharpe ([1964](https://arxiv.org/html/2507.16717v1#bib.bib35)) and Lintner ([1965](https://arxiv.org/html/2507.16717v1#bib.bib21)), who introduced the Market Portfolio and the Sharpe ratio. Subsequently, new models have been introduced, such as the Black-Litterman model of Black ([1990](https://arxiv.org/html/2507.16717v1#bib.bib4)), the Three Factors Model of Fama and French ([1992](https://arxiv.org/html/2507.16717v1#bib.bib10)), and the Risk Parity and Budgeting of Roncalli ([2014](https://arxiv.org/html/2507.16717v1#bib.bib31)), including research that allows us to stabilize the optimized portfolio, such as denoising the covariance matrix or shrinkage methods (Roncalli, [2013](https://arxiv.org/html/2507.16717v1#bib.bib30)).

On the other hand, recent research in portfolio optimization has proposed new risk measures such as the value-at-risk (VaR, Jorion ([2001](https://arxiv.org/html/2507.16717v1#bib.bib18))), the Conditional Value-at-risk (CVar, Rockafellar and Uryasev ([2000](https://arxiv.org/html/2507.16717v1#bib.bib29))) or the Conditional Drawdown-at-risk (CDaR, Chekhlov et al. ([2004](https://arxiv.org/html/2507.16717v1#bib.bib6))). These advancements have allowed for a more nuanced understanding of risk, catering to various investor preferences and constraints. Nowadays, portfolio optimization has evolved beyond this traditional single-objective framework of balancing risk and return. Investors and portfolio managers aim to address multiple and conflicting objectives simultaneously. For example, one might seek to minimize CVaR while maximizing the Sharpe ratio and imposing constraints such as limiting the number of assets in the portfolio or enforcing sector diversification.

This approach reflects the complexity of real-world investment scenarios and has led to a multi-objective optimization (MO) scenario. MO is the problem of simultaneously optimizing two or more conflicting objectives with certain constraints. Thus, multi-objective portfolio optimization (MPO), which integrates the principles of MO into portfolio management, provides a robust framework for developing modern portfolio construction. However, there is no optimal solution that maximizes or minimizes each objective to its fullest in MPO (Gandibleux and Ehrgott, [2005](https://arxiv.org/html/2507.16717v1#bib.bib14)) since the various objective functions in the problem are usually in conflict with each other. Therefore, MPO aims to find efficient solutions that provide a trade-off between the different objectives (Zitzler et al., [2000](https://arxiv.org/html/2507.16717v1#bib.bib39), [2002](https://arxiv.org/html/2507.16717v1#bib.bib40)).

During the last decades, MPO has attracted attention from academics and investors to address the growing complexity of financial markets. Researchers have developed advanced mathematical models and computational algorithms, such as evolutionary algorithms (EA) and machine learning (ML) techniques. Instead of using these complex models, this paper aims to present a benchmark for MPO based on gradient descent optimization. This approach performs efficient and scalable MPO, enabling the identification of optimal solutions by iteratively adjusting portfolio weights with the gradient descent technique. The automatic differentiation provided by Tensorflow (Abadi et al., [2015](https://arxiv.org/html/2507.16717v1#bib.bib1)) and PyTorch (Paszke et al., [2019](https://arxiv.org/html/2507.16717v1#bib.bib28)), two of the most popular Deep Learning frameworks, offer computational simplicity and adaptability to solve large-scale portfolio problems for real-world scenarios.

Our approach supports any financial objective and constraint, including regulatory rules (e.g., UCITS), risk metrics (e.g., CVaR), and practical portfolio construction rules (e.g., maximum tracking error allowed, asset limits, etc.). We show that this framework achieves competitive performance compared to standard optimization tools while offering greater modeling flexibility. In addition, we provide open-source implementations in [B](https://arxiv.org/html/2507.16717v1#A2 "Appendix B Differentiation techniques in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent") and [C](https://arxiv.org/html/2507.16717v1#A3 "Appendix C Constraints implementation in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent") to facilitate reproducibility 1 1 1 We provide a Github repository with the code used in this paper: https://github.com/pventura1976/mpo-gradient-descent and future research.

The paper is organized as follows. Section [2](https://arxiv.org/html/2507.16717v1#S2 "2 Related Work ‣ Multi-objective Portfolio Optimization Via Gradient Descent") reviews the existing literature on portfolio optimization methods, with a focus on evolutionary and gradient-based techniques. Section [3](https://arxiv.org/html/2507.16717v1#S3 "3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent") describes our proposed framework in detail, including the optimizer, the formulation of the loss function, and the technical considerations for handling constraints in a differentiable setting to ensure compatibility with gradient descent. Section [4](https://arxiv.org/html/2507.16717v1#S4 "4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") presents experimental results across six scenarios, from simple objectives to complex multi-objective constrained problems. Finally, in section [5](https://arxiv.org/html/2507.16717v1#S5 "5 Conclusion and Future Work ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), we summarize the main conclusions and outlines directions for future work.

2 Related Work
--------------

Exact solution algorithms can tackle PO problems since they have a quadratic structure. However, when considering additional constraints and objectives, or when the dimension of the problem increases, exact solution algorithms may face trouble while handling MPO problems (Kalayci et al., [2019](https://arxiv.org/html/2507.16717v1#bib.bib19)). Recent research has used inexact techniques to solve more realistic MPO problems that incorporate non-convex constraints in the mathematical formulation, thus turning the problem NP-hard (Moral-Escudero et al., [2006](https://arxiv.org/html/2507.16717v1#bib.bib27)).

Traditional research is based on EAs, usually Genetic algorithms (GAs). These algorithms are based on emulating the mechanisms of natural selection to solve optimization problems. These are population-based stochastic optimization heuristics inspired by Darwin’s evolution theory. GAs search through a solution space by evaluating possible solutions (individuals). They start with a random initial population, and then the fitness of individuals is determined by evaluating the objective function. The best individuals survive, and new individuals for the next generation are created by combining their parents and altering their genes through random mutations. This cycle repeats until a breaking criterion is completed. Schaffer ([1985a](https://arxiv.org/html/2507.16717v1#bib.bib32), [b](https://arxiv.org/html/2507.16717v1#bib.bib33)); Schaffer and Grefenstette ([1985](https://arxiv.org/html/2507.16717v1#bib.bib34)) introduced the first implementation of a multi-objective evolutionary algorithm: the Vector Evaluation Genetic Algorithm (VEGA). In reality, it is just a simple genetic algorithm with a modified survivor selection mechanism, but they opened the way to MO with GAs. Fonseca and Fleming ([1993](https://arxiv.org/html/2507.16717v1#bib.bib12)); Horn et al. ([1994](https://arxiv.org/html/2507.16717v1#bib.bib16)); Srinivas and Deb ([1994](https://arxiv.org/html/2507.16717v1#bib.bib36)); Zitzler and Thiele ([1999](https://arxiv.org/html/2507.16717v1#bib.bib41)); Knowles and Corne ([2000](https://arxiv.org/html/2507.16717v1#bib.bib20)) followed this approach and proposed different multi-objective genetic algorithms that extend the traditional VEGA by using the Pareto domination ranking and fitness.

The ability of GAs to deal with a set of possible solutions makes them naturally suitable for MPO problems. Arnone et al. ([1993](https://arxiv.org/html/2507.16717v1#bib.bib3)) were the first to use GAs to optimize investment portfolios. They implemented the Markowitz model, but proposed to use lower partial moments as a risk measure. The use of downside risk makes the problem non-convex, so quadratic solvers can not find exact solutions. This opened a new line of research, and many works appeared a few years later. Foster and Shoaf ([1996](https://arxiv.org/html/2507.16717v1#bib.bib13)); Vedarajan et al. ([1997](https://arxiv.org/html/2507.16717v1#bib.bib37)); Chang et al. ([2000](https://arxiv.org/html/2507.16717v1#bib.bib5)), among others, consider the Markowitz model with different variations that incorporate many constraints. We suggest Metaxiotis and Liagkouras ([2012](https://arxiv.org/html/2507.16717v1#bib.bib25)) for a more extended review of traditional GAs applied to MPO.

However, while genetic algorithms are the most preferred evolutionary algorithms in mean-variance portfolio optimization, swarm-based algorithms (SAs) have increased in popularity when facing an MPO problem. SAs are based on the study of computational systems inspired by the behaviors of animals living in their natural environment. Swarms, such as flocks of birds or colonies of ants, reflect their cooperation in computational systems. Chen et al. ([2006](https://arxiv.org/html/2507.16717v1#bib.bib7)) were the first to apply particle swarm optimization to the MPO problem. García et al. ([2012](https://arxiv.org/html/2507.16717v1#bib.bib15)); Mishra et al. ([2016](https://arxiv.org/html/2507.16717v1#bib.bib26)) proposed robust multi-objective swarm approaches and compared their results with traditional genetic algorithms, reporting that swarm optimization outperforms the others, providing the best Pareto optimal solutions. Kalayci et al. ([2019](https://arxiv.org/html/2507.16717v1#bib.bib19)) report that particle swarm optimization and artificial bee colony algorithms are the most popular SAs for the MPO problem.

Nevertheless, it is surprising that Machine Learning (ML), which has reached a high popularity, has not been extensively explored in the context of MPO problems. Fernández and Gómez ([2007](https://arxiv.org/html/2507.16717v1#bib.bib11)) applied Hopfield networks and compared their results with GAs and SAs, reporting higher performance in larger instances. Yu et al. ([2008](https://arxiv.org/html/2507.16717v1#bib.bib38)) proposed a neural network based on mean–variance–skewness and report high performance on MPO problems. Kalayci et al. ([2019](https://arxiv.org/html/2507.16717v1#bib.bib19)) show that more than 80% of the research related to MPO uses metaheuristics (including swarm optimization and genetic algorithms), while ML approaches only cover a 12%, Deng et al. ([2024](https://arxiv.org/html/2507.16717v1#bib.bib8)), have introduced a general multi-objective framework using reinforcement learning reporting improved profitability and risk resistance, as well as better generalization ability across different financial market. In summary, it seems that population-based algorithms, such as SAs and GAs, are winning the battle and dominating the field of MPO research due to their flexibility and ease of implementation.

3 Materials and Methods
-----------------------

In this section, we present the benchmark for multi-objective portfolio optimization problems based on the gradient descent technique. This framework is designed to address the challenges of balancing multiple conflicting objectives in portfolio optimization, such as maximizing Sharpe ratio while minimizing CVaR with some constraints (see section [3.5.5](https://arxiv.org/html/2507.16717v1#S3.SS5.SSS5 "3.5.5 Case 5: Joint optimization of Sharpe ratio and CVaR under Tracking Error, UCITS, Minimum Active Asset Weights, and Active Asset Count Constraints ‣ 3.5 Experiments ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). We aim to provide a systematic approach to solve MPO problems, while maintaining computational efficiency. The following sections detail the optimization techniques (sec. [3.1](https://arxiv.org/html/2507.16717v1#S3.SS1 "3.1 Gradient Descent Optimizer ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), the formulation of the multi-objective loss function (sec. [3.2](https://arxiv.org/html/2507.16717v1#S3.SS2 "3.2 Multi-objective Loss ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), some technical considerations for differentiation (sec. [3.3](https://arxiv.org/html/2507.16717v1#S3.SS3 "3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), the definitions of the tested objectives and constraints (sec. [3.4](https://arxiv.org/html/2507.16717v1#S3.SS4 "3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), and the experimental setup used to validate the gradient descent benchmark (section [3.5](https://arxiv.org/html/2507.16717v1#S3.SS5 "3.5 Experiments ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")).

### 3.1 Gradient Descent Optimizer

Portfolio optimization is the process of selecting the best possible mix of assets to achieve specific financial goals, such as maximizing returns while minimizing risk. The most widely known method of portfolio optimization is the mean-variance optimization (Markowitz, [1959](https://arxiv.org/html/2507.16717v1#bib.bib23)), which maximizes the expected return for a given level of risk by considering the expected return of the assets and their covariances:

m⁢a⁢x(𝔼⁢[𝐑]−λ⁢V⁢a⁢r⁢(𝐑)),𝑚 𝑎 𝑥 𝔼 delimited-[]𝐑 𝜆 𝑉 𝑎 𝑟 𝐑 max\quad(\mathbb{E}[\mathbf{R}]-\lambda Var(\mathbf{R})),italic_m italic_a italic_x ( blackboard_E [ bold_R ] - italic_λ italic_V italic_a italic_r ( bold_R ) ) ,(1)

where 𝐑 𝐑\mathbf{R}bold_R represents the portfolio returns and:

𝔼⁢[𝐑]=∑i=1 n 𝐰 i⁢𝔼⁢[𝐫 i]=𝐰 T⁢𝔼⁢[𝐫],𝔼 delimited-[]𝐑 superscript subscript 𝑖 1 𝑛 subscript 𝐰 𝑖 𝔼 delimited-[]subscript 𝐫 𝑖 superscript 𝐰 𝑇 𝔼 delimited-[]𝐫\mathbb{E}[\mathbf{R}]=\sum_{i=1}^{n}\mathbf{w}_{i}\mathbb{E}[\mathbf{r}_{i}]=% \mathbf{w}^{T}\mathbb{E}[\mathbf{r}],blackboard_E [ bold_R ] = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT blackboard_E [ bold_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] = bold_w start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT blackboard_E [ bold_r ] ,(2)

V⁢a⁢r⁢(𝐑)=∑i=1 n∑j=1 n 𝐰 i⁢C⁢o⁢v⁢(𝐫 i,𝐫 j)⁢𝐰 j=𝐰 T⁢Σ⁢𝐰,𝑉 𝑎 𝑟 𝐑 superscript subscript 𝑖 1 𝑛 superscript subscript 𝑗 1 𝑛 subscript 𝐰 𝑖 𝐶 𝑜 𝑣 subscript 𝐫 𝑖 subscript 𝐫 𝑗 subscript 𝐰 𝑗 superscript 𝐰 𝑇 Σ 𝐰 Var(\mathbf{R})=\sum_{i=1}^{n}\sum_{j=1}^{n}\mathbf{w}_{i}Cov(\mathbf{r}_{i},% \mathbf{r}_{j})\mathbf{w}_{j}=\mathbf{w}^{T}\Sigma\mathbf{w},italic_V italic_a italic_r ( bold_R ) = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_C italic_o italic_v ( bold_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , bold_r start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) bold_w start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = bold_w start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT roman_Σ bold_w ,(3)

over the constraint ∑i=1 n 𝐰 i=1 superscript subscript 𝑖 1 𝑛 subscript 𝐰 𝑖 1\sum_{i=1}^{n}\mathbf{w}_{i}=1∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = 1. In the previous equations, 𝔼⁢[𝐑]𝔼 delimited-[]𝐑\mathbb{E}[\mathbf{R}]blackboard_E [ bold_R ] is the expected return of the portfolio, 𝔼⁢[𝐫 i]𝔼 delimited-[]subscript 𝐫 𝑖\mathbb{E}[\mathbf{r}_{i}]blackboard_E [ bold_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ] is the expected return of the i 𝑖 i italic_i th asset in the portfolio, 𝐰 i subscript 𝐰 𝑖\mathbf{w}_{i}bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is the weight of the i 𝑖 i italic_i th asset in the portfolio, Σ Σ\Sigma roman_Σ is the covariance matrix, and n 𝑛 n italic_n is the number of assets in the portfolio. The investor’s risk aversion is reflected in the parameter λ 𝜆\lambda italic_λ. The higher the value of λ 𝜆\lambda italic_λ, the higher the risk aversion. For simplicity, we will not allow short positions; therefore, asset weights must be non-negative: 𝐰 i≥0∀i subscript 𝐰 𝑖 0 for-all 𝑖\mathbf{w}_{i}\geq 0\quad\forall i bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ≥ 0 ∀ italic_i. Therefore, the portfolio optimization problem consists of determining the weights 𝐰 i subscript 𝐰 𝑖\mathbf{w}_{i}bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT of the portfolio that maximize equation [1](https://arxiv.org/html/2507.16717v1#S3.E1 "In 3.1 Gradient Descent Optimizer ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent") with the two mentioned constraints.

However, in practice, we do not typically work with just expected returns; instead, we work with time series of returns for all assets, such as daily returns for each asset. Therefore, if we already know the weights 𝐰 𝐰\mathbf{w}bold_w (or we are in the process of optimizing them), we can treat the portfolio return as another asset and estimate it using the following equation:

R t=∑i=1 n 𝐰 i⁢𝐫 i,t,subscript 𝑅 𝑡 superscript subscript 𝑖 1 𝑛 subscript 𝐰 𝑖 subscript 𝐫 𝑖 𝑡 R_{t}=\sum_{i=1}^{n}\mathbf{w}_{i}\mathbf{r}_{i,t},italic_R start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ,(4)

where R t subscript 𝑅 𝑡 R_{t}italic_R start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT is the return of the portfolio at time t 𝑡 t italic_t, 𝐫 i,t subscript 𝐫 𝑖 𝑡\mathbf{r}_{i,t}bold_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT is the return of the i 𝑖 i italic_i th asset at time t 𝑡 t italic_t, and 𝐰 i subscript 𝐰 𝑖\mathbf{w}_{i}bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is its corresponding weight in the portfolio. This way, we can calculate the expected return and the variance of the portfolio in a simpler way:

𝔼⁢[𝐑]=𝐑¯=1 T⁢∑t=1 T R t,𝔼 delimited-[]𝐑¯𝐑 1 𝑇 superscript subscript 𝑡 1 𝑇 subscript 𝑅 𝑡\mathbb{E}[\mathbf{R}]=\overline{\mathbf{R}}=\frac{1}{T}\sum_{t=1}^{T}R_{t},blackboard_E [ bold_R ] = over¯ start_ARG bold_R end_ARG = divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT italic_R start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ,(5)

V⁢a⁢r⁢(𝐑)=σ 2⁢(𝐑)=1 T⁢∑t=1 T(R t−𝐑¯)2 𝑉 𝑎 𝑟 𝐑 superscript 𝜎 2 𝐑 1 𝑇 superscript subscript 𝑡 1 𝑇 superscript subscript 𝑅 𝑡¯𝐑 2 Var(\mathbf{R})=\sigma^{2}(\mathbf{R})=\frac{1}{T}\sum_{t=1}^{T}(R_{t}-% \overline{\mathbf{R}})^{2}italic_V italic_a italic_r ( bold_R ) = italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ( bold_R ) = divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT ( italic_R start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT - over¯ start_ARG bold_R end_ARG ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT(6)

Once we have the return series 𝐫 i,t subscript 𝐫 𝑖 𝑡\mathbf{r}_{i,t}bold_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT, we can proceed with the optimization process to build the benchmark. First, we define the variables to be optimized, the pre-weights 𝐳 𝐳\mathbf{z}bold_z for each asset in the portfolio, which are randomly initialized. In addition, to ensure that the portfolio weights are positive and sum to 1, we apply the softmax function to the pre-weights:

𝐰 i=s⁢o⁢f⁢t⁢m⁢a⁢x i⁢(𝐳)=e⁢x⁢p⁢(𝐳 i)∑j=1 n e⁢x⁢p⁢(𝐳 j)subscript 𝐰 𝑖 𝑠 𝑜 𝑓 𝑡 𝑚 𝑎 subscript 𝑥 𝑖 𝐳 𝑒 𝑥 𝑝 subscript 𝐳 𝑖 superscript subscript 𝑗 1 𝑛 𝑒 𝑥 𝑝 subscript 𝐳 𝑗\mathbf{w}_{i}=softmax_{i}(\mathbf{z})=\frac{exp(\mathbf{z}_{i})}{\sum_{j=1}^{% n}exp(\mathbf{z}_{j})}bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_s italic_o italic_f italic_t italic_m italic_a italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_z ) = divide start_ARG italic_e italic_x italic_p ( bold_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) end_ARG start_ARG ∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT italic_e italic_x italic_p ( bold_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) end_ARG(7)

The softmax function converts the pre-weights 𝐳 𝐳\mathbf{z}bold_z into a valid probability distribution, ensuring that the sum of the portfolio weights 𝐰 𝐰\mathbf{w}bold_w equals 1 and that they are non-negative. This step allows us to perform gradient descent optimization while maintaining the constraints. However, softmax has an intrinsic limitation: the resulting probability distribution always has full support. In other words, s⁢o⁢f⁢t⁢m⁢a⁢x i⁢(𝐳)≠0 𝑠 𝑜 𝑓 𝑡 𝑚 𝑎 subscript 𝑥 𝑖 𝐳 0 softmax_{i}(\mathbf{z})\neq 0 italic_s italic_o italic_f italic_t italic_m italic_a italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_z ) ≠ 0 for almost every 𝐳 𝐳\mathbf{z}bold_z and i 𝑖 i italic_i. This is a disadvantage when a sparse probability distribution is desired, such as in the MPO problem. In these cases, it is common to define a threshold below which small probability values are truncated to zero. Thus, we propose to apply an alternative to the softmax function: the sparsemax (Martins and Astudillo, [2016](https://arxiv.org/html/2507.16717v1#bib.bib24)), which is defined as follows:

𝐰=s⁢p⁢a⁢r⁢s⁢e⁢m⁢a⁢x⁢(𝐳)=[𝐳−τ⁢(𝐳)]+,𝐰 𝑠 𝑝 𝑎 𝑟 𝑠 𝑒 𝑚 𝑎 𝑥 𝐳 subscript delimited-[]𝐳 𝜏 𝐳\mathbf{w}=sparsemax(\mathbf{z})=[\mathbf{z}-\tau(\mathbf{z})]_{+},bold_w = italic_s italic_p italic_a italic_r italic_s italic_e italic_m italic_a italic_x ( bold_z ) = [ bold_z - italic_τ ( bold_z ) ] start_POSTSUBSCRIPT + end_POSTSUBSCRIPT ,(8)

where [⋅]+=max⁡(0,⋅)subscript delimited-[]⋅0⋅[\cdot]_{+}=\max(0,\cdot)[ ⋅ ] start_POSTSUBSCRIPT + end_POSTSUBSCRIPT = roman_max ( 0 , ⋅ ), and τ⁢(𝐳)𝜏 𝐳\tau(\mathbf{z})italic_τ ( bold_z ) is the threshold function that satisfies ∑j=1 n 𝐰 i=1 superscript subscript 𝑗 1 𝑛 subscript 𝐰 𝑖 1\sum_{j=1}^{n}\mathbf{w}_{i}=1∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = 1 for every 𝐳 𝐳\mathbf{z}bold_z, which is expressed as:

τ⁢(𝐳)=(∑j≤k⁢(𝐳)𝐳 j)−1 k⁢(𝐳),𝜏 𝐳 subscript 𝑗 𝑘 𝐳 subscript 𝐳 𝑗 1 𝑘 𝐳\tau(\mathbf{z})=\frac{\left(\sum_{j\leq k(\mathbf{z})}\mathbf{z}_{j}\right)-1% }{k(\mathbf{z})},italic_τ ( bold_z ) = divide start_ARG ( ∑ start_POSTSUBSCRIPT italic_j ≤ italic_k ( bold_z ) end_POSTSUBSCRIPT bold_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) - 1 end_ARG start_ARG italic_k ( bold_z ) end_ARG ,(9)

where k⁢(𝐳):=max⁡{k∈[K]⁢|1+k⁢𝐳 k>⁢∑j≤k 𝐳 j}assign 𝑘 𝐳 𝑘 delimited-[]𝐾 ket 1 𝑘 subscript 𝐳 𝑘 subscript 𝑗 𝑘 subscript 𝐳 𝑗 k(\mathbf{z}):=\max\{k\in[K]|1+k\mathbf{z}_{k}>\sum_{j\leq k}\mathbf{z}_{j}\}italic_k ( bold_z ) := roman_max { italic_k ∈ [ italic_K ] | 1 + italic_k bold_z start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT > ∑ start_POSTSUBSCRIPT italic_j ≤ italic_k end_POSTSUBSCRIPT bold_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT }. In other words, the sparsemax function defines a threshold τ i⁢(𝐳)subscript 𝜏 𝑖 𝐳\tau_{i}(\mathbf{z})italic_τ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( bold_z ) for each pre-weight 𝐳 i subscript 𝐳 𝑖\mathbf{z}_{i}bold_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, so that the components lower than its corresponding threshold will be truncated to zero, and the remaining weights sum to 1. In [A](https://arxiv.org/html/2507.16717v1#A1 "Appendix A Softmax vs Sparsemax ‣ Multi-objective Portfolio Optimization Via Gradient Descent") we compare the softmax and the sparsemax functions with the same weights vector. With this setup, we can now define the following loss function for the optimization:

L⁢(𝐳,λ)=−(𝐑¯−λ⁢V⁢a⁢r⁢(𝐑)),𝐿 𝐳 𝜆¯𝐑 𝜆 𝑉 𝑎 𝑟 𝐑 L(\mathbf{z},\lambda)=-(\overline{\mathbf{R}}-\lambda Var(\mathbf{R})),italic_L ( bold_z , italic_λ ) = - ( over¯ start_ARG bold_R end_ARG - italic_λ italic_V italic_a italic_r ( bold_R ) ) ,(10)

which is equivalent to the proposal of Markowitz ([1959](https://arxiv.org/html/2507.16717v1#bib.bib23)) (see equation [1](https://arxiv.org/html/2507.16717v1#S3.E1 "In 3.1 Gradient Descent Optimizer ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). Note the minus sign in the loss function as we aim to maximize the objective. The optimization procedure applies gradient descent to optimize the portfolio pre-weights 𝐳 𝐳\mathbf{z}bold_z. It iteratively adjusts 𝐳 𝐳\mathbf{z}bold_z to minimize the loss function L 𝐿 L italic_L (maximize the objective due to the minus sign), and the sparsemax (also works with softmax) function ensures that the portfolio weights remain valid throughout the iterations.

### 3.2 Multi-objective Loss

Up to this point, we have presented a gradient descent optimizer for the Markowitz mean-variance model. This framework provides a foundational approach to portfolio optimization, but operates under a single-objective paradigm. However, in real-world scenarios, investors face multiple conflicting objectives, adding various constraints such as budget, sector exposure, or regulatory requirements. To address these complexities, MPO extends the traditional Markowitz framework employing weighted terms (similarly to the regularization penalty in the loss function for neural networks) to introduce and manage restrictions effectively. This allows us to incorporate these constraints directly into the loss function:

L⁢(𝐳,λ,λ 1,λ 2,…,λ N)=−(𝐑¯−λ⁢Var⁢(𝐑))+∑i=1 N λ i⁢C i 𝐿 𝐳 𝜆 subscript 𝜆 1 subscript 𝜆 2…subscript 𝜆 𝑁¯𝐑 𝜆 Var 𝐑 superscript subscript 𝑖 1 𝑁 subscript 𝜆 𝑖 subscript 𝐶 𝑖 L(\mathbf{z},\lambda,\lambda_{1},\lambda_{2},...,\lambda_{N})=-(\overline{% \mathbf{R}}-\lambda\,\mathrm{Var}(\mathbf{R}))+\sum_{i=1}^{N}\lambda_{i}C_{i}italic_L ( bold_z , italic_λ , italic_λ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_λ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_λ start_POSTSUBSCRIPT italic_N end_POSTSUBSCRIPT ) = - ( over¯ start_ARG bold_R end_ARG - italic_λ roman_Var ( bold_R ) ) + ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT italic_λ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT(11)

where C i subscript 𝐶 𝑖 C_{i}italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT represents the i 𝑖 i italic_i th objective or constraint, and λ i subscript 𝜆 𝑖\lambda_{i}italic_λ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is the multiplier associated with C i subscript 𝐶 𝑖 C_{i}italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. These multipliers reflect the relative importance of each objective, allowing the model to explore the trade-off between these multiple objectives. Note that the objective function could be replaced by any other of interest. For all other aspects, the gradient descent optimizer described in the previous section remains the same, and the only thing that must be modified is the loss function according to our needs.

### 3.3 Technical considerations for differentiation

When implementing the constraints, we have to pay attention to the technical nuances of differentiation. Certain constraints introduce non-differentiable operations that interfere with the gradient-based optimization framework. This section addresses two common situations that require special care: the enforcement of threshold constraints (sec. [3.3.1](https://arxiv.org/html/2507.16717v1#S3.SS3.SSS1 "3.3.1 Differentiation with threshold constraints ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), and the use of conditional masks for asset selection (sec. [3.3.2](https://arxiv.org/html/2507.16717v1#S3.SS3.SSS2 "3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). In both cases, we implement a technique that maintains compatibility with our optimizer. For reproducibility, the Tensorflow implementation code can be found in [B](https://arxiv.org/html/2507.16717v1#A2 "Appendix B Differentiation techniques in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

#### 3.3.1 Differentiation with threshold constraints

Many constraints ensure that a variable remains below a certain threshold. For example, an asset weight 𝐰 i subscript 𝐰 𝑖\mathbf{w}_{i}bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT cannot exceed 10%percent 10 10\%10 % because we want to force some diversification (w i≤0.1∀i subscript 𝑤 𝑖 0.1 for-all 𝑖 w_{i}\leq 0.1\quad\forall i italic_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ≤ 0.1 ∀ italic_i). This kind of constraints, which include a threshold α=0.1 𝛼 0.1\alpha=0.1 italic_α = 0.1, makes the processing not differentiable. To solve this, we introduce a penalty term C 𝐶 C italic_C that activates when the weights 𝐰 𝐰\mathbf{w}bold_w exceed the threshold α 𝛼\alpha italic_α, as follows:

C=[𝐰−α]+=R⁢e⁢L⁢U⁢(𝐰−α),𝐶 subscript delimited-[]𝐰 𝛼 𝑅 𝑒 𝐿 𝑈 𝐰 𝛼 C=[\mathbf{w}-\alpha]_{+}=ReLU(\mathbf{w}-\alpha),italic_C = [ bold_w - italic_α ] start_POSTSUBSCRIPT + end_POSTSUBSCRIPT = italic_R italic_e italic_L italic_U ( bold_w - italic_α ) ,(12)

where [⋅]+=max⁡(0,⋅)=R⁢e⁢L⁢U⁢(⋅)subscript delimited-[]⋅0⋅𝑅 𝑒 𝐿 𝑈⋅[\cdot]_{+}=\max(0,\cdot)=ReLU(\cdot)[ ⋅ ] start_POSTSUBSCRIPT + end_POSTSUBSCRIPT = roman_max ( 0 , ⋅ ) = italic_R italic_e italic_L italic_U ( ⋅ ). This constraint is non-zero when any 𝐰 i>α subscript 𝐰 𝑖 𝛼\mathbf{w}_{i}>\alpha bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT > italic_α, so the loss function is penalized when this happens.

#### 3.3.2 Differentiation of asset conditional masks

In many MPO problems, conditional masks are often necessary, for instance, in asset selection. For example, if we want to select assets whose weights exceed 50%percent 50 50\%50 %, we can introduce a conditional mask based on a sigmoid function:

m⁢a⁢s⁢k=σ^⁢(x−α)=r⁢o⁢u⁢n⁢d⁢(s⁢i⁢g⁢m⁢o⁢i⁢d⁢(x−α))𝑚 𝑎 𝑠 𝑘^𝜎 𝑥 𝛼 𝑟 𝑜 𝑢 𝑛 𝑑 𝑠 𝑖 𝑔 𝑚 𝑜 𝑖 𝑑 𝑥 𝛼 mask=\hat{\sigma}(x-\alpha)=round(sigmoid(x-\alpha))italic_m italic_a italic_s italic_k = over^ start_ARG italic_σ end_ARG ( italic_x - italic_α ) = italic_r italic_o italic_u italic_n italic_d ( italic_s italic_i italic_g italic_m italic_o italic_i italic_d ( italic_x - italic_α ) )(13)

where x 𝑥 x italic_x is the input variable and α 𝛼\alpha italic_α is a threshold parameter (0.5 0.5 0.5 0.5 in the example). When x>α 𝑥 𝛼 x>\alpha italic_x > italic_α, the rounded sigmoid function is equal to 1 1 1 1, and otherwise it is 0 0, as shown in figure [1](https://arxiv.org/html/2507.16717v1#S3.F1 "Figure 1 ‣ 3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"). The r⁢o⁢u⁢n⁢d 𝑟 𝑜 𝑢 𝑛 𝑑 round italic_r italic_o italic_u italic_n italic_d function ensures that the mask takes binary values, effectively converting the smooth sigmoid into a boolean mask.

![Image 1: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/sigmoid_vs_round_sigmoid.png)

Figure 1: Round effect on the sigmoid activation function, following the equation [13](https://arxiv.org/html/2507.16717v1#S3.E13 "In 3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"). Note that the vertical orange line is exactly at α=0.5 𝛼 0.5\alpha=0.5 italic_α = 0.5, where the mask switches its value.

This function, of course, presents a problem when applying gradient descent: the r⁢o⁢u⁢n⁢d 𝑟 𝑜 𝑢 𝑛 𝑑 round italic_r italic_o italic_u italic_n italic_d function is not differentiable. However, this issue has an easy solution if, for gradient computation during training, we use the derivative of the original sigmoid function before rounding:

s⁢i⁢g⁢m⁢o⁢i⁢d⁢(x)⁢(1−s⁢i⁢g⁢m⁢o⁢i⁢d⁢(x))𝑠 𝑖 𝑔 𝑚 𝑜 𝑖 𝑑 𝑥 1 𝑠 𝑖 𝑔 𝑚 𝑜 𝑖 𝑑 𝑥 sigmoid(x)(1-sigmoid(x))italic_s italic_i italic_g italic_m italic_o italic_i italic_d ( italic_x ) ( 1 - italic_s italic_i italic_g italic_m italic_o italic_i italic_d ( italic_x ) )(14)

This mask (eq. [13](https://arxiv.org/html/2507.16717v1#S3.E13 "In 3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")) allows the optimizer to filter assets under specific conditions while maintaining differentiability during optimization. Gradient updates remain well defined, enabling automatic differentiation through backpropagation, and ensuring that asset constraints are respected while preserving the flexibility of our gradient-based optimization framework. We show the Tensorflow implementation of the r⁢o⁢u⁢n⁢d 𝑟 𝑜 𝑢 𝑛 𝑑 round italic_r italic_o italic_u italic_n italic_d function and its derivative in Table [11](https://arxiv.org/html/2507.16717v1#A2.T11 "Table 11 ‣ Appendix B Differentiation techniques in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), and the mask in Table [12](https://arxiv.org/html/2507.16717v1#A2.T12 "Table 12 ‣ Appendix B Differentiation techniques in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

In summary, these two methods (equations [12](https://arxiv.org/html/2507.16717v1#S3.E12 "In 3.3.1 Differentiation with threshold constraints ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent") and [13](https://arxiv.org/html/2507.16717v1#S3.E13 "In 3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), together with the basic mathematical operations, are sufficient to handle the typical constraints in MPO problems without requiring more complex adjustments, as shown in the following section (sec. [3.4](https://arxiv.org/html/2507.16717v1#S3.SS4 "3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). The constraints are formulated to penalize the loss function, guiding the optimizer to learn how to satisfy the restrictions effectively.

### 3.4 Objectives and constraints

This section provides the primary objectives and constraints considered in the evaluation of our gradient descent optimizer benchmark for MPO. Following the technical considerations for differentiation described in section [3.3](https://arxiv.org/html/2507.16717v1#S3.SS3 "3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), we can easily define different objectives, such as maximizing Sharpe ratio (sec. [3.4.1](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS1 "3.4.1 Objective: Sharpe ratio ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")) or minimizing CVaR (sec. [3.4.2](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS2 "3.4.2 Objective: Conditional Value at Risk (CVaR) ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), and also different constraints: tracking error limitations (sec. [3.4.3](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS3 "3.4.3 Constraint: Maximum Tracking error allowed ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), the adherence to a simplified version of the UCITS directive (sec. [3.4.4](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS4 "3.4.4 Constraint: UCITS directive ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), and conditions on asset weights (sec. [3.4.5](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS5 "3.4.5 Constraint: Active weights over a minimum ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")) and their active range (sec. [3.4.6](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS6 "3.4.6 Constraint: Number of active assets within a range ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). By combining and balancing these objectives and constraints, we aim to consolidate the efficacy of the optimizer for MPO problems. The Tensorflow implementation of these objectives and constraints can be found in [C](https://arxiv.org/html/2507.16717v1#A3 "Appendix C Constraints implementation in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

#### 3.4.1 Objective: Sharpe ratio

Instead of generating the entire set of efficient portfolios by varying the risk aversion parameter λ 𝜆\lambda italic_λ in the standard mean-variance optimization formulation (see equation [1](https://arxiv.org/html/2507.16717v1#S3.E1 "In 3.1 Gradient Descent Optimizer ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), we directly search for the efficient portfolio that maximizes the risk-return trade-off. This is achieved by maximizing the Sharpe ratio, a widely used metric in finance that evaluates portfolio performance relative to its risk. The Sharpe ratio quantifies the difference between the expected return of the portfolio and the risk-free rate, relative to the portfolio’s risk, which is represented by its standard deviation:

S⁢h⁢a⁢r⁢p⁢e⁢R⁢a⁢t⁢i⁢o⁢(𝐑,r f)=𝐑¯−r f σ⁢(𝐑),𝑆 ℎ 𝑎 𝑟 𝑝 𝑒 𝑅 𝑎 𝑡 𝑖 𝑜 𝐑 subscript 𝑟 𝑓¯𝐑 subscript 𝑟 𝑓 𝜎 𝐑 SharpeRatio(\mathbf{R},r_{f})=\frac{\overline{\mathbf{R}}-r_{f}}{\sigma(% \mathbf{R})},italic_S italic_h italic_a italic_r italic_p italic_e italic_R italic_a italic_t italic_i italic_o ( bold_R , italic_r start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT ) = divide start_ARG over¯ start_ARG bold_R end_ARG - italic_r start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT end_ARG start_ARG italic_σ ( bold_R ) end_ARG ,(15)

where 𝐑¯¯𝐑\overline{\mathbf{R}}over¯ start_ARG bold_R end_ARG represents the mean of the portfolio returns 𝐑 𝐑\mathbf{R}bold_R, r f subscript 𝑟 𝑓 r_{f}italic_r start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT is the risk-free rate, and σ⁢(𝐑)𝜎 𝐑\sigma(\mathbf{R})italic_σ ( bold_R ) is the standard deviation of the portfolio returns 𝐑 𝐑\mathbf{R}bold_R, which measures the volatility of the investment. A higher Sharpe ratio indicates that the investment has a higher return relative to its risk. The Sharpe ratio implementation can be found in Table [13](https://arxiv.org/html/2507.16717v1#A3.T13 "Table 13 ‣ Appendix C Constraints implementation in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

#### 3.4.2 Objective: Conditional Value at Risk (CVaR)

The Conditional Value-at-Risk (CVaR) is a risk measure that quantifies the mean of the losses that exceed the Value-at-Risk (VaR) cutoff point:

C⁢V⁢a⁢R⁢(𝐑,α)=V⁢a⁢R α+1 α⁢𝔼⁢[max⁡(−𝐑−V⁢a⁢R α,0)],𝐶 𝑉 𝑎 𝑅 𝐑 𝛼 𝑉 𝑎 subscript 𝑅 𝛼 1 𝛼 𝔼 delimited-[]𝐑 𝑉 𝑎 subscript 𝑅 𝛼 0 CVaR(\mathbf{R},\alpha)=VaR_{\alpha}+\frac{1}{\alpha}\mathbb{E}[\max(-\mathbf{% R}-VaR_{\alpha},0)],italic_C italic_V italic_a italic_R ( bold_R , italic_α ) = italic_V italic_a italic_R start_POSTSUBSCRIPT italic_α end_POSTSUBSCRIPT + divide start_ARG 1 end_ARG start_ARG italic_α end_ARG blackboard_E [ roman_max ( - bold_R - italic_V italic_a italic_R start_POSTSUBSCRIPT italic_α end_POSTSUBSCRIPT , 0 ) ] ,(16)

where:

V⁢a⁢R α=−inf{r:P⁢(𝐑≤r)≥α},𝑉 𝑎 subscript 𝑅 𝛼 infimum conditional-set 𝑟 𝑃 𝐑 𝑟 𝛼 VaR_{\alpha}=-\inf\{r:P(\mathbf{R}\leq r)\geq\alpha\},italic_V italic_a italic_R start_POSTSUBSCRIPT italic_α end_POSTSUBSCRIPT = - roman_inf { italic_r : italic_P ( bold_R ≤ italic_r ) ≥ italic_α } ,(17)

where V⁢a⁢R α 𝑉 𝑎 subscript 𝑅 𝛼 VaR_{\alpha}italic_V italic_a italic_R start_POSTSUBSCRIPT italic_α end_POSTSUBSCRIPT is the value-at-risk at the confidence level α 𝛼\alpha italic_α, which is the minimum value such that the cumulative distribution function (CDF) of the portfolio’s returns 𝐑 𝐑\mathbf{R}bold_R, P⁢(𝐑≤r)𝑃 𝐑 𝑟 P(\mathbf{R}\leq r)italic_P ( bold_R ≤ italic_r ), is at least equal to a certain confidence level α 𝛼\alpha italic_α. The CVaR implementation can be also found in Table [13](https://arxiv.org/html/2507.16717v1#A3.T13 "Table 13 ‣ Appendix C Constraints implementation in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

#### 3.4.3 Constraint: Maximum Tracking error allowed

The Tracking error (TE) measures the differences between the returns of the portfolio and its benchmark (for example, a reference index) to which it is being compared, as described in the following equation:

T⁢r⁢a⁢c⁢k⁢i⁢n⁢g⁢E⁢r⁢r⁢o⁢r⁢(𝐑,𝐈)=σ⁢(𝐑−𝐈),𝑇 𝑟 𝑎 𝑐 𝑘 𝑖 𝑛 𝑔 𝐸 𝑟 𝑟 𝑜 𝑟 𝐑 𝐈 𝜎 𝐑 𝐈 TrackingError(\mathbf{R},\mathbf{I})=\sigma(\mathbf{R}-\mathbf{I}),italic_T italic_r italic_a italic_c italic_k italic_i italic_n italic_g italic_E italic_r italic_r italic_o italic_r ( bold_R , bold_I ) = italic_σ ( bold_R - bold_I ) ,(18)

where 𝐈 𝐈\mathbf{I}bold_I represents the index’s returns, and σ⁢(⋅)𝜎⋅\sigma(\cdot)italic_σ ( ⋅ ) represents the standard deviation. In other words, TE measures the volatility of the difference between the portfolio and index returns. Thus, if we want to limit the maximum TE allowed (T⁢E m⁢a⁢x 𝑇 subscript 𝐸 𝑚 𝑎 𝑥 TE_{max}italic_T italic_E start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT), we must define the following constraint:

C T⁢E=R⁢e⁢L⁢U⁢(T⁢r⁢a⁢c⁢k⁢i⁢n⁢g⁢E⁢r⁢r⁢o⁢r⁢(𝐑,𝐈)−T⁢E m⁢a⁢x)subscript 𝐶 𝑇 𝐸 𝑅 𝑒 𝐿 𝑈 𝑇 𝑟 𝑎 𝑐 𝑘 𝑖 𝑛 𝑔 𝐸 𝑟 𝑟 𝑜 𝑟 𝐑 𝐈 𝑇 subscript 𝐸 𝑚 𝑎 𝑥 C_{TE}=ReLU(TrackingError(\mathbf{R},\mathbf{I})-TE_{max})italic_C start_POSTSUBSCRIPT italic_T italic_E end_POSTSUBSCRIPT = italic_R italic_e italic_L italic_U ( italic_T italic_r italic_a italic_c italic_k italic_i italic_n italic_g italic_E italic_r italic_r italic_o italic_r ( bold_R , bold_I ) - italic_T italic_E start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT )(19)

The Tracking Error implementation in Tensorflow can be found in Table [13](https://arxiv.org/html/2507.16717v1#A3.T13 "Table 13 ‣ Appendix C Constraints implementation in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent"). This constraint implementation and all the remaining ones can be found in Table [14](https://arxiv.org/html/2507.16717v1#A3.T14 "Table 14 ‣ Appendix C Constraints implementation in Tensorflow ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

#### 3.4.4 Constraint: UCITS directive

The Undertaking for Collective Investment in Transferable Securities (UCITS) is a set of regulatory rules in the European Union that allows collective investment schemes to operate freely throughout the EU on the basis of a single authorization from one member state. Essentially, UCITS constraints are designed to ensure that funds are not overly concentrated in a small number of assets or sectors. For simplicity, we focus on the following two constraints: (1) The weight of each asset must be below 10%percent 10 10\%10 %, and (2) the sum of the weights exceeding a lower limit (5%percent 5 5\%5 %) must not exceed an upper limit (40%percent 40 40\%40 %). This way, we must define two additional constraints:

C 10%=∑i=1 n R⁢e⁢L⁢U⁢(𝐰 i−0.1),subscript 𝐶 percent 10 superscript subscript 𝑖 1 𝑛 𝑅 𝑒 𝐿 𝑈 subscript 𝐰 𝑖 0.1 C_{10\%}=\sum_{i=1}^{n}ReLU(\mathbf{w}_{i}-0.1),italic_C start_POSTSUBSCRIPT 10 % end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT italic_R italic_e italic_L italic_U ( bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - 0.1 ) ,(20)

C 5−40%=R⁢e⁢L⁢U⁢((∑i=1 n 𝐰^i)−0.4),subscript 𝐶 5 percent 40 𝑅 𝑒 𝐿 𝑈 superscript subscript 𝑖 1 𝑛 subscript^𝐰 𝑖 0.4 C_{5-40\%}=ReLU\left((\sum_{i=1}^{n}{\mathbf{\hat{w}}_{i}})-0.4\right),italic_C start_POSTSUBSCRIPT 5 - 40 % end_POSTSUBSCRIPT = italic_R italic_e italic_L italic_U ( ( ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT over^ start_ARG bold_w end_ARG start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) - 0.4 ) ,(21)

where:

𝐰^=𝐰⊙σ^⁢(𝐰−0.05),^𝐰 direct-product 𝐰^𝜎 𝐰 0.05\hat{\mathbf{w}}=\mathbf{w}\odot\hat{\sigma}(\mathbf{w}-0.05),over^ start_ARG bold_w end_ARG = bold_w ⊙ over^ start_ARG italic_σ end_ARG ( bold_w - 0.05 ) ,(22)

where ⊙direct-product\odot⊙ represents the element-wise product, and σ^^𝜎\hat{\sigma}over^ start_ARG italic_σ end_ARG is the r⁢o⁢u⁢n⁢d 𝑟 𝑜 𝑢 𝑛 𝑑 round italic_r italic_o italic_u italic_n italic_d sigmoid function described in equation [13](https://arxiv.org/html/2507.16717v1#S3.E13 "In 3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

#### 3.4.5 Constraint: Active weights over a minimum

The following constraint forces the weights of the active assets to exceed a minimum value m 𝑚 m italic_m. Thus, we must penalize those weights in the range (0,m)0 𝑚(0,m)( 0 , italic_m ), so we define a mask that filters the corresponding weights and then apply a penalty, as shown in the following equations:

𝐰^=𝐰⊙σ^⁢(m−𝐰)^𝐰 direct-product 𝐰^𝜎 𝑚 𝐰\mathbf{\hat{w}}=\mathbf{w}\odot\hat{\sigma}(m-\mathbf{w})over^ start_ARG bold_w end_ARG = bold_w ⊙ over^ start_ARG italic_σ end_ARG ( italic_m - bold_w )(23)

C m⁢i⁢n=∑i=1 n w^i subscript 𝐶 𝑚 𝑖 𝑛 superscript subscript 𝑖 1 𝑛 subscript^𝑤 𝑖 C_{min}=\sum_{i=1}^{n}\hat{w}_{i}italic_C start_POSTSUBSCRIPT italic_m italic_i italic_n end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT over^ start_ARG italic_w end_ARG start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT(24)

#### 3.4.6 Constraint: Number of active assets within a range

The following constraint requires the number of active assets to be within the range (l⁢o⁢w,h⁢i⁢g⁢h)𝑙 𝑜 𝑤 ℎ 𝑖 𝑔 ℎ(low,high)( italic_l italic_o italic_w , italic_h italic_i italic_g italic_h ). To solve this, we must define a mask 𝐦 𝐦\mathbf{m}bold_m that allows us to count the number of selected assets, and then define a penalty by combining two thresholds, as shown in the following equations:

𝐦=σ^⁢(𝐰),𝐦^𝜎 𝐰\mathbf{m}=\hat{\sigma}(\mathbf{w}),bold_m = over^ start_ARG italic_σ end_ARG ( bold_w ) ,(25)

C r⁢a⁢n⁢g⁢e=R⁢e⁢L⁢U⁢((l⁢o⁢w−∑i=1 n 𝐦 i)⁢(h⁢i⁢g⁢h−∑i=1 n 𝐦 i)),subscript 𝐶 𝑟 𝑎 𝑛 𝑔 𝑒 𝑅 𝑒 𝐿 𝑈 𝑙 𝑜 𝑤 superscript subscript 𝑖 1 𝑛 subscript 𝐦 𝑖 ℎ 𝑖 𝑔 ℎ superscript subscript 𝑖 1 𝑛 subscript 𝐦 𝑖 C_{range}=ReLU\left((low-\sum_{i=1}^{n}\mathbf{m}_{i})(high-\sum_{i=1}^{n}% \mathbf{m}_{i})\right),italic_C start_POSTSUBSCRIPT italic_r italic_a italic_n italic_g italic_e end_POSTSUBSCRIPT = italic_R italic_e italic_L italic_U ( ( italic_l italic_o italic_w - ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT bold_m start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ( italic_h italic_i italic_g italic_h - ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT bold_m start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ) ,(26)

where l⁢o⁢w 𝑙 𝑜 𝑤 low italic_l italic_o italic_w and h⁢i⁢g⁢h ℎ 𝑖 𝑔 ℎ high italic_h italic_i italic_g italic_h define the allowed range, 𝐦 𝐦\mathbf{m}bold_m defines the vector mask (𝐦 i=0 subscript 𝐦 𝑖 0\mathbf{m}_{i}=0 bold_m start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = 0 if w i<0 subscript 𝑤 𝑖 0 w_{i}<0 italic_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT < 0, else 𝐦 i=1 subscript 𝐦 𝑖 1\mathbf{m}_{i}=1 bold_m start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = 1).

#### 3.4.7 Constraint: Sum of the weights of the assets selected by a mask

The following constraint requires that the total weight assigned to the assets belonging to each predefined group (or mask) does not exceed a specified threshold. Each mask may contain an arbitrary number of assets and multiple masks can be defined. The combined weight across all masks must remain within the feasible budget, typically constrained by the total portfolio weight (e.g., not exceeding 1).

Let 𝐌∈{0,1}n×m 𝐌 superscript 0 1 𝑛 𝑚\mathbf{M}\in\{0,1\}^{n\times m}bold_M ∈ { 0 , 1 } start_POSTSUPERSCRIPT italic_n × italic_m end_POSTSUPERSCRIPT be a binary mask matrix, where n 𝑛 n italic_n is the number of assets and m 𝑚 m italic_m the number of masks. An element 𝐌 i⁢j=1 subscript 𝐌 𝑖 𝑗 1\mathbf{M}_{ij}=1 bold_M start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = 1 if the i 𝑖 i italic_i th asset belongs to the j 𝑗 j italic_j th mask, and 0 0 otherwise. Also, let 𝐦 max∈ℝ m subscript 𝐦 max superscript ℝ 𝑚\mathbf{m}_{\text{max}}\in\mathbb{R}^{m}bold_m start_POSTSUBSCRIPT max end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_m end_POSTSUPERSCRIPT represent the maximum total weight allowed for each mask. The penalty for deviations from the constraint is given by:

C mask=∑j=1 m|𝐦 max,j−∑i=1 n 𝐰 i⁢𝐌 i⁢j|subscript 𝐶 mask superscript subscript 𝑗 1 𝑚 subscript 𝐦 max 𝑗 superscript subscript 𝑖 1 𝑛 subscript 𝐰 𝑖 subscript 𝐌 𝑖 𝑗 C_{\text{mask}}=\sum_{j=1}^{m}|\mathbf{m}_{\text{max},j}-\sum_{i=1}^{n}\mathbf% {w}_{i}\mathbf{M}_{ij}|italic_C start_POSTSUBSCRIPT mask end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_j = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_m end_POSTSUPERSCRIPT | bold_m start_POSTSUBSCRIPT max , italic_j end_POSTSUBSCRIPT - ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT bold_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT bold_M start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT |(27)

Note that the sum of the maximum total weights allowed for each mask must not exceed 1, although it may be lower than 1 (∑i=1 m 𝐦 m⁢a⁢x,i≤1 superscript subscript 𝑖 1 𝑚 subscript 𝐦 𝑚 𝑎 𝑥 𝑖 1\sum_{i=1}^{m}\mathbf{m}_{max,i}\leq 1∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_m end_POSTSUPERSCRIPT bold_m start_POSTSUBSCRIPT italic_m italic_a italic_x , italic_i end_POSTSUBSCRIPT ≤ 1). This allows the optimizer the flexibility to allocate the remaining portion as it deems optimal.

### 3.5 Experiments

For the experiments, we use the assets from the S&P 500 index, which includes approximately 500 of the largest publicly traded companies in the U.S., offering a diverse set of assets that are ideal for testing and evaluating optimization models. This index is widely recognized as one of the most common benchmarks in portfolio optimization and finance in general, and is used frequently by investors, analysts, and researchers alike. The S&P 500 provides a sufficient number of assets to build different scenarios under various constraints and objectives; therefore, it is a natural choice for benchmarking MPO problems. Furthermore, it is readily available to anyone through platforms like Yahoo Finance, which offers easy access to historical data through its libraries. In this scenario, for the sake of evaluation of our gradient descent optimizer benchmark, we propose six different situations where the only difference between the optimizers is the loss function.

#### 3.5.1 Case 1: Maximize Sharpe ratio

For the first case, we consider a single-objective optimization problem closely related to the Markowitz model (see Equation [1](https://arxiv.org/html/2507.16717v1#S3.E1 "In 3.1 Gradient Descent Optimizer ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). Specifically, the problem seeks to maximize the Sharpe ratio, as detailed in Section [3.4.1](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS1 "3.4.1 Objective: Sharpe ratio ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"). The loss function that optimizes the Sharpe ratio is defined as:

L⁢(𝐳)=−𝐑¯−r f σ⁢(𝐑)𝐿 𝐳¯𝐑 subscript 𝑟 𝑓 𝜎 𝐑 L(\mathbf{z})=-\frac{\overline{\mathbf{R}}-r_{f}}{\sigma(\mathbf{R})}italic_L ( bold_z ) = - divide start_ARG over¯ start_ARG bold_R end_ARG - italic_r start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT end_ARG start_ARG italic_σ ( bold_R ) end_ARG

where the objective function L⁢(𝐳)𝐿 𝐳 L(\mathbf{z})italic_L ( bold_z ) is designed for minimization, so its negative formulation ensures that maximizing the Sharpe ratio is equivalent to minimizing L⁢(𝐳)𝐿 𝐳 L(\mathbf{z})italic_L ( bold_z ). This simple scenario allows us to compare the results of our proposal with those achieved by the famous CVXPY convex optimization library (Diamond and Boyd, [2016](https://arxiv.org/html/2507.16717v1#bib.bib9); Agrawal et al., [2018](https://arxiv.org/html/2507.16717v1#bib.bib2)).

#### 3.5.2 Case 2: Minimize CVaR

The second scenario is also a single-objective optimization, but this time the objective is to minimize CVaR (see section [3.4.2](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS2 "3.4.2 Objective: Conditional Value at Risk (CVaR) ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), that means to minimize extreme losses. The loss function is the following:

L⁢(𝐳)=C⁢V⁢a⁢R⁢(𝐑,α),𝐿 𝐳 𝐶 𝑉 𝑎 𝑅 𝐑 𝛼 L(\mathbf{z})=CVaR(\mathbf{R},\alpha),italic_L ( bold_z ) = italic_C italic_V italic_a italic_R ( bold_R , italic_α ) ,(28)

where 𝐑 𝐑\mathbf{R}bold_R is the series of returns in the portfolio and α 𝛼\alpha italic_α is the confidence level to calculate the VaR cutoff point (typically α=0.05 𝛼 0.05\alpha=0.05 italic_α = 0.05). In this case, the objective function L⁢(𝐳)𝐿 𝐳 L(\mathbf{z})italic_L ( bold_z ) minimizes CVaR, which is equivalent to minimizing extreme losses. Once again, this single-objective scenario will be useful for comparing the benchmark with exact algorithms, implemented in this case by the SKFOLIO library (Hugo Delatte, [2023](https://arxiv.org/html/2507.16717v1#bib.bib17)).

#### 3.5.3 Case 3: Minimize CVaR with UCITS constraints

In this scenario, our objective is to minimize CVaR while simultaneously satisfying a simplified version of the UCITS constraints, as detailed in Section [3.4.4](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS4 "3.4.4 Constraint: UCITS directive ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"). This is our first multi-objective portfolio optimization, where some constraints are used. The loss function is defined as follows:

L⁢(𝐳,λ 10%,λ 5−40%)=C⁢V⁢a⁢R⁢(𝐰,α)+λ 10%⁢C 10%+λ 5−40%⁢C 5−40%,𝐿 𝐳 subscript 𝜆 percent 10 subscript 𝜆 5 percent 40 𝐶 𝑉 𝑎 𝑅 𝐰 𝛼 subscript 𝜆 percent 10 subscript 𝐶 percent 10 subscript 𝜆 5 percent 40 subscript 𝐶 5 percent 40 L(\mathbf{z},\lambda_{10\%},\lambda_{5-40\%})=CVaR(\mathbf{w},\alpha)+\lambda_% {10\%}\,C_{10\%}+\lambda_{5-40\%}\,C_{5-40\%},italic_L ( bold_z , italic_λ start_POSTSUBSCRIPT 10 % end_POSTSUBSCRIPT , italic_λ start_POSTSUBSCRIPT 5 - 40 % end_POSTSUBSCRIPT ) = italic_C italic_V italic_a italic_R ( bold_w , italic_α ) + italic_λ start_POSTSUBSCRIPT 10 % end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT 10 % end_POSTSUBSCRIPT + italic_λ start_POSTSUBSCRIPT 5 - 40 % end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT 5 - 40 % end_POSTSUBSCRIPT ,(29)

where λ 10%subscript 𝜆 percent 10\lambda_{10\%}italic_λ start_POSTSUBSCRIPT 10 % end_POSTSUBSCRIPT and λ 5−40%subscript 𝜆 5 percent 40\lambda_{5-40\%}italic_λ start_POSTSUBSCRIPT 5 - 40 % end_POSTSUBSCRIPT are scaling hyperparameters that quantify the penalty for constraints and the objective. Lower values of these hyperparameters result in more permissive enforcement, while higher values impose stricter penalties. Note that the incorporation of additional constraints is achieved by introducing a new hyperparameter λ x subscript 𝜆 𝑥\lambda_{x}italic_λ start_POSTSUBSCRIPT italic_x end_POSTSUBSCRIPT multiplied by the corresponding constraint C x subscript 𝐶 𝑥 C_{x}italic_C start_POSTSUBSCRIPT italic_x end_POSTSUBSCRIPT.

#### 3.5.4 Case 4: Minimize CVaR with a tracking error constraint

In this scenario, our objective is to minimize CVaR while limiting the maximum allowable tracking error (see the details of the constraint in Section [3.4.3](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS3 "3.4.3 Constraint: Maximum Tracking error allowed ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). The loss function is defined as follows:

L⁢(𝐳,λ T⁢E)=C⁢V⁢a⁢R⁢(𝐑,α)+λ T⁢E⁢C T⁢E,𝐿 𝐳 subscript 𝜆 𝑇 𝐸 𝐶 𝑉 𝑎 𝑅 𝐑 𝛼 subscript 𝜆 𝑇 𝐸 subscript 𝐶 𝑇 𝐸 L(\mathbf{z},\lambda_{TE})=CVaR(\mathbf{R},\alpha)+\lambda_{TE}\,C_{TE},italic_L ( bold_z , italic_λ start_POSTSUBSCRIPT italic_T italic_E end_POSTSUBSCRIPT ) = italic_C italic_V italic_a italic_R ( bold_R , italic_α ) + italic_λ start_POSTSUBSCRIPT italic_T italic_E end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_T italic_E end_POSTSUBSCRIPT ,(30)

where λ T⁢E subscript 𝜆 𝑇 𝐸\lambda_{TE}italic_λ start_POSTSUBSCRIPT italic_T italic_E end_POSTSUBSCRIPT is the scaling hyperparameter that controls the penalty for exceeding the maximum allowed value T⁢E m⁢a⁢x 𝑇 subscript 𝐸 𝑚 𝑎 𝑥 TE_{max}italic_T italic_E start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT. If it is critical that the tracking error does not exceed the maximum limit, λ T⁢E subscript 𝜆 𝑇 𝐸\lambda_{TE}italic_λ start_POSTSUBSCRIPT italic_T italic_E end_POSTSUBSCRIPT should be set to a high value. In contrast, if a more flexible approach to this constraint is acceptable, it can be reduced.

#### 3.5.5 Case 5: Joint optimization of Sharpe ratio and CVaR under Tracking Error, UCITS, Minimum Active Asset Weights, and Active Asset Count Constraints

In this case, we pursue two objectives: maximizing the Sharpe ratio and minimizing CVaR, while simultaneously enforcing several constraints. Specifically, the tracking error must not exceed an upper limit, the portfolio must comply with UCITS regulations, each selected asset must meet a minimum weight requirement, and the total number of selected assets must fall within a specified range. Given the dual objectives, they are balanced through lambda hyperparameters, resulting in the following loss function:

L⁢(𝐳,λ 1,λ 2,λ¯,C¯)=−λ 1⁢𝐑¯−r f σ⁢(𝐑)+λ 2⁢C⁢V⁢a⁢R⁢(𝐑,α)+λ¯⁢C¯T,𝐿 𝐳 subscript 𝜆 1 subscript 𝜆 2¯𝜆¯𝐶 subscript 𝜆 1¯𝐑 subscript 𝑟 𝑓 𝜎 𝐑 subscript 𝜆 2 𝐶 𝑉 𝑎 𝑅 𝐑 𝛼¯𝜆 superscript¯𝐶 𝑇 L(\mathbf{z},\lambda_{1},\lambda_{2},\overline{\lambda},\overline{C})=-\lambda% _{1}\frac{\overline{\mathbf{R}}-r_{f}}{\sigma(\mathbf{R})}+\lambda_{2}\,CVaR(% \mathbf{R},\alpha)+\overline{\lambda}\,\overline{C}^{T},italic_L ( bold_z , italic_λ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_λ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , over¯ start_ARG italic_λ end_ARG , over¯ start_ARG italic_C end_ARG ) = - italic_λ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT divide start_ARG over¯ start_ARG bold_R end_ARG - italic_r start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT end_ARG start_ARG italic_σ ( bold_R ) end_ARG + italic_λ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT italic_C italic_V italic_a italic_R ( bold_R , italic_α ) + over¯ start_ARG italic_λ end_ARG over¯ start_ARG italic_C end_ARG start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT ,(31)

where λ 1 subscript 𝜆 1\lambda_{1}italic_λ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, λ 2 subscript 𝜆 2\lambda_{2}italic_λ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT, and λ¯={λ T⁢E,λ 10%,λ 5−40%,λ m⁢i⁢n,λ r⁢a⁢n⁢g⁢e}¯𝜆 subscript 𝜆 𝑇 𝐸 subscript 𝜆 percent 10 subscript 𝜆 5 percent 40 subscript 𝜆 𝑚 𝑖 𝑛 subscript 𝜆 𝑟 𝑎 𝑛 𝑔 𝑒\overline{\lambda}=\{\lambda_{TE},\lambda_{10\%},\lambda_{5-40\%},\lambda_{min% },\lambda_{range}\}over¯ start_ARG italic_λ end_ARG = { italic_λ start_POSTSUBSCRIPT italic_T italic_E end_POSTSUBSCRIPT , italic_λ start_POSTSUBSCRIPT 10 % end_POSTSUBSCRIPT , italic_λ start_POSTSUBSCRIPT 5 - 40 % end_POSTSUBSCRIPT , italic_λ start_POSTSUBSCRIPT italic_m italic_i italic_n end_POSTSUBSCRIPT , italic_λ start_POSTSUBSCRIPT italic_r italic_a italic_n italic_g italic_e end_POSTSUBSCRIPT } balance the importance of objectives and constraints in the loss function, and C¯={C T⁢E,C 10%,C 5−40%,C m⁢i⁢n,C r⁢a⁢n⁢g⁢e}¯𝐶 subscript 𝐶 𝑇 𝐸 subscript 𝐶 percent 10 subscript 𝐶 5 percent 40 subscript 𝐶 𝑚 𝑖 𝑛 subscript 𝐶 𝑟 𝑎 𝑛 𝑔 𝑒\overline{C}=\{C_{TE},C_{10\%},C_{5-40\%},C_{min},C_{range}\}over¯ start_ARG italic_C end_ARG = { italic_C start_POSTSUBSCRIPT italic_T italic_E end_POSTSUBSCRIPT , italic_C start_POSTSUBSCRIPT 10 % end_POSTSUBSCRIPT , italic_C start_POSTSUBSCRIPT 5 - 40 % end_POSTSUBSCRIPT , italic_C start_POSTSUBSCRIPT italic_m italic_i italic_n end_POSTSUBSCRIPT , italic_C start_POSTSUBSCRIPT italic_r italic_a italic_n italic_g italic_e end_POSTSUBSCRIPT } is the vector constraints, whose components are defined in equations [19](https://arxiv.org/html/2507.16717v1#S3.E19 "In 3.4.3 Constraint: Maximum Tracking error allowed ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), [20](https://arxiv.org/html/2507.16717v1#S3.E20 "In 3.4.4 Constraint: UCITS directive ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), [21](https://arxiv.org/html/2507.16717v1#S3.E21 "In 3.4.4 Constraint: UCITS directive ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), [24](https://arxiv.org/html/2507.16717v1#S3.E24 "In 3.4.5 Constraint: Active weights over a minimum ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), and [26](https://arxiv.org/html/2507.16717v1#S3.E26 "In 3.4.6 Constraint: Number of active assets within a range ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"). Note the negative sign preceding the Sharpe ratio term, which reflects its maximization objective. This larger case, compared to the previous cases, serves as an example of how multi-objective optimization can be performed just by adding further objectives or constraints to the loss function.

#### 3.5.6 Case 6: Minimize Risk with weight constraints in assets subsets

Finally, we have a single objective: minimize volatility measured as the standard deviation of daily logarithmic returns, while enforcing upper bounds on the total weights assigned to specific subsets of assets (see section [3.4.7](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS7 "3.4.7 Constraint: Sum of the weights of the assets selected by a mask ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). This type of constraint can be particularly useful for limiting the portfolio’s exposure to certain sectors or themes.

In this last case, we define four asset masks, each containing ten randomly chosen assets from the investment universe. Each mask represents a subset of assets for which we want to control the aggregate weight. A random maximum weight is assigned to each mask, and the total combined weight allowed in all four masks is 90% of the portfolio. This allows the optimizer to allocate the remaining 10% of the portfolio to assets outside of the defined masks. It is important to mention that this constraint does not require the optimizer to assign weights to every asset within each mask, but only enforces that the total weight within each mask does not exceed its respective upper limit. The resulting loss function is defined as:

L⁢(𝐳,λ m⁢a⁢s⁢k)=σ⁢(𝐑)−λ m⁢a⁢s⁢k⁢C m⁢a⁢s⁢k,𝐿 𝐳 subscript 𝜆 𝑚 𝑎 𝑠 𝑘 𝜎 𝐑 subscript 𝜆 𝑚 𝑎 𝑠 𝑘 subscript 𝐶 𝑚 𝑎 𝑠 𝑘 L(\mathbf{z},\lambda_{mask})=\sigma(\mathbf{R})-\lambda_{mask}\,C_{mask},italic_L ( bold_z , italic_λ start_POSTSUBSCRIPT italic_m italic_a italic_s italic_k end_POSTSUBSCRIPT ) = italic_σ ( bold_R ) - italic_λ start_POSTSUBSCRIPT italic_m italic_a italic_s italic_k end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_m italic_a italic_s italic_k end_POSTSUBSCRIPT ,(32)

where λ m⁢a⁢s⁢k subscript 𝜆 𝑚 𝑎 𝑠 𝑘\lambda_{mask}italic_λ start_POSTSUBSCRIPT italic_m italic_a italic_s italic_k end_POSTSUBSCRIPT balances the importance of the objective and constraint in the loss function, 𝐑 𝐑\mathbf{R}bold_R denotes the portfolio return, and C m⁢a⁢s⁢k subscript 𝐶 𝑚 𝑎 𝑠 𝑘 C_{mask}italic_C start_POSTSUBSCRIPT italic_m italic_a italic_s italic_k end_POSTSUBSCRIPT is the constraint defined in equation [27](https://arxiv.org/html/2507.16717v1#S3.E27 "In 3.4.7 Constraint: Sum of the weights of the assets selected by a mask ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

Through these six cases, our aim is to demonstrate the flexibility of our optimization framework in handling single- and multi-objective portfolio optimization problems with diverse constraints. The following section shows the effectiveness of our gradient descent optimizer in MPO.

4 Results and Analysis
----------------------

This section presents the experimental evaluation of our framework for MPO problems. We evaluate the performance of the optimizer in the six distinct scenarios detailed in the previous section. The results are analyzed by comparing our gradient descent (GD) optimizer with well-established optimization tools such as CVXPY and SKFOLIO in the simpler scenarios. For more complex constrained cases, we focus on verifying its ability to satisfy regulatory constraints while maintaining strong risk-return profiles. Performance metrics such as Sharpe ratio, Tracking Error, Value-at-Risk (VaR), and Conditional Value-at-Risk (CVaR) are used to evaluate the resulting portfolios. In all cases, the hyperparameters for the optimizer are given. The training dataset is composed by the S&P 500 Index and its components from 2019-12-31 to 2020-12-31.

### 4.1 Case 1: Maximize Sharpe ratio

In this first experiment, we compare the exact solution obtained using the CVXPY library (Diamond and Boyd, [2016](https://arxiv.org/html/2507.16717v1#bib.bib9); Agrawal et al., [2018](https://arxiv.org/html/2507.16717v1#bib.bib2)), which performs convex optimization, with the results from our gradient descent-based optimizer, trained with a learning rate of 0.01 0.01 0.01 0.01 for 500 500 500 500 epochs. Figure[2](https://arxiv.org/html/2507.16717v1#S4.F2 "Figure 2 ‣ 4.1 Case 1: Maximize Sharpe ratio ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") presents the portfolio weights resulting from both optimization methods, displaying only the assets with non-zero allocations.

![Image 2: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_01/MaxSharpe_portfolio_weights_comparison.png)

Figure 2: Weights comparison for selected assets (CVXPY vs GD) of S&P 500, whose weights are greater than 0 0.

As shown in the figure, the weights for the selected assets are almost equal in both optimizations, with some slight differences shown in Table [1](https://arxiv.org/html/2507.16717v1#S4.T1 "Table 1 ‣ 4.1 Case 1: Maximize Sharpe ratio ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), which shows the exact values.

Table 1: Weights comparison for selected assets (CVXPY vs GD) of S&P 500, whose weights are greater than 0 0. The weights are almost equal, differing from the 6th decimal.

In this first experiment, we show that our gradient descent optimizer obtains almost the same exact solution as the CVXPY convex optimization. For an additional comparison, we show in Table [2](https://arxiv.org/html/2507.16717v1#S4.T2 "Table 2 ‣ 4.1 Case 1: Maximize Sharpe ratio ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") some performance metrics: the Sharpe ratio of the portfolio, its Tracking Error evaluated over the S&P 500 index, its value-at-risk, and the conditional value-at-risk, where there are no differences between them.

Table 2: Metrics comparison (CVXPY vs GD) when maximizing Sharpe ratio. The obtained values are totally equal.

To ensure the robustness of the results, we conducted K=100 𝐾 100 K=100 italic_K = 100 random simulations to measure the similarity between both methods. In each simulation, both the set of assets and the training time window are randomly selected, with a universe of 10 assets minimum. For each k∈{1,…,K}𝑘 1…𝐾 k\in\{1,\dots,K\}italic_k ∈ { 1 , … , italic_K }, let 𝐮(k)=(u 1(k),…,u d(k))superscript 𝐮 𝑘 superscript subscript 𝑢 1 𝑘…superscript subscript 𝑢 𝑑 𝑘\mathbf{u}^{(k)}=(u_{1}^{(k)},\dots,u_{d}^{(k)})bold_u start_POSTSUPERSCRIPT ( italic_k ) end_POSTSUPERSCRIPT = ( italic_u start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_k ) end_POSTSUPERSCRIPT , … , italic_u start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_k ) end_POSTSUPERSCRIPT ) represent the optimal weight vector obtained from CVXPY, and 𝐯(k)=(v 1(k),…,v d(k))superscript 𝐯 𝑘 superscript subscript 𝑣 1 𝑘…superscript subscript 𝑣 𝑑 𝑘\mathbf{v}^{(k)}=(v_{1}^{(k)},\dots,v_{d}^{(k)})bold_v start_POSTSUPERSCRIPT ( italic_k ) end_POSTSUPERSCRIPT = ( italic_v start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_k ) end_POSTSUPERSCRIPT , … , italic_v start_POSTSUBSCRIPT italic_d end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_k ) end_POSTSUPERSCRIPT ) the corresponding vector from GD. The dimension d 𝑑 d italic_d denotes the number of assets in the portfolio for simulation k 𝑘 k italic_k.

To measure the discrepancy between both methods, we compute the Euclidean distance between the vectors u 𝑢 u italic_u and v 𝑣 v italic_v on average. This metric provides a direct quantification of how close the GD-based and CVXPY-based optimizations are in terms of the overall weight allocation. A low MSE indicates that both methods yield nearly identical portfolio compositions, while a high MSE reveals a significant divergence.

The result of the simulation is an MSE between CVXPY and GD portfolios of 3.6179×10−5 3.6179 superscript 10 5 3.6179\times 10^{-5}3.6179 × 10 start_POSTSUPERSCRIPT - 5 end_POSTSUPERSCRIPT, which means identical results in practice.

### 4.2 Case 2: Minimize CVaR

In the second experiment, our objective is to minimize the Conditional Value-at-Risk (CVaR) for the portfolio. We compare exact portfolio allocations and performance metrics obtained by the SKFOLIO optimization method with our gradient descent optimizer, trained with a learning rate of 0.001 0.001 0.001 0.001 for 2000 2000 2000 2000 epochs. Figure[3](https://arxiv.org/html/2507.16717v1#S4.F3 "Figure 3 ‣ 4.2 Case 2: Minimize CVaR ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") shows the portfolio weights for the selected assets as determined by both methods. The exact weight values are provided in Table[3](https://arxiv.org/html/2507.16717v1#S4.T3 "Table 3 ‣ 4.2 Case 2: Minimize CVaR ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

![Image 3: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_02/MinCVaR_portfolio_weights_comparison.png)

Figure 3: Weights comparison for selected assets (SKFOLIO vs GD).

Table 3: Weights comparison for selected assets (SKFOLIO vs GD) of S&P 500, whose weights are greater than 0 in one option at least.

As observed in the figure and table, there are clear differences in the individual allocations, including different assets selected and different weights. For example, SKFOLIO chooses Erie Indemnity Company (ERIE) to invest in, while our gradient descent optimizer ignores it and prefers to invest in Hormel Foods Corporation (HRL) and Viatris Inc. (VTRS). In addition, the portfolio weights are different for each optimizer. However, it must also be considered that these 11 assets have been chosen from more than 400 assets in the dataset, and both methods share 8 of them. The key to understanding these differences appears when we show, in addition to the weights, the performance of the portfolios. First, we examine the accumulated returns, as shown in Figure[4](https://arxiv.org/html/2507.16717v1#S4.F4 "Figure 4 ‣ 4.2 Case 2: Minimize CVaR ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), supporting the similarity in performance between the two portfolios.

![Image 4: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_02/MinCVaR_portfolio_acum_returns_comparison.png)

Figure 4: Acummulated returns comparison (SKFOLIO vs GD)

Second, we analyze the key performance metrics, shown in Table[4](https://arxiv.org/html/2507.16717v1#S4.T4 "Table 4 ‣ 4.2 Case 2: Minimize CVaR ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), where we include Sharpe ratio, Tracking Error, VaR, and CVaR. The metrics indicate that, while there are differences, the GD optimizer closely approximates the SKFOLIO solution in terms of CVaR, which is the objective of the optimization, showing differences in the 5th decimal.

Table 4: Metrics comparison (SKFOLIO vs GD) when minimizing CVaR. The obtained values differ, but the objective to minimize (CVaR) is almost equal, differing in the 5th decimal.

Although the performance metrics and accumulated returns of both methods appear comparable, especially in terms of CVaR, where the difference is negligible, there is a fundamental reason why the gradient descent optimizer does not exactly replicate the SKFOLIO solution. The CVaR, in its standard form (see section [3.4.2](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS2 "3.4.2 Objective: Conditional Value at Risk (CVaR) ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")), is known to lead to a generally non-convex optimization problem. However, it is possible to transform the problem into a linear program in order to use convex optimization algorithms, like CVXPY or SKFOLIO, and find the global solution.

Our gradient descent optimizer is an iterative algorithm that solves convex and non-convex problems indistinctly, yet it still achieves results that closely approximate those of the global solution while avoiding the additional complexity in the mathematical transformations. The results shown in table [4](https://arxiv.org/html/2507.16717v1#S4.T4 "Table 4 ‣ 4.2 Case 2: Minimize CVaR ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") and figure [4](https://arxiv.org/html/2507.16717v1#S4.F4 "Figure 4 ‣ 4.2 Case 2: Minimize CVaR ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") are just an example of how our algorithm converges to suboptimal local minima. Considering the following cases, where constrained non-convex multi-objective problems are presented, our gradient descent benchmark is completely valid and reliable.

Consistent with the procedure in Case 1, we ran 100 random simulations and evaluated the mean squared error (MSE) across them. In addition, we have calculated the difference of both the SKFOLIO and GD CVaR of each simulation to evaluate the discrepancy between the exact method (SKFOLIO) and our GD benchmark, since there are small variations in the results shown in figure [3](https://arxiv.org/html/2507.16717v1#S4.F3 "Figure 3 ‣ 4.2 Case 2: Minimize CVaR ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent").

The MSE between the weights of the SKFOLIO and GD portfolios was 0.003646 0.003646 0.003646 0.003646, and the MSE of the CVaR of each of them was 8.7775×10−10 8.7775 superscript 10 10 8.7775\times 10^{-10}8.7775 × 10 start_POSTSUPERSCRIPT - 10 end_POSTSUPERSCRIPT, which means almost identical weights and the same achievement of the CVaR objective.

### 4.3 Case 3: Minimize CVaR with UCITS constraints

In this experiment, we extend the CVaR optimization problem by incorporating simplified UCITS constraints, emulating that the portfolio complies with regulatory requirements. In this case, this is a non-DCP (Disciplined Convex Programming) problem, thus CVXPY and SKFOLIO can not find an exact solution, and here is where our benchmark highlights. It was trained with a learning rate of 0.001 0.001 0.001 0.001 for 2000 2000 2000 2000 epochs. The objective and constraints hyperparameters were set to λ 10%=1.0 subscript 𝜆 10%1.0\lambda_{\text{10\%}}=1.0 italic_λ start_POSTSUBSCRIPT 10% end_POSTSUBSCRIPT = 1.0, and λ 5-40%=1.0 subscript 𝜆 5-40%1.0\lambda_{\text{5-40\%}}=1.0 italic_λ start_POSTSUBSCRIPT 5-40% end_POSTSUBSCRIPT = 1.0. The training process of the gradient descent-based optimizer is illustrated in Figure[5](https://arxiv.org/html/2507.16717v1#S4.F5 "Figure 5 ‣ 4.3 Case 3: Minimize CVaR with UCITS constraints ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), which shows the evolution of the training metrics and confirms the convergence of the optimization algorithm. From left to right, the first plot (Loss) shows the optimization global loss, which follows equation [29](https://arxiv.org/html/2507.16717v1#S3.E29 "In 3.5.3 Case 3: Minimize CVaR with UCITS constraints ‣ 3.5 Experiments ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), and the remaining show the terms involved in the optimization process. The results shown in the plots represent the product λ x⁢C x subscript 𝜆 𝑥 subscript 𝐶 𝑥\lambda_{x}C_{x}italic_λ start_POSTSUBSCRIPT italic_x end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_x end_POSTSUBSCRIPT.

![Image 5: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_03/MinCVaR_UCITS_gd_optimizer_training_metrics.png)

Figure 5: Minimizing CVaR with simplified UCITS constraints training metrics. The first figure (Loss) shows the optimization global loss, which follows equation [29](https://arxiv.org/html/2507.16717v1#S3.E29 "In 3.5.3 Case 3: Minimize CVaR with UCITS constraints ‣ 3.5 Experiments ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), and the remaining show the terms involved in the optimization process.

As illustrated in the figures, the optimizer responds when the cumulative weight of assets that exceed 5% surpasses the 40% threshold (see the peaks in figure Weights over 5% which exceed 40%), while the individual weight limit of 10% is never breached (figure Weights over 10% is always 0). In summary, the result of this optimization is shown in Figure[6](https://arxiv.org/html/2507.16717v1#S4.F6 "Figure 6 ‣ 4.3 Case 3: Minimize CVaR with UCITS constraints ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), which shows the portfolio weights along with the 5% and 10% UCITS limits. The upper figure shows the weights assigned to every single asset of the more than 400 assets universe, and the lower figure shows the same portfolio weights, but only for the 30 selected assets. Finally, Table[5](https://arxiv.org/html/2507.16717v1#S4.T5 "Table 5 ‣ 4.3 Case 3: Minimize CVaR with UCITS constraints ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") presents the exact weights assigned by the GD optimizer to each asset.

![Image 6: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_03/MinCVaR_UCITS_gd_optimizer_portfolio_weights.png)

![Image 7: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_03/MinCVaR_UCITS_gd_optimizer_portfolio_weights_no_zeros.png)

Figure 6: Top figure: Weight assigned to every single asset of the universe along with 5% and 10% UCITS limits. Bottom figure: The same portfolio weights for the 23 selected assets. The combined weight of assets exceeding 5% is 39.98%.

Table 5: Weights for selected assets. Weights that exceed 5% are remarked. The sum of remarked weights is 39.9793%, which is lower than the 40% threshold of the UCITS constraint.

By analyzing the figures and table, we confirm that the UCITS constraints have been satisfied. The highlighted values in the table correspond to asset allocations exceeding 5%, but their combined weight does not exceed the limit of 40% imposed by the UCITS constraint. In addition, we observe that several other assets have allocations close to the 5% threshold without exceeding it (CAG, CLX, ERIE, WMT), further ensuring compliance with the constraint. For further analysis, we show in Table [6](https://arxiv.org/html/2507.16717v1#S4.T6 "Table 6 ‣ 4.3 Case 3: Minimize CVaR with UCITS constraints ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") a comparison of performance when minimizing CVaR with or without the UCITS constraint.

Table 6: Metrics comparison when minimizing CVaR. Case 2 does not apply any constraint, while case 3 apply UCITS constraints (see section [3.4.4](https://arxiv.org/html/2507.16717v1#S3.SS4.SSS4 "3.4.4 Constraint: UCITS directive ‣ 3.4 Objectives and constraints ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). In both cases, the optimization objective is to minimize CVaR.

Despite the UCITS constraints in case 3, the resulting CVaR remains very close to that of case 2, demonstrating that the gradient descent optimizer is capable of finding a solution that balances regulatory compliance constraints with the risk minimization objective.

To evaluate the impact of the regularization parameters λ 10%subscript 𝜆 10%\lambda_{\text{10\%}}italic_λ start_POSTSUBSCRIPT 10% end_POSTSUBSCRIPT and λ 5-40%subscript 𝜆 5-40%\lambda_{\text{5-40\%}}italic_λ start_POSTSUBSCRIPT 5-40% end_POSTSUBSCRIPT on both CVaR and the regulatory compliance of the optimized portfolio, we performed a grid search across a range of values for these hyperparameters. Specifically, λ 10%subscript 𝜆 10%\lambda_{\text{10\%}}italic_λ start_POSTSUBSCRIPT 10% end_POSTSUBSCRIPT and λ 5-40%subscript 𝜆 5-40%\lambda_{\text{5-40\%}}italic_λ start_POSTSUBSCRIPT 5-40% end_POSTSUBSCRIPT varied independently in the set {0.0,0.001,0.01,0.1,1.0,10.0}0.0 0.001 0.01 0.1 1.0 10.0\{0.0,0.001,0.01,0.1,1.0,10.0\}{ 0.0 , 0.001 , 0.01 , 0.1 , 1.0 , 10.0 }.

![Image 8: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_03/MinCVaR_UCITS_gd_optimizer_cvar_loss_heatmap.png)

Figure 7: CVaR corresponding to each combination of regularization parameters λ 10%subscript 𝜆 10%\lambda_{\text{10\%}}italic_λ start_POSTSUBSCRIPT 10% end_POSTSUBSCRIPT and λ 5-40%subscript 𝜆 5-40%\lambda_{\text{5-40\%}}italic_λ start_POSTSUBSCRIPT 5-40% end_POSTSUBSCRIPT during the grid search algorithm. Cells marked with an red X mean that basic UCITS constraints are not satisfied, so that scenario could not be taken into account.

Figure [7](https://arxiv.org/html/2507.16717v1#S4.F7 "Figure 7 ‣ 4.3 Case 3: Minimize CVaR with UCITS constraints ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") displays the CVaR of the resulting optimized portfolio for each combination of parameters. Each cell in the heat map represents the CVaR for a specific pair (λ 10%,λ 5-40%)subscript 𝜆 10%subscript 𝜆 5-40%(\lambda_{\text{10\%}},\lambda_{\text{5-40\%}})( italic_λ start_POSTSUBSCRIPT 10% end_POSTSUBSCRIPT , italic_λ start_POSTSUBSCRIPT 5-40% end_POSTSUBSCRIPT ). Red crosses indicate portfolios that do not comply with basic UCITS constraints and are therefore considered infeasible. Note that the minimum CVaR is obtained when both regularization parameters are set to zero (0.02725). However, this configuration does not satisfy the regulatory requirements. Compliance is achieved only when λ 5-40%subscript 𝜆 5-40%\lambda_{\text{5-40\%}}italic_λ start_POSTSUBSCRIPT 5-40% end_POSTSUBSCRIPT takes a value within the range between 1.0 and 10.0. Thus, the optimizer is forced to find the proper tradeoff between minimizing CVaR and satisfying the constraints, with a difference of 0.00178.

### 4.4 Case 4: Minimize CVaR with Tracking Error constraint

Now, the objective is to minimize CVaR while enforcing a maximum Tracking Error constraint. The optimizer was trained with a learning rate of 0.00005 0.00005 0.00005 0.00005 for 6000 6000 6000 6000 epochs. The maximum allowed tracking error was set to 0.004 0.004 0.004 0.004. The objective and constraints hyperparameters were λ TE=3 subscript 𝜆 TE 3\lambda_{\text{TE}}=3 italic_λ start_POSTSUBSCRIPT TE end_POSTSUBSCRIPT = 3. The training process is illustrated in Figure[8](https://arxiv.org/html/2507.16717v1#S4.F8 "Figure 8 ‣ 4.4 Case 4: Minimize CVaR with Tracking Error constraint ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), where the evolution of the training metrics indicates a stable convergence of the optimization algorithm under the tracking error constraint.

![Image 9: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_04/MinCVaR_Tracking_gd_optimizer_training_metrics.png)

Figure 8: Minimize CVaR with Tracking Error constraint training metrics

As shown in the figures, both the CVaR and the Tracking Error constraints are effectively satisfied, resulting in a portfolio that not only minimizes tail risk, but also closely follows the benchmark index (S&P 500) due to an explicit limitation on tracking error equal to 0.4%, which ensures that the portfolio does not deviate from the reference index. The comparison of accumulated returns in Figure [9](https://arxiv.org/html/2507.16717v1#S4.F9 "Figure 9 ‣ 4.4 Case 4: Minimize CVaR with Tracking Error constraint ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") and Table [7](https://arxiv.org/html/2507.16717v1#S4.T7 "Table 7 ‣ 4.4 Case 4: Minimize CVaR with Tracking Error constraint ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") show this idea, where the resulting portfolio follows the reference index.

![Image 10: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_04/MinCVaR_Tracking_portfolio_acum_returns_comparison.png)

Figure 9: Accumulated returns comparison between the resulting portfolio and the reference index S&P 500 when minimizing CVaR with Tracking Error constraint of T⁢E m⁢a⁢x=0.004 𝑇 subscript 𝐸 𝑚 𝑎 𝑥 0.004 TE_{max}=0.004 italic_T italic_E start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT = 0.004.

Table 7: Metrics comparison (GD vs S&P 500 index) when minimizing CVaR with a maximum Tracking Error constraint of T⁢E m⁢a⁢x=0.004 𝑇 subscript 𝐸 𝑚 𝑎 𝑥 0.004 TE_{max}=0.004 italic_T italic_E start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT = 0.004. Note that the optimizer reaches an exceeding of 10−6 superscript 10 6 10^{-6}10 start_POSTSUPERSCRIPT - 6 end_POSTSUPERSCRIPT with respect to the T⁢E m⁢a⁢x 𝑇 subscript 𝐸 𝑚 𝑎 𝑥 TE_{max}italic_T italic_E start_POSTSUBSCRIPT italic_m italic_a italic_x end_POSTSUBSCRIPT.

From the figure and the table, we can extract two main ideas. First, by minimizing CVaR and restricting the Tracking Error relative to the index, the portfolio effectively follows the benchmark, while it reduces the exposure to extreme losses. As shown in Table [7](https://arxiv.org/html/2507.16717v1#S4.T7 "Table 7 ‣ 4.4 Case 4: Minimize CVaR with Tracking Error constraint ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), the gradient descent optimizer portfolio achieves a higher Sharpe ratio, indicating a better risk-return trade-off, while simultaneously lowering CVaR (better performance and lower losses in Figure [9](https://arxiv.org/html/2507.16717v1#S4.F9 "Figure 9 ‣ 4.4 Case 4: Minimize CVaR with Tracking Error constraint ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent")). Second, the results in this case show that incorporating a Tracking Error constraint along with the CVaR minimization objective leads to portfolios that maintain competitive risk and return characteristics while ensuring lower extreme losses. This finding underscores the potential of the GD optimizer to achieve a robust balance between constraints and the objective.

### 4.5 Case 5: Maximize Sharpe ratio, minimize CVaR with Tracking Error, UCITS, minimum asset weight constraints, and assets number within a certain range

In this experiment, the goal is to simultaneously maximize the Sharpe ratio and minimize CVaR, while incorporating several practical constraints: a Tracking Error limit, UCITS constraints, a minimum asset weight threshold, and a restriction on the number of selected assets (between 20 and 30). The optimizer was trained with a learning rate of 0.001 0.001 0.001 0.001 for 1000 1000 1000 1000 epochs. The optimization hyperparameters were set to λ CVaR=100 subscript 𝜆 CVaR 100\lambda_{\text{CVaR}}=100 italic_λ start_POSTSUBSCRIPT CVaR end_POSTSUBSCRIPT = 100, λ Sharpe=10 subscript 𝜆 Sharpe 10\lambda_{\text{Sharpe}}=10 italic_λ start_POSTSUBSCRIPT Sharpe end_POSTSUBSCRIPT = 10, λ TE=0.004 subscript 𝜆 TE 0.004\lambda_{\text{TE}}=0.004 italic_λ start_POSTSUBSCRIPT TE end_POSTSUBSCRIPT = 0.004, λ 10%=10 subscript 𝜆 10%10\lambda_{\text{10\%}}=10 italic_λ start_POSTSUBSCRIPT 10% end_POSTSUBSCRIPT = 10, λ 5-40%=10 subscript 𝜆 5-40%10\lambda_{\text{5-40\%}}=10 italic_λ start_POSTSUBSCRIPT 5-40% end_POSTSUBSCRIPT = 10, λ m⁢i⁢n=10 subscript 𝜆 𝑚 𝑖 𝑛 10\lambda_{min}=10 italic_λ start_POSTSUBSCRIPT italic_m italic_i italic_n end_POSTSUBSCRIPT = 10, and λ r⁢a⁢n⁢g⁢e=0.00001 subscript 𝜆 𝑟 𝑎 𝑛 𝑔 𝑒 0.00001\lambda_{range}=0.00001 italic_λ start_POSTSUBSCRIPT italic_r italic_a italic_n italic_g italic_e end_POSTSUBSCRIPT = 0.00001. The maximum Tracking Error is set to 0.004 0.004 0.004 0.004 and the minimum weight an asset could have is 1%percent 1 1\%1 %. Figure[10](https://arxiv.org/html/2507.16717v1#S4.F10 "Figure 10 ‣ 4.5 Case 5: Maximize Sharpe ratio, minimize CVaR with Tracking Error, UCITS, minimum asset weight constraints, and assets number within a certain range ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") shows the training process, where the global loss is minimized while balancing Sharpe ratio and CVaR.

![Image 11: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_05/MaxSharpe_MinCVaR_Tracking_MinWeight_AssetRange_UCITS_gd_optimizer_training_metrics.png)

Figure 10: Maximize Sharpe ratio, minimize CVaR with Tracking Error, UCITS, minimum asset weight constraints, and assets number within a range training metrics.

The figure shows the convergence behavior of the gradient descent optimizer under the complex multi-objective setup. During the epochs, it can be observed that the overall loss decreases steadily, indicating that the optimizer is successfully balancing the objectives (maximizing the Sharpe ratio and minimizing CVaR) while satisfying the additional constraints (tracking error, UCITS, minimum asset weight, and asset count).

In addition Figure[11](https://arxiv.org/html/2507.16717v1#S4.F11 "Figure 11 ‣ 4.5 Case 5: Maximize Sharpe ratio, minimize CVaR with Tracking Error, UCITS, minimum asset weight constraints, and assets number within a certain range ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") provides a dual perspective. In the top figure, the weights allocated across the entire S&P 500 universe are shown, highlighting the asset selection process. In the bottom figure, we zoom in on the 23 selected assets. Here, each asset receives at least a 1% allocation (meeting the minimum weight constraint, represented in green), and the cumulative weight of assets with allocations above 5% totals 39.66%. This layout confirms that the UCITS constraints are respected and that the portfolio is diversified and focused.

![Image 12: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_05/MaxSharpe_MinCVaR_Tracking_MinWeight_AssetRange_UCITS_gd_optimizer_portfolio_weights.png)

![Image 13: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_05/MaxSharpe_MinCVaR_Tracking_MinWeight_AssetRange_UCITS_gd_optimizer_portfolio_weights_no_zeros.png)

Figure 11: Top figure: Selected assets from the whole S&P 500 universe. Bottom figure: Portfolio weights for the 23 selected assets. The minimum weight threshold is at 1%. The allowed range of number of selected assets is between 20 and 30. The sum of weights above 5% is 39.66%

Finally, Figure[12](https://arxiv.org/html/2507.16717v1#S4.F12 "Figure 12 ‣ 4.5 Case 5: Maximize Sharpe ratio, minimize CVaR with Tracking Error, UCITS, minimum asset weight constraints, and assets number within a certain range ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") compares the performance of the portfolio against the S&P 500 index. The GD portfolio closely tracks the benchmark while exhibiting a superior risk-return profile, as indicated by the improved Sharpe ratio and lower CVaR. This is a direct consequence of the Tracking Error constraint and the optimization objectives (maximizing the Sharpe ratio and minimization of CVaR). Table[8](https://arxiv.org/html/2507.16717v1#S4.T8 "Table 8 ‣ 4.5 Case 5: Maximize Sharpe ratio, minimize CVaR with Tracking Error, UCITS, minimum asset weight constraints, and assets number within a certain range ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") shows the performance metrics of the portfolio compared to S&P 500.

![Image 14: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_05/MaxSharpe_MinCVaR_Tracking_MinWeight_AssetRange_UCITS_gd_optimizer_portfolio_acum_returns.png)

Figure 12: Accumulated returns comparison between portfolio optimized using Gradient Descent and the reference index S&P 500 in 2020 when facing case 5: maximize Sharpe ratio, minimize CVaR, and adding the Tracking Error, UCITS, minimum asset weight, and assets number within a certain range constraints.

Table 8: Comparison of performance metrics for Cases 4, 5 and S&P 500 Index.

Compared to Case 4, where the objective was solely to minimize CVaR under a maximum Tracking Error constraint, Case 5 also introduces the maximization of the Sharpe ratio, while also satisfying other constraints. This approach leads to a significant improvement in performance. As shown in the table, the portfolio in Case 5 achieves a higher Sharpe ratio (0.051612 vs 0.031009) in balance with a slight loss with CVaR (0.047519 vs 0.043603), this is a reasonable trade-off considering the notable increase in risk-adjusted return. Moreover, both cases accomplish with the Tracking Error constraint, confirming that the performance gain in Case 5 is not a result of increased deviation from the benchmark. We refer to the figures [9](https://arxiv.org/html/2507.16717v1#S4.F9 "Figure 9 ‣ 4.4 Case 4: Minimize CVaR with Tracking Error constraint ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") and [12](https://arxiv.org/html/2507.16717v1#S4.F12 "Figure 12 ‣ 4.5 Case 5: Maximize Sharpe ratio, minimize CVaR with Tracking Error, UCITS, minimum asset weight constraints, and assets number within a certain range ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") in order to visually compare the performance of both portfolios.

Overall, the figures and performance metrics together illustrate that the multi-objective and multi-constraint optimization framework effectively constructs a portfolio with balanced risk and return, ensuring compliance with regulations. The lambda hyperparameters balance the trade-off between different objectives and constraints, being the final user who decides which values work better for his investment strategy.

### 4.6 Case 6: Minimize Risk with weight constraints in assets subsets

In this last case, the goal is to minimize portfolio risk, measured by the portfolio’s standard deviation, while enforcing specific weight constraints on four predefined asset subsets (masks). The optimizer was trained with a learning rate of 0.0009 0.0009 0.0009 0.0009 for 8000 8000 8000 8000 epochs. The hyperparameters were set to λ m⁢a⁢s⁢k=0.1 subscript 𝜆 𝑚 𝑎 𝑠 𝑘 0.1\lambda_{mask}=0.1 italic_λ start_POSTSUBSCRIPT italic_m italic_a italic_s italic_k end_POSTSUBSCRIPT = 0.1. The masks were created by randomly choosing 10 assets per group without replacement, and assigning maximum allowed weights to each. The selected assets per mask are listed in Table[9](https://arxiv.org/html/2507.16717v1#S4.T9 "Table 9 ‣ 4.6 Case 6: Minimize Risk with weight constraints in assets subsets ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"), and the masks weights constraints are shown in Table[10](https://arxiv.org/html/2507.16717v1#S4.T10 "Table 10 ‣ 4.6 Case 6: Minimize Risk with weight constraints in assets subsets ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent"). Note that the sum of weights of all masks is 90%, giving the optimizer flexibility to allocate the remaining 10% outside the defined masks.

Table 9: Asset indexes randomly chosen without replace for each mask.

Table 10: Mask weight constraints randomly selected. The sum of the wights of all masks is 0.9.

Figure[13](https://arxiv.org/html/2507.16717v1#S4.F13 "Figure 13 ‣ 4.6 Case 6: Minimize Risk with weight constraints in assets subsets ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") shows the training metrics. The loss, volatility (standard deviation), and mask-related penalty terms all converge smoothly, suggesting that the optimizer effectively balances the goal of risk minimization with the satisfaction of the mask constraints.

![Image 15: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_06/MinStd_AssetSubsetWeight_gd_optimizer_training_metrics.png)

Figure 13: Minimize Risk with weight constraints in assets subsets training metrics

Figure[14](https://arxiv.org/html/2507.16717v1#S4.F14 "Figure 14 ‣ 4.6 Case 6: Minimize Risk with weight constraints in assets subsets ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") illustrates how each of the four asset masks distributes its weights. Each subplot includes the maximum allowed weight (objective), the actual accumulated weight achieved by the optimizer, and the difference. This shows that the model adheres tightly to the constraints, and that all masks achieve their targets with negligible deviations (below 0.01%). Note that the optimizer does not assign weight to every asset included in each mask.

![Image 16: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_06/MinStd_AssetSubsetWeight_gd_optimizer_mask_1_weights.png)

![Image 17: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_06/MinStd_AssetSubsetWeight_gd_optimizer_mask_2_weights.png)

![Image 18: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_06/MinStd_AssetSubsetWeight_gd_optimizer_mask_3_weights.png)

![Image 19: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_06/MinStd_AssetSubsetWeight_gd_optimizer_mask_4_weights.png)

Figure 14: Asset weights selected by each Mask, showing the target objective for the optimizer and the value achieved.

Finally, Figure[15](https://arxiv.org/html/2507.16717v1#S4.F15 "Figure 15 ‣ 4.6 Case 6: Minimize Risk with weight constraints in assets subsets ‣ 4 Results and Analysis ‣ Multi-objective Portfolio Optimization Via Gradient Descent") displays the optimized portfolio weights, highlighting the 13 assets with non-zero allocations. While the selected assets are drawn from different masks, not every asset within each mask is used. Notably, the optimizer has assigned weights of 2.61% to DPZ and 7.39% to KR, together accounting for exactly 10% of the portfolio—matching the portion of total weight not constrained by the predefined masks.

![Image 20: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/Case_06/MinStd_AssetSubsetWeight_gd_optimizer_portfolio_colored_weights_no_zeros.png)

Figure 15: Portfolio weights for 13 selected assets.

Overall, this case confirms that the proposed optimization setup can effectively reduce portfolio volatility while adhering to strict, mask-based allocation constraints. This kind of setup is especially useful in scenarios where asset exposures must follow sector, thematic, or regulatory grouping structures.

5 Conclusion and Future Work
----------------------------

In this work, we have presented a framework for multi-objective portfolio optimization (MPO) based on gradient descent with automatic differentiation in Tensorflow. Rather than proposing a novel optimization strategy or aiming to outperform existing portfolio construction techniques, our goal has been to present a flexible, scalable, and extensible tool that can support a wide range of financial objectives and constraints. This benchmark will allow both researchers and practitioners to model complex investment scenarios in a straightforward and customizable manner.

Our experimental evaluation across six scenarios, ranging from single-objective optimization to highly constrained multi-objective cases, shows that the proposed framework performs consistently well. With this optimizer, one can explore the effect of the relative importance of each objective and constraint in multi-objective strategies. This capability of customizing the optimizer is completely useful in real-world financial applications, and the results show that the multi-objective and multi-constraint optimization framework effectively build portfolios with balanced risk and return, ensuring the constraints. The lambda hyperparameters in the loss function balance the trade-off between different objectives and constraints, where the final user decides which values work better for his investment strategy.

This work opens up several promising directions for further research. First, a natural extension would be to perform a comparison with other metaheuristic methods, such as genetic and swarm algorithms. This evaluation would deepen the strengths of gradient-based optimization in terms of convergence speed, performance, and solution quality. Second, our gradient-based approach allows the optimization of multiple scenarios at the same time. Investigating whether training in batches across multiple temporal windows would help to discover globally suboptimal portfolios that could perform robustly in all scenarios at once. In addition, exploring the incorporation of forward-looking models, such as Black-Litterman, or integrating risk-based frameworks like Risk Parity, would guide better allocation decisions. With this work, we hope this benchmark will serve as a foundation for research in portfolio optimization.

Acknowledgements
----------------

This research was supported by grant PID2023-149669NB-I00 (MCIN/AEI and ERDF - “A way of making Europe”).

References
----------

*   Abadi et al. (2015) Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G.S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., Zheng, X., 2015. TensorFlow: Large-scale machine learning on heterogeneous systems. URL: [https://www.tensorflow.org/](https://www.tensorflow.org/). software available from tensorflow.org. 
*   Agrawal et al. (2018) Agrawal, A., Verschueren, R., Diamond, S., Boyd, S., 2018. A rewriting system for convex optimization problems. Journal of Control and Decision 5, 42–60. 
*   Arnone et al. (1993) Arnone, S., Loraschi, A., Tettamanzi, A., 1993. A genetic approach to portfolio selection. Neural Network World - International Journal on Neural and Mass-Parallel Computing and Information Systems 3, 597–604. 
*   Black (1990) Black, F., 1990. Asset allocation: Combining investor views with market equilibrium. Fixed Income Research/Goldman Sachs & Company . 
*   Chang et al. (2000) Chang, T.J., Meade, N., Beasley, J.E., Sharaiha, Y.M., 2000. Heuristics for cardinality constrained portfolio optimisation. Comput. Oper. Res. 27, 1271–1302. 
*   Chekhlov et al. (2004) Chekhlov, A., Uryasev, S., Zabarankin, M., 2004. Portfolio Optimization With Drawdown Constraints, in: Pardalos, P.M., Migdalas, A., Baourakis, G. (Eds.), Supply Chain And Finance. World Scientific Publishing Co. Pte. Ltd.. World Scientific Book Chapters. chapter 13, pp. 209–228. 
*   Chen et al. (2006) Chen, W., Zhang, R.T., Cai, Y.M., Xu, F., 2006. Particle swarm optimization for constrained portfolio selection problems, in: Proceedings of the 2006 International Conference on Machine Learning and Cybernetics, pp. 2425 – 2429. 
*   Deng et al. (2024) Deng, L., Wang, T., Zhao, Y., Zheng, K., 2024. Million: A general multi-objective framework with controllable risk for portfolio management. URL: [https://arxiv.org/abs/2412.03038](https://arxiv.org/abs/2412.03038), [arXiv:2412.03038](http://arxiv.org/abs/2412.03038). 
*   Diamond and Boyd (2016) Diamond, S., Boyd, S., 2016. CVXPY: A Python-embedded modeling language for convex optimization. Journal of Machine Learning Research 17, 1–5. 
*   Fama and French (1992) Fama, E.F., French, K.R., 1992. The cross-section of expected stock returns. The Journal of Finance 47, 427–465. URL: [http://www.jstor.org/stable/2329112](http://www.jstor.org/stable/2329112). 
*   Fernández and Gómez (2007) Fernández, A., Gómez, S., 2007. Portfolio selection using neural networks. Computers & Operations Research 34, 1177–1191. 
*   Fonseca and Fleming (1993) Fonseca, C., Fleming, P., 1993. Genetic algorithms for multiobjective optimization: Formulation, discussion and generalization, in: Proceedings of the 5th International Conference on Genetic Algorithms, Morgan Kaufmann Publishers Inc., San Francisco, CA, USA. pp. 416–423. 
*   Foster and Shoaf (1996) Foster, J.A., Shoaf, J., 1996. A genetic algorithm solution to the efficient set problem: a technique for portfolio selection based on the markowitz model, in: 1996 Annual Meeting, Decision Sciences Institute. 
*   Gandibleux and Ehrgott (2005) Gandibleux, X., Ehrgott, M., 2005. 1984-2004 - 20 years of multiobjective metaheuristics. but what about the solution of combinatorial problems with multiple objectives?, in: International Conference on Evolutionary Multi-Criterion Optimization. 
*   García et al. (2012) García, S., Quintana, D., Galván, I.M., Isasi, P., 2012. Time-stamped resampling for robust evolutionary portfolio optimization. Expert Systems with Applications 39, 10722–10730. 
*   Horn et al. (1994) Horn, J.D., Nafpliotis, N., Goldberg, D.E., 1994. A niched pareto genetic algorithm for multiobjective optimization. Proceedings of the First IEEE Conference on Evolutionary Computation. IEEE World Congress on Computational Intelligence , 82–87 vol.1. 
*   Hugo Delatte (2023) Hugo Delatte, C.N., 2023. skfolio. URL: [https://github.com/skfolio/skfolio](https://github.com/skfolio/skfolio). 
*   Jorion (2001) Jorion, P., 2001. Value at risk. 2. ed. ed., McGraw-Hill, New York [u.a.]. 
*   Kalayci et al. (2019) Kalayci, C.B., Ertenlice, O., Akbay, M.A., 2019. A comprehensive review of deterministic models and applications for mean-variance portfolio optimization. Expert Syst. Appl. 125, 345–368. 
*   Knowles and Corne (2000) Knowles, J.D., Corne, D.W., 2000. Approximating the nondominated front using the pareto archived evolution strategy. Evol. Comput. 8, 149–172. 
*   Lintner (1965) Lintner, J., 1965. The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets. The Review of Economics and Statistics 47, 13–37. 
*   Markowitz (1952) Markowitz, H., 1952. Portfolio selection. The Journal of Finance 7, 77–91. 
*   Markowitz (1959) Markowitz, H.M., 1959. Portfolio Selection: Efficient Diversification of Investments. Yale University Press. 
*   Martins and Astudillo (2016) Martins, A., Astudillo, R., 2016. From softmax to sparsemax: A sparse model of attention and multi-label classification, in: Balcan, M.F., Weinberger, K.Q. (Eds.), Proceedings of The 33rd International Conference on Machine Learning, PMLR, New York, New York, USA. pp. 1614–1623. 
*   Metaxiotis and Liagkouras (2012) Metaxiotis, K., Liagkouras, K., 2012. Multiobjective evolutionary algorithms for portfolio management: A comprehensive literature review. Expert Systems with Applications 39, 11685–11698. 
*   Mishra et al. (2016) Mishra, S., Panda, G., Majhi, B., 2016. Prediction based mean-variance model for constrained portfolio assets selection using multiobjective evolutionary algorithms. Swarm and Evolutionary Computation 28. 
*   Moral-Escudero et al. (2006) Moral-Escudero, R., Ruiz-Torrubiano, R., Suárez, A., 2006. Selection of optimal investment portfolios with cardinality constraints, in: IEEE International Conference on Evolutionary Computation, CEC 2006, part of WCCI 2006, Vancouver, BC, Canada, 16-21 July 2006, IEEE. pp. 2382–2388. doi:[10.1109/CEC.2006.1688603](http://dx.doi.org/10.1109/CEC.2006.1688603). 
*   Paszke et al. (2019) Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S., 2019. Pytorch: An imperative style, high-performance deep learning library, in: Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., Garnett, R. (Eds.), Advances in Neural Information Processing Systems, Curran Associates, Inc. 
*   Rockafellar and Uryasev (2000) Rockafellar, R.T., Uryasev, S., 2000. Optimization of conditional value-at-risk. Journal of Risk 3, 21–41. 
*   Roncalli (2013) Roncalli, T., 2013. Introduction to risk parity and budgeting. SSRN Electronic Journal doi:[10.2139/ssrn.2272973](http://dx.doi.org/10.2139/ssrn.2272973). 
*   Roncalli (2014) Roncalli, T., 2014. Introduction to risk parity and budgeting. URL: [https://arxiv.org/abs/1403.1889](https://arxiv.org/abs/1403.1889), [arXiv:1403.1889](http://arxiv.org/abs/1403.1889). 
*   Schaffer (1985a) Schaffer, J., 1985a. Multiple objective optimization with vector evaluated genetic algorithms., pp. 93–100. 
*   Schaffer (1985b) Schaffer, J., 1985b. Some Experiments in Machine Learning Using Vector Evaluated Genetic Algorithms. Ph.D. thesis. 
*   Schaffer and Grefenstette (1985) Schaffer, J.D., Grefenstette, J.J., 1985. Multi-objective learning via genetic algorithms, in: Proceedings of the 9th International Joint Conference on Artificial Intelligence - Volume 1, Morgan Kaufmann Publishers Inc., San Francisco, CA, USA. p. 593–595. 
*   Sharpe (1964) Sharpe, W.F., 1964. Capital asset prices: A theory of market equilibrium under conditions of risk. The Journal of Finance 19, 425–442. 
*   Srinivas and Deb (1994) Srinivas, N., Deb, K., 1994. Muiltiobjective optimization using nondominated sorting in genetic algorithms. Evol. Comput. 2, 221–248. 
*   Vedarajan et al. (1997) Vedarajan, G., Chan, L.C., Goldberg, D., 1997. Investment portfolio optimization using genetic algorithms, in: Koza, J.R. (Ed.), Late Breaking Papers at the 1997 Genetic Programming Conference, Stanford Bookstore, Stanford University, CA, USA. pp. 255–263. 
*   Yu et al. (2008) Yu, L., Wang, S., Lai, K.K., 2008. Neural network-based mean–variance–skewness model for portfolio selection. Computers & Operations Research 35, 34–46. Part Special Issue: Applications of OR in Finance. 
*   Zitzler et al. (2000) Zitzler, E., Deb, K., Thiele, L., 2000. Comparison of multiobjective evolutionary algorithms: Empirical results. Evol. Comput. 8, 173–195. 
*   Zitzler et al. (2002) Zitzler, E., Laumanns, M., Thiele, L., Fonseca, C.M., da Fonseca, V.G., 2002. Why quality assessment of multiobjective optimizers is difficult, in: Proceedings of the 4th Annual Conference on Genetic and Evolutionary Computation, Morgan Kaufmann Publishers Inc., San Francisco, CA, USA. p. 666–674. 
*   Zitzler and Thiele (1999) Zitzler, E., Thiele, L., 1999. Multiobjective evolutionary algorithms: a comparative case study and the strength pareto approach. Trans. Evol. Comp 3, 257–271. 

Appendix A Softmax vs Sparsemax
-------------------------------

In this paper, we explore the use of the Softmax and Sparsemax functions as mechanisms to assign portfolio weights in the context of optimization algorithms. These functions are evaluated based on their ability to enforce non-negativity and normalization constraints while enabling sparsity in the solution.

The Softmax function transforms an input vector z∈ℝ K z superscript ℝ 𝐾\textbf{z}\in\mathbb{R}^{K}z ∈ blackboard_R start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT into a probability distribution:

σ⁢(z)j=e z j∑k=1 K e z k for⁢j=1,…,K formulae-sequence 𝜎 subscript z 𝑗 superscript 𝑒 subscript 𝑧 𝑗 superscript subscript 𝑘 1 𝐾 superscript 𝑒 subscript 𝑧 𝑘 for 𝑗 1…𝐾\sigma(\textbf{z})_{j}=\frac{e^{z_{j}}}{\sum_{k=1}^{K}e^{z_{k}}}\quad\text{for% }j=1,\ldots,K italic_σ ( z ) start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = divide start_ARG italic_e start_POSTSUPERSCRIPT italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_POSTSUPERSCRIPT end_ARG start_ARG ∑ start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT italic_e start_POSTSUPERSCRIPT italic_z start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUPERSCRIPT end_ARG for italic_j = 1 , … , italic_K

To prevent numerical overflow, the function is often stabilized using a constant M=max⁡(z)𝑀 z M=\max(\textbf{z})italic_M = roman_max ( z ):

σ⁢(z)j=e z j−M∑k=1 K e z k−M 𝜎 subscript z 𝑗 superscript 𝑒 subscript 𝑧 𝑗 𝑀 superscript subscript 𝑘 1 𝐾 superscript 𝑒 subscript 𝑧 𝑘 𝑀\sigma(\textbf{z})_{j}=\frac{e^{z_{j}-M}}{\sum_{k=1}^{K}e^{z_{k}-M}}italic_σ ( z ) start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = divide start_ARG italic_e start_POSTSUPERSCRIPT italic_z start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_M end_POSTSUPERSCRIPT end_ARG start_ARG ∑ start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT italic_e start_POSTSUPERSCRIPT italic_z start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_M end_POSTSUPERSCRIPT end_ARG

This ensures that the output vector has all elements strictly positive and sums to one. However, a key drawback in financial applications is that Softmax tends to allocate non-zero weights to all elements, which is not ideal when sparsity is desired in asset allocation.

Sparsemax Function: As a remedy to this limitation, the Sparsemax function, introduced by Martins and Astudillo ([2016](https://arxiv.org/html/2507.16717v1#bib.bib24)), offers a projection onto the probability simplex that can yield exact zeros.

The resulting vector maintains the key properties of non-negativity and summing to one, but unlike Softmax, allows for true sparsity, assigning exactly zero weights to less relevant elements.

Figure [16](https://arxiv.org/html/2507.16717v1#A1.F16 "Figure 16 ‣ Appendix A Softmax vs Sparsemax ‣ Multi-objective Portfolio Optimization Via Gradient Descent") shows that Sparsemax can output weight vectors with many exact zeros, promoting interpretable and concentrated portfolios. When dealing with hundreds of assets, Sparsemax avoids capital dilution by zeroing out low-scoring assets. Unlike ad-hoc truncation and renormalization applied to Softmax outputs, Sparsemax respects the original structure of the optimized vector.

In the context of portfolio optimization, the Softmax function is a convenient but dense mapping. Sparsemax offers an alternative that encourages sparsity, improving both interpretability and practicality in large-scale asset allocation.

![Image 21: Refer to caption](https://arxiv.org/html/2507.16717v1/extracted/6644229/images/results/SoftMax_vs_SparseMax/SoftMax_vs_SparseMax.png)

Figure 16: Softmax vs Sparsemax.

Appendix B Differentiation techniques in Tensorflow
---------------------------------------------------

1

2 def round_sigmoid(decimals=0):

3@tf.custom_gradient

4 def rs(x):

5 z=tf.nn.sigmoid(x)

6 scale=10**decimals

7 output=tf.math.round(z*scale)/scale

8

9 def backward(dy):

10 custom_grad=z*(1-z)

11 return dy*custom_grad

12

13 return output,backward

14 return rs

Table 11: Tensorflow implementation of the r⁢o⁢u⁢n⁢d 𝑟 𝑜 𝑢 𝑛 𝑑 round italic_r italic_o italic_u italic_n italic_d of the sigmoid function, following the equation [13](https://arxiv.org/html/2507.16717v1#S3.E13 "In 3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent") and its derivative (eq. [14](https://arxiv.org/html/2507.16717v1#S3.E14 "In 3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent")).

1

2 def mask_lower_than(x,threshold,epsilon=1 e-16):

3 return round_sigmoid()(x-threshold-epsilon)

4

5 def mask_greater_than(x,threshold,epsilon=1 e-16):

6 return round_sigmoid()(threshold+epsilon-x)

Table 12: Tensorflow implementation of the mask described in equation [13](https://arxiv.org/html/2507.16717v1#S3.E13 "In 3.3.2 Differentiation of asset conditional masks ‣ 3.3 Technical considerations for differentiation ‣ 3 Materials and Methods ‣ Multi-objective Portfolio Optimization Via Gradient Descent") using the round sigmoid function (mask_lower_than) and its analogous mask_greater_than.

Appendix C Constraints implementation in Tensorflow
---------------------------------------------------

1

2 def SharpeRatio(x,r_f=0.0):

3 return(tf.math.reduce_mean(x)-r_f)/tf.math.reduce_std(x)

4

5 def VaR(x,alpha=0.05):

6 return-tfp.stats.percentile(x,q=alpha*100)

7

8 def CVaR(x,alpha=0.05):

9 var=VaR(x,alpha=alpha)

10 shortfall=tf.nn.relu(-x-var_alpha)

11 mean_shortfall=tf.math.reduce_mean(shortfall)/alpha

12 return var+mean_shortfall

13

14 def TrackingError(x,y):

15 return tfp.stats.stddev(x-y)

16

17 def Std(x):

18 return tf.math.reduce_std(x)

Table 13: Tensorflow implementation of portfolio risk and performance objectives. SharpeRatio maximizes return-to-risk efficiency, VaR and CVaR estimate downside risk at a given confidence level, TrackingError measures the deviation from a reference benchmark, and Std measures the volatility of the portfolio returns.

1

2 def ConstraintUCITS_1(w):

3 return tf.math.reduce_sum(exceeding_threshold(w,0.1))

4

5 def ConstraintUCITS_2(w):

6 mask=mask_lower_than(w,0.05)

7 return exceeding_threshold(tf.math.reduce_sum(w*mask)-0.4)

8

9 def ConstraintTrackingError(x,y,TE_max):

10 return exceeding_threshold(TrackingError(x,y),TE_max)

11

12 def ConstraintMinWeights(w,min_value):

13 mask=mask_greater_than(w,min_value)

14 return tf.math.reduce_sum(w*mask)

15

16 def ConstraintRange(w,low,high):

17 mask=mask_lower_than(w,0.0)

18 lower=low-tf.math.reduce_sum(mask)

19 higher=high-tf.math.reduce_sum(mask)

20 return exceeding_threshold(lower*higher)

21

22 def ConstraintSubsets(w,M,m):

23 return tf.math.reduce_sum(tf.abs((m-w@M)))

Table 14: Tensorflow implementation of portfolio constraints. These include simplified UCITS regulations (ConstraintUCITS_1 and ConstraintUCITS_2), Tracking Error cap (ConstraintTrackingError), minimum asset weights (ConstraintMinWeights), number of assets within a valid range (ConstraintRange), and mask-based subset constraints (ConstraintSubsets).

