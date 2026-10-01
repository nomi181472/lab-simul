Title: AI-Powered Energy Algorithmic Trading: Integrating Hidden Markov Models with Neural Networks

URL Source: https://arxiv.org/html/2407.19858

Markdown Content:
###### Abstract

In quantitative finance, machine-learning methods are key for alpha generation. This study introduces a refreshing approach that combines Hidden Markov Models (HMM) and neural networks integrated with Black-Litterman portfolio optimization. The approach was tested during the COVID period (2019–2022), where it achieved an 83% return with a Sharpe ratio of 0.77. Two risk models were incorporated to enhance risk management, particularly during the volatile periods. The methodology was implemented on the QuantConnect platform, which was chosen for its robust framework and experimental reproducibility. The system is designed to predict future price movements and includes a three-year warm-up period to ensure the proper use of the algorithm. It focuses on highly liquid, large-cap energy stocks to ensure stable and reliable results, while also accounting for broker payments. The dual-model alpha system utilizes log returns to select the optimal state based on historical performance. It combines state predictions with neural network outputs derived from historical data to generate trading signals. This study presents an in-depth examination of the trading system’s architecture, data pre-processing, training, and performance. The full code and backtesting data are available under QuantConnect’s terms.

Keywords: Hidden Markov Models; Neural Networks; Algorithmic Trading; Machine Learning; Energy Sector; Portfolio Optimization

JEL Codes: C45, C53, G11, G17

1 Introduction
--------------

Algorithmic trading influences financial markets by enabling the rapid and exact execution of trading strategies beyond human capability [dananjayan2023]. Machine learning integration has further reinvented this field by providing novel techniques for pattern recognition and predictive modeling [afua2024].

QuantConnect, chosen for this study, offers data access, back-testing, and a powerful algorithmic trading framework. This simplifies the development, testing, and deployment of trading strategies. The environment also ensures result replication, which is crucial for assessing the advanced trading algorithms.

This study outlines a novel methodology that fuses the HMM and neural networks to create a dual-model alpha generation system. The aim is to leverage the HMM’s ability to capture temporal dependencies and market regimes, along with the power of neural networks to learn subtle patterns from historical price data [oelschlager2020, elmorr2022]. The strategy uses Black-Litterman portfolio optimization combined with two risk management models.

The system employs a low-frequency buying strategy that is activated by converging signals from various AI models. A buy decision is made only when different model signals align.

The main objective is to develop an adaptable trading strategy that predicts price fluctuations and optimizes trading decisions. The results indicate the potential of this unified approach, with the algorithm achieving a 83% return and a Sharpe ratio of 0.77 during the COVID period (2019–2022).

2 Background
------------

HMMs, deep learning, and multi-model AI are crucial for optimizing algorithmic trading strategies. HMMs simulate systems with partially observable states, whereas deep learning utilizes neural networks to capture complex patterns. Multi-model AI integrates various machine learning models to improve robustness and accuracy [giudici2024].

The use of HMMs in financial markets has evolved from simple regime-switching models to advanced volatility modeling methods [gorynin2017]. Deep learning has progressed from basic neural networks to sophisticated architectures, such as LSTM, CNN, and transformer-based models [shiri2023]. Multi-model AI, employing ensemble methods, surpasses single ML algorithms in terms of robustness [aelgani2023]. These theories form the basis for advanced trading strategies. HMMs provide insights into market regimes and volatility, deep learning models forecast price movements, and multi-model AI enhances the overall performance.

### 2.1 Previous Research

Previous research has highlighted the utility of HMMs in identifying market regimes and in volatility modeling. In addition, deep learning techniques have shown potential for time-series forecasting and sentiment analysis [larabenitez2021]. Multi-model AI approaches, such as stacking and boosting, have improved prediction accuracy across various fields [odegua2019]. HMM studies often use regime-switching models and probabilistic analysis, whereas deep learning focuses on neural network architectures and training methods. Multi-model AI research combines different algorithms through ensemble techniques to leverage their unique strengths.

### 2.2 Challenges

Despite their advantages, HMMs face challenges in capturing the market complexity. Simple HMMs may not ideally interpret market dynamics because of their inability to capture both immediate and extended trends [oelschlager2021]. Deep learning models capture nonlinear relationships but require substantial data and computational resources. Although they often outperform traditional methods, their benefits can be modest and context-dependent, especially when data are limited or computational costs are high [jiang2020]. Multi-model AI approaches offer enhanced robustness and predictive performance but are more complex to implement [wang2021]. Variability in data, methods, and implementation can lead to inconsistencies, complicating the reproduction of results [chen2022].

### 2.3 Research Gaps and Future Directions

Researchers often downplay the benefits of HMMs, deep learning, and multi-model AI in trading strategies owing to the lack of standardized frameworks for strategy standardization and data pre-processing. This absence challenges the creation of single and multi-model systems [afua2024b]. Bridging these gaps is essential to craft trading strategies capable of traversing complex financial markets. Future studies should explore integrating techniques to capture both linear and nonlinear market behavior, and investigate the impact of real-time market conditions on model performance [bharath2023].

### 2.4 Tools and Hypotheses

QuantConnect facilitates advanced trading by supporting multi-model deployment using the LEAN open-source engine. It standardizes data and algorithm development, improving efficiency and reproducibility, while addressing current limitations [rashid2010].

The execution of the hypotheses depends on incorporating HMM, deep learning, and multi-model AI to optimize algorithmic trading strategies. A comprehensive literature review clarifies each approach’s advantages and limitations, forming the following research questions: How does integrating HMM, deep learning, and multi-model AI affect the effectiveness of algorithmic trading strategies? What computational challenges does this integration present, and how can they be mitigated? How does the unified model perform under market conditions compared with the S&P 500, a standard benchmark for evaluating equity trading algorithms?

These hypotheses and research questions aim to develop an adaptive trading strategy that addresses the stated research gaps and takes advantage of HMM, deep learning, and multi-model AI strengths.

3 Methodology
-------------

![Image 1: Refer to caption](https://arxiv.org/html/2407.19858v7/workflow.png)

Figure 1: Workflow of Algorithmic Trading System

Note: This figure illustrates the complete workflow from data collection through signal generation to portfolio execution.

This section explains how the dual-model system for generating trading signals was created using the HMM and neural networks. The QuantConnect platform was used in this project because it provides significant market data and streamlines strategy testing and deployment. Since 2013, it has been refined through over two million algorithms, ensuring reproducibility and transparency [quantconnect2024].

However, QuantConnect has limitations, such as data availability, which might lead to missing market scenarios required for training deep learning models. Future studies can use distributed computing to address these limitations. This study covers aspects such as model design, data management, training, signal generation, and portfolio strategies, as illustrated in figure [1](https://arxiv.org/html/2407.19858v7#S3.F1 "Figure 1 ‣ 3 Methodology ‣ AI-Powered Energy Algorithmic Trading: Integrating Hidden Markov Models with Neural Networks"). A transparent and reproducible approach was used to confirm the reliability and success of the system.

Full code and backtesting data are available and licensed under the QuantConnect terms.

### 3.1 Universe Selection

This study used a dynamic universe with two filters to select tradable securities. The first filter evaluates liquidity through the dollar volume by selecting stocks with the highest total traded value. The second filter focuses on energy sector stocks based on market capitalization and identifies the top 20 companies in this sector. Larger companies typically exhibit more stable and predictable performance, beneficial for analysis [sauberschwarz2017]. This way, this dynamic universe prioritizes liquidity and market capitalization. Establishing a solid foundation for scientific models to operate effectively without artificial complexity [rakovic2018].

### 3.2 Alpha Generation

#### 3.2.1 Model Architecture

The system leverages the fusion of an HMM and a neural network based on PyTorch to examine the historical price data, generate trading insights, and facilitate their integration.

##### Hidden Markov Model

![Image 2: Refer to caption](https://arxiv.org/html/2407.19858v7/hmm.png)

Figure 2: HMM Architecture for Simulating Stock Price Fluctuations

Note: The model uses five hidden states to capture different market regimes including bull, bear, and transitional phases.

The HMM is a stochastic framework for simulating stock price fluctuations (figure [2](https://arxiv.org/html/2407.19858v7#S3.F2 "Figure 2 ‣ Hidden Markov Model ‣ 3.2.1 Model Architecture ‣ 3.2 Alpha Generation ‣ 3 Methodology ‣ AI-Powered Energy Algorithmic Trading: Integrating Hidden Markov Models with Neural Networks")). An HMM is formally defined by the tuple λ=(S,O,A,B,π)\lambda=(S,O,A,B,\pi) where:

*   •S={s 1,s 2,…,s N}S=\{s_{1},s_{2},\ldots,s_{N}\} is the set of N N hidden states (in this study, N=5 N=5) 
*   •O={o 1,o 2,…,o T}O=\{o_{1},o_{2},\ldots,o_{T}\} is the sequence of observations (log returns) 
*   •A={a i​j}A=\{a_{ij}\} is the state transition probability matrix, where

(1)a i​j=P​(s t=j|s t−1=i),∑j=1 N a i​j=1 a_{ij}=P(s_{t}=j|s_{t-1}=i),\quad\sum_{j=1}^{N}a_{ij}=1 
*   •B={b j​(o t)}B=\{b_{j}(o_{t})\} is the observation probability distribution 
*   •π={π i}\pi=\{\pi_{i}\} is the initial state distribution, where π i=P​(s 1=i)\pi_{i}=P(s_{1}=i) 

The model is trained on log returns, calculated as:

(2)r t=ln⁡(P t P t−1)r_{t}=\ln\left(\frac{P_{t}}{P_{t-1}}\right)

where P t P_{t} is the closing price at time t t.

For a Gaussian HMM with full covariance, the emission probability is:

(3)b j​(o t)=𝒩​(o t;μ j,Σ j)=1(2​π)d​|Σ j|​exp⁡(−1 2​(o t−μ j)T​Σ j−1​(o t−μ j))b_{j}(o_{t})=\mathcal{N}(o_{t};\mu_{j},\Sigma_{j})=\frac{1}{\sqrt{(2\pi)^{d}|\Sigma_{j}|}}\exp\left(-\frac{1}{2}(o_{t}-\mu_{j})^{T}\Sigma_{j}^{-1}(o_{t}-\mu_{j})\right)

where μ j\mu_{j} and Σ j\Sigma_{j} are the mean vector and covariance matrix for state j j.

The optimal state sequence is determined using the Viterbi algorithm, which finds:

(4)s∗=arg⁡max s 1:T⁡P​(s 1:T|o 1:T,λ)s^{*}=\arg\max_{s_{1:T}}P(s_{1:T}|o_{1:T},\lambda)

The best state for prediction is selected based on historical mean return:

(5)s best=arg⁡max j⁡𝔼​[r t|s t=j]s_{\text{best}}=\arg\max_{j}\mathbb{E}[r_{t}|s_{t}=j]

The model runs for up to 10 iterations during fitting to ensure convergence while preserving computational efficiency. The five hidden states represent various market conditions, such as bull, bear, and transitional phases, offering a broader market view.

The HMM can overfit past data, affecting the prediction accuracy with respect to abrupt market changes. Future research should explore regularization techniques and hybrid models that combine HMM with unsupervised machine learning to navigate market transitions.

##### PyTorch-Based Neural Network

![Image 3: Refer to caption](https://arxiv.org/html/2407.19858v7/nn.png)

Figure 3: Neural Network Architecture for Detecting Complex Patterns in Historical Price Data

Note: The feedforward network consists of four hidden layers with ReLU activations, trained using the Adam optimizer.

The neural network aims to detect complex patterns in the historical price data, as shown in figure [3](https://arxiv.org/html/2407.19858v7#S3.F3 "Figure 3 ‣ PyTorch-Based Neural Network ‣ 3.2.1 Model Architecture ‣ 3.2 Alpha Generation ‣ 3 Methodology ‣ AI-Powered Energy Algorithmic Trading: Integrating Hidden Markov Models with Neural Networks"). The architecture is a fully connected feedforward neural network with the following mathematical formulation:

The input layer receives a feature vector 𝐱∈ℝ 5\mathbf{x}\in\mathbb{R}^{5} representing five consecutive historical price differences:

(6)𝐱=[P t−4−P t−5,P t−3−P t−4,…,P t−P t−1]T\mathbf{x}=[P_{t-4}-P_{t-5},P_{t-3}-P_{t-4},\ldots,P_{t}-P_{t-1}]^{T}

The network architecture consists of four hidden layers with ReLU activation functions and one output layer:

(7)𝐡 1\displaystyle\mathbf{h}_{1}=ReLU​(W 1​𝐱+𝐛 1),𝐡 1∈ℝ 10\displaystyle=\text{ReLU}(W_{1}\mathbf{x}+\mathbf{b}_{1}),\quad\mathbf{h}_{1}\in\mathbb{R}^{10}
(8)𝐡 2\displaystyle\mathbf{h}_{2}=ReLU​(W 2​𝐡 1+𝐛 2),𝐡 2∈ℝ 10\displaystyle=\text{ReLU}(W_{2}\mathbf{h}_{1}+\mathbf{b}_{2}),\quad\mathbf{h}_{2}\in\mathbb{R}^{10}
(9)𝐡 3\displaystyle\mathbf{h}_{3}=ReLU​(W 3​𝐡 2+𝐛 3),𝐡 3∈ℝ 10\displaystyle=\text{ReLU}(W_{3}\mathbf{h}_{2}+\mathbf{b}_{3}),\quad\mathbf{h}_{3}\in\mathbb{R}^{10}
(10)𝐡 4\displaystyle\mathbf{h}_{4}=ReLU​(W 4​𝐡 3+𝐛 4),𝐡 4∈ℝ 5\displaystyle=\text{ReLU}(W_{4}\mathbf{h}_{3}+\mathbf{b}_{4}),\quad\mathbf{h}_{4}\in\mathbb{R}^{5}
(11)y^\displaystyle\hat{y}=W 5​𝐡 4+b 5,y^∈ℝ\displaystyle=W_{5}\mathbf{h}_{4}+b_{5},\quad\hat{y}\in\mathbb{R}

where W i W_{i} and 𝐛 i\mathbf{b}_{i} are the weight matrices and bias vectors for layer i i, and ReLU​(z)=max⁡(0,z)\text{ReLU}(z)=\max(0,z) is the Rectified Linear Unit activation function.

The network is trained to minimize the mean squared error (MSE) loss function:

(12)ℒ​(θ)=1 n​∑i=1 n(y i−y^i)2\mathcal{L}(\theta)=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}

where θ={W 1,𝐛 1,…,W 5,b 5}\theta=\{W_{1},\mathbf{b}_{1},\ldots,W_{5},b_{5}\} represents all trainable parameters, y i y_{i} is the actual price change, and y^i\hat{y}_{i} is the predicted price change.

The Adam optimizer is employed for parameter updates with adaptive learning rates:

(13)m t\displaystyle m_{t}=β 1​m t−1+(1−β 1)​∇θ ℒ t\displaystyle=\beta_{1}m_{t-1}+(1-\beta_{1})\nabla_{\theta}\mathcal{L}_{t}
(14)v t\displaystyle v_{t}=β 2​v t−1+(1−β 2)​(∇θ ℒ t)2\displaystyle=\beta_{2}v_{t-1}+(1-\beta_{2})(\nabla_{\theta}\mathcal{L}_{t})^{2}
(15)m^t\displaystyle\hat{m}_{t}=m t 1−β 1 t,v^t=v t 1−β 2 t\displaystyle=\frac{m_{t}}{1-\beta_{1}^{t}},\quad\hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}
(16)θ t+1\displaystyle\theta_{t+1}=θ t−α​m^t v^t+ϵ\displaystyle=\theta_{t}-\alpha\frac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}}+\epsilon}

where α=0.001\alpha=0.001 is the learning rate, β 1=0.9\beta_{1}=0.9 and β 2=0.999\beta_{2}=0.999 are the exponential decay rates, and ϵ=10−8\epsilon=10^{-8} is a small constant for numerical stability. The model was trained for five epochs to balance training time and risk of overfitting.

The complete network architecture can be represented as:

(17)f N​N:ℝ 5→W 1 ℝ 10→W 2 ℝ 10→W 3 ℝ 10→W 4 ℝ 5→W 5 ℝ f_{NN}:\mathbb{R}^{5}\xrightarrow{W_{1}}\mathbb{R}^{10}\xrightarrow{W_{2}}\mathbb{R}^{10}\xrightarrow{W_{3}}\mathbb{R}^{10}\xrightarrow{W_{4}}\mathbb{R}^{5}\xrightarrow{W_{5}}\mathbb{R}

Neural networks can encounter challenges in hyperparameter tuning and overfitting. Overcoming these challenges can be achieved by cross-validation and automated hyperparameter optimization.

#### 3.2.2 Data Management

Effective data management is crucial for optimizing the model’s forecasting performance. The algorithm uses a rolling window approach to update historical price data continuously. At each time step t t, a window of size k k is maintained:

(18)𝒲 t={P t−k+1,P t−k+2,…,P t}\mathcal{W}_{t}=\{P_{t-k+1},P_{t-k+2},\ldots,P_{t}\}

Feature extraction differs between the two models to leverage their respective strengths:

For the HMM, log returns are computed as:

(19)r t=ln⁡(P t P t−1)r_{t}=\ln\left(\frac{P_{t}}{P_{t-1}}\right)

For the neural network, price differences are used as features:

(20)Δ​P t=P t−P t−1\Delta P_{t}=P_{t}-P_{t-1}

Data normalization is applied to ensure numerical stability:

(21)x norm=x−μ σ x_{\text{norm}}=\frac{x-\mu}{\sigma}

where μ\mu and σ\sigma are the mean and standard deviation computed over the rolling window.

Data integrity was ensured through validation, adding only trade bars with valid closing prices, and using QuantConnect’s infrastructure for data consistency. The validation criterion requires:

(22)P t>0 and|r t|<τ P_{t}>0\quad\text{and}\quad|r_{t}|<\tau

where τ\tau is a threshold for detecting anomalous returns (e.g., τ=0.5\tau=0.5 representing 50% change).

Regular retraining occurs at frequency f f (measured in days) to ensure the models adapt to evolving market conditions:

(23)Retrain at​t​if​t mod f=0\text{Retrain at }t\text{ if }t\bmod f=0

This approach ensures that models have fresh data for predictions while maintaining computational efficiency via error handling and robust data validation across market conditions.

#### 3.2.3 Models Training

HMM training uses a Gaussian HMM with several components to capture and adapt to different market regimes. Neural network training occurs in mini-batches over multiple epochs and involves forward propagation, loss calculation, backpropagation, and parameter updates. This repeatable process enables the network to learn from data, refine parameters, and improve accuracy. A three-year preparation period provides historical data to help models recognize patterns before making predictions.

Cross-validation and regularization techniques are essential for preventing underfitting and overfitting. Incorporating validation metrics and evaluation methods further enhances model performance assessment.

#### 3.2.4 Insight Generation

The dual-model system generates trading insights by integrating predictions from both the HMM and neural network through a consensus-based decision framework.

The HMM produces a state prediction for the next time step using the forward algorithm:

(24)s^t+1=arg⁡max j∈S⁡P​(s t+1=j|o 1:t,λ)\hat{s}_{t+1}=\arg\max_{j\in S}P(s_{t+1}=j|o_{1:t},\lambda)

For each state j j, the expected return is computed from historical data:

(25)𝔼​[r|s=j]=1|T j|​∑t∈T j r t\mathbb{E}[r|s=j]=\frac{1}{|T_{j}|}\sum_{t\in T_{j}}r_{t}

where T j={t:s t=j}T_{j}=\{t:s_{t}=j\} is the set of time steps when the system was in state j j.

The Neural Network predicts the next price change:

(26)P^t+1=P t+f N​N​(𝐱 t;θ)\hat{P}_{t+1}=P_{t}+f_{NN}(\mathbf{x}_{t};\theta)

where f N​N f_{NN} is the trained neural network function with parameters θ\theta.

The trading signal is generated by combining both predictions through the following decision rule:

(27)Signal t={+1(Buy) if​𝔼​[r|s=s^t+1]>ϵ​AND​P^t+1>P t−1(Sell) if​𝔼​[r|s=s^t+1]<−ϵ​AND​P^t+1<P t 0(Hold) otherwise\text{Signal}_{t}=\begin{cases}+1&\text{(Buy) if }\mathbb{E}[r|s=\hat{s}_{t+1}]>\epsilon\text{ AND }\hat{P}_{t+1}>P_{t}\\ -1&\text{(Sell) if }\mathbb{E}[r|s=\hat{s}_{t+1}]<-\epsilon\text{ AND }\hat{P}_{t+1}<P_{t}\\ 0&\text{(Hold) otherwise}\end{cases}

where ϵ\epsilon is a threshold parameter that filters out weak signals (typically ϵ=0.001\epsilon=0.001 or 0.1%).

The confidence level of the signal can be quantified as:

(28)Confidence t=|𝔼[r|s=s^t+1]|×P(s t+1=s^t+1|o 1:t)\text{Confidence}_{t}=\left|\mathbb{E}[r|s=\hat{s}_{t+1}]\right|\times P(s_{t+1}=\hat{s}_{t+1}|o_{1:t})

This consensus mechanism ensures that a trading signal is only generated when both models agree on the direction of price movement, thereby reducing false signals and improving overall strategy reliability. The approach can be formalized as:

(29)Execute t={True if sign​(𝔼​[r|s=s^t+1])=sign​(P^t+1−P t)False otherwise\text{Execute}_{t}=\begin{cases}\text{True}&\text{if }\text{sign}(\mathbb{E}[r|s=\hat{s}_{t+1}])=\text{sign}(\hat{P}_{t+1}-P_{t})\\ \text{False}&\text{otherwise}\end{cases}

Possible conflicts between model predictions can be addressed using ensemble techniques, such as stacking, weighted voting, and gradient boosting, which enhance prediction accuracy and resilience by leveraging the strengths of each model.

### 3.3 Portfolio Construction

Proper portfolio construction is crucial for the success of the algorithm. The Black-Litterman model was chosen for its innovative approach, combining expected returns and risk. Created by Fischer Black and Robert Litterman, this model merges investor views with market equilibrium for better asset allocation [cayirli2019].

The Black-Litterman model combines market equilibrium returns with investor views to produce posterior expected returns. The model starts with the market equilibrium return vector Π\Pi, derived from market capitalization weights:

(30)Π=λ​Σ​w mkt\Pi=\lambda\Sigma w_{\text{mkt}}

where λ\lambda is the risk aversion coefficient, Σ\Sigma is the covariance matrix of asset returns, and w mkt w_{\text{mkt}} is the market capitalization weight vector.

The investor views are expressed through a picking matrix P P and view portfolio returns Q Q, with uncertainty in the views captured by the diagonal matrix Ω\Omega. The posterior expected return is then computed as:

(31)𝔼​[R]=[(τ​Σ)−1+P T​Ω−1​P]−1​[(τ​Σ)−1​Π+P T​Ω−1​Q]\mathbb{E}[R]=\left[(\tau\Sigma)^{-1}+P^{T}\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\Pi+P^{T}\Omega^{-1}Q\right]

where τ\tau is a scalar parameter representing the uncertainty in the prior (typically τ∈[0.01,0.05]\tau\in[0.01,0.05]).

The posterior covariance matrix is:

(32)Σ post=Σ+[(τ​Σ)−1+P T​Ω−1​P]−1\Sigma_{\text{post}}=\Sigma+\left[(\tau\Sigma)^{-1}+P^{T}\Omega^{-1}P\right]^{-1}

Optimal portfolio weights are determined by maximizing the expected utility:

(33)w∗=arg⁡max w⁡{w T​𝔼​[R]−λ 2​w T​Σ post​w}w^{*}=\arg\max_{w}\left\{w^{T}\mathbb{E}[R]-\frac{\lambda}{2}w^{T}\Sigma_{\text{post}}w\right\}

This yields the optimal weight vector:

(34)w∗=1 λ​(Σ post)−1​𝔼​[R]w^{*}=\frac{1}{\lambda}(\Sigma_{\text{post}})^{-1}\mathbb{E}[R]

subject to the constraint ∑i=1 N w i=1\sum_{i=1}^{N}w_{i}=1 (fully invested portfolio).

The expected portfolio return and variance are:

(35)R p\displaystyle R_{p}=∑i=1 N w i∗​R i=(w∗)T​𝔼​[R]\displaystyle=\sum_{i=1}^{N}w_{i}^{*}R_{i}=(w^{*})^{T}\mathbb{E}[R]
(36)σ p 2\displaystyle\sigma_{p}^{2}=(w∗)T​Σ post​w∗\displaystyle=(w^{*})^{T}\Sigma_{\text{post}}w^{*}

This model increases algorithm modularity, separating alpha generation from portfolio management [tzang2020]. Unlike signal-based portfolio construction, it reduces the probability of inaccurate signals, preventing inefficient management in achieving the optimal Sharpe ratio. The Black-Litterman model improves system reliability by clearly dividing alpha generation and portfolio management. It directs alpha model insights into risk management and execution models and minimizes unnecessary trades and risks.

### 3.4 Risk Management and Execution Models

This study integrates two risk management models to enhance portfolio robustness. The selected models are the Maximum Drawdown Percent per security and the trailing-stop risk management model.

##### Maximum Drawdown Model

The maximum drawdown (MDD) for a security measures the largest peak-to-trough decline in portfolio value. For a given security i i, the drawdown at time t t is:

(37)D​D i​(t)=max s∈[0,t]⁡V i​(s)−V i​(t)max s∈[0,t]⁡V i​(s)DD_{i}(t)=\frac{\max_{s\in[0,t]}V_{i}(s)-V_{i}(t)}{\max_{s\in[0,t]}V_{i}(s)}

where V i​(t)V_{i}(t) is the value of security i i at time t t.

The maximum drawdown over the entire period is:

(38)MDD i=max t∈[0,T]⁡D​D i​(t)\text{MDD}_{i}=\max_{t\in[0,T]}DD_{i}(t)

A position in security i i is automatically closed if:

(39)D​D i​(t)>MDD threshold DD_{i}(t)>\text{MDD}_{\text{threshold}}

where MDD threshold\text{MDD}_{\text{threshold}} is typically set between 10% and 20%, depending on risk tolerance. This model protects investor capital by automatically exiting positions with excessive losses, thereby ensuring disciplined risk controls.

##### Trailing Stop Model

The trailing-stop model dynamically adjusts stop-loss levels to capture gains while protecting against losses. For a long position in security i i, the trailing stop price is updated as:

(40)Stop i​(t)=max⁡(P i​(t)×(1−δ),Stop i​(t−1))\text{Stop}_{i}(t)=\max\left(P_{i}(t)\times(1-\delta),\text{Stop}_{i}(t-1)\right)

where δ\delta is the trailing stop percentage (typically δ∈[0.05,0.15]\delta\in[0.05,0.15]), and P i​(t)P_{i}(t) is the current price.

The position is closed if:

(41)P i​(t)≤Stop i​(t)P_{i}(t)\leq\text{Stop}_{i}(t)

For short positions, the trailing stop is computed as:

(42)Stop i​(t)=min⁡(P i​(t)×(1+δ),Stop i​(t−1))\text{Stop}_{i}(t)=\min\left(P_{i}(t)\times(1+\delta),\text{Stop}_{i}(t-1)\right)

The trailing-stop model locks in profits in trending markets while providing downside protection. The expected profit captured by the trailing stop can be approximated as:

(43)𝔼​[Profit]≈max t∈[t entry,t exit]⁡[P i​(t)−P i​(t entry)]×(1−δ)\mathbb{E}[\text{Profit}]\approx\max_{t\in[t_{\text{entry}},t_{\text{exit}}]}\left[P_{i}(t)-P_{i}(t_{\text{entry}})\right]\times(1-\delta)

These risk management models, combined with the lean engine’s execution model, ensure timely trade execution and capitalize on market opportunities. This approach supports robust prediction models and effective asset allocation, balancing the alpha generation with disciplined risk management. The combined risk-adjusted return can be expressed as:

(44)R adj=R p−λ risk×Risk penalty R_{\text{adj}}=R_{p}-\lambda_{\text{risk}}\times\text{Risk}_{\text{penalty}}

where λ risk\lambda_{\text{risk}} is the risk aversion parameter and Risk penalty\text{Risk}_{\text{penalty}} captures drawdown and volatility concerns.

4 Results and Discussion
------------------------

### 4.1 Key Statistics

Table 1: Key Trading Metrics

Note: Performance metrics cover the full backtesting period from January 2019 to January 2022, encompassing the COVID-19 market volatility.

This table presents the essential metrics used to evaluate the performance and risk of an investment portfolio. It covers Runtime Days over a period of 1096 days to assess performance sustainability and volatility. The Drawdown indicates a maximum loss from peak to trough of 17.1%, reflecting the risk of potential declines. A Portfolio Turnover of 3.29% signifies trading frequency, with lower figures suggesting a more passive management strategy aimed at minimizing transaction costs.

Risk-adjusted performance was assessed using several ratios, as summarized in table [1](https://arxiv.org/html/2407.19858v7#S4.T1 "Table 1 ‣ 4.1 Key Statistics ‣ 4 Results and Discussion ‣ AI-Powered Energy Algorithmic Trading: Integrating Hidden Markov Models with Neural Networks"). These metrics provide a comprehensive evaluation of the trading strategy’s effectiveness:

The Sharpe Ratio measures risk-adjusted returns:

(45)S=𝔼​[R p−R f]σ p S=\frac{\mathbb{E}[R_{p}-R_{f}]}{\sigma_{p}}

where R p R_{p} is the portfolio return, R f R_{f} is the risk-free rate, and σ p\sigma_{p} is the portfolio standard deviation. The probabilistic Sharpe ratio at 41% quantifies returns relative to risk, with higher ratios denoting more efficient risk management.

The Compounded Annual Growth Rate (CAGR) at 22.2% represents the geometric mean return:

(46)CAGR=(V final V initial)1 n−1\text{CAGR}=\left(\frac{V_{\text{final}}}{V_{\text{initial}}}\right)^{\frac{1}{n}}-1

where V final=$​182,761 V_{\text{final}}=\mathdollar 182,761, V initial=$​100,000 V_{\text{initial}}=\mathdollar 100,000, and n=1096/365=3 n=1096/365=3 years.

The Sortino Ratio of 0.6 focuses on downside risk:

(47)S sortino=𝔼​[R p−R f]σ downside S_{\text{sortino}}=\frac{\mathbb{E}[R_{p}-R_{f}]}{\sigma_{\text{downside}}}

where σ downside=𝔼[min(R p−R f,0)2]\sigma_{\text{downside}}=\sqrt{\mathbb{E}[\min(R_{p}-R_{f},0)^{2}]} measures downside deviation.

The Information Ratio of −0.1-0.1 compares performance to a benchmark:

(48)I​R=𝔼​[R p−R b]σ R p−R b IR=\frac{\mathbb{E}[R_{p}-R_{b}]}{\sigma_{R_{p}-R_{b}}}

where R b R_{b} is the benchmark return (S&P 500) and σ R p−R b\sigma_{R_{p}-R_{b}} is the tracking error.

Further details include the Capacity (USD) at $10 million, which represents the optimal investment level for strategy sustainability and achieving target returns.

### 4.2 Detailed Trading Metrics

Table 2: Detailed Trading Metrics and Performance Indicators

Note: The low trading frequency (40 total orders) reflects the conservative, signal-based approach of the dual-model system.

The metrics evaluate trading outcomes over a specified period, revealing a relatively low frequency, with only 40 trades. An average win rate of 7.70% and an average loss of −3.29%-3.29\%, with a 60% success rate, highlight the effectiveness of the strategy. The initial equity of $100,000 increased to $182,761.12, indicating substantial gain. A Sharpe Ratio of 0.77 demonstrates a risk-return trade-off.

An alpha of 0.13 suggests modest active returns, and a beta of 0.175 indicates lower market volatility than broader indices, as detailed in table [2](https://arxiv.org/html/2407.19858v7#S4.T2 "Table 2 ‣ 4.2 Detailed Trading Metrics ‣ 4 Results and Discussion ‣ AI-Powered Energy Algorithmic Trading: Integrating Hidden Markov Models with Neural Networks"). An annual standard deviation of 0.21 and a variance of 0.044 quantified the return variability. A tracking error of 0.256 indicates how the portfolio’s returns deviate from the benchmark, whereas a Treynor Ratio of 0.951 indicates good risk-adjusted returns. Fees amounting to $971.62, are relatively small compared to growth in equity.

These data indicate a robust trading strategy that can effectively manage risk. Optimizing hyperparameters and preprocessing data can enhance model performance and improve results. However, extensive Monte Carlo simulations are necessary to ensure model reliability. This reinforces the importance of rigorous data analysis and strategy refinement in trading.

### 4.3 Cumulative Returns

![Image 4: Refer to caption](https://arxiv.org/html/2407.19858v7/returns.png)

Figure 4: Cumulative Returns Comparison (January 2019–January 2022)

Note: The light blue line represents the backtested strategy while the gray line shows the S&P 500 benchmark. Notable outperformance occurred during the COVID-19 market volatility in May 2020 and September 2021.

The Cumulative Returns chart illustrates the performance comparison of a backtested investment strategy against a benchmark from January 2019 to January 2022. The x-axis represents the timeline, marked in months, while the y-axis quantifies cumulative returns in percentage terms. The backtested strategy is depicted by a light blue line, whereas the benchmark (presumed to be the S&P 500) is shown in gray.

Throughout the observation period, the strategy generally aligns closely with the benchmark, indicating a strong market correlation. Notably, the strategy outperformed the benchmark at specific points, such as during the onset of the COVID-19 pandemic in May 2020 and September 2021, as illustrated in figure [4](https://arxiv.org/html/2407.19858v7#S4.F4 "Figure 4 ‣ 4.3 Cumulative Returns ‣ 4 Results and Discussion ‣ AI-Powered Energy Algorithmic Trading: Integrating Hidden Markov Models with Neural Networks"). These surges suggest effective strategy adjustments in response to market volatility.

Despite some instances of notable growth, the strategy often mirrors the benchmark’s movements, highlighting its dependence on broader market trends rather than consistently outperforming it. During the COVID-19 crisis, the strategy demonstrated resilience, maintaining its value better than the benchmark and taking advantage of the recovery phase, which is indicative of adept risk management and opportunity utilization.

### 4.4 Advantages of the Approach

Utilizing the HMM alongside neural networks offers notable advantages for predictive modeling in financial markets. HMMs capture temporal dependencies crucial for sequential data like stock prices, while neural networks excel in recognizing complex patterns and relationships [alwateer2023, qamar2023]. By combining these approaches, a more complete and accurate prediction model is achieved, leveraging the strengths of both strategies. This hybrid approach enhances the robustness by reducing the biases inherent in any single model, resulting in more reliable trading signals. Risk management strategies also play a crucial role in guarding against potential losses and improving the overall portfolio stability. The adaptability gained from merging HMMs and neural networks is essential for maintaining a high performance across different market conditions.

### 4.5 Future Research Directions

Future research should explore various approaches to enhance dual-model methodology by combining HMMs and neural networks.

##### Wavelet Transform for Enhanced Market Pattern Recognition

One potential approach is to apply a wavelet transform to the pre-processing strategy of the data. Wavelet transforms capture both time and frequency domain information, improving the potential of the model to recognize patterns in price data, which is crucial to ensuring its success [kwon2022]. This is vital as it accounts for the dynamic nature of the market, making it more effective than static methods such as Pearson’s correlation coefficient.

##### Monte Carlo Simulations for Model Generalizability

The use of extensive Monte Carlo simulations across diverse temporal frames is another promising area of research. These simulations help analyze the generalization capabilities of the model, evaluating its reliability and identifying possible cases of overfitting or incorrect market predictions [zillich2009]. This ensures the stability and accuracy of the algorithmic trading system in a real trading scenario.

##### Hyperparameter Tuning and Ensemble Methods

Employing novel neural network architectures along with automated hyperparameter fine-tuning can enhance forecasting capabilities by finding the optimal architecture for a specific dataset [wu2024]. Additionally, incorporating an ensemble machine learning method for HMM and neural network systems can enable the model to adapt dynamically to market fluctuations, improving overall alpha generation performance [ong2011].

##### Diversifying Equities for Growth and Stability

Increasing sector diversification is essential to enhance portfolio resilience. For example, combining the 25 largest stocks from high-potential sectors such as technology, biomedicine, pharmaceuticals, and energy can enhance diversification and access growth alternatives. This approach helps offset sector-specific risks, leverages the strengths of each sector, and theoretically leads to more stable returns.

### 4.6 QuantConnect for Reproducibility

QuantConnect was chosen for its powerful framework and user-friendly design, which simplified research replication. Its extensive access to data assures precise training and testing of the model. The integrated lean open-source algorithmic trading framework allows the implementation and testing of complex strategies, accelerating the development and validation of innovative models. This platform enhances research replicability, enabling other researchers and practitioners to validate and extend the outcomes. Which is essential for the advancement of scientific research.

5 Conclusion
------------

The combined use of the HMM and neural networks in a dual-model approach has shown success in enhancing the accuracy and resilience of trading signals. During the COVID period (2019–2022), the dual-model system achieved an 83% return and a Sharpe ratio of 0.77, highlighting its practical potential.

A three-year preparation period provides historical data to help models recognize patterns and understand market dynamics before making predictions. However, replicating these outcomes in complex financial markets is harder. These results suggest that combining the HMM and neural networks can strengthen algorithmic trading by generating more precise and reliable signals. Additionally, integration of multiple risk models improves the robustness of the system.

This study underscores the benefits of combining HMMs and neural networks for signal trading. Future research could explore wavelet transform pre-processing, extensive Monte Carlo simulations, and adaptive machine learning methods to further improve model performance. Using QuantConnect can enhance the reproducibility of results.

The full code and backtesting data are available and licensed under the QuantConnect terms.

Acknowledgments
---------------

The author acknowledges the QuantConnect platform for providing the infrastructure and data access necessary for this research.

Conflict of interest
--------------------

The author declares no conflicts of interest in this paper.

