# 2303.00356

Source: https://arxiv.org/html/2303.00356



A Deep Reinforcement Learning Trader without Offline Training

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
1 Introduction
2 The algorithm

2.1 Double Q-learning

2.1.1 State
2.1.2 Actions and rewards
2.1.3 Terminal state
2.1.4 Policy and hyperparameters

2.2 Fast learning network

2.2.1 Preliminaries
2.2.2 Weight renormalisation

3 Implementation details and testing

3.1 Observing, trading, hyperparameters
3.2 Testing

4 Conclusion
A Mathematica code
References

 License: arXiv.org perpetual non-exclusive license
 

arXiv:2303.00356v1 [q-fin.CP] 01 Mar 2023

A Deep Reinforcement Learning Trader without Offline Training

Boian Lazov

Thanks: blazov_fte@uacg.bg

Affiliation: Department of Mathematics, University of Architecture, Civil Engineering and Geodesy, 1164 Sofia, Bulgaria

Abstract
In this paper we pursue the question of a fully online trading algorithm (i.e. one that does not need offline training on previously gathered data). For this task we use Double Deep QQ-learning in the episodic setting with Fast Learning Networks approximating the expected reward QQ. Additionally, we define the possible terminal states of an episode in such a way as to introduce a mechanism to conserve some of the money in the trading pool when market conditions are seen as unfavourable. Some of these money are taken as profit and some are reused at a later time according to certain criteria. After describing the algorithm, we test it using the 1-minute-tick data for Cardano’s price on Binance. We see that the agent performs better than trading with randomly chosen actions on each timestep. And it does so when tested on the whole dataset as well as on different subsets, capturing different market trends.

1 Introduction

In recent years algorithmic trading on financial markets is increasingly replacing humans [1]. One can find numerous estimates for the market share of automated traders with some sources giving over 73%73\% for US equity trading [1], while others citing as high as 92%92\% for forex trading.11
1
 
 
 See for example https://www.quantifiedstrategies.com/what-percentage-of-trading-is-algorithmic/.
Such sources, however, do not seem that reliable, since data is generally not openly available. Nevertheless, there are many (paid) reports that give a general idea of the scope of automated trading:
https://www.grandviewresearch.com/industry-analysis/algorithmic-trading-market-report,
https://www.mordorintelligence.com/industry-reports/algorithmic-trading-market, 
https://www.alliedmarketresearch.com/algorithmic-trading-market-A08567.. Regardless of the actual figures, intelligent automation is increasingly used in our world and promises to be applicable in some very complex domains, where analytic solutions are either not known or very hard to obtain.
There are many possible approaches to developing a trading algorithm, but recently one direction of research has been receiving much attention, namely machine learning based approaches. In particular, Reinforcement Learning (RL) has been a really promising way to solve some very difficult problems in other areas (like learning to play various games like Go [2] and StarCraft II [3] at the expert level) and is now being adapted to make decisions and execute trades in the trading setting. This area of research is very active and fairly new (see for example [4]).
As promising as it is, RL suffers from one big problem, namely the generalisation one (as does all of machine learning in fact). More specifically, once the agent (or neural network) is trained on a set of data and good performance is achieved, it is generally hard to translate this training to a new dataset and keep the performance. Furthermore, the training set usually needs to be very large and the agent needs to replay it many times. This is obviously not a good situation for a trading algorithm, since the market is considered a stochastic system and as such it changes rapidly and continuously. If we hope to be able to predict its movement, it should mostly be short term. For this the trader should be able to adapt quickly to current information.
There are many proposed ways to try to deal with said problem of generalisation (both in supervised learning and RL), but one that seems both promising and simple is the idea to learn only the output weights of a neural network. It is implemented in partiular in two algorithms – Extreme Learning Machine (ELM) [5, 6] and Fast Learning Network (FLM) [7], and one can borrow the structure of the FLN to use as an approximator to the QQ-function. As we will see later this will be a success and we will obtain a RL agent that performs better than random even when the market’s overall trend is downward.
This paper is organised as follows: in section 2 we will describe the algorithm in detail, namely all of the components of the RL (section 2.1) as well as the neural network (section 2.2); in section 3 we will discuss briefly how to implement the algorithm and then we will test it on historical market data against an algorithm, that takes random actions on each timestep; finally, we will end with some concluding remarks (section 4).

2 The algorithm

Our goal is to build a simple trading algorithm, which uses a pool of money to trade for some asset, by observing the state of the market. More precisely our agent needs to collect some information about the market, then open a position (ideally, executing a trade). Then it needs to wait for more information to calculate whether the trade was a good one and the process repeats.
We will build our trader in the framework of RL. More precisely we will use a double QQ-learning algorithm with approximation [8]. It is generally thought that combining QQ-learning with approximation should be avoided, because of instabilities, but there are examples of successfully using such algorithms [9, 10]. To approximate the QQ-functions we will use the structure of a FLN [7]. On top of that we will also propose a savings mechanism to deal with “bearish” markets. The idea is to take money out of the trading pool, so that part of it is never used again and can be taken as profit and another part is returned to the trading pool when the market conditions seem favourable. We will now go through all the specific components one by one.

2.1 Double Q-learning

2.1.1 State

We will use a standard double Q-learning algorithm. It will be episodic with a continuous state space. As is usual, we will add an index tt to variables to denote the current time step and t+1t+1 for the next time step. First we will see how to construct the state of the environment. As we will be using approximaiton of the QQ-function by a neural network the state will be defined by a feature vector.
The algorithm is initialised with 3 pools of money, denoted by m​o​nmon, s​a​vsav and r​e​sres. m​o​nmon refers to the current pool with which to trade. s​a​vsav denotes an amount of money, that are saved and never again used. This gives a convenient way to use the profit, without disturbing the operation of the trader and also safeguards somewhat against big losses. r​e​sres refers to a pool of money, that are stored for later use when some conditions are met. Finally, there are also the assets that the agent will buy, denoted by c​n​scns. m​o​nmon and c​n​scns will be included in the calculation of the state.
Next, we need a variable, which will be used to determine when to move money to s​a​vsav and r​e​sres. We denote this by m​l​i​mmlim. We will describe this in more detail later, but briefly s​a​vsav and r​e​sres will be increased (and m​o​nmon decreased), when the value of m​o​nmon becomes greater than m​l​i​mmlim.
To calculate the state, our trader also needs information for the market. It consists of one price, recorded at the beginning of the episode, denoted by i​p​ripr, and 55 consecutive prices, denoted by p​r1pr_{1}, p​r2pr_{2}, p​r3pr_{3}, p​r4pr_{4} and p​r5pr_{5}, recorded at some intervals.
The next thing that is needed to calculate the state is the trading volume. More precisely, the trader records the volumes from the intervals immediately preceding the ones with recorded prices. These 55 volumes are then used to calculate a few averages, that will be included in the feature vector. These are a simple moving average of the previous 100100 values of the volume, denoted by a​vav as well as the average volume of the current state’s data points, denoted by c​a​vcav. The last recorded volume (associated with p​r5pr_{5}) is also used in the feature vector and it is denoted by v​o​l5vol_{5}.
Finally, we also include in the feature vector the Relative Strength Index (RSI), calculated with the recorded prices (15 prices, as is standard), as well as the relative movements of the price and relative movements of the movements and so on, i.e.

n​m​di(1)\displaystyle nmd^{(1)}_{i}
=p​ri+1−p​rip​ri,i∈{1,2,3,4},\displaystyle=\frac{pr_{i+1}-pr_{i}}{pr_{i}},\ \ \ i\in\{1,2,3,4\},

(2.1)

n​m​dk(2)\displaystyle nmd^{(2)}_{k}
=n​m​dk+1(1)−n​m​dk(1)n​m​dk(1),k∈{1,2,3},\displaystyle=\frac{nmd^{(1)}_{k+1}-nmd^{(1)}_{k}}{nmd^{(1)}_{k}},\ \ \ k\in\{1,2,3\},

(2.2)

n​m​dl(3)\displaystyle nmd^{(3)}_{l}
=n​m​dl+1(2)−n​m​dl(2)n​m​dl(2),l∈{1,2},\displaystyle=\frac{nmd^{(2)}_{l+1}-nmd^{(2)}_{l}}{nmd^{(2)}_{l}},\ \ \ l\in\{1,2\},

(2.3)

n​m​d(4)\displaystyle nmd^{(4)}
=n​m​d2(3)−n​m​d1(3)n​m​d1(3).\displaystyle=\frac{nmd^{(3)}_{2}-nmd^{(3)}_{1}}{nmd^{(3)}_{1}}.

(2.4)

With all of the above the feature vector has the following form:

f​e​a​t=\displaystyle feat=
(1,p​r1,…,p​r5,i​p​r,p​r5−i​p​ri​p​r,m​o​n,c​n​s,c​a​v,a​v,c​a​v−a​va​v,v​o​l5−a​va​vCLOSE,\displaystyle\left(1,pr_{1},...,pr_{5},ipr,\frac{pr_{5}-ipr}{ipr},mon,cns,cav,av,\frac{cav-av}{av},\frac{vol_{5}-av}{av},\right.

(2.5)

OPENv​o​l5−c​a​vc​a​v,r​s​i,n​m​d1(1),…,n​m​d4(1),n​m​d1(2),…,n​m​d3(2),…,n​m​d(4),m​l​i​m)T.\displaystyle\left.\frac{vol_{5}-cav}{cav},rsi,nmd^{(1)}_{1},...,nmd^{(1)}_{4},nmd^{(2)}_{1},...,nmd^{(2)}_{3},...,nmd^{(4)},mlim\right)^{T}.

2.1.2 Actions and rewards

Next, we want to define the set of actions that the agent can take. They are simple – buy, sell or hold. We denote the set of actions as {a1,a2,…,a19}\{a_{1},a_{2},...,a_{19}\}. Actions a1a_{1} to a9a_{9} denote buying. Action a1a_{1} means the agent buys coins for 1010 money, action a2a_{2} – for 2020 and so on. Actions a10a_{10} to a18a_{18} denote selling for a fixed amount of money – again at increments of 1010. Finally, action a19a_{19} denotes holding. Actions, of course, can fail due to insufficient funds and this will be reflected in the reward.
This leads us to the next ingredient of the RL algorithm – the reward signal. In order to calculate the reward we first need to keep a record of the total wealth w​t​hwth of the agent, associated with a given state,

w​t​h=m​o​n+p​r5​c​n​s.\displaystyle wth=mon+pr_{5}\ cns.

(2.6)

Thus after trading and recording the next 55 prices, the wealth changes. Using this, we choose the reward to be polynomial in the change of the wealth, i.e.

r​e​w=(w​t​ht+1−w​t​ht)−(w​t​ht+1−w​t​ht2)2.\displaystyle rew=(wth_{t+1}-wth_{t})-\left(\frac{wth_{t+1}-wth_{t}}{2}\right)^{2}.

(2.7)

The above formula helps to discourage too risky actions (i.e. ones with changes in the wealth that are too big). Additionally, if the attempted action fails, the reward is instead

r​e​w=(w​t​ht+1−w​t​ht)−(w​t​ht+1−w​t​ht2)2−0.1.\displaystyle rew=(wth_{t+1}-wth_{t})-\left(\frac{wth_{t+1}-wth_{t}}{2}\right)^{2}-0.1.

(2.8)

2.1.3 Terminal state

As we mentioned earlier, we are using a savings mechanism. It ties in with the terminal state. We will consider three different terminal states. The first terminal state is the one in which

m​o​nt+1>m​l​i​m,\displaystyle mon_{t+1}>mlim,

(2.9)

where m​l​i​mmlim is a parameter that changes when encountering a terminal state. This means that the current money pool of the agent is greater than some threshold. Before beginning the new episode, the extra money m​d​f=m​o​nt+1−m​l​i​mmdf=mon_{t+1}-mlim is distributed between three pools: 0.340.34 goes to a savings pool s​a​vsav, which is never again used to trade; 0.330.33 – to a reserves pool r​e​sres, which might be used again later; and 0.330.33 is left in the money pool (so that after this still m​o​nt+1>m​l​i​mmon_{t+1}>mlim). Then m​l​i​mmlim is increased to the current value of m​o​nmon plus m​d​fmdf. The reward for going into this state is modified – it is the usual plus the amount of money that was added to s​a​vsav, i.e. 0.34​m​d​f0.34\,mdf.
The second terminal state is determined by four conditions:

m​o​nt+1<m​l​i​m;\displaystyle mon_{t+1}<mlim;

(2.10)

w​t​ht+1<m​l​i​m​n;\displaystyle wth_{t+1}<mlimn;

(2.11)

Q⁡(st,at)>0;\displaystyle Q(s_{t},a_{t})>0;

(2.12)

r​s​it+1>70.\displaystyle rsi_{t+1}>70.

(2.13)

Here m​l​i​m​nmlimn is a hyperparameter, which sets the lowest possible value of m​l​i​mmlim. In general the meaning of RSI is open to interpretation [11, 12], but if all of the above conditions are met, we take this as an indication that the market conditions are favourable (while the agent is low on money), so half of the money in r​e​sres are redistributed to the money pool to be used for trading again. After this m​l​i​mmlim is again changed – this time to max⁡{m​l​i​m​n,m​o​nt+1+r​e​s2}\max\{mlimn,mon_{t+1}+\frac{res}{2}\}, and the new episode begins.
The final terminal state is determined by the following:

m​o​nt+1<m​l​i​m;\displaystyle mon_{t+1}<mlim;

(2.14)

w​t​ht+1≥m​l​i​m​n;\displaystyle wth_{t+1}\geq mlimn;

(2.15)

Q⁡(st,at)<0;\displaystyle Q(s_{t},a_{t})<0;

(2.16)

r​s​it+1<30.\displaystyle rsi_{t+1}<30.

(2.17)

Here the market is seen as unfavourable so the only thing to do is to change m​l​i​mmlim to w​t​ht+1wth_{t+1}, so that it will be easier to redistribute some of the money later.

2.1.4 Policy and hyperparameters

As is standard we use an ε\varepsilon-greedy policy. It picks actions based on the value of the average of the two QQ-functions in a given state. In general the choice of ε\varepsilon is not a trivial task and the performance of the algorithm can vary greatly depending on this choice. There are many suggestions on how to successfully manage this balance of exploration and exploitation (using for example decay of ε\varepsilon [8], change point detection [13], adaptation based on value differences [14], etc.). What we want here is to be able to explore sufficiently when the market conditions change, which is very important for a fully online algorithm. In line with this we use a simple decay of ε\varepsilon, but mixed with a probabilistic reset to a larger value. More precisely, first initialise a counter iεi_{\varepsilon} to 00. Before each choice of an action iεi_{\varepsilon} is either incremented by 11 or with probability p​r​o​bεprob_{\varepsilon} it is reset to ⌈e5−25⌉\left\lceil\frac{e^{5}-2}{5}\right\rceil, if iε≥⌈e5−25⌉i_{\varepsilon}\geq\left\lceil\frac{e^{5}-2}{5}\right\rceil (this resets ε\varepsilon to about 0.20.2). Afterwards, ε\varepsilon is calculated according to the formula

ε=1ln⁡(5​iε+2).\displaystyle\varepsilon=\frac{1}{\ln\left(5i_{\varepsilon}+2\right)}.

(2.18)

Next, we want to choose a learning rate α\alpha. It is well-known that the learning rate in gradient descent methods greatly affects the performance of a neural network (or of the RL algorithm using it) [15]. To avoid fixing the learning rate manually, we choose to use a cyclical one [16, 17]. More precisely, a counter iαi_{\alpha} is initialised to 00. Then, before taking the previously chosen action α\alpha is calculated using the formula [17]

α=αm​i​n+12​(αm​a​x−αm​i​n)​(1+cos⁡(iαTα​π)),\displaystyle\alpha=\alpha_{min}+\frac{1}{2}\left(\alpha_{max}-\alpha_{min}\right)\left(1+\cos\left(\frac{i_{\alpha}}{T_{\alpha}}\pi\right)\right),

(2.19)

after which iαi_{\alpha} is incremented by 11. This means that α\alpha varies between αm​a​x\alpha_{max} and αm​i​n\alpha_{min} with a period of 2​Tα2T_{\alpha} steps.

2.2 Fast learning network

2.2.1 Preliminaries

Now we need to describe the neural network, that will approximate the QQ-function, namely FLN. FLNs use a parallel connection of two feedforward neural networks – one has a single hidden layer, while the other has none [7]. The hidden layer weights are random and fixed and only the output weights are learned. If, in addition, we choose the output neurons’ activation function to be the identity function and fix all the biases to zero, this effectively means that the approximating function is linear in the feature vector with additional fixed nonlinear terms from the hidden layer.
More precisely, we can denote the input and output as XX and YY, respectively:

X=(x1x2...xn),Y=(y1y2...ym),\displaystyle X=\left(\begin{matrix}x_{1}\\
x_{2}\\
...\\
x_{n}\end{matrix}\right),\ \ \ Y=\left(\begin{matrix}y_{1}\\
y_{2}\\
...\\
y_{m}\end{matrix}\right),

(2.20)

where nn and mm are the respective sizes of the input and the output. Also, for shortness of notation, we denote the hidden layer output as

g⁡(Z)=(g⁡(z1)g⁡(z2)...g⁡(zr)).\displaystyle g(Z)=\left(\begin{matrix}g(z_{1})\\
g(z_{2})\\
...\\
g(z_{r})\end{matrix}\right).

(2.21)

Here gg is the activation function of the hidden layer and ZZ is the input to the hidden layer. rr is the hidden layer size.
The weights are denoted by WoiW^{\mathrm{oi}} (input to output layers), WhiW^{\mathrm{hi}} (input to hidden layers) and WohW^{\mathrm{oh}} (hidden to output layers):

Woi=(w11oi...w1​noi...wm​1oi...wm​noi),Whi=(w11hi...w1​nhi...wr​1hi...wr​nhi),Woh=(w11oh...w1​roh...wm​1oh...wm​roh).\displaystyle W^{\mathrm{oi}}=\left(\begin{matrix}w^{\mathrm{oi}}_{11}&...&w^{\mathrm{oi}}_{1n}\\
...\\
w^{\mathrm{oi}}_{m1}&...&w^{\mathrm{oi}}_{mn}\end{matrix}\right),\ \ \ W^{\mathrm{hi}}=\left(\begin{matrix}w^{\mathrm{hi}}_{11}&...&w^{\mathrm{hi}}_{1n}\\
...\\
w^{\mathrm{hi}}_{r1}&...&w^{\mathrm{hi}}_{rn}\end{matrix}\right),\ \ \ W^{\mathrm{oh}}=\left(\begin{matrix}w^{\mathrm{oh}}_{11}&...&w^{\mathrm{oh}}_{1r}\\
...\\
w^{\mathrm{oh}}_{m1}&...&w^{\mathrm{oh}}_{mr}\end{matrix}\right).

(2.22)

Now the kk-th component of the output vector is calculated by the following formula [7]:

yk=∑s=1nwk​soi​xs+∑l=1rwk​loh​g​(∑t=1nwl​thi​xt).\displaystyle y_{k}=\sum_{s=1}^{n}w^{\mathrm{oi}}_{ks}x_{s}+\sum_{l=1}^{r}w^{\mathrm{oh}}_{kl}g\left(\sum_{t=1}^{n}w^{\mathrm{hi}}_{lt}x_{t}\right).

(2.23)

We can shorten the above to

Y=Woi​X+Woh​g​(Whi​X).\displaystyle Y=W^{\mathrm{oi}}X+W^{\mathrm{oh}}g\left(W^{\mathrm{hi}}X\right).

(2.24)

The optimisation is then performed only with respect to WoiW^{\mathrm{oi}} and WohW^{\mathrm{oh}}.

2.2.2 Weight renormalisation

One common problem that we can encounter is that the weights in the neural network may diverge. This is especially true when the learning rate is large (but a large learning rate might help with adaptation). The above is a problem, since it is generally accepted that very large weights correlate with overfitting the training set and poor generalisation [18]. There are many ways to try to deal with this, but one simple method is to just renormalise the weight vector [19]. We do something similar with the output weights WoiW^{\mathrm{oi}} and WohW^{\mathrm{oh}}.
More precisely, consider the output yky_{k}. It is obtained by scalar multiplication of the weight vector

(wk​1oi,wk​2oi,…,wk​noi,wk​1oh,wk​2oh,…,wk​roh)T\displaystyle\left(w^{\mathrm{oi}}_{k1},w^{\mathrm{oi}}_{k2},...,w^{\mathrm{oi}}_{kn},w^{\mathrm{oh}}_{k1},w^{\mathrm{oh}}_{k2},...,w^{\mathrm{oh}}_{kr}\right)^{T}

(2.25)

with the concatenation of the input XX and the hidden layer output g⁡(Z)g(Z). The vector (2.25) itself is the concatenation of the kk-th row of WoiW^{\mathrm{oi}} and the kk-th row of WohW^{\mathrm{oh}} and it is learned by stochastic gradient descent. A record of the maximal value of its norm m​a​x​wmaxw is kept. If the weight vector is longer than 11 after an update, it is rescaled by a factor of 1m​a​x​w\frac{1}{maxw}. This keeps the weights from diverging and allows us to use large learning rates (the exact values of αm​i​n\alpha_{min} and αm​a​x\alpha_{max} are hyperparameters and will be specified later, but them being larger should help the agent adapt quickly).

3 Implementation details and testing

3.1 Observing, trading, hyperparameters

Before implementing the algorithm we need to consider a few points, namely how to record prices, how to trade, how exactly to structure the FLN and the values of the hyperparameters. The first question that needs to be answered is how often to record a price (and volume) for the feature vector. In principle the intervals can be of any length. One advantage of automated trading is that it can react quickly to the market. In line with this we want the intervals to be short, e.g. 11 minute (more precisely the price is recorded in the beginning of the 11-minute interval). However, a trade occurs right after observing 55 prices, which in practice means that trades are performed every 55 minutes. This means that, depending on the volatility of the market, consecutive trades might happen on similar (often the same) prices and the profit from this is very small. Possibly too small to compensate for the trading fee. To counter this the observed price passes through a filter before being recorded, such that the relative change between two prices is greater than 0.010.01.
The next question is how exactly to trade. In testing we just assume that the trade occurs at the last recorded price p​r5pr_{5}. To ensure this in practice one should use limit orders instead of market orders to avoid slippage. However, this poses the problem that the trade might not be executed at all (or at least not before new 55 prices are recorded and it’s time to trade again). To ensure that it has up-to-date information the trader should cancel the order before the next 55 prices are recorded, e.g. after recording p​r4pr_{4}. Additionally, in such cases one can include the same negative reward as for trade failure due to insufficient funds to try to discourage orders that are later cancelled.
Next, we need to describe the neural network in more detail. It’s input is the feature vector (2.5), representing the state, while in its output we include one node for each action, i.e.

X=f​e​a​t,Y=(Q⁡(s,a1)Q⁡(s,a2)...Q⁡(s,a19)).\displaystyle X=feat,\ \ \ Y=\left(\begin{matrix}Q(s,a_{1})\\
Q(s,a_{2})\\
...\\
Q(s,a_{19})\end{matrix}\right).

(3.1)

This means that there are 2727 input nodes and 1919 output nodes. The size of the hidden layer is a hyperparameter of the algorithm and it is fixed to r=50r=50. Then the weight martices WoiW^{\mathrm{oi}}, WhiW^{\mathrm{hi}} and WohW^{\mathrm{oh}} are 19×2719\times 27, 50×2750\times 27 and 19×5019\times 50, respectively, while the weight vector (2.25) has 7777 components.
From the above we see that there is a separate weight vector (2.25) for each action aka_{k}. After choosing and taking the action aka_{k} only the respective weight vector should be updated. So the gradient of Q⁡(s,ak)Q(s,a_{k}) with respect to the weights is just the concatenation of XX and g⁡(Z)g(Z) and it is the same for all actions.
For the neuron activation function we choose to use the logistic function, i.e.

g⁡(z)=11+e−z,\displaystyle g(z)=\frac{1}{1+e^{-z}},

(3.2)

and the feature vector (2.5) is scaled, so that its norm is 66, before feeding it into the neural network.
Finally, we need to fix the rest of the hyperparameters (in addition to the hidden layer size). For the discount factor we choose γ=0.05\gamma=0.05. The lowest possible value of m​l​i​mmlim is fixed to m​l​i​m​n=75mlimn=75. While we have eliminated the need to choose ε\varepsilon, there is still a hyperparameter to fix and it is the probability for a reset of ε\varepsilon. We choose this to be p​r​o​bε=10−4prob_{\varepsilon}=10^{-4}. Likewise, we are not choosing the learning rate α\alpha. Nevertheless, there are still hyperparameters to fix there also, namely αm​i​n\alpha_{min}, αm​a​x\alpha_{max} and TαT_{\alpha}. We choose the following values: αm​i​n=10−3\alpha_{min}=10^{-3}, αm​a​x=1\alpha_{max}=1 and Tα=103T_{\alpha}=10^{3}.

3.2 Testing

With the above considerations in mind here we present the results of testing the algorithm (the whole code for which is written in Mathematica and is included in appendix A) on historical market data. Because we are using previously recorded prices, as already mentioned, there are a few things we can’t account for, one of which is that in testing the order and the trade are the same, i.e. the order is always fully fulfilled at exactly the recorded price. Also, the precision is much higher when testing as we may use numbers with many digits. In real world applications one needs to round appropriately (e.g. when using part of r​e​sres, when placing an order, etc.).
In principle nothing stops us from usign the trader in any market, but our tests are performed on historical data for the ADA/USDT cryptocurrency pair on Binance from the pair listing on 17.04.201817.04.2018 to 06.08.202106.08.2021. We first pass the data through a filter as described in section 3.1. Then we use 44 subsets of the filtered data. One is the whole dataset (figure 1(a)), while the other three are attempts to capture different market conditions – a “bearish” (figure 1(b)), a “bullish” (figure 1(c)) and a “mixed” (figure 1(d)) market.

(a) The full dataset from 17.04.201817.04.2018 to 06.08.202106.08.2021.

(b) A subset capturing a “bearish” market between 16.05.202116.05.2021 and 06.08.202106.08.2021.

(c) A subset capturing a “bullish” market from 25.02.202125.02.2021 to 16.05.202116.05.2021.

(d) A subset capturing a “mixed” market between 27.02.202127.02.2021 and 29.06.202129.06.2021.

Figure 1: Different subsets of the filtered dataset. The ticks along the upper frames are chosen to show the dates for 10 equally spaced (in terms of the datapoint number in the subset) points. A larger distance between two dates represents the lower volatility of the market (i.e. the smaller changes in price) in this period, which leads to more data points in the unfiltered data (and thus the greater distance) for the same amount of points in the filtered data.

For each dataset we perform 10001000 runs of the algorithm. Each run starts with m​o​n=100mon=100, c​n​s=0cns=0, s​a​v=0sav=0, r​e​s=0res=0 and m​l​i​m=m​o​nmlim=mon and we record the performance in terms of the sum of the wealth (2.6) and the s​a​vsav and r​e​sres pools, i.e.

t​w​t​h=w​t​h+s​a​v+r​e​s,\displaystyle twth=wth+sav+res,

(3.3)

as well as the value of s​a​vsav alone, at the end of the run. In order to evaluate the effectiveness of the algorithm we also perform 10001000 runs with randomly selected actions on each time step. In this case t​w​t​h=w​t​htwth=wth, as the savings mechanism depends on multiple reinforcement learning ingredients.
After this we arrange the data in histograms (figures 2, 3, 4 and 5), which also include the sample minimum and maximum, and calculate the sample means, medians, standard deviations, as well as the empirical probabilities for finishing a run with t​w​t​h≤100twth\leq 100. The last is included as a measure of the risk of losing money after a run. All of these are arranged in tables 1, 2, 3 and 4. As can be seen, our algorithm performs better than random in all datasets.
In particular, for the full dataset we observe an increase in the mean value of t​w​t​htwth of about 39%39\% when taking non-random actions versus random ones. The median also increases – by 58%58\%. Additionally, the probability for finishing a run with t​w​t​h≤100twth\leq 100 (i.e. for losing money) is 86%86\% smaller when taking non-random actions.

(a) t​w​t​htwth when taking non-random actions.

(b) t​w​t​htwth when taking random actions.

(c) s​a​vsav (when taking non-random actions).

Figure 2: Histograms for the full dataset.

Mean [USDT]
Median [USDT]
St. Dev. [USDT]
P⁡(t​w​t​h≤100)P(twth\leq 100)

Non-random t​w​t​htwth
263.928
241.903
113.334
0.031

Random t​w​t​htwth
189.703
153.028
121.777
0.226

s​a​vsav
68.192
64.020
32.131
−-

Table 1: Sample means, medians and standard deviations of t​w​t​htwth when taking random or non-random actions and of s​a​vsav for the full dataset. The probability for finishing a run with t​w​t​h≤100twth\leq 100 is also included.

Similar calculations can be made for the rest of the datasets. In all of the cases the algorithm has higher mean and median values of t​w​t​htwth when compared to random, as well as lower values of P⁡(t​w​t​h≤100)P(twth\leq 100), i.e. it makes more money on average and has a lower chance to lose money. Probably the most interesting case is the “bearish” market one, as there it is the hardest to make a profit. This is reflected in our results as the differences with random are the smallest. More specifically, the mean and median of t​w​t​htwth are about 4%4\% larger and the probability of t​w​t​h≤100twth\leq 100 – about 5%5\% smaller.

(a) t​w​t​htwth when taking non-random actions.

(b) t​w​t​htwth when taking random actions.

(c) s​a​vsav (when taking non-random actions).

Figure 3: Histograms for the “bearish” dataset.

Mean [USDT]
Median [USDT]
St. Dev. [USDT]
P⁡(t​w​t​h≤100)P(twth\leq 100)

Non-random t​w​t​htwth
78.99978.999
77.53877.538
16.68916.689
0.8840.884

Random t​w​t​htwth
76.13976.139
74.90574.905
15.16915.169
0.9260.926

s​a​vsav
3.0603.060
1.9821.982
3.3243.324
−-

Table 2: Descriptive statistics of t​w​t​htwth when taking random or non-random actions and of s​a​vsav for the “bearish” dataset.

For completeness we also include the relative performance in the other two datasets. In terms of the mean of t​w​t​htwth our algorithm performs about 5%5\% better in the “bullish” dataset and 12%12\% better in the “mixed” dataset. In terms of the median the increases are 6%6\% and 14%14\% for the “bullish” and “mixed” cases, respectively. Finally, in terms of P⁡(t​w​t​h≤100)P(twth\leq 100) the decreases are 74%74\% and 26%26\%, respectively, for the two cases.

(a) t​w​t​htwth when taking non-random actions.

(b) t​w​t​htwth when taking random actions.

(c) s​a​vsav (when taking non-random actions).

Figure 4: Histograms for the “bullish” dataset.

Mean [USDT]
Median [USDT]
St. Dev. [USDT]
P⁡(t​w​t​h≤100)P(twth\leq 100)

Non-random t​w​t​htwth
152.267152.267
150.601150.601
25.37225.372
0.0070.007

Random t​w​t​htwth
145.245145.245
142.256142.256
28.56128.561
0.0270.027

s​a​vsav
10.16910.169
9.7589.758
4.7554.755
−-

Table 3: Descriptive statistics for the “bullish” dataset.

(a) t​w​t​htwth when taking non-random actions.

(b) t​w​t​htwth when taking random actions.

(c) s​a​vsav (when taking non-random actions).

Figure 5: Histograms for the “mixed” dataset.

Mean [USDT]
Median [USDT]
St. Dev. [USDT]
P⁡(t​w​t​h≤100)P(twth\leq 100)

Non-random t​w​t​htwth
104.338104.338
100.545100.545
26.77126.771
0.4910.491

Random t​w​t​htwth
93.20293.202
88.56988.569
26.99426.994
0.6640.664

s​a​vsav
11.18211.182
10.61010.610
6.0396.039
−-

Table 4: Descriptive statistics for the “mixed” dataset.

4 Conclusion

In this paper we attempted to tackle the challenging problem of market prediction using machine learning. More specifically, we introduced a deep reinforcement learning agent that is meant to adapt to market conditions and trade fully online. The main components of our algorithm are a more or less standard Double QQ-learning framework coupled with a Fast Learning Network, used to approximate the QQ-functions. On top of that we added a mechanism, which takes money out of the trading pool, both as a means to take profit and to boost performance by reusing some of it at a more favourable moment.
After this we tested the algorithm on historical market data, which was chosen so that it captures different market conditions. We observed that our agent performs better than random on all datasets – both in terms of profit and probability of loss at the end of a run through the data. Furthermore, it did so even in a “bearish” market, when the overall market trend is downward and, most importantly, without any prior learning on big offline dataset. We can view the latter as the main strength of our algorithm.

Appendix A Mathematica code

CloseKernels[];LaunchKernels[];ClearAll[“Global`*”]SetDirectory[NotebookDirectory[]](* Read the CSV file with the prices and volumes. *)
data=Import[“ADA-USDT.csv”];uprices=Table[data[[𝒌,𝟐]],{𝒌,𝟐,Length[data]}];simlength=Length[uprices]uvolumes=Table[data[[𝒌,𝟔]]/𝟏𝟎𝟎𝟎𝟎𝟎𝟎𝟎,{𝒌,𝟐,Length[data]}];(* Pass the prices through a filer and make a list of the new prices and preceding volumes. *)
prices=Table[𝟎,{𝒌,𝟏,simlength}];volumes=Table[𝟎,{𝒌,𝟏,simlength}];pinit=uprices[[𝟐]];prices[[𝟏]]=uprices[[𝟐]];volumes[[𝟏]]=uvolumes[[𝟏]];𝒍=𝟐;𝒊=𝟑;While[𝒊≤simlength,If[Abs[uprices[[𝒊]]−pinit]/pinit>0.01,pinit=uprices[[𝒊]];prices[[𝒍]]=uprices[[𝒊]];volumes[[𝒍]]=uvolumes[[𝒊−𝟏]];𝒍=𝒍+𝟏];𝒊=𝒊+𝟏];prices=DeleteCases[prices,𝟎];simlength=Length[prices]volumes=DeleteCases[volumes,𝟎];(* Plot the prices and volumes. *)
ListPlot[prices,Joined→True,PlotRange→Full]ListPlot[volumes,Joined→True,PlotRange→Full](* Parameters. *)
vsize=𝟓;(* Number of data points for each state. *)
fsize=𝟏+vsize+𝟏+𝟏+𝟏+𝟏+𝟏+𝟏+𝟏+𝟏+𝟏+𝟏+𝟒+𝟑+𝟐+𝟏+𝟏(* Size of the feature vector. *)
hlsize=𝟓𝟎;(* Size of the hidden layer. *)
gamma=0.05;(* Discount factor. *)
probeps=0.0001;(* Probability to reset the value of epsilon. *)
mlimn=𝟕𝟓.;(* The lowest possible value of mlim. *)
runs=𝟏𝟎𝟎𝟎;(* Number of test runs. *)

feat=Table[𝟎.,{𝒌,𝟏,fsize}];(* Feature vector. *)
pr=Table[𝟎.,{𝒌,𝟏,vsize}];(* List of prices in the state. *)

(* Hidden layer outputs. *)
hlayer1=Table[𝟎.,{𝒌,𝟏,hlsize}];hlayer2=Table[𝟎.,{𝒌,𝟏,hlsize}];(* Gradients of the Q-functions. *)
gradq1=Table[𝟎.,{𝒌,𝟏,hlsize+fsize}];gradq1new=Table[𝟎.,{𝒌,𝟏,hlsize+fsize}];gradq2=Table[𝟎.,{𝒌,𝟏,hlsize+fsize}];gradq2new=Table[𝟎.,{𝒌,𝟏,hlsize+fsize}];(* Q-functions. *)
qarr1=Table[𝟎.,{𝒌,𝟏,𝟏𝟗}];qarr2=Table[𝟎.,{𝒌,𝟏,𝟏𝟗}];qarr=Table[𝟎.,{𝒌,𝟏,𝟏𝟗}];qarr1new=Table[𝟎.,{𝒌,𝟏,𝟏𝟗}];qarr2new=Table[𝟎.,{𝒌,𝟏,𝟏𝟗}];(* Lists to write sav and twth at the end of each run. *)
savarray=Table[𝟎.,{𝒌,𝟏,runs}];twtharray=Table[𝟎.,{𝒌,𝟏,runs}];(* Relative movements of the price, etc. *)
nmd1=Table[𝟎.,{𝒌,𝟏,vsize−𝟏}];nmd2=Table[𝟎.,{𝒌,𝟏,vsize−𝟐}];nmd3=Table[𝟎.,{𝒌,𝟏,vsize−𝟑}];nmd4=Table[𝟎.,{𝒌,𝟏,vsize−𝟒}];(* The activation function. *)
Plot[𝟏./(𝟏.+Exp[−𝒙]),{𝒙,−𝟔.,𝟔.}]𝒎=𝟏;While[𝒎≤runs,(* Initial weights are randomly generated. Then the weights between the input and hidden layers are rescaled. *)
whi1=Table[RandomReal[{−𝟏.,𝟏.}],{𝒌,𝟏,hlsize},{𝒍,𝟏,fsize}];𝒋=𝟏;While[𝒋≤hlsize,whi1[[𝒋]]=whi1[[𝒋]]/Norm[whi1[[𝒋]]];𝒋=𝒋+𝟏];whi2=Table[RandomReal[{−𝟏.,𝟏.}],{𝒌,𝟏,hlsize},{𝒍,𝟏,fsize}];𝒋=𝟏;While[𝒋≤hlsize,whi2[[𝒋]]=whi2[[𝒋]]/Norm[whi2[[𝒋]]];𝒋=𝒋+𝟏];wout1=Table[RandomReal[{−𝟏.,𝟏.}],{𝒍,𝟏,𝟏𝟗},{𝒌,𝟏,hlsize+fsize}];wout2=Table[RandomReal[{−𝟏.,𝟏.}],{𝒍,𝟏,𝟏𝟗},{𝒌,𝟏,hlsize+fsize}];(* Initial values of the counters for epsilon and alpha. *)
ialpha=−𝟏;ieps=𝟎;(* Tables to be used for calculating RSI. *)
rsip=Table[𝟎.,{𝒍,𝟏,𝟏𝟓}];rsiu=Table[𝟎.,{𝒍,𝟏,𝟏𝟒}];rsid=Table[𝟎.,{𝒍,𝟏,𝟏𝟒}];(* Variables for the average volumes and the max norms of the output weights. *)
mv=Table[𝟎.,{𝒋,𝟏,𝟐𝟎}];nv=𝟎;av=𝟎.;max1=𝟏.;max2=𝟏.;(* Initial mon, cns, sav, res, mlim. *)
mon=𝟏𝟎𝟎.;cns=𝟎.;sav=𝟎.;res=𝟎.;mlim=mon;mdf=𝟎;𝒊=𝟏;Label[episode];(* Label used to start a new episode. *)

(* Initial price in the episode. *)
ipr=prices[[𝒊∗𝟓−vsize+𝟏]];(* Calculate the average volumes. *)
cav=(volumes[[𝒊∗𝟓−𝟒]]+volumes[[𝒊∗𝟓−𝟑]]+volumes[[𝒊∗𝟓−𝟐]]+volumes[[𝒊∗𝟓−𝟏]]+volumes[[𝒊∗𝟓]])/𝟓.;𝒋=𝟏;While[𝒋≤𝟏𝟗,mv[[𝒋]]=mv[[𝒋+𝟏]];𝒋=𝒋+𝟏];mv[[𝒋]]=cav;av=Mean[mv];(* Observe the prices in the current state. *)
𝒋=𝟏;While[𝒋≤vsize,pr[[𝒋]]=prices[[𝒊∗𝟓−vsize+𝒋]];𝒋=𝒋+𝟏];(* Relative price movements. *)
𝒋=𝟏;While[𝒋≤vsize−𝟏,nmd1[[𝒋]]=(pr[[𝒋+𝟏]]−pr[[𝒋]])/pr[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟐,nmd2[[𝒋]]=(nmd1[[𝒋+𝟏]]−nmd1[[𝒋]])/Abs[nmd1[[𝒋]]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟑,nmd3[[𝒋]]=(nmd2[[𝒋+𝟏]]−nmd2[[𝒋]])/Abs[nmd2[[𝒋]]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟒,nmd4[[𝒋]]=(nmd3[[𝒋+𝟏]]−nmd3[[𝒋]])/Abs[nmd3[[𝒋]]];𝒋=𝒋+𝟏];(* Calculate RSI for 15 points. *)
𝒋=𝟏;While[𝒋≤𝟏𝟎,rsip[[𝒋]]=rsip[[𝒋+𝟓]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize,rsip[[𝟏𝟎+𝒋]]=pr[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤𝟏𝟒,If[rsip[[𝒋+𝟏]]>rsip[[𝒋]],rsiu[[𝒋]]=rsip[[𝒋+𝟏]]−rsip[[𝒋]];rsid[[𝒋]]=𝟎.];If[rsip[[𝒋+𝟏]]<rsip[[𝒋]],rsid[[𝒋]]=rsip[[𝒋]]−rsip[[𝒋+𝟏]];rsiu[[𝒋]]=𝟎.];If[rsip[[𝒋+𝟏]]==rsip[[𝒋]],rsiu[[𝒋]]=𝟎.;rsid[[𝒋]]=𝟎.];𝒋=𝒋+𝟏];If[Mean[rsid]==𝟎.,rsi=𝟏𝟎𝟎.,rsi=𝟏𝟎𝟎.−(𝟏𝟎𝟎./(𝟏.+(Mean[rsiu]/Mean[rsid])))];(* Construct the feature vector (same for all actions). *)
feat[[𝟏]]=𝟎.;𝒋=𝟏;𝒌=𝟏;While[𝒋≤vsize,feat[[𝒌+𝟏]]=pr[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];feat[[𝒌+𝟏]]=ipr;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=(pr[[vsize]]−ipr)/ipr;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=mon;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=cns;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=cav;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=av;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=(cav−av)/av;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=(volumes[[𝒊∗𝟓]]−av)/av;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=(volumes[[𝒊∗𝟓]]−cav)/cav;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=rsi;𝒌=𝒌+𝟏;𝒋=𝟏;While[𝒋≤vsize−𝟏,feat[[𝒌+𝟏]]=nmd1[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟐,feat[[𝒌+𝟏]]=nmd2[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟑,feat[[𝒌+𝟏]]=nmd3[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟒,feat[[𝒌+𝟏]]=nmd4[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];feat[[𝒌+𝟏]]=mlim;𝒌=𝒌+𝟏;(* Rescale the feature vector, so its norm is 6. *)
feat=𝟔.∗(feat/Norm[feat]);feat[[𝟏]]=𝟏.;(* Output of the hidden layers. *)
𝒏=𝟏;While[𝒏≤hlsize,𝒒=𝟎;𝒋=𝟏;While[𝒋≤fsize,𝒒=𝒒+whi1[[𝒏,𝒋]]∗feat[[𝒋]];𝒋=𝒋+𝟏];hlayer1[[𝒏]]=𝟏./(𝟏.+Exp[−𝒒]);𝒒=𝟎;𝒋=𝟏;While[𝒋≤fsize,𝒒=𝒒+whi2[[𝒏,𝒋]]∗feat[[𝒋]];𝒋=𝒋+𝟏];hlayer2[[𝒏]]=𝟏./(𝟏.+Exp[−𝒒]);𝒏=𝒏+𝟏];(* Gradients of the Q-functions, taken in the current state (same for all actions), which are just
concatenations of the feature vector with the hidden layer outputs. *)
𝒋=𝟏;While[𝒋≤fsize,gradq1[[𝒋]]=feat[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤hlsize,gradq1[[fsize+𝒋]]=hlayer1[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤fsize,gradq2[[𝒋]]=feat[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤hlsize,gradq2[[fsize+𝒋]]=hlayer2[[𝒋]];𝒋=𝒋+𝟏];(* Calculate wth in the current state. *)
wth=mon+(cns∗prices[[𝒊∗𝟓]]);While[(𝒊+𝟏)∗𝟓≤simlength,(* Calculate Q1 and Q2 for every action in the given state. *)
𝒏=𝟏;While[𝒏≤𝟏𝟗,𝒒=𝟎;𝒋=𝟏;While[𝒋≤hlsize+fsize,𝒒=𝒒+wout1[[𝒏,𝒋]]∗gradq1[[𝒋]];𝒋=𝒋+𝟏];qarr1[[𝒏]]=𝒒;𝒒=𝟎;𝒋=𝟏;While[𝒋≤hlsize+fsize,𝒒=𝒒+wout2[[𝒏,𝒋]]∗gradq2[[𝒋]];𝒋=𝒋+𝟏];qarr2[[𝒏]]=𝒒;𝒏=𝒏+𝟏];(* Calculate the average of Q1 and Q2, which is needed to choose an action. *)
qarr=(qarr1+qarr2)/𝟐.;(* Update epsilon. *)
rand=RandomReal[];If[((rand≤probeps)&&(ieps≥Ceiling[(Exp[𝟏./0.2]−𝟐.)/𝟓.])),ieps=Ceiling[(Exp[𝟏./0.2]−𝟐.)/𝟓.],ieps=ieps+𝟏];eps=𝟏./Log[ieps∗𝟓.+𝟐.];(* Choose an action. *)
rand=RandomReal[];If[rand≤eps,𝒂=RandomInteger[{𝟏,𝟏𝟗}],𝒂=Ordering[qarr,−𝟏][[𝟏]]];(* Update alpha. *)
ialpha=ialpha+𝟏;alpha=0.001+(𝟏./𝟐.)∗(𝟏.−0.001)∗(𝟏.+Cos[Pi∗ialpha/𝟏𝟎𝟎𝟎.]);(* Take the chosen action. *)
fff=𝟏.;(* This tracks for insufficient mon or cns in order to reflect it in the reward later. *)
If[𝒂==𝟏,If[mon<𝟏𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟏𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟏𝟎.];If[𝒂==𝟐,If[mon<𝟐𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟐𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟐𝟎.];If[𝒂==𝟑,If[mon<𝟑𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟑𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟑𝟎.];If[𝒂==𝟒,If[mon<𝟒𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟒𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟒𝟎.];If[𝒂==𝟓,If[mon<𝟓𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟓𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟓𝟎.];If[𝒂==𝟔,If[mon<𝟔𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟔𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟔𝟎.];If[𝒂==𝟕,If[mon<𝟕𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟕𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟕𝟎.];If[𝒂==𝟖,If[mon<𝟖𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟖𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟖𝟎.];If[𝒂==𝟗,If[mon<𝟗𝟎.,fff=−𝟏.;Goto[fail]];cns=cns+((99.9/𝟏𝟎𝟎.)∗(𝟗𝟎./prices[[𝒊∗𝟓]]));mon=mon−𝟗𝟎.];If[𝒂==𝟏𝟎,If[prices[[𝒊∗𝟓]]∗cns<𝟏𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟏𝟎.));cns=cns−(𝟏𝟎./prices[[𝒊∗𝟓]])];If[𝒂==𝟏𝟏,If[prices[[𝒊∗𝟓]]∗cns<𝟐𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟐𝟎.));cns=cns−(𝟐𝟎./prices[[𝒊∗𝟓]])];If[𝒂==𝟏𝟐,If[prices[[𝒊∗𝟓]]∗cns<𝟑𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟑𝟎.));cns=cns−(𝟑𝟎./prices[[𝒊∗𝟓]])];If[𝒂==𝟏𝟑,If[prices[[𝒊∗𝟓]]∗cns<𝟒𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟒𝟎.));cns=cns−(𝟒𝟎./prices[[𝒊∗𝟓]])];If[𝒂==𝟏𝟒,If[prices[[𝒊∗𝟓]]∗cns<𝟓𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟓𝟎.));cns=cns−(𝟓𝟎./prices[[𝒊∗𝟓]])];If[𝒂==𝟏𝟓,If[prices[[𝒊∗𝟓]]∗cns<𝟔𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟔𝟎.));cns=cns−(𝟔𝟎./prices[[𝒊∗𝟓]])];If[𝒂==𝟏𝟔,If[prices[[𝒊∗𝟓]]∗cns<𝟕𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟕𝟎.));cns=cns−(𝟕𝟎./prices[[𝒊∗𝟓]])];If[𝒂==𝟏𝟕,If[prices[[𝒊∗𝟓]]∗cns<𝟖𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟖𝟎.));cns=cns−(𝟖𝟎./prices[[𝒊∗𝟓]])];If[𝒂==𝟏𝟖,If[prices[[𝒊∗𝟓]]∗cns<𝟗𝟎.,fff=−𝟏.;Goto[fail]];mon=mon+((99.9/𝟏𝟎𝟎.)∗(𝟗𝟎.));cns=cns−(𝟗𝟎./prices[[𝒊∗𝟓]])];(* If[𝒂==𝟏𝟗,HOLD]; *)
Label[fail];If[((mon<𝟎.)∥(cns<𝟎.)),Print[“Negative mon or cns!”]];(* Just to be sure. *)

(* Observe the reward (by calculating wth in the next state). *)
wthnew=mon+(cns∗prices[[(𝒊+𝟏)∗𝟓]]);rew=wthnew−wth;If[fff<𝟎.,rew=rew−((rew/𝟐.)𝟐∧)−0.1,rew=rew−((rew/𝟐.)𝟐∧)];(* Observe the prices in the next state. *)
𝒋=𝟏;While[𝒋≤vsize,pr[[𝒋]]=prices[[(𝒊+𝟏)∗𝟓−vsize+𝒋]];𝒋=𝒋+𝟏];(* Calculate RSI for 15 points. *)
𝒋=𝟏;While[𝒋≤𝟏𝟎,rsip[[𝒋]]=rsip[[𝒋+𝟓]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize,rsip[[𝟏𝟎+𝒋]]=pr[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤𝟏𝟒,If[rsip[[𝒋+𝟏]]>rsip[[𝒋]],rsiu[[𝒋]]=rsip[[𝒋+𝟏]]−rsip[[𝒋]];rsid[[𝒋]]=𝟎.];If[rsip[[𝒋+𝟏]]<rsip[[𝒋]],rsid[[𝒋]]=rsip[[𝒋]]−rsip[[𝒋+𝟏]];rsiu[[𝒋]]=𝟎.];If[rsip[[𝒋+𝟏]]==rsip[[𝒋]],rsiu[[𝒋]]=𝟎.;rsid[[𝒋]]=𝟎.];𝒋=𝒋+𝟏];If[Mean[rsid]==𝟎.,rsi=𝟏𝟎𝟎.,rsi=𝟏𝟎𝟎.−(𝟏𝟎𝟎./(𝟏.+(Mean[rsiu]/Mean[rsid])))];(* Check for the three possible terminal states. *)
If[mon>mlim,(* Terminal state 1. *)
rew=rew+((mon−mlim)∗0.34);rand=RandomReal[];If[rand≤0.5,wout1[[𝒂]]=wout1[[𝒂]]+alpha∗(rew−qarr1[[𝒂]])∗gradq1;max1=Max[max1,Norm[wout1[[𝒂]]]];If[Norm[wout1[[𝒂]]]>𝟏.,wout1[[𝒂]]=wout1[[𝒂]]/max1],wout2[[𝒂]]=wout2[[𝒂]]+alpha∗(rew−qarr2[[𝒂]])∗gradq2;max2=Max[max2,Norm[wout2[[𝒂]]]];If[Norm[wout2[[𝒂]]]>𝟏.,wout2[[𝒂]]=wout2[[𝒂]]/max2]];mdf=mon−mlim;sav=sav+(mdf∗0.34);res=res+(mdf∗0.33);mon=mlim+(mdf∗0.33);mlim=mon+mdf;𝒊=𝒊+𝟐;If[𝒊∗𝟓>simlength,Goto[break]];Goto[episode],If[((wthnew<mlimn)&&(qarr[[𝒂]]>𝟎.)&&(rsi>𝟕𝟎.)),(* Terminal state 2. *)
rand=RandomReal[];If[rand≤0.5,wout1[[𝒂]]=wout1[[𝒂]]+alpha∗(rew−qarr1[[𝒂]])∗gradq1;max1=Max[max1,Norm[wout1[[𝒂]]]];If[Norm[wout1[[𝒂]]]>𝟏.,wout1[[𝒂]]=wout1[[𝒂]]/max1],wout2[[𝒂]]=wout2[[𝒂]]+alpha∗(rew−qarr2[[𝒂]])∗gradq2;max2=Max[max2,Norm[wout2[[𝒂]]]];If[Norm[wout2[[𝒂]]]>𝟏.,wout2[[𝒂]]=wout2[[𝒂]]/max2]];mon=mon+(res/𝟐.);res=res−(res/𝟐.);If[mon≥mlimn,mlim=mon,mlim=mlimn];𝒊=𝒊+𝟐;If[𝒊∗𝟓>simlength,Goto[break]];Goto[episode]];If[((wthnew≥mlimn)&&(qarr[[𝒂]]<𝟎.)&&(rsi<𝟑𝟎.)),(* Terminal state 3. *)
rand=RandomReal[];If[rand≤0.5,wout1[[𝒂]]=wout1[[𝒂]]+alpha∗(rew−qarr1[[𝒂]])∗gradq1;max1=Max[max1,Norm[wout1[[𝒂]]]];If[Norm[wout1[[𝒂]]]>𝟏.,wout1[[𝒂]]=wout1[[𝒂]]/max1],wout2[[𝒂]]=wout2[[𝒂]]+alpha∗(rew−qarr2[[𝒂]])∗gradq2;max2=Max[max2,Norm[wout2[[𝒂]]]];If[Norm[wout2[[𝒂]]]>𝟏.,wout2[[𝒂]]=wout2[[𝒂]]/max2]];mlim=wthnew;𝒊=𝒊+𝟐;If[𝒊∗𝟓>simlength,Goto[break]];Goto[episode]]];(* Calculate the average volumes. *)
cav=(volumes[[(𝒊+𝟏)∗𝟓−𝟒]]+volumes[[(𝒊+𝟏)∗𝟓−𝟑]]+volumes[[(𝒊+𝟏)∗𝟓−𝟐]]+volumes[[(𝒊+𝟏)∗𝟓−𝟏]]+volumes[[(𝒊+𝟏)∗𝟓]])/𝟓.;𝒋=𝟏;While[𝒋≤𝟏𝟗,mv[[𝒋]]=mv[[𝒋+𝟏]];𝒋=𝒋+𝟏];mv[[𝒋]]=cav;av=Mean[mv];(* Relative price movements. *)
𝒋=𝟏;While[𝒋≤vsize−𝟏,nmd1[[𝒋]]=(pr[[𝒋+𝟏]]−pr[[𝒋]])/pr[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟐,nmd2[[𝒋]]=(nmd1[[𝒋+𝟏]]−nmd1[[𝒋]])/Abs[nmd1[[𝒋]]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟑,nmd3[[𝒋]]=(nmd2[[𝒋+𝟏]]−nmd2[[𝒋]])/Abs[nmd2[[𝒋]]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟒,nmd4[[𝒋]]=(nmd3[[𝒋+𝟏]]−nmd3[[𝒋]])/Abs[nmd3[[𝒋]]];𝒋=𝒋+𝟏];(* Construct the feature vector (same for all actions). *)
feat[[𝟏]]=𝟎.;𝒋=𝟏;𝒌=𝟏;While[𝒋≤vsize,feat[[𝒌+𝟏]]=pr[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];feat[[𝒌+𝟏]]=ipr;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=(pr[[vsize]]−ipr)/ipr;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=mon;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=cns;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=cav;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=av;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=(cav−av)/av;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=(volumes[[(𝒊+𝟏)∗𝟓]]−av)/av;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=(volumes[[(𝒊+𝟏)∗𝟓]]−cav)/cav;𝒌=𝒌+𝟏;feat[[𝒌+𝟏]]=rsi;𝒌=𝒌+𝟏;𝒋=𝟏;While[𝒋≤vsize−𝟏,feat[[𝒌+𝟏]]=nmd1[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟐,feat[[𝒌+𝟏]]=nmd2[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟑,feat[[𝒌+𝟏]]=nmd3[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤vsize−𝟒,feat[[𝒌+𝟏]]=nmd4[[𝒋]];𝒌=𝒌+𝟏;𝒋=𝒋+𝟏];feat[[𝒌+𝟏]]=mlim;𝒌=𝒌+𝟏;(* Rescale the feature vector, so its norm is 6. *)
feat=𝟔.∗(feat/Norm[feat]);feat[[𝟏]]=𝟏.;(* Output of the hidden layers. *)
𝒏=𝟏;While[𝒏≤hlsize,𝒒=𝟎;𝒋=𝟏;While[𝒋≤fsize,𝒒=𝒒+whi1[[𝒏,𝒋]]∗feat[[𝒋]];𝒋=𝒋+𝟏];hlayer1[[𝒏]]=𝟏./(𝟏.+Exp[−𝒒]);𝒒=𝟎;𝒋=𝟏;While[𝒋≤fsize,𝒒=𝒒+whi2[[𝒏,𝒋]]∗feat[[𝒋]];𝒋=𝒋+𝟏];hlayer2[[𝒏]]=𝟏./(𝟏.+Exp[−𝒒]);𝒏=𝒏+𝟏];(* Gradients of the Q-functions, taken in the next state (same for all actions), which are just
concatenations of the feature vector with the hidden layer outputs. *)
𝒋=𝟏;While[𝒋≤fsize,gradq1new[[𝒋]]=feat[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤hlsize,gradq1new[[fsize+𝒋]]=hlayer1[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤fsize,gradq2new[[𝒋]]=feat[[𝒋]];𝒋=𝒋+𝟏];𝒋=𝟏;While[𝒋≤hlsize,gradq2new[[fsize+𝒋]]=hlayer2[[𝒋]];𝒋=𝒋+𝟏];(* Calculate Q1 and Q2 for every action in the next state. *)
𝒏=𝟏;While[𝒏≤𝟏𝟗,𝒒=𝟎;𝒋=𝟏;While[𝒋≤hlsize+fsize,𝒒=𝒒+wout1[[𝒏,𝒋]]∗gradq1new[[𝒋]];𝒋=𝒋+𝟏];qarr1new[[𝒏]]=𝒒;𝒒=𝟎;𝒋=𝟏;While[𝒋≤hlsize+fsize,𝒒=𝒒+wout2[[𝒏,𝒋]]∗gradq2new[[𝒋]];𝒋=𝒋+𝟏];qarr2new[[𝒏]]=𝒒;𝒏=𝒏+𝟏];(* Update the output weights. *)
rand=RandomReal[];If[rand≤0.5,amax=Ordering[qarr1new,−𝟏][[𝟏]];wout1[[𝒂]]=wout1[[𝒂]]+alpha∗(rew+gamma∗qarr2new[[amax]]−qarr1[[𝒂]])∗gradq1;max1=Max[max1,Norm[wout1[[𝒂]]]];If[Norm[wout1[[𝒂]]]>𝟏.,wout1[[𝒂]]=wout1[[𝒂]]/max1],amax=Ordering[qarr2new,−𝟏][[𝟏]];wout2[[𝒂]]=wout2[[𝒂]]+alpha∗(rew+gamma∗qarr1new[[amax]]−qarr2[[𝒂]])∗gradq2;max2=Max[max2,Norm[wout2[[𝒂]]]];If[Norm[wout2[[𝒂]]]>𝟏.,wout2[[𝒂]]=wout2[[𝒂]]/max2]];(* Make the next value of wth and the next states current. *)
wth=wthnew;gradq1=gradq1new;gradq2=gradq2new;𝒊=𝒊+𝟏;];Label[break];(* Label used to exit the loop, if there are no more points in the test data. *)
savarray[[𝒎]]=sav;(* Write sav at the end of the current run. *)
twtharray[[𝒎]]=sav+wthnew+res;(* Write twth at the end of the current run. *)

𝒎=𝒎+𝟏];Export[“sav.txt”,savarray]Export[“total.txt”,twtharray]\boldsymbol{\text{CloseKernels}[];}\\
\boldsymbol{\text{LaunchKernels}[];}\\
\boldsymbol{\text{ClearAll}[\text{{``}Global$\grave{}$*{''}}]}\\
\boldsymbol{\text{SetDirectory}[\text{NotebookDirectory}[]]}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Read the CSV file with the prices and volumes. *)}}\\
\boldsymbol{\text{data}=\text{Import}[\text{{``}ADA-USDT.csv{''}}];}\\
\boldsymbol{\text{uprices}=\text{Table}[\text{data}[[k,2]],\{k,2,\text{Length}[\text{data}]\}];\text{simlength}=\text{Length}[\text{uprices}]}\\
\boldsymbol{\text{uvolumes}=\text{Table}[\text{data}[[k,6]]/10000000,\{k,2,\text{Length}[\text{data}]\}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Pass the prices through a filer and make a list of the new prices and preceding volumes. *)}}\\
\boldsymbol{\text{prices}=\text{Table}[0,\{k,1,\text{simlength}\}];}\\
\boldsymbol{\text{volumes}=\text{Table}[0,\{k,1,\text{simlength}\}];}\\
\boldsymbol{\text{pinit}=\text{uprices}[[2]];}\\
\boldsymbol{\text{prices}[[1]]=\text{uprices}[[2]];}\\
\boldsymbol{\text{volumes}[[1]]=\text{uvolumes}[[1]];}\\
\boldsymbol{l=2;}\\
\boldsymbol{i=3;}\\
\boldsymbol{\text{While}[i\leq\text{simlength},}\\
\boldsymbol{\text{If}[\text{Abs}[\text{uprices}[[i]]-\text{pinit}]/\text{pinit}>0.01,\text{pinit}=\text{uprices}[[i]];\text{prices}[[l]]=\text{uprices}[[i]];\text{volumes}[[l]]=\text{uvolumes}[[i-1]];l=l+1];}\\
\boldsymbol{i=i+1}\\
\boldsymbol{];}\\
\boldsymbol{\text{prices}=\text{DeleteCases}[\text{prices},0];}\\
\boldsymbol{\text{simlength}=\text{Length}[\text{prices}]}\\
\boldsymbol{\text{volumes}=\text{DeleteCases}[\text{volumes},0];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Plot the prices and volumes. *)}}\\
\boldsymbol{\text{ListPlot}[\text{prices},\text{Joined}\to\text{True},\text{PlotRange}\to\text{Full}]}\\
\boldsymbol{\text{ListPlot}[\text{volumes},\text{Joined}\to\text{True},\text{PlotRange}\to\text{Full}]}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Parameters. *)}}\\
\boldsymbol{\text{vsize}=5;\text{(* Number of data points for each state. *)}}\\
\boldsymbol{\text{fsize}=1+\text{vsize}+1+1+1+1+1+1+1+1+1+1+4+3+2+1+1\text{(* Size of the feature vector. *)}}\\
\boldsymbol{\text{hlsize}=50;\text{(* Size of the hidden layer. *)}}\\
\boldsymbol{\text{gamma}=0.05;\text{(* Discount factor. *)}}\\
\boldsymbol{\text{probeps}=0.0001;\text{(* Probability to reset the value of epsilon. *)}}\\
\boldsymbol{\text{mlimn}=75.;\text{(* The lowest possible value of mlim. *)}}\\
\boldsymbol{\text{runs}=1000;\text{(* Number of test runs. *)}}\\
\boldsymbol{}\\
\boldsymbol{\text{feat}=\text{Table}[0.,\{k,1,\text{fsize}\}];\text{(* Feature vector. *)}}\\
\boldsymbol{\text{pr}=\text{Table}[0.,\{k,1,\text{vsize}\}];\text{(* List of prices in the state. *)}}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Hidden layer outputs. *)}}\\
\boldsymbol{\text{hlayer1}=\text{Table}[0.,\{k,1,\text{hlsize}\}];}\\
\boldsymbol{\text{hlayer2}=\text{Table}[0.,\{k,1,\text{hlsize}\}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Gradients of the Q-functions. *)}}\\
\boldsymbol{\text{gradq1}=\text{Table}[0.,\{k,1,\text{hlsize}+\text{fsize}\}];}\\
\boldsymbol{\text{gradq1new}=\text{Table}[0.,\{k,1,\text{hlsize}+\text{fsize}\}];}\\
\boldsymbol{\text{gradq2}=\text{Table}[0.,\{k,1,\text{hlsize}+\text{fsize}\}];}\\
\boldsymbol{\text{gradq2new}=\text{Table}[0.,\{k,1,\text{hlsize}+\text{fsize}\}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Q-functions. *)}}\\
\boldsymbol{\text{qarr1}=\text{Table}[0.,\{k,1,19\}];}\\
\boldsymbol{\text{qarr2}=\text{Table}[0.,\{k,1,19\}];}\\
\boldsymbol{\text{qarr}=\text{Table}[0.,\{k,1,19\}];}\\
\boldsymbol{\text{qarr1new}=\text{Table}[0.,\{k,1,19\}];}\\
\boldsymbol{\text{qarr2new}=\text{Table}[0.,\{k,1,19\}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Lists to write sav and twth at the end of each run. *)}}\\
\boldsymbol{\text{savarray}=\text{Table}[0.,\{k,1,\text{runs}\}];}\\
\boldsymbol{\text{twtharray}=\text{Table}[0.,\{k,1,\text{runs}\}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Relative movements of the price, etc. *)}}\\
\boldsymbol{\text{nmd1}=\text{Table}[0.,\{k,1,\text{vsize}-1\}];}\\
\boldsymbol{\text{nmd2}=\text{Table}[0.,\{k,1,\text{vsize}-2\}];}\\
\boldsymbol{\text{nmd3}=\text{Table}[0.,\{k,1,\text{vsize}-3\}];}\\
\boldsymbol{\text{nmd4}=\text{Table}[0.,\{k,1,\text{vsize}-4\}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* The activation function. *)}}\\
\boldsymbol{\text{Plot}[1./(1.+\text{Exp}[-x]),\{x,-6.,6.\}]}\\
\boldsymbol{}\\
\boldsymbol{m=1;\text{While}[m\leq\text{runs},}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Initial weights are randomly generated. Then the weights between the input and hidden layers are rescaled. *)}}\\
\boldsymbol{\text{whi1}=\text{Table}[\text{RandomReal}[\{-1.,1.\}],\{k,1,\text{hlsize}\},\{l,1,\text{fsize}\}];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{hlsize},\text{whi1}[[j]]=\text{whi1}[[j]]/\text{Norm}[\text{whi1}[[j]]];j=j+1];}\\
\boldsymbol{\text{whi2}=\text{Table}[\text{RandomReal}[\{-1.,1.\}],\{k,1,\text{hlsize}\},\{l,1,\text{fsize}\}];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{hlsize},\text{whi2}[[j]]=\text{whi2}[[j]]/\text{Norm}[\text{whi2}[[j]]];j=j+1];}\\
\boldsymbol{\text{wout1}=\text{Table}[\text{RandomReal}[\{-1.,1.\}],\{l,1,19\},\{k,1,\text{hlsize}+\text{fsize}\}];}\\
\boldsymbol{\text{wout2}=\text{Table}[\text{RandomReal}[\{-1.,1.\}],\{l,1,19\},\{k,1,\text{hlsize}+\text{fsize}\}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Initial values of the counters for epsilon and alpha. *)}}\\
\boldsymbol{\text{ialpha}=-1;}\\
\boldsymbol{\text{ieps}=0;}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Tables to be used for calculating RSI. *)}}\\
\boldsymbol{\text{rsip}=\text{Table}[0.,\{l,1,15\}];}\\
\boldsymbol{\text{rsiu}=\text{Table}[0.,\{l,1,14\}];}\\
\boldsymbol{\text{rsid}=\text{Table}[0.,\{l,1,14\}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Variables for the average volumes and the max norms of the output weights. *)}}\\
\boldsymbol{\text{mv}=\text{Table}[0.,\{j,1,20\}];}\\
\boldsymbol{\text{nv}=0;}\\
\boldsymbol{\text{av}=0.;}\\
\boldsymbol{\text{max1}=1.;}\\
\boldsymbol{\text{max2}=1.;}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Initial mon, cns, sav, res, mlim. *)}}\\
\boldsymbol{\text{mon}=100.;\text{cns}=0.;\text{sav}=0.;\text{res}=0.;\text{mlim}=\text{mon};\text{mdf}=0;}\\
\boldsymbol{}\\
\boldsymbol{i=1;}\\
\boldsymbol{}\\
\boldsymbol{\text{Label}[\text{episode}];\text{(* Label used to start a new episode. *)}}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Initial price in the episode. *)}}\\
\boldsymbol{\text{ipr}=\text{prices}[[i*5-\text{vsize}+1]];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Calculate the average volumes. *)}}\\
\boldsymbol{\text{cav}=(\text{volumes}[[i*5-4]]+\text{volumes}[[i*5-3]]+\text{volumes}[[i*5-2]]+\text{volumes}[[i*5-1]]+\text{volumes}[[i*5]])/5.;}\\
\boldsymbol{j=1;\text{While}[j\leq 19,\text{mv}[[j]]=\text{mv}[[j+1]];j=j+1];}\\
\boldsymbol{\text{mv}[[j]]=\text{cav};}\\
\boldsymbol{\text{av}=\text{Mean}[\text{mv}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Observe the prices in the current state. *)}}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize},\text{pr}[[j]]=\text{prices}[[i*5-\text{vsize}+j]];j=j+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Relative price movements. *)}}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-1,}\\
\boldsymbol{\text{nmd1}[[j]]=(\text{pr}[[j+1]]-\text{pr}[[j]])/\text{pr}[[j]];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-2,}\\
\boldsymbol{\text{nmd2}[[j]]=(\text{nmd1}[[j+1]]-\text{nmd1}[[j]])/\text{Abs}[\text{nmd1}[[j]]];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-3,}\\
\boldsymbol{\text{nmd3}[[j]]=(\text{nmd2}[[j+1]]-\text{nmd2}[[j]])/\text{Abs}[\text{nmd2}[[j]]];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-4,}\\
\boldsymbol{\text{nmd4}[[j]]=(\text{nmd3}[[j+1]]-\text{nmd3}[[j]])/\text{Abs}[\text{nmd3}[[j]]];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Calculate RSI for 15 points. *)}}\\
\boldsymbol{j=1;\text{While}[j\leq 10,\text{rsip}[[j]]=\text{rsip}[[j+5]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize},\text{rsip}[[10+j]]=\text{pr}[[j]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq 14,}\\
\boldsymbol{\text{If}[\text{rsip}[[j+1]]>\text{rsip}[[j]],\text{rsiu}[[j]]=\text{rsip}[[j+1]]-\text{rsip}[[j]];\text{rsid}[[j]]=0.];}\\
\boldsymbol{\text{If}[\text{rsip}[[j+1]]<\text{rsip}[[j]],\text{rsid}[[j]]=\text{rsip}[[j]]-\text{rsip}[[j+1]];\text{rsiu}[[j]]=0.];}\\
\boldsymbol{\text{If}[\text{rsip}[[j+1]]==\text{rsip}[[j]],\text{rsiu}[[j]]=0.;\text{rsid}[[j]]=0.];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{\text{If}[\text{Mean}[\text{rsid}]==0.,\text{rsi}=100.,\text{rsi}=100.-(100./(1.+(\text{Mean}[\text{rsiu}]/\text{Mean}[\text{rsid}])))];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Construct the feature vector (same for all actions). *)}}\\
\boldsymbol{\text{feat}[[1]]=0.;}\\
\boldsymbol{j=1;k=1;\text{While}[j\leq\text{vsize},\text{feat}[[k+1]]=\text{pr}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{\text{feat}[[k+1]]=\text{ipr};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=(\text{pr}[[\text{vsize}]]-\text{ipr})/\text{ipr};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{mon};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{cns};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{cav};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{av};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=(\text{cav}-\text{av})/\text{av};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=(\text{volumes}[[i*5]]-\text{av})/\text{av};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=(\text{volumes}[[i*5]]-\text{cav})/\text{cav};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{rsi};k=k+1;}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-1,\text{feat}[[k+1]]=\text{nmd1}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-2,\text{feat}[[k+1]]=\text{nmd2}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-3,\text{feat}[[k+1]]=\text{nmd3}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-4,\text{feat}[[k+1]]=\text{nmd4}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{\text{feat}[[k+1]]=\text{mlim};k=k+1;}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Rescale the feature vector, so its norm is 6. *)}}\\
\boldsymbol{\text{feat}=6.*(\text{feat}/\text{Norm}[\text{feat}]);}\\
\boldsymbol{\text{feat}[[1]]=1.;}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Output of the hidden layers. *)}}\\
\boldsymbol{n=1;\text{While}[n\leq\text{hlsize},}\\
\boldsymbol{q=0;j=1;\text{While}[j\leq\text{fsize},q=q+\text{whi1}[[n,j]]*\text{feat}[[j]];j=j+1];}\\
\boldsymbol{\text{hlayer1}[[n]]=1./(1.+\text{Exp}[-q]);}\\
\boldsymbol{q=0;j=1;\text{While}[j\leq\text{fsize},q=q+\text{whi2}[[n,j]]*\text{feat}[[j]];j=j+1];}\\
\boldsymbol{\text{hlayer2}[[n]]=1./(1.+\text{Exp}[-q]);}\\
\boldsymbol{n=n+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Gradients of the Q-functions, taken in the current state (same for all actions), which are just}}\\
\boldsymbol{\text{concatenations of the feature vector with the hidden layer outputs. *)}}\\
\boldsymbol{j=1;\text{While}[j\leq\text{fsize},\text{gradq1}[[j]]=\text{feat}[[j]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{hlsize},\text{gradq1}[[\text{fsize}+j]]=\text{hlayer1}[[j]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{fsize},\text{gradq2}[[j]]=\text{feat}[[j]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{hlsize},\text{gradq2}[[\text{fsize}+j]]=\text{hlayer2}[[j]];j=j+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Calculate wth in the current state. *)}}\\
\boldsymbol{\text{wth}=\text{mon}+(\text{cns}*\text{prices}[[i*5]]);}\\
\boldsymbol{}\\
\boldsymbol{\text{While}[(i+1)*5\leq\text{simlength},}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Calculate Q1 and Q2 for every action in the given state. *)}}\\
\boldsymbol{n=1;\text{While}[n\leq 19,}\\
\boldsymbol{q=0;j=1;\text{While}[j\leq\text{hlsize}+\text{fsize},q=q+\text{wout1}[[n,j]]*\text{gradq1}[[j]];j=j+1];}\\
\boldsymbol{\text{qarr1}[[n]]=q;}\\
\boldsymbol{q=0;j=1;\text{While}[j\leq\text{hlsize}+\text{fsize},q=q+\text{wout2}[[n,j]]*\text{gradq2}[[j]];j=j+1];}\\
\boldsymbol{\text{qarr2}[[n]]=q;}\\
\boldsymbol{n=n+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Calculate the average of Q1 and Q2, which is needed to choose an action. *)}}\\
\boldsymbol{\text{qarr}=(\text{qarr1}+\text{qarr2})/2.;}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Update epsilon. *)}}\\
\boldsymbol{\text{rand}=\text{RandomReal}[];}\\
\boldsymbol{\text{If}[((\text{rand}\leq\text{probeps})\&\&(\text{ieps}\geq\text{Ceiling}[(\text{Exp}[1./0.2]-2.)/5.])),\text{ieps}=\text{Ceiling}[(\text{Exp}[1./0.2]-2.)/5.],\text{ieps}=\text{ieps}+1];}\\
\boldsymbol{\text{eps}=1./\text{Log}[\text{ieps}*5.+2.];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Choose an action. *)}}\\
\boldsymbol{\text{rand}=\text{RandomReal}[];}\\
\boldsymbol{\text{If}[\text{rand}\leq\text{eps},a=\text{RandomInteger}[\{1,19\}],a=\text{Ordering}[\text{qarr},-1][[1]]];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Update alpha. *)}}\\
\boldsymbol{\text{ialpha}=\text{ialpha}+1;}\\
\boldsymbol{\text{alpha}=0.001+(1./2.)*(1.-0.001)*(1.+\text{Cos}[\text{Pi}*\text{ialpha}/1000.]);}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Take the chosen action. *)}}\\
\boldsymbol{\text{fff}=1.;\text{(* This tracks for insufficient mon or cns in order to reflect it in the reward later. *)}}\\
\boldsymbol{\text{If}[a==1,\text{If}[\text{mon}<10.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(10./\text{prices}[[i*5]]));\text{mon}=\text{mon}-10.];}\\
\boldsymbol{\text{If}[a==2,\text{If}[\text{mon}<20.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(20./\text{prices}[[i*5]]));\text{mon}=\text{mon}-20.];}\\
\boldsymbol{\text{If}[a==3,\text{If}[\text{mon}<30.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(30./\text{prices}[[i*5]]));\text{mon}=\text{mon}-30.];}\\
\boldsymbol{\text{If}[a==4,\text{If}[\text{mon}<40.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(40./\text{prices}[[i*5]]));\text{mon}=\text{mon}-40.];}\\
\boldsymbol{\text{If}[a==5,\text{If}[\text{mon}<50.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(50./\text{prices}[[i*5]]));\text{mon}=\text{mon}-50.];}\\
\boldsymbol{\text{If}[a==6,\text{If}[\text{mon}<60.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(60./\text{prices}[[i*5]]));\text{mon}=\text{mon}-60.];}\\
\boldsymbol{\text{If}[a==7,\text{If}[\text{mon}<70.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(70./\text{prices}[[i*5]]));\text{mon}=\text{mon}-70.];}\\
\boldsymbol{\text{If}[a==8,\text{If}[\text{mon}<80.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(80./\text{prices}[[i*5]]));\text{mon}=\text{mon}-80.];}\\
\boldsymbol{\text{If}[a==9,\text{If}[\text{mon}<90.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{cns}=\text{cns}+((99.9/100.)*(90./\text{prices}[[i*5]]));\text{mon}=\text{mon}-90.];}\\
\boldsymbol{\text{If}[a==10,\text{If}[\text{prices}[[i*5]]*\text{cns}<10.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(10.));\text{cns}=\text{cns}-(10./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{If}[a==11,\text{If}[\text{prices}[[i*5]]*\text{cns}<20.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(20.));\text{cns}=\text{cns}-(20./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{If}[a==12,\text{If}[\text{prices}[[i*5]]*\text{cns}<30.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(30.));\text{cns}=\text{cns}-(30./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{If}[a==13,\text{If}[\text{prices}[[i*5]]*\text{cns}<40.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(40.));\text{cns}=\text{cns}-(40./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{If}[a==14,\text{If}[\text{prices}[[i*5]]*\text{cns}<50.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(50.));\text{cns}=\text{cns}-(50./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{If}[a==15,\text{If}[\text{prices}[[i*5]]*\text{cns}<60.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(60.));\text{cns}=\text{cns}-(60./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{If}[a==16,\text{If}[\text{prices}[[i*5]]*\text{cns}<70.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(70.));\text{cns}=\text{cns}-(70./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{If}[a==17,\text{If}[\text{prices}[[i*5]]*\text{cns}<80.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(80.));\text{cns}=\text{cns}-(80./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{If}[a==18,\text{If}[\text{prices}[[i*5]]*\text{cns}<90.,\text{fff}=-1.;\text{Goto}[\text{fail}]];\text{mon}=\text{mon}+((99.9/100.)*(90.));\text{cns}=\text{cns}-(90./\text{prices}[[i*5]])];}\\
\boldsymbol{\text{(* }\text{If}[a==19,\text{HOLD}];\text{ *)}}\\
\boldsymbol{\text{Label}[\text{fail}];}\\
\boldsymbol{\text{If}[((\text{mon}<0.)\|(\text{cns}<0.)),\text{Print}[\text{{``}Negative mon or cns!{''}}]];\text{(* Just to be sure. *)}}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Observe the reward (by calculating wth in the next state). *)}}\\
\boldsymbol{\text{wthnew}=\text{mon}+(\text{cns}*\text{prices}[[(i+1)*5]]);}\\
\boldsymbol{\text{rew}=\text{wthnew}-\text{wth};}\\
\boldsymbol{\text{If}[\text{fff}<0.,}\\
\boldsymbol{\text{rew}=\text{rew}-((\text{rew}/2.){}^{\wedge}2)-0.1,}\\
\boldsymbol{\text{rew}=\text{rew}-((\text{rew}/2.){}^{\wedge}2)];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Observe the prices in the next state. *)}}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize},\text{pr}[[j]]=\text{prices}[[(i+1)*5-\text{vsize}+j]];j=j+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Calculate RSI for 15 points. *)}}\\
\boldsymbol{j=1;\text{While}[j\leq 10,\text{rsip}[[j]]=\text{rsip}[[j+5]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize},\text{rsip}[[10+j]]=\text{pr}[[j]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq 14,}\\
\boldsymbol{\text{If}[\text{rsip}[[j+1]]>\text{rsip}[[j]],\text{rsiu}[[j]]=\text{rsip}[[j+1]]-\text{rsip}[[j]];\text{rsid}[[j]]=0.];}\\
\boldsymbol{\text{If}[\text{rsip}[[j+1]]<\text{rsip}[[j]],\text{rsid}[[j]]=\text{rsip}[[j]]-\text{rsip}[[j+1]];\text{rsiu}[[j]]=0.];}\\
\boldsymbol{\text{If}[\text{rsip}[[j+1]]==\text{rsip}[[j]],\text{rsiu}[[j]]=0.;\text{rsid}[[j]]=0.];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{\text{If}[\text{Mean}[\text{rsid}]==0.,\text{rsi}=100.,\text{rsi}=100.-(100./(1.+(\text{Mean}[\text{rsiu}]/\text{Mean}[\text{rsid}])))];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Check for the three possible terminal states. *)}}\\
\boldsymbol{\text{If}[\text{mon}>\text{mlim},\text{(* Terminal state 1. *)}}\\
\boldsymbol{\text{rew}=\text{rew}+((\text{mon}-\text{mlim})*0.34);}\\
\boldsymbol{\text{rand}=\text{RandomReal}[];}\\
\boldsymbol{\text{If}[\text{rand}\leq 0.5,}\\
\boldsymbol{\text{wout1}[[a]]=\text{wout1}[[a]]+\text{alpha}*(\text{rew}-\text{qarr1}[[a]])*\text{gradq1};}\\
\boldsymbol{\text{max1}=\text{Max}[\text{max1},\text{Norm}[\text{wout1}[[a]]]];}\\
\boldsymbol{\text{If}[\text{Norm}[\text{wout1}[[a]]]>1.,\text{wout1}[[a]]=\text{wout1}[[a]]/\text{max1}],}\\
\boldsymbol{\text{wout2}[[a]]=\text{wout2}[[a]]+\text{alpha}*(\text{rew}-\text{qarr2}[[a]])*\text{gradq2};}\\
\boldsymbol{\text{max2}=\text{Max}[\text{max2},\text{Norm}[\text{wout2}[[a]]]];}\\
\boldsymbol{\text{If}[\text{Norm}[\text{wout2}[[a]]]>1.,\text{wout2}[[a]]=\text{wout2}[[a]]/\text{max2}]];}\\
\boldsymbol{\text{mdf}=\text{mon}-\text{mlim};}\\
\boldsymbol{\text{sav}=\text{sav}+(\text{mdf}*0.34);}\\
\boldsymbol{\text{res}=\text{res}+(\text{mdf}*0.33);}\\
\boldsymbol{\text{mon}=\text{mlim}+(\text{mdf}*0.33);}\\
\boldsymbol{\text{mlim}=\text{mon}+\text{mdf};}\\
\boldsymbol{i=i+2;}\\
\boldsymbol{\text{If}[i*5>\text{simlength},\text{Goto}[\text{break}]];}\\
\boldsymbol{\text{Goto}[\text{episode}],}\\
\boldsymbol{\text{If}[((\text{wthnew}<\text{mlimn})\&\&(\text{qarr}[[a]]>0.)\&\&(\text{rsi}>70.)),\text{(* Terminal state 2. *)}}\\
\boldsymbol{\text{rand}=\text{RandomReal}[];}\\
\boldsymbol{\text{If}[\text{rand}\leq 0.5,}\\
\boldsymbol{\text{wout1}[[a]]=\text{wout1}[[a]]+\text{alpha}*(\text{rew}-\text{qarr1}[[a]])*\text{gradq1};}\\
\boldsymbol{\text{max1}=\text{Max}[\text{max1},\text{Norm}[\text{wout1}[[a]]]];}\\
\boldsymbol{\text{If}[\text{Norm}[\text{wout1}[[a]]]>1.,\text{wout1}[[a]]=\text{wout1}[[a]]/\text{max1}],}\\
\boldsymbol{\text{wout2}[[a]]=\text{wout2}[[a]]+\text{alpha}*(\text{rew}-\text{qarr2}[[a]])*\text{gradq2};}\\
\boldsymbol{\text{max2}=\text{Max}[\text{max2},\text{Norm}[\text{wout2}[[a]]]];}\\
\boldsymbol{\text{If}[\text{Norm}[\text{wout2}[[a]]]>1.,\text{wout2}[[a]]=\text{wout2}[[a]]/\text{max2}]];}\\
\boldsymbol{\text{mon}=\text{mon}+(\text{res}/2.);\text{res}=\text{res}-(\text{res}/2.);}\\
\boldsymbol{\text{If}[\text{mon}\geq\text{mlimn},\text{mlim}=\text{mon},\text{mlim}=\text{mlimn}];}\\
\boldsymbol{i=i+2;}\\
\boldsymbol{\text{If}[i*5>\text{simlength},\text{Goto}[\text{break}]];}\\
\boldsymbol{\text{Goto}[\text{episode}]];}\\
\boldsymbol{\text{If}[((\text{wthnew}\geq\text{mlimn})\&\&(\text{qarr}[[a]]<0.)\&\&(\text{rsi}<30.)),\text{(* Terminal state 3. *)}}\\
\boldsymbol{\text{rand}=\text{RandomReal}[];}\\
\boldsymbol{\text{If}[\text{rand}\leq 0.5,}\\
\boldsymbol{\text{wout1}[[a]]=\text{wout1}[[a]]+\text{alpha}*(\text{rew}-\text{qarr1}[[a]])*\text{gradq1};}\\
\boldsymbol{\text{max1}=\text{Max}[\text{max1},\text{Norm}[\text{wout1}[[a]]]];}\\
\boldsymbol{\text{If}[\text{Norm}[\text{wout1}[[a]]]>1.,\text{wout1}[[a]]=\text{wout1}[[a]]/\text{max1}],}\\
\boldsymbol{\text{wout2}[[a]]=\text{wout2}[[a]]+\text{alpha}*(\text{rew}-\text{qarr2}[[a]])*\text{gradq2};}\\
\boldsymbol{\text{max2}=\text{Max}[\text{max2},\text{Norm}[\text{wout2}[[a]]]];}\\
\boldsymbol{\text{If}[\text{Norm}[\text{wout2}[[a]]]>1.,\text{wout2}[[a]]=\text{wout2}[[a]]/\text{max2}]];}\\
\boldsymbol{\text{mlim}=\text{wthnew};}\\
\boldsymbol{i=i+2;}\\
\boldsymbol{\text{If}[i*5>\text{simlength},\text{Goto}[\text{break}]];}\\
\boldsymbol{\text{Goto}[\text{episode}]]];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Calculate the average volumes. *)}}\\
\boldsymbol{\text{cav}=(\text{volumes}[[(i+1)*5-4]]+\text{volumes}[[(i+1)*5-3]]+\text{volumes}[[(i+1)*5-2]]+\text{volumes}[[(i+1)*5-1]]+\text{volumes}[[(i+1)*5]])/5.;}\\
\boldsymbol{j=1;\text{While}[j\leq 19,\text{mv}[[j]]=\text{mv}[[j+1]];j=j+1];}\\
\boldsymbol{\text{mv}[[j]]=\text{cav};}\\
\boldsymbol{\text{av}=\text{Mean}[\text{mv}];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Relative price movements. *)}}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-1,}\\
\boldsymbol{\text{nmd1}[[j]]=(\text{pr}[[j+1]]-\text{pr}[[j]])/\text{pr}[[j]];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-2,}\\
\boldsymbol{\text{nmd2}[[j]]=(\text{nmd1}[[j+1]]-\text{nmd1}[[j]])/\text{Abs}[\text{nmd1}[[j]]];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-3,}\\
\boldsymbol{\text{nmd3}[[j]]=(\text{nmd2}[[j+1]]-\text{nmd2}[[j]])/\text{Abs}[\text{nmd2}[[j]]];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-4,}\\
\boldsymbol{\text{nmd4}[[j]]=(\text{nmd3}[[j+1]]-\text{nmd3}[[j]])/\text{Abs}[\text{nmd3}[[j]]];}\\
\boldsymbol{j=j+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Construct the feature vector (same for all actions). *)}}\\
\boldsymbol{\text{feat}[[1]]=0.;}\\
\boldsymbol{j=1;k=1;\text{While}[j\leq\text{vsize},\text{feat}[[k+1]]=\text{pr}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{\text{feat}[[k+1]]=\text{ipr};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=(\text{pr}[[\text{vsize}]]-\text{ipr})/\text{ipr};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{mon};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{cns};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{cav};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{av};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=(\text{cav}-\text{av})/\text{av};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=(\text{volumes}[[(i+1)*5]]-\text{av})/\text{av};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=(\text{volumes}[[(i+1)*5]]-\text{cav})/\text{cav};k=k+1;}\\
\boldsymbol{\text{feat}[[k+1]]=\text{rsi};k=k+1;}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-1,\text{feat}[[k+1]]=\text{nmd1}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-2,\text{feat}[[k+1]]=\text{nmd2}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-3,\text{feat}[[k+1]]=\text{nmd3}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{vsize}-4,\text{feat}[[k+1]]=\text{nmd4}[[j]];k=k+1;j=j+1];}\\
\boldsymbol{\text{feat}[[k+1]]=\text{mlim};k=k+1;}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Rescale the feature vector, so its norm is 6. *)}}\\
\boldsymbol{\text{feat}=6.*(\text{feat}/\text{Norm}[\text{feat}]);}\\
\boldsymbol{\text{feat}[[1]]=1.;}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Output of the hidden layers. *)}}\\
\boldsymbol{n=1;\text{While}[n\leq\text{hlsize},}\\
\boldsymbol{q=0;j=1;\text{While}[j\leq\text{fsize},q=q+\text{whi1}[[n,j]]*\text{feat}[[j]];j=j+1];}\\
\boldsymbol{\text{hlayer1}[[n]]=1./(1.+\text{Exp}[-q]);}\\
\boldsymbol{q=0;j=1;\text{While}[j\leq\text{fsize},q=q+\text{whi2}[[n,j]]*\text{feat}[[j]];j=j+1];}\\
\boldsymbol{\text{hlayer2}[[n]]=1./(1.+\text{Exp}[-q]);}\\
\boldsymbol{n=n+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Gradients of the Q-functions, taken in the next state (same for all actions), which are just}}\\
\boldsymbol{\text{concatenations of the feature vector with the hidden layer outputs. *)}}\\
\boldsymbol{j=1;\text{While}[j\leq\text{fsize},\text{gradq1new}[[j]]=\text{feat}[[j]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{hlsize},\text{gradq1new}[[\text{fsize}+j]]=\text{hlayer1}[[j]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{fsize},\text{gradq2new}[[j]]=\text{feat}[[j]];j=j+1];}\\
\boldsymbol{j=1;\text{While}[j\leq\text{hlsize},\text{gradq2new}[[\text{fsize}+j]]=\text{hlayer2}[[j]];j=j+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Calculate Q1 and Q2 for every action in the next state. *)}}\\
\boldsymbol{n=1;\text{While}[n\leq 19,}\\
\boldsymbol{q=0;j=1;\text{While}[j\leq\text{hlsize}+\text{fsize},q=q+\text{wout1}[[n,j]]*\text{gradq1new}[[j]];j=j+1];}\\
\boldsymbol{\text{qarr1new}[[n]]=q;}\\
\boldsymbol{q=0;j=1;\text{While}[j\leq\text{hlsize}+\text{fsize},q=q+\text{wout2}[[n,j]]*\text{gradq2new}[[j]];j=j+1];}\\
\boldsymbol{\text{qarr2new}[[n]]=q;}\\
\boldsymbol{n=n+1];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Update the output weights. *)}}\\
\boldsymbol{\text{rand}=\text{RandomReal}[];}\\
\boldsymbol{\text{If}[\text{rand}\leq 0.5,}\\
\boldsymbol{\text{amax}=\text{Ordering}[\text{qarr1new},-1][[1]];}\\
\boldsymbol{\text{wout1}[[a]]=\text{wout1}[[a]]+\text{alpha}*(\text{rew}+\text{gamma}*\text{qarr2new}[[\text{amax}]]-\text{qarr1}[[a]])*\text{gradq1};}\\
\boldsymbol{\text{max1}=\text{Max}[\text{max1},\text{Norm}[\text{wout1}[[a]]]];}\\
\boldsymbol{\text{If}[\text{Norm}[\text{wout1}[[a]]]>1.,\text{wout1}[[a]]=\text{wout1}[[a]]/\text{max1}],}\\
\boldsymbol{\text{amax}=\text{Ordering}[\text{qarr2new},-1][[1]];}\\
\boldsymbol{\text{wout2}[[a]]=\text{wout2}[[a]]+\text{alpha}*(\text{rew}+\text{gamma}*\text{qarr1new}[[\text{amax}]]-\text{qarr2}[[a]])*\text{gradq2};}\\
\boldsymbol{\text{max2}=\text{Max}[\text{max2},\text{Norm}[\text{wout2}[[a]]]];}\\
\boldsymbol{\text{If}[\text{Norm}[\text{wout2}[[a]]]>1.,\text{wout2}[[a]]=\text{wout2}[[a]]/\text{max2}]];}\\
\boldsymbol{}\\
\boldsymbol{\text{(* Make the next value of wth and the next states current. *)}}\\
\boldsymbol{\text{wth}=\text{wthnew};}\\
\boldsymbol{\text{gradq1}=\text{gradq1new};}\\
\boldsymbol{\text{gradq2}=\text{gradq2new};}\\
\boldsymbol{}\\
\boldsymbol{i=i+1;}\\
\boldsymbol{}\\
\boldsymbol{];}\\
\boldsymbol{}\\
\boldsymbol{\text{Label}[\text{break}];\text{(* Label used to exit the loop, if there are no more points in the test data. *)}}\\
\boldsymbol{\text{savarray}[[m]]=\text{sav};\text{(* Write sav at the end of the current run. *)}}\\
\boldsymbol{\text{twtharray}[[m]]=\text{sav}+\text{wthnew}+\text{res};\text{(* Write twth at the end of the current run. *)}}\\
\boldsymbol{}\\
\boldsymbol{m=m+1}\\
\boldsymbol{}\\
\boldsymbol{];}\\
\boldsymbol{}\\
\boldsymbol{\text{Export}[\text{{``}sav.txt{''}},\text{savarray}]}\\
\boldsymbol{\text{Export}[\text{{``}total.txt{''}},\text{twtharray}]}

References

[1]
 P. Treleaven, M. Galas, V. Lalchand, Algorithmic trading review, Communications of the ACM 56(11), 76-85 (2013).

[2]
 D. Silver, A. Huang, C. J. Maddison et al., Mastering the game of Go with deep neural networks and tree search, Nature 529, 484-489 (2016).

[3]
 O. Vinyals, I. Babuschkin, W. M. Czarnecki et al., Grandmaster level in StarCraft II using multi-agent reinforcement learning, Nature 575, 350-354 (2019).

[4]
 A. Millea, Deep Reinforcement Learning for Trading—A Critical Survey, Data 6(11), 119 (2021).

[5]
 G.-B. Huang, Q.-Y. Zhu, C.-K. Siew, Extreme learning machine: a new learning scheme of feedforward neural networks, 2004 IEEE International Joint Conference on Neural Networks, Budapest, Hungary, Volume 2, 985-990 (2004).

[6]
 S. Ding, X. Xu, R. Nie, Extreme learning machine and its applications, Neural Computing and Applications 25, 549-556 (2014).

[7]
 G. Li, P. Niu, X. Duan, X. Zhang, Fast learning network: a novel artificial neural network with a fast learning speed, Neural Computing and Applications 24, 1683-1695 (2014).

[8]
 R. S. Sutton, A. G. Barto, Reinforcement Learning: An Introduction, 2nd Ed., The MIT Press, Cambridge, MA (2018).

[9]
 H. van Hasselt, Y. Doron, F. Strub, M. Hessel, N. Sonnerat, J. Modayil, Deep Reinforcement Learning and the Deadly Triad, arXiv:1812.02648 (2018).

[10]
 H. van Hasselt, A. Guez, D. Silver, Deep reinforcement learning with double Q-learning, AAAI’16: Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, 2094-2100 (2016).

[11]
 J. W. Wilder, New concepts in technical trading systems, Trend Research, Greensboro, N.C. (1978).

[12]
 C. Brown, Technical Analysis for the Trading Professional, Second Edition, McGraw-Hill (2011).

[13]
 C. Hartland, N. Baskiotis, S. Gelly, M. Sebag, O. Teytaud, Change Point Detection and Meta-Bandits for Online Learning in Dynamic Environments, CAp 2007, pp. 237-250 (2007).

[14]
 M. Tokic, Adaptive ε\varepsilon-Greedy Exploration in Reinforcement Learning Based on Value Differences, KI 2010: Advances in Artificial Intelligence (2010).

[15]
 L. N. Smith, Cyclical Learning Rates for Training Neural Networks, 2017 IEEE Winter Conference on Applications of Computer Vision (WACV), 464-472 (2017).

[16]
 R. Gulde, M. Tuscher, A.Csiszar, O. Riedel, A. Verl, Deep Reinforcement Learning using Cyclical Learning Rates, 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 32-35 (2020).

[17]
 A. Gotmare, N. S. Keskar, C. Xiong, R. Socher, A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation, 7th International Conference on Learning Representations, ICLR 2019 (2019).

[18]
 I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, The MIT Press (2016).

[19]
 T. Salimans, D. P. Kingma, Weight normalisation: a simple reparameterization to accelerate training of deep neural networks, NIPS’16: Proceedings of the 30th International Conference on Neural Information Processing Systems (2016).

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

