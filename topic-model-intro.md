## 主题模型简介

### gamma函数

Gamma函数为：$\Gamma(x)=\int_{0}^{\infty}t^{x-1}e^{-t}dt$
分部积分有性质：$\Gamma(x+1)=x\Gamma(x)$
容易得到：$\Gamma(n)=(n-1)!$

Beta函数：$B(m,n)=\int_0^1x^{m-1}(1-x)^{n-1}dx$
Beta函数与Gamma函数有如下关系：$B(m,n)=\frac{\Gamma(m)\Gamma(n)}{\Gamma(m+n)}$

digamma函数：$\Psi(x)=\frac{d\log\Gamma(x)}{dx}$
digamma函数有如下性质：$\Psi(x+1)=\Psi(x)+\frac{1}{x}$

从二项分布到Gamma分布

### 共轭分布：

贝叶斯估计过程：**先验分布+数据的知识=后验分布**

**Beta-Binomial共轭：**
$Beta(p|\alpha,\beta)+BinomCount(m_1,m_2)=Beta(p|\alpha+m_1,\beta+m_2)$

**Dirichlet-multinomial共轭：**
$Dir(p|\vec\alpha)+MultCount(\vec{m})=Dir(p|\vec\alpha+\vec{m})$

**Dirichlet分布：**$Dir(p|\vec\alpha)=\frac{\Gamma(\sum_{k=1}^{K}\alpha_k)}{\prod_{k=1}^K\Gamma(\alpha_k)}\prod_{k=1}^Kp_k^{\alpha_k-1}$
**Multinomial分布：**$Mult(\vec n|\vec p, \vec N)=\tbinom{N}{\vec n}\prod_{k=1}^Kp_k^{n_k}$

### Beta/Dirichlet分布的一个性质

如果$p \sim Beta(t|\alpha,\beta)$，则
$$
\begin{align*}
E(p)&=\int_0^1t*Beta(t|\alpha,\beta)dt \\
&=\int_0^1t*\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}t^{\alpha-1}(1-t)^{\beta-1}dt \\
&=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\int_0^1t^{\alpha}(1-t)^{\beta-1}dt
\end{align*}
$$
上式右边的积分对应到概率分布$Beta(t|\alpha+1, \beta)$，对于这个分布我们有
$$\int_0^1\frac{\Gamma(\alpha+\beta+1)}{\Gamma(\alpha+1)\Gamma(\beta)}t^{\alpha}(1-t)^{\beta-1}dt=1$$
把上式代入到$E(p)$的计算公式，得到
$$
\begin{aligned}
E(p)&=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\cdot \frac{\Gamma(\alpha+1)\Gamma(\beta)}{\Gamma(\alpha+\beta+1)} \\
&=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha+\beta+1)} \cdot \frac{\Gamma(\alpha+1)}{\Gamma(\alpha)} \\
&=\frac{\alpha}{\alpha+\beta}
\end{aligned}
$$
这说明对于$Beta$分布的随机变量，其均值可以用$\frac{\alpha}{\alpha+\beta}$来估计。$Dirichlet$分布也有类似结论，如果$\vec p\sim Dir(\vec t|\vec \alpha)$，同样可以证明
$$
E(\vec p)=\left(\frac{\alpha_1}{\sum_{i=1}^{K}\alpha_i},\frac{\alpha_2}{\sum_{i=1}^{K}\alpha_i},...,\frac{\alpha_k}{\sum_{i=1}^{K}\alpha_i} \right)
$$
以上两个结论很重要，因为我们在后面的LDA数学推导中需要使用这个结论。

### 马氏链及平稳分布

马氏链定义如下：
$$P(X_{t+1}=x|X_t,X_{t-1},...)=P(X_{t+1}=x|X_t)$$
即：状态转移的概率只依赖于前一个状态。

**马氏链的收敛定理：**如果一个非周期马氏链具有转移概率矩阵$P$，且它的任何两个状态是连通的，那么$\lim_{n \to \infty}P_{ij}^n$存在且与$i$无关，记$\lim_{n \to \infty}P_{ij}^n=\pi(j)$，我们有
1.$$\lim_{n \to \infty}P^n=\left[ 
\begin{array}{ccccc} 
\pi(1) & \pi(2) & ... & \pi(j) &... \\
\pi(1) & \pi(2) & ... & \pi(j) &... \\
... & ... & ... & ... &... \\
\pi(1) & \pi(2) & ... & \pi(j) &... \\
... & ... & ... & ... &...
\end{array} 
\right]$$
2.$\pi(j)=\sum_{i=0}^\infty\pi(i)P(ij)$
3.$\pi$是方程$\pi P=\pi$的唯一非负解，其中：$\pi=[\pi(1),\pi(2),...,\pi(j),...], \sum_{i=0}^{\infty}\pi_i=1$，$\pi$称为马氏链的平稳分布。
这个马氏链的平稳分布非常重要，**所有的MCMC(Markov Chain Monto Carlo)方法都是以这个定理作为基础的.**

**定理(细致平稳条件)：**如果非周期马氏链的转移矩阵$P$和分布$\pi(x)$满足
$$\pi(i)P_{ij}=\pi(j)P_{ji} \quad for \ all \quad i,j$$
则$\pi(x)$是马氏链的平稳分布，上式被称为细致平稳条件(detailed balance condition)。

### Markov Chain Monte Carlo
假设我们已经有一个转移矩阵为$Q$马氏链($q(i,j)$表示从状态$i$转移到状态$j$的概率)，显然，通常情况下：
$$p(i)q(i,j) \neq p(j)p(j,i)$$
也就是细致平稳条件不成立，所以$p(x)$不太可能是这个马氏链的平稳分布。我们可否对马氏链做一个改造，使得细致平稳条件成立呢？譬如，我们引入一个$\alpha(i,j)$，我们希望
$$
\begin{equation}
p(i)p(i,j)\alpha(i,j)=p(j)p(j,i)\alpha(j,i)
\end{equation}
$$
取什么样的$\alpha(i,j)$使上式成立呢？最简单的，按照对称性，我们可以取：
$$\alpha(i,j)=p(j)q(j,i) \quad \alpha(j,i)=p(i)p(i,j)$$
于是(84)式就成立了。所以有：
$$
p(i)  \underbrace{q(i,j)\alpha(i,j)}_{Q{'}(i,j)}=p(j)\underbrace{q(j,i)\alpha(j,i)}_{Q{'}(j,i)}
$$
于是我们把原来具有转移矩阵$Q$的一个很普通的马氏链，改造为具有转移矩阵$Q{'}$的马氏链，而$Q{'}$恰好满足细致平稳条件，由此马氏链$Q{'}$的平稳分布就是$p(x)$!
在改造$Q$的过程中引入的$\alpha(i,j)$称为接受率，物理意义可以理解为在原来的马氏链上，从状态$i$以$q(i,j)$概率跳转到状态$j$的时候，我们以$\alpha(i,j)$的概率接受这个转移，于是得到了新的马氏链$Q{'}$的转移概率为$q(i,j)\alpha(i,j)$。
![Alt text](./1524694853429.png)
<center>马氏链转移和接受概率

对于原始的马氏转移矩阵$Q$，其对应的采样算法如下：

**Algorithm 1.MCMC采样算法**
1. 初始化马氏链状态$X_0=x_0$
2. 对于$t=0,1,2,...$，进行以下过程进行采样：
    * 第$t$个时刻马氏链状态为$X_t=x_t$，采样$y \sim q(x|x_t)$
    * 从均匀分布采样$u \sim Uniform(0,1)$
    * 如果$u<\alpha(x_t,y)=p(y)q(x_t,y)$则接受转移$x_t \to y$，即$X_{t+1}=y$
    * 否则不接受转移，即$X_{t+1}=x_t$
该算法有个问题，马氏链$Q$在转移过程中的接受率$\alpha(i,j)$可能偏小，这样的采样过程中马氏链容易原地踏步，拒绝大量的跳转，使得马氏链遍历所有的状态空间需要花费太长的时间，收敛到平稳分布$p(x)$太慢。
我们可以把平稳条件中的$\alpha(i,j)$和$\alpha(j,i)$同比例放大，使得两数中最大的放大到1，这样就提高了采样中跳转的接受率。所以，我们取：
$$\alpha(i,j)=\min\left(\frac{p(j)q(j,i)}{p(i),q(i,j)},1\right)$$
于是我们得到了经典的Metropolis-Hastings算法
对于分布$p(x)$，我们构造转移矩阵$Q{'}$使其满足细致平稳条件
$$p(x)Q{'}(x \to y)=p(y)Q{'}(y\to x)$$
此处$x$可扩展到多维的情况，对于高维空间的$p(\textbf x)$，如果满足细致平稳条件
$$p(\textbf x)Q{'}(\textbf x \to \textbf y)=p(\textbf y)Q{'}(\textbf y \to \textbf x)$$

**Algorithm 2. Metropolis-Hastings算法**
1. 初始化马氏链状态$X_0=x_0$
2. 对$t=0,1,2,...$，循环以下过程进行采样：
    * 第$t$个时刻的马氏链状态为$X_t=x_t$，采样$y \sim q(x|x_t)$
    * 从均匀分布采样$u \sim Uniform(0,1)$
    * 如果$u<\alpha(x_t,y)=min\left(\frac{p(y)q(x_t,y)}{p(x_t)q(y|x_t)},1\right)$则接受转移$x \to y$，即$X_{t+1}=y$
    * 否则不接受转移，即$X_{t+1}=x_t$

### Gibbs Sampling

对于高维的情况，由于接受率$\alpha$的存在(通常\alpha<1)，以上Metropolis-Hastings算法的效率不够高。能否找到一个转移矩阵$Q$使得接受率$\alpha=1$呢？我们先看看二维的情形，假设有一个概率分布$p(x,y)$，考察$x$坐标相同的两上点$A(x_1,y_1)$,$B(x_1,y_2)$，我们发现
$$
p(x_1,y_1)p(y_2|x_1)=p(x_1)p(y_1|x_1)p(y_2|x_1) \\
p(x_1,y_2)p(y_1|x_1)=p(x_1)p(y_2|x_1)p(y_1|x_1)
$$
所以得到
$$
p(x_1,y_1)p(y_2|x_1)=p(x_1,y_2)p(y_1|x_1)
$$
即
$$p(A)p(y_2|x_1)=p(B)p(y_1|x_1)$$
基于上式，我们发现，在$x=x_1$这条平行于$y$轴的直线上，如果使用条件分布$p(y|x_1)$做为任何两个点之间的转移概率，那么任何两个点之间的转移满足细致平稳条件。同样的，我们在$y=y_1$这条直线上任意取两个点$A(x_1,y_1)$,C(x_2,y_1)，也有如下等式：
$$p(A)p(x_2|y_1)=p(C)p(x_1|y_1)$$
![Alt text](./1524697730505.png)

<center>平面上马氏链转移矩阵的构造</center>
于是我们可以如下构造平面上任意两点之间的转移概率矩阵$Q$
$$
\begin{align}
Q(A\to B)&=p(y_B|x_1) \qquad 如果\quad x_A=x_B=x_1 \\
Q(A\to C)&=p(y_C|x_1) \qquad 如果\quad y_A=y_C=y_1 \\
Q(A\to D)&=0 \qquad \qquad \quad 其他
\end{align}
$$
有了如上的转移矩阵$Q$，我们很容易验证对平面上任意两点$X,Y$，满足细致平稳条件：
$$p(X)Q(X\to Y)=p(Y)Q(Y \to X)$$
于是这个二维空间上的马氏链将收敛到平稳分布$p(x,y)$。而这个算法就称为Gibbs Sampling算法，由物理学家Gibbs最先给出的。
**Algorithm 3. 二维Gibbs Sampling算法**

1. 随机初始化$X_0=x_0,Y_0=y_0$
2. 对于$t=0,1,2,...$，循环采样
    * $y_{t+1} \sim p(y|x_t)$
    * $x_{t+1} \sim p(x|y_{t+1}$

以上的采样过程中，如图所示，马氏链的转移只是转换的沿着坐标$x$轴和$y$轴做转移，于是得到样本$(x_0,y_0)$,$(x_0,y_1)$,$(x_1,y_1)$,$(x_1,y_2)$,$(x_2,y_2)$,$...$，马氏链收敛后，最终得到的样本就是$p(x,y)$的样本，而收敛之前的阶段称为burn-in period。另Gibbs Sampling算法一般是在坐标轴上轮换采样的，但是其实不强制要求的。最一般的情形是，在$t$时刻，可以在$x$轴和$y$轴之间随机的选取一个坐标轴，然后按条件概率做转移，马氏链也一样收敛的。轮换坐标只是一种方便的形式。
将二维情况推广到$n$维情况，得到$n$维Gibbs Sampling算法

**Algorithm 4. n维Gibbs Sampling算法**

1. 随机初始化$\{x_i:i=1,2,...,n\}$
2. 对于$t=0,1,2,...$循环采样
    1. $x_1^{t+1} \sim p(x_1|x_2^{(t)},x_3^{(t)},...,x_n^{(t)})$
    2. $x_2^{t+1} \sim p(x_2|x_1^{(t+1)},x_3^{(t)},...,x_n^{(t)})$
    3. ...
    1. $x_j^{t+1} \sim p(x_j|x_1^{(t+1)},...,x_{j-1}^{(t+1)},x_{j+1}^{(t)}...,x_n^{(t)})$
    1. ...
    1. $x_n^{t+1} \sim p(x_j|x_1^{(t+1)},x_2^{(t+1)},...,x_{n-1}^{(t)})$

同理，在n维情况下坐标轴轮换也可以引入随机性，这时假转移矩阵$Q$中任何两个点的转移概率中就会包含坐标由选择的概率。而在通常的Gibbs Sampling算法中，坐标轴轮换是一个确定性过程，也就是在给定时刻$t$，在一根固定的坐标轴上转移的概率是1.

### 文本建模

每篇文本就是有序的词的序列$d=(w_1,w_2,...,w_n)$，统计文本建模的目的就是要问这些观察到的语料库中的词序列是如何生成的？

#### Unigram Model

对于一篇文档$d=\vec w=(w_1,w_2,...,w_n)$，该文档的生成概率为：
$$p(\vec w)=p(w_1,w_2,...,w_n)=p(w_1)p(w_2)...p(w_n)$$
而文档和文档之间是独立的，所以如果语料中有多篇文档$W=(\vec w_1,\vec w_2,...,\vec w_m)$，则该语料的概率为：
$$p(W)=p(\vec w_1)p(\vec w_2)...p(\vec w_m)$$
在Unigram Model中，我们假设了文档之间是独立可交换的，而文档中的词也是独立可交换的，所以一篇文章类似于一个袋子，里面装了一些词，而词的顺序信息无关紧要了，这样的模型也称为词袋模型(Bag-of-words)

假设语料中总的词频是$N$，在所有的$N$个词中，如果我们关注了某个词$V_i$的发生次数$n_i$，那么$\vec n=(n_1,n_2,...,n_V)$正好是一个多项式分布
$$p(\vec n)=Mult(\vec n|\vec p,N)=\tbinom{N}{\vec n}\prod_{k=1}^Vp_k^{n_k}$$
此时，语料的概率是
$$p(W)=p(\vec w_1)p(\vec w_2)...p(\vec w_m)=\prod_{k=1}^Vp_k^{n_k}$$
这个模型中我们需要估计参数$\vec p$，按照统计学家中频率派的观点，使用最大似然估计最大化$P(W)$，于是参数$p_i$的估计值就是
$$p_i=\frac{n_i}{N}$$

在贝叶斯学派看来，一切都是随机变量，词分布概率$\vec p$不是唯一固定的，它也是随机变量。以贝叶斯学派来看需要给分布$p(\vec n)$增加一个先验分布，由于$p(\vec n)$是一个多项分布，所以先验分布选择与之对应的共轭分布，即Dirichlet分布
$$Dir(\vec p|\vec \alpha)=\frac{1}{\Delta(\vec \alpha)}\prod_{k=1}^Vp_k^{\alpha_k-1} \quad \vec\alpha=(\alpha_1,\alpha_2,...,\alpha_V) $$
此处，$\Delta(\vec \alpha)$就是归一化因子$Dir(\vec \alpha)$，即
$$\Delta(\vec \alpha)=\int\prod_{k=1}^Vp_k^{a_k-1}d\vec p$$

![Alt text](./1524744225581.png)
<center>贝叶斯Unigram Model的概率图模型

由共轭分布有：**Dirichlet分布$+$多项式分布的数据$\to$后验分布为Dirichlet分布**
$$Dir(\vec p|\vec \alpha)+MultCount(\vec n)=Dir(\vec p|\vec \alpha + \vec n)$$
于是，在给定了参数$\vec p$的先验分布$Dir(\vec p|\vec \alpha)$的时候，各个词出现频次的数据$\vec n \sim Mult(\vec n|\vec p, N)$为多项分布，所以无须计算，我们就可以推出后验分布是
$$p(\vec p|W,\vec \alpha)=Dir(\vec p|\vec n + \vec \alpha)=\frac{1}{\Delta(\vec n+\vec \alpha)}\prod_{k=1}^Vp_k^{n_k+\alpha_k-1}d\vec p$$
在贝叶斯的框架下，参数$\vec p$如何估计呢？由于我们已经有了参数的后验分布，所以合理的方式是使用后验分布的极大值点，或者是参数在后验分布下的平均值。在此，我们取平均值作为参数的估计值，使用前文的结论，由于$\vec p$的后验分布为$Dir(\vec p|\vec n+\vec\alpha)$，于是有：
$$E(\vec p)=\left(\frac{n_1+\alpha_1}{\sum_{i=1}^V(n_i+\alpha_i)},\frac{n_2+\alpha_2}{\sum_{i=1}^V(n_i+\alpha_i)},...,\frac{n_V+\alpha_V}{\sum_{i=1}^V(n_i+\alpha_i)}\right)$$
也就是说对每一个$p_i$，我们用下式做参数估计：
$$\hat p_i=\frac{n_i+\alpha_i}{\sum_{i=1}^{V}(n_i+\alpha_i)}$$
考虑到$\alpha_i$在Dirichlet分布中的物理意义是事件的先验的伪计数，这个估计式子的意义很直观：每个参数的估计值是其对应事件的先验的伪计数和数据中的计数的和在整体计数中的比例。
进一步，我们可以计算出文本语料的产生概率为：
$$
\begin{aligned}
p(W|\vec \alpha)&=\int p(W|\vec p)p(\vec p|\vec \alpha)d\vec p \\
&=\int\prod_{k=1}^Vp_k^{n_k}Dir(\vec p|\vec \alpha)d\vec p \\
&=\int\prod_{k=1}^Vp_k^{n_k}\frac{1}{\Delta \vec\alpha}\prod_{k=1}^Vp_k^{\alpha_k-1}d\vec p \\
&=\frac{1}{\Delta(\vec \alpha)}\int\prod_{k=1}^Vp_k^{n_k+\alpha_k-1}d\vec p\\
&=\frac{\Delta(\vec n+\vec\alpha)}{\Delta(\vec\alpha)}
\end{aligned}
$$

#### Topic Model和PLSA

思考如何写篇文章，一般先确定要写几个主题，譬如构思一篇自然语言处理的文章，可能40%会谈论语言学、30%谈论概率统计、20%谈论计算机、还有10%谈论其他主题；没有主题下又会有相应语的概率。这种直观的想法由Hoffmn于1999年给出的PLSA(Probabilistic Latent Semantic Analysis)模型中首先进行了明确的数学化。Hoffmn认为一篇文档(Document)可以由多个主题(Topic)混合而成，而每个Topic都是词汇上的概率分布，文章中的每个词都是由一个固定的Topic生成的。如下图所示：
![Alt text](./1524749616930.png)
<center>Topic就是Vocab上的概率分布

**PLSA Topic Model**
1. 上帝有两类骰子，一类是doc-topic骰子，每个骰子有K个面，每个面是一个topic的编号；一类是topic-word骰子，每个骰子有V个面，每个面对应一个词；
2. 上帝一共有K个topic-word骰子，每个骰子有一个编号，编号从1到K;
3. 生成每篇文章之前，上帝都先为这篇文章制造一个特定的doc-topic骰子，然后重复如下过程生成文档中的词
    * 投郑这个doc-topic骰子，得到一个topic编号$z$
    * 选择$k$个topic-word骰子中编号为$z$的那个，投郑那个骰子，于是得到一个词
以上PLSA模型的文档生成的过程可以图形化的表示为
![Alt text](./1524750324185.png)

可以发现，文档和文档之间是独立可交换的，同一个文档中的词也是独立可交换的，还是一个bag-of-words模型。$K$个topic-words分布可标记为$\vec\varphi_1,\vec\varphi_2,...,\vec\varphi_K$，对于包含$M$篇文章的语料$C=(d_1,d_2,...,d_M)$中的每篇文章$d_m$，都会有一个特定的doc-topic骰子$\vec\theta_m$，所有对应的骰子记为$\vec\theta_1,...,\vec\theta_M$。为了方便，我们假设每个词$w$都是一个编号，对应到topic-word骰子的面。于是在PLSA模型中，每$m$篇文档$d_m$中每个词的生成概率为：
$$p(w|d_m)=\sum_{z=1}^Kp(w|z)p(z|d_m)=\sum_{z=1}^K\varphi_{zw}\theta_{mz}$$
所以整篇文章的生成概率为
$$p(\vec w|d_m)=\prod_{i=1}^n\sum_{z=1}^Kp(w_i|z)p(z|d_m)=\prod_{i=1}^n\sum_{z=1}^K\varphi_{zw_i}\theta_{dz}$$
求解PLSA的模型参数可以使用著名的EM算法。

#### LDA文本建模

对于上述PLSA模型，类似对Unigram Model的贝叶斯改造，将其贝叶斯化就得到了LDA模型。由于$\vec\varphi_k$和$\vec\theta_m$都对应到多项式分布，所以先验分布都选Dirichlet分布。

**LDA topic model**
1. 上帝有两类骰子，一类是doc-topic骰子，一类是topic-word骰子
2. 上帝随机从第二类骰子中独立地抽取了K个topic-word骰子，编号为1到K
3. 每次生成一篇新文档前，上帝先从第一类骰子中随机抽取一个doc-topic骰子，然后重复如下过程生成文档中的词：
    * 投郑这个doc-topic骰子，得到一个topic的编号$z$
    * 选择K个topic-word骰子中编号为$z$的那个，投掷这个骰子，于是得到一个词

#### 物理过程分解

使用概率图模型表示如下：
![Alt text](./1524751767623.png)
<center>LDA的图模型表示
这个过程可分解为两个主要的物理过程
1. $\vec\alpha\to\vec\theta_m\to z_{m,n}$，这个过程表示在生成第$m$篇文档的时候，先从第一个类骰子中抽取了一个doc-topic骰子$\vec\theta_m$，然后投掷这个骰子生成了文档中第$n$个词的topic编号$z_{m,n}$;
2. $\vec\beta\to\vec\varphi_k\to w_{m,n}|k=z_{m,n}$，这个过程表示如下动作生成语料中第$m$篇文档的第$n$个词：在上帝手头的K个topic-word骰子$\vec\varphi_k$中，挑选编号为$k=z_{m,n}$的那个骰子进行投郑，然后生成word $w_{m,n}$.

由于LDA也是bag-of-words模型，有一些物理过程是独立可交换的。由此，在LDA生成模型中，M篇文档会对应于M个独立的Dirichlet-Multinomial共轭结构；K个topic会对应K个独立的Dirichlet-Multinomial共轭结构。下面来看看LDA模型如何被分解为M+K个Dirichlet-Multinomial共轭结构的。

由第一个物理过程，我们知道$\vec\alpha \to \vec \theta_m\to\vec z_m$表示生成第$m$篇文档中的所有词对应的topics，显然$\vec\alpha\to\vec\theta_m$对应于Dirichlet分布，$\vec\theta_m\to\vec z_m$对应于Multinomial分布，所以整体是一个Dirichlet共轭结构：
![Alt text](./1524782350222.png)
由前文中的计算有：
$$p(\vec z_m|\vec\alpha)=\frac{\Delta(\vec n_m+\vec\alpha)}{\Delta\vec\alpha}$$
其中$\vec n_m=\left(n_m^{(1)},...,n_m^{(K)}\right)$，$n_m^{(k)}$表示第$m$篇文档中第$k$个topic产生的词的个数。进一步，利用Dirichlet-Multinomial共轭结构，我们得到参数$\vec\theta_m$的后验分布恰好是
$$Dir(\vec\theta_m|\vec n_m+\vec\alpha)$$
由于语料中M篇文档的topics生成过程相互独立，所以我们得到M个相互独立的Dirichlet-Multinomial共轭结构，从而可以得到整个语料中topics生成概率
$$
\begin{aligned}
p(\vec{\textbf z}|\vec\alpha)&=\prod_{m=1}^{M}p(\vec z_m|\vec\alpha)\\
&=\prod_{m=1}^M\frac{\Delta(\vec n_m+\vec\alpha)}{\Delta(\vec\alpha)}
\end{aligned}
$$
上帝是顺序处理每篇文章的，处理完一篇再去处理下一篇。文档中每个词的生成都要抛两次骰子，第一次抛doc-topic骰子得到topic，第二次抛topic-word骰子得到word，每次生成词时这两次抛骰子的动作都是紧邻轮换进行的。如果语料中共有N个词，则上帝共要轮换doc-topic和topic-word骰子抛2N次。实际上有一些骰子的顺序是可交换的，可以得到如下次序：前N次只抛doc-topic骰子得到语料中所有词的topic，然后基于得到的每个词的topic编号，后N次只抛topic-word骰子生成N个word。

**LDA Topic Model2**
1. 上帝有两类骰子，一类是doc-topic骰子，一类是topic-word骰子
2. 上帝随机的从第二类骰子中独立地抽取了K个topic-word骰子，编号从1到K
3. 每次生成一篇新文档前，上帝先从第一类骰子中随机抽取一个doc-topic骰子，然后重复投掷这个doc-topic骰子，为每个词生成一个topic编号$z$；重复如上过程处理每篇文章档，生成语料中每个词的topic编号，但词尚未生成
4. 从头到尾，对语料中的每篇文档中的每个topic编号$z$，选择K个topic-word骰子中编号为$z$的那个，投掷这个骰子，于是生成对应的word


#### Gibbs Sampling
$z_i=k,w_i=t$的概率只和两个Dirichlet-Multinomail共轭结构关联。而最终得到的$\hat\theta_{m,k},\hat\varphi_{k,t}$就是对应的两个Dirichlet后验分布在贝叶斯框架下的参数估计。于是有：
$$
\hat\theta_{mk}=\frac{n_{m,\neg i}^{(t)}+\alpha_k}{\sum_{k=1}^K(n_{m,\neg i}^{(t)}\alpha_k)} \\
\hat\varphi_{kt}=\frac{n_{k,\neg i}^{(t)}+\beta_t}{\sum_{t=1}^V(n_{k,\neg i}^{(t)}+\beta_t)}
$$
于是我们得到了LDA模型的Gibbs Sampling公式
$$
p(z_i=k|\vec {\textbf z}_{\neg i},\vec{\textbf w}) \propto \frac{n_{m,\neg i}^{(k)}+\alpha_k}{\sum_{k=1}^K\left(n_{m,\neg i}^{(t)}+\alpha_k\right)} \cdot \frac{n_{k,\neg i}^{(t)}+\beta_t}{\sum_{t=1}^V\left(n_{k,\neg i}^{(t)}+\beta_t \right)}
$$
这个公式是很漂亮的，右边其实就是$p(topic|doc)\cdot p(word|topic)$，这个概率其实是$doc \to topic \to word$的路径概率，由于topic有$K$个，所以Gibbs Sampling公式的物理意义其实就是在这$K$条路径中进行采样。
![Alt text](./1524755019105.png)
<center>doc-toic-word路径概率

#### Training and Inference

**LDA Traing**
1. 随机初始化：对于语料中每篇文档中的每个词$w$，随机赋予一个topic编号$z$;
2. 重新扫描语料库，对每个词$w$，按照Gibbs Sampling公式重新采样它的topic，在语料中更新其topic；
3. 重复以上语料库的重新采样过程直到Gibbs Sampling收敛；
4. 统计语料库的topic-word共现频率矩阵，该矩阵就是LDA的模型。

由这个topic-word频率矩阵我们可以计算每个$p(word|topic)$概率，从而计算出模型参数$\vec\varphi_1,...,\vec\varphi_K$，这就是上帝用的K个topic-word骰子。当然，语料中的文档对应的骰子参数$\vec\theta_1,...,\vec\theta_M$在以上的训练过程中也可以计算出来的，只要在Gibbs Sampling收敛之后，统计每篇文档中的topic的频率分布，我们就可以计算每个$p(topic|doc)$概率，于是就可以计算出每一个$\vec\theta_m$。由于$\vec\theta_m$是和训练语料中每篇文档相关的，对于我们理解新的文档并无用处，所以工程上最终存储LDA模型时一般没必要保留。通常，在LDA训练过程中，我们取Gibbs Sampling收敛之后的$n$个迭代的结果进行平均来做参数估计，这样模型质量更高。

有了LDA模型，对于新的文档$doc_{new}$，我们如何做该文档的语义分布计算呢？基本上inference过程和training过程完全类似。对于新的文档，我们只要认为Gibbs Sampling公式中的$\hat\varphi_{kt}$部分是稳定不变的，是由训练语料得到的模型提供的，所以采样过程中我们只要估计该文档的topic分布$\vec\theta_{new}$就好了。

**LDA Inference**
1. 随机初始化：对当前文档中的每个词$w$，随机的赋一个topic编号$z$;
2. 重新扫描当前文档，按照Gibbs Sampling公式，对每个词$w$，重新采样它的topic；
3. 重复以上过程直到Gibbs Sampling收敛；
4. 统计文档中的topic分布，该分布就是$\vec\theta_{new}$

