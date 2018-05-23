---
title: 算法总结
toc: true
date: 2018-01-21 19:59:59
tags: 算法, 总结
categories: 
---

## 朴素贝叶斯

有以下几个地方需要注意：

1. 如果给出的特征向量长度可能不同，这是需要归一化为通长度的向量（这里以文本分类为例），比如说是句子单词的话，则长度为整个词汇量的长度，对应位置是该单词出现的次数

2. 计算公式如下：

$$p(c_i|w)= \frac {p(w|c_i)p(c_i)}{p(w)}$$

 其中一项条件概率可以通过朴素贝叶斯条件独立展开。要注意一点就是$p(w|c_i)$的计算方法，而由朴素贝叶斯的前提假设可知，$p(w_0,w_1,w_2...w_n|c_i)=p(w_0|c_i)p(w_1|c_i)p(w_2|c_i)...p(w_n|c_i) $ ，因此一般有两种，一种是在类别为$c_i$的那些样本集中，找到$w_j$出现次数的总和，然后除以该样本的总和；第二种方法是类别为$c_i$的那些样本集中，找到$w_j$出现次数的总和，然后除以该样本中所有特征出现次数的总和

3.如果$p(w|c_i)$中的某一项为0，则其联合概率的乘积也可能为0，即2中公式的分子为0，为了避免这种现象出现，一般情况下会将这一项初始化为1，当然为了保证概率相等，分母应对应初始化为2（这里因为是2类，所以加2，如果是k类就需要加k，术语上叫做laplace光滑, 分母加k的原因是使之满足全概率公式）

### 优点：

* 对小规模的数据表现很好，适合多分类任务，适合增量式训练。

### 缺点：

* 对输入数据的表达形式很敏感
* 没有考虑组合特征，简化为特征间是互相独立的

## 决策树

  决策树中很重要的一点就是选择一个属性进行分枝，因此要注意一下信息增益的计算公式，并深入理解它。
  信息熵的计算公式如下:
  $$H = -\sum_{i=1}^n p(x_i)log_2p(x_i)$$

其中的$n$代表有$n$个分类类别（比如假设是2类问题，那么n=2）。分别计算这2类样本在总样本中出现的概率$p_1$和$p_2$，这样就可以计算出未选中属性分枝前的信息熵。

　　现在选中一个属性$x_i$用来进行分枝，此时分枝规则是：如果$x_i=v_x$的话，将样本分到树的一个分支；如果不相等则进入另一个分支。很显然，分支中的样本很有可能包括2个类别，分别计算这2个分支的熵$H_1$和$H_2$,计算出分枝后的总信息熵$H'=p_1*H_1+p_2*H_2$.，则此时的信息增益$\delta H=H-H'$。以信息增益为原则，把所有的属性都测试一边，选择一个使增益最大的属性作为本次分枝属性

### 优点：

* 计算量简单，可解释性强，比较适合处理有缺失属性值的样本，能够处理不相关的特征；

### 缺点：

* 容易过拟合（后续出现了随机森林，减小了过拟合现象）

## Logistic回归：

Logistic是用来分类的，是一种线性分类器，需要注意的地方有
1. logistic函数表达式为：$$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$$
where $$g(z)=\frac{1}{1+e^{-z}}$$
其导数形式为：

{% math %}
\begin{aligned}
g\prime(z)=\frac{d}{dz}\frac{1}{1+e^{-z}} \\
          =\frac{1}{(1+e^{-z})^2}(e^{-z}) \\
          =\frac{1}{1+e^{-z}}*(1-\frac{1}{1+e^{-z}}) \\
          =g(z)(1-g(z)
\end{aligned}
{% endmath %}

2. logsitc回归方法主要是用最大似然估计来学习的，所以单个样本的后验概率为：

$
p(y|x;\theta)=(1-h_\theta(x))^{1-y} * (h_\theta(x))^y
$

到整个样本的后验概率：

$
L(\theta)=p(\vec{y}|X;\theta)=\prod_{i=1}^{m}p(y_i|x_i;\theta)=\prod_{i=1}^{m}(h_\theta(x_i))^{y_i}(1-h_\theta(x_i))^{1-y_i}
$

$
\begin{aligned}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{aligned}
$

其中：

$
P(y=1|x; \theta)=h_\theta(x) \\
P(y=0|x; \theta)=1-h_\theta(x)
$

通过进一步简化：

$
l(\theta)=\log L(\theta)=\sum_{i=1}^{m}y^i\log h(x^i)+(1-y_i)\log(1-h(x_i))
$

3. 其实它的$loss function$为$-l(θ)$，因此我们需使$lossfunction$最小，可采用梯度下降法得到。梯度下降法公式为:

### 优点：

* 实现简单；
* 分类时计算量非常小，速度很快，存储资源低；

### 缺点：

* 容易欠拟合，一般准确度不太高
* 只能处理两分类问题（在此基础上衍生出来的softmax可以用于多分类），且必须线性可分；

## 线性回归：

线性回归才是真正用于回归的，而不像logistic回归是用于分类，其基本思想是用梯度下降法对最小二乘法形式的误差函数进行优化，当然也可以用normal equation直接求得参数的解，结果为：
$$\hat{w}=(X^TX)^{-1}X^Ty$$
而在LWLR（局部加权线性回归）中，参数的计算表达式为:
$$\hat{w}=(X^TWX)^{-1}X^TWy$$
因为此时优化的是：
1. Fit $\theta$ to minimize $\sum_i w^i(y_i-\theta^Tx^i)^2$
2. Output $\theta^Tx$

由此可见LWLR与LR不同，LWLR是一个非参数模型，因为每次进行回归计算都要遍历训练样本至少一次。

### 优点：

* 实现简单，计算简单；

### 缺点：

* 不能拟合非线性数据；

## KNN算法：

KNN即最近邻算法，其主要过程为：

1. 计算训练样本和测试样本中每个样本点的距离（常见的距离度量有欧式距离，马氏距离等）；
2. 对上面所有的距离值进行排序；
3. 选前k个最小距离的样本；
4. 根据这k个样本的标签进行投票，得到最后的分类类别；

　如何选择一个最佳的K值，这取决于数据。一般情况下，在分类时较大的K值能够减小噪声的影响。但会使类别之间的界限变得模糊。一个较好的K值可通过各种启发式技术来获取，比如，交叉验证。另外噪声和非相关性特征向量的存在会使K近邻算法的准确性减小。

　近邻算法具有较强的一致性结果。随着数据趋于无限，算法保证错误率不会超过贝叶斯算法错误率的两倍。对于一些好的K值，K近邻保证错误率不会超过贝叶斯理论误差率。

　注：马氏距离一定要先给出样本集的统计性质，比如均值向量，协方差矩阵等。关于马氏距离的介绍如下

### 优点：

* 思想简单，理论成熟，既可以用来做分类也可以用来做回归；
* 可用于非线性分类；
* 训练时间复杂度为O(n)；
* 准确度高，对数据没有假设，对outlier不敏感；

### 缺点：

* 计算量大；
* 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
* 需要大量的内存

## 支持向量机(Support Vector Machine)
### 基本型

![Alt text](./1524785098026.png)
<center>存在多个划分超平面将两类训练样本分开

在样本空间中，划分超平面可通过如下线性方程来描述：

\begin{gathered}
\boldsymbol{w}^Tx+b=0
\end{gathered}

其中$\boldsymbol{w}=(w_1,w_2,...,w_d)$为法向量，决定了超平面的方向；$b$为位移项，决定了超平面与原点之间的距离。显然划分超平面可被法向量$\boldsymbol{w}$和位移$b$确定，下面我们将其记为$(\boldsymbol{w},b)$.样本空间中任意点$x$到超平面$(\boldsymbol{w},b)$的距离可写为

\begin{gathered}
r=\frac{|\boldsymbol{w}^Tx+b|}{||\boldsymbol{w}||}
\end{gathered}

假设超平面$(\boldsymbol{w},b)$能将训练样本正确分类，即对于$(\boldsymbol{x_i},y_i) \in D$，若$y_i=+1$，则有$\boldsymbol{w}^T\boldsymbol{x}_i+b>0$；若$y_i=-1$，则有$\boldsymbol{x}^T\boldsymbol{x}_i+b<0$.令:

\begin{cases}
\boldsymbol{w}^T\boldsymbol{x_i}+b \ge +1, y_i=+1; \\
\boldsymbol{w}^T\boldsymbol{x_i}+b \le -1, y_i=-1.
\end{cases}

如图所示，距离超平面最近的这几个训练样本点使上式的等号成立，它们被称为『支持向量』(support vector)，两个异类支持向量到超平面的距离之和为
$$
\gamma=\frac{2}{\Vert\boldsymbol{w}\Vert}
$$
被称为『间隔』(margin).
![Alt text](./1524785212683.png)
<center>支持向量与间隔

欲找到具有『最大间隔』(maximum margin)的划分超平面，也就是要找到能满足上式中约束的参数$\boldsymbol{w}$和$b$，使得$\gamma$最大，即：

\begin{gathered}
max_{\boldsymbol{w},b}      \frac{2}{\Vert\boldsymbol{w}\Vert} \\
s.t.        y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b) \ge 1, i=1,2,...,m.
\end{gathered}

其对偶问题可写成如下：

\begin{gathered}
min_{\boldsymbol{w},b}\frac{1}{2}\Vert\boldsymbol{w}\Vert \\
s.t.  y_i(\boldsymbol{w}T\boldsymbol{x}_i+b) \le 1, i=1,2,...,m.
\end{gathered}

这就是支持向量机的基本型.

### 对偶问题
划分超平面对应的模型为：$f(\boldsymbol{w}^T\boldsymbol{x}+b)$，其中$\boldsymbol{w}$和$b$是模型参数。使用拉格朗日乘子法可得到其对偶问题(dual problem).则该问题的拉格朗日函数可写为：

\begin{gathered}
L(\boldsymbol{x},b,\boldsymbol\alpha)=\frac{1}{2}\Vert{w}\Vert^2+\sum_{i=1}^m\alpha_i(1-y_i(\boldsymbol{w}^T)\boldsymbol{x}_i+b)
\end{gathered}

其中$\boldsymbol\alpha=(\alpha_1,\alpha_2,...,\alpha_m)$.令$L(\boldsymbol{w},b\boldsymbol\alpha)$对$\boldsymbol{w}$和$b$的偏导为令可得

\begin{gathered}
\boldsymbol{w}=\sum_{i=1}^m\alpha_iy_i\boldsymbol{x}_i \\
0=\sum_{i=1}^m\alpha_iy_i
\end{gathered}

将上式代入可得对偶问题：

\begin{gathered}
max_\boldsymbol\alpha\sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{y=1}^m\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i^T\boldsymbol{x}_j \\
s.t. \sum_{i=1}^m\alpha_iy_i=0, \\
\alpha_i \le 0, i=1,2,...,m
\end{gathered}

求出$\boldsymbol{w}$与$b$即可得到模型

\begin{aligned}
f(\boldsymbol{x})&=\boldsymbol{w}^T\boldsymbol{x}+b \\
&=\sum_{i=1}^m\alpha_iy_i\boldsymbol{x}_i^T\boldsymbol{x}+b
\end{aligned}

其对应的$KKT$条件为：

\begin{cases}
\alpha_i=0 \\
y_if(\boldsymbol{x_i})-1 \le 0 \\
\alpha_i(y_if(\boldsymbol{x}_i)-1).
\end{cases}

>其含义如下：*对任意训练样本$(\boldsymbol{x}_i,y_i)$，总有$\alpha_i=0$或$y_if(\boldsymbol{x}_i)=1$.若$\alpha=0$，则该样本将不会在式中的求和中出现，也就不会对$f(\boldsymbol{x})$有任何影响；若$\alpha_i>0$，则必有$y_if(\boldsymbol{x}_i)=1$，所对应的样本点位于最大间隔的边界上，是一个支持向量*。这显示出支持向量机的一个重要性质：*训练完成后，大部分的训练样本都不需要保留，最终模型仅与支持向量有关*

### 软间隔与松弛变量
![Alt text](./1524785298277.png)
<center>软间隔示意图.红色圈出了一些不满足约束的样本

软间隔允许某些样本不满足约束
$$y_i(\boldsymbol w^T \boldsymbol {x}_i+b) \ge 1 $$
于是优化目标可写为：
$$\min_{w,b}\frac{1}{2}\Vert\boldsymbol{w}\Vert ^2+C\sum_{i=1}^m l_{0/1}\left(y_i\left(\boldsymbol w^T \boldsymbol x_i+b\right)-1\right) $$
其中$C\gt 0$是一个常数，$l_{0/1}$是“0/1损失函数”

hinge损失：$l_{hinge}(z)=max(0, 1-z)$
引入松弛变量$\xi_i \ge 0$后，原优化问题可写为：

\begin{gathered}
min_{\boldsymbol{w},b,\xi_i} \frac{1}{2}\Vert{w}\Vert^2+C\sum_{i=1}^m\xi_i \\
s.t. y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b) \le 1-\xi_i \\
\xi_i \ge 0, i=1,2,...,m.
\end{gathered}

通过拉格朗日乘子法可得到如下的拉格朗日函数

\begin{aligned}
L(\boldsymbol{w},b,\boldsymbol\alpha,\boldsymbol\xi,\boldsymbol\mu)&=\frac{1}{2}\Vert\boldsymbol{w}\Vert^2+C\sum_{i=1}^m\xi_i \\
&+\sum_{i=1}^m\alpha_i(1-\xi_i-y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b))-\sum_{i=1}^m\mu_i\xi_i,
\end{aligned}

其中$\alpha_i \ge 0, \mu_i \ge 0$是拉格朗日乘子.令$L(\boldsymbol{w},b,\boldsymbol\alpha,\boldsymbol\xi,\boldsymbol\mu)$对$\boldsymbol{w}$,$b$,$\xi_i$的偏导为零可得

\begin{aligned}
\boldsymbol{w}&=\sum_{i=1}^m\alpha_iy_i\boldsymbol{x}_i, \\
0&=\sum_{i=1}^m\alpha_iy_i, \\
C&=\alpha_i+\mu_i.
\end{aligned}

将上式代入可得到对偶问题：

\begin{gathered}
max_\boldsymbol\alpha\sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i^T\boldsymbol{x}_j \\
s.t. \sum_{i=1}^m \alpha_iy_i=0, \\
0 \le \alpha_i \le C, i=1,2,...,m
\end{gathered}

类似的，对软间隔支持向量机，其$KKT$条件如下：

\begin{cases}
\alpha_i \ge 0, \mu_i \ge 0, \\
y_if(\boldsymbol{x}_i) -1+\xi_i \ge 0, \\
\alpha_i(y_if(\boldsymbol{x}_i)-1+\xi_i)=0, \\
\xi_i \ge 0, \mu_i\xi_i=0.
\end{cases}




## 参考资料
> - [机器学习&数据挖掘笔记_16（常见面试之机器学习算法思想简单梳理）](http://www.cnblogs.com/tornadomeet/p/3395593.html)
> - []()