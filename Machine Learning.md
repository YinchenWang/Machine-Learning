# Machine Learning

### Week 1

#### Introduction

##### Supervised Learning

we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

***Supervised Learning*** is also called ***Regression(回归问题)*** which is to predict continuous valued output(such as price).

##### Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

#### Model and Cost Function

##### Model Representation

some symbols:
**m** = Number of training examples
**x**'s = "input" variable / features
**y**'s = "output" variable / "target" variable

***Linear regression model***

<img src="C:\Users\王尹晨\Desktop\2.PNG" alt="捕获" style="zoom:50%;" />

##### Cost Function

We can measure the accuracy of our hypothesis function by using a **cost function**. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

in the last example of house price:
$$
Hypothesis(假设函数):  h_\theta(x) = \theta_0 + \theta_1x
$$

**θ**'s:     Parameters

**How to chose $\theta's$?**

Chose $θ_0，θ_1$ so that $h_θ(x)$ is close to $y$ for our training examples$(x,y)$
so the cost function in this example is
$$
minimize(\theta_0,\theta_1)\space\space\frac{1}{2m}\sum^m_{i=1}(h_\theta(x_i)-y_i)^2
$$

  which is 
$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum^m_{i=1}(h_\theta(x_i)-y_i)^2
$$

(minimize function $J(\theta_0,\theta_1)$ over $\theta_0,\theta_1$)  		
this is the cost function also called the squared error function.(the most common used cost function in regression!)

**Examples of what cost function doing:**

[examples of θ~0~=0](https://www.coursera.org/learn/machine-learning/supplement/u3qF5/cost-function-intuition-i)

cost function is meant to find the most fitted $\theta_0,\theta_1$ to make a straight line (defined by $h_\theta(x)$ which passes through these scattered data points.

#### Parameter Learning

##### Gradient Descent(梯度下降)

we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

**situation：**

Have some function of $J(\theta_0,\theta_1)$
         Want ${min}_{\theta_0,\theta_1} J(\theta_0,\theta_1)$

**outline:**

Start with some $\theta_0,\theta_1$
        Keep changing $\theta_0,\theta_1$ to reduce $J(\theta_0,\theta_1)$,until we hopefully end up at a minimum.

<img src="C:\Users\王尹晨\Desktop\1.PNG" alt="捕获" style="zoom:50%;" />

***Gradient descent algorithm:***
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)
$$
**$\alpha:$ learning rate( It basically controls how big a step we take downhill with creating descent.**

in this algorithm,we need simultaneous update the $\theta_0,\theta_1$,the correct step is blew:

(for $j = 0$ and $j = 1$)
        $temp0 := \theta_0 - \alpha \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)$
        $temp1 :=  \theta_1 - \alpha \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1)$  
        $\theta_0 := temp0$
        $\theta_1 := temp1$
update them both!

##### Gradient Descent intuition

If $\alpha$ is too small, gradient descent can be slow, but if $\alpha$ is too large,gradient descent overshoot the minimum,it may fail to converge,or even diverge.

<img src="C:\Users\王尹晨\Desktop\捕获.PNG" alt="捕获" style="zoom:50%;" />

if your parameters are already at a local minimum ,gradient descent will do nothing and keep your parameters in the local minimum.

Gradient descent can converge to a local minimum,even with the learning rate $\alpha$ fixed. As we approach a local minimum,**gradient descent will automatically take smaller steps(because of the $\frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$ became smaller and smaller)**. So, no need to decrease $\alpha$ over time. 

##### Gradient Descent For Linear Regression