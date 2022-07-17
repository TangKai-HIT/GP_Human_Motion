
## An Intuitive Tutorial to Gaussian Processes Regression

[Jie Wang](mailto:jie.wang@queensu.ca), [Offroad Robotics](https://offroad.engineering.queensu.ca/), Queen's University, Kingston, Canada

The notebook can be executed at

<a href="https://colab.research.google.com/github/jwangjie/Gaussian-Processes-Regression-Tutorial/blob/master/gpr_tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#### A [formal paper](https://arxiv.org/abs/2009.10862) of the notebook:
```
@misc{wang2020intuitive,
    title={An Intuitive Tutorial to Gaussian Processes Regression},
    author={Jie Wang},
    year={2020},
    eprint={2009.10862},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```

The audience of this tutorial is the one who wants to use GP but not feels comfortable using it. This happens to me after finishing reading the first two chapters of the textbook **Gaussian Process for Machine Learning** [[1](#Reference)]. There is a gap between the usage of GP and feel comfortable using it due to the difficulties in understanding the theory. When I was reading the textbook and watching tutorial videos online, I can follow the majority without too many difficulties. The content kind of makes sense to me. But even when I am trying to talk to myself what GP is, the big picture is blurry. After keep trying to understand GP from various recourses, including textbooks, blog posts, and open-sourced codes, I get my understandings sorted and summarize them up from my perspective. 

One thing I realized the difficulties in understanding GP is due to background varies, everyone has different knowledge. To understand GP, even to the intuitive level, needs to know multivariable Gaussian, kernel, conditional probability. If you familiar with these, start reading from [III. Math](#III.-Math). Entry or medium-level in deep learning (application level), without a solid understanding in machine learning theory, even cause more confusion in understanding GP. 

<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/gpr_animation_wide.gif?raw=1" width="1000"/> [10]

## I.   Motivation

First of all, why use Gaussian Process to do regression? Or even, what is regression? Regression is a common machine learning task that can be described as Given some observed data points (training dataset), finding a function that represents the dataset as close as possible, then using the function to make predictions at new data points. Regression can be conducted with polynomials, and it's common there is more than one possible function that fits the observed data. Besides getting predictions by the function, we also want to know how certain these predictions are. Moreover, quantifying uncertainty is super valuable to achieve an efficient learning process. The areas with the least certainty should be explored more. 

In a word, GP can be used to make predictions at new data points and can tell us how certain these predictions are. 

<div id="image-table">
    <table>
        <tr>
            <td style="padding:10px">
                <img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/regression1.png?raw=1" width="400"/> [2]
            </td>
            <td style="padding:10px">
                <img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/regression2.png?raw=1" width="550"/>
            </td>
        </tr>
    </table>
</div> 

## II. Basics 

### A.  Gaussian (Normal) Distribution  

Let's talk about Gaussian. 

A random variable <img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> is said to be normally distributed with mean <img src="./tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> and variance <img src="./tex/e6718aa5499c31af3ff15c3c594a7854.svg?invert_in_darkmode&sanitize=true" align=middle width=16.535428799999988pt height=26.76175259999998pt/> if its probability density function (PDF) is 
<p align="center"><img src="./tex/e550b422b54a4cca377140c08eb745ae.svg?invert_in_darkmode&sanitize=true" align=middle width=240.58308119999998pt height=49.315569599999996pt/></p>

Here, <img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> represents random variables and <img src="./tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> is the real argument. The Gaussian or Normal distribution of <img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> is usually represented by <img src="./tex/07bdc2a7c5138580e2f4b172dc9aedff.svg?invert_in_darkmode&sanitize=true" align=middle width=125.6777049pt height=26.76175259999998pt/>. 

A <img src="./tex/18c86fe07e798736bb4ea0f824b20d97.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/> Gaussian PDF is plotted below. We generate `n` number random sample points from a <img src="./tex/18c86fe07e798736bb4ea0f824b20d97.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/> Gaussian distribution on `x` axis. 


```python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```


```python
# Plot 1-D gaussian
n = 1         # n number of independent 1-D gaussian 
m= 1000       # m points in 1-D gaussian 
f_random = np.random.normal(size=(n, m)) 
# more information about 'size': https://www.sharpsightlabs.com/blog/numpy-random-normal/ 
#print(f_random.shape)

for i in range(n):
    #sns.distplot(f_random[i], hist=True, rug=True, vertical=True, color="orange")
    sns.distplot(f_random[i], hist=True, rug=True)

plt.title('1 random samples from a 1-D Gaussian distribution')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.show()
```


![png](./img/codes_plot_output/output_10_1.png)


We generated data points that follow the normal distribution. On the other hand, we can model data points, assume these points are Gaussian, model as a function, and do regression using it. As shown above, a kernel density and histogram of the generated points were estimated. The kernel density estimation looks a normal distribution due to there are plenty `(m=1000)` observation points to get this Gaussian looking PDF. In regression, even we don't have that many observation data, we can model the data as a function that follows a normal distribution if we assume a Gaussian prior. 

The Gaussian PDF <img src="./tex/b7d5ffe2d856ef70136d42b025daa3d7.svg?invert_in_darkmode&sanitize=true" align=middle width=63.26343869999999pt height=26.76175259999998pt/> is completely characterized by the two parameters <img src="./tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> and <img src="./tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/>, they can be obtained from the PDF as [3]

<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/1dGaussian.png?raw=1" width="550"/> 

We have a random generated dataset in <img src="./tex/18c86fe07e798736bb4ea0f824b20d97.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/> <img src="./tex/3c61e97f3a775c71752c1376d7c563bd.svg?invert_in_darkmode&sanitize=true" align=middle width=175.62573599999996pt height=29.190975000000005pt/>. We sampled the generated dataset and got a <img src="./tex/18c86fe07e798736bb4ea0f824b20d97.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/>Gaussian bell curve. 

Now, if we project all points <img src="./tex/443fbe95e05f23fd37ec921303ee71c8.svg?invert_in_darkmode&sanitize=true" align=middle width=139.21046369999996pt height=29.190975000000005pt/> on the x-axis to another **space**. In this space, We treat all points <img src="./tex/443fbe95e05f23fd37ec921303ee71c8.svg?invert_in_darkmode&sanitize=true" align=middle width=139.21046369999996pt height=29.190975000000005pt/> as a vector <img src="./tex/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/>, and plot <img src="./tex/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> on the new <img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> axis at <img src="./tex/b6903d0bfe9fdb18f618c3811752bda9.svg?invert_in_darkmode&sanitize=true" align=middle width=45.04550654999999pt height=22.465723500000017pt/>. 


```python
n = 1         # n number of independent 1-D gaussian 
m= 1000       # m points in 1-D gaussian  
f_random = np.random.normal(size=(n, m))

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

plt.clf()
plt.plot(Xshow, f_random, 'o', linewidth=1, markersize=1, markeredgewidth=2)
plt.xlabel('<img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>')
plt.ylabel('<img src="./tex/161805ece9a8142e4ebe9d356fd0f763.svg?invert_in_darkmode&sanitize=true" align=middle width=37.51151249999999pt height=24.65753399999998pt/>')
plt.show()
```


![png](./img/codes_plot_output/output_14_0.png)


It's clear that the vector <img src="./tex/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> is Gaussian. It looks like we did nothing but vertically plot the vector points <img src="./tex/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/>. 
Next, we can plot multiple independent Gaussian in the <img src="./tex/a0070c759142f7a0b5723afd57c22639.svg?invert_in_darkmode&sanitize=true" align=middle width=48.196244249999985pt height=22.465723500000017pt/> coordinates. For example, put vector <img src="./tex/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> at at <img src="./tex/b6903d0bfe9fdb18f618c3811752bda9.svg?invert_in_darkmode&sanitize=true" align=middle width=45.04550654999999pt height=22.465723500000017pt/> and another vector <img src="./tex/f6fac43e354f1b2ca85658091df26df1.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> at at <img src="./tex/3fc7e158ad67836edf7a548295603ba3.svg?invert_in_darkmode&sanitize=true" align=middle width=45.04550654999999pt height=22.465723500000017pt/>. 


```python
n = 2          
m = 1000
f_random = np.random.normal(size=(n, m))

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

plt.clf()
plt.plot(Xshow, f_random, 'o', linewidth=1, markersize=1, markeredgewidth=2)
plt.xlabel('<img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>')
plt.ylabel('<img src="./tex/161805ece9a8142e4ebe9d356fd0f763.svg?invert_in_darkmode&sanitize=true" align=middle width=37.51151249999999pt height=24.65753399999998pt/>')
plt.show()
```


![png](./img/codes_plot_output/output_16_0.png)


Keep in mind that both vecotr <img src="./tex/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> and <img src="./tex/f6fac43e354f1b2ca85658091df26df1.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> are Gaussian. 

<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/2gaussian.png?raw=1" width="500"/> 

Let's do something interesting. Let's connect points of <img src="./tex/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> and <img src="./tex/f6fac43e354f1b2ca85658091df26df1.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> by lines. For now, we only generate 10 random points for <img src="./tex/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/> and <img src="./tex/f6fac43e354f1b2ca85658091df26df1.svg?invert_in_darkmode&sanitize=true" align=middle width=20.17129784999999pt height=22.465723500000017pt/>, and then join them up as 10 lines. Keep in mind, these random generated 10 points are Gaussian. 


```python
n = 2          
m = 10
f_random = np.random.normal(size=(n, m))

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

plt.clf()
plt.plot(Xshow, f_random, '-o', linewidth=2, markersize=4, markeredgewidth=2)
plt.xlabel('<img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>')
plt.ylabel('<img src="./tex/161805ece9a8142e4ebe9d356fd0f763.svg?invert_in_darkmode&sanitize=true" align=middle width=37.51151249999999pt height=24.65753399999998pt/>')
plt.show()
```


![png](./img/codes_plot_output/output_19_0.png)


Going back to think about regression. These lines look like **functions** for each pair of points. On the other hand, the plot also looks like we are sampling the region <img src="./tex/e88c070a4a52572ef1d5792a341c0900.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/> with 10 linear functions even there are only two points on each line. In the sampling perspective, the <img src="./tex/e88c070a4a52572ef1d5792a341c0900.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/> domain is our region of interest, i.e. the specific region we do our regression. This sampling looks even more clear if we generate more independent Gaussian and connecting points in order by lines. 


```python
n = 20          
m = 10
f_random = np.random.normal(size=(n, m))

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

plt.clf()
plt.plot(Xshow, f_random, '-o', linewidth=1, markersize=3, markeredgewidth=2)
plt.xlabel('<img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>')
plt.ylabel('<img src="./tex/161805ece9a8142e4ebe9d356fd0f763.svg?invert_in_darkmode&sanitize=true" align=middle width=37.51151249999999pt height=24.65753399999998pt/>')
plt.show()
```


![png](./img/codes_plot_output/output_21_0.png)


Wait for a second, what we are trying to do by connecting random generated independent Gaussian points? Even these lines look like functions, but they are too noisy. If <img src="./tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> is our input space, these functions are meaningless for the regression task. We can do no prediction by using these functions. The functions should be smoother, meaning input points that are close to each other should have similar values of the function. 

Thus, functions by connecting independent Gaussian are not proper for regression, we need Gaussians that correlated to each other. How to describe joint Gaussian? Multivariable Gaussian.  

### B. Multivariate Normal Distribution (MVN)

In some situations, a system (set of data) has to be described by more than more feature variables <img src="./tex/f21e44741a10198e5e42671a58a486a5.svg?invert_in_darkmode&sanitize=true" align=middle width=104.84956844999999pt height=24.65753399999998pt/>, and these variables are correlated. If we want to model the data all in one go as Gaussian, we need multivariate Gaussian. Here are examples of the <img src="./tex/7c91fa1fa7be856b248f729bd78b5f6f.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/> Gaussian. A data center is monitored by the CPU load <img src="./tex/277fbbae7d4bc65b6aa601ea481bebcc.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> and memory use <img src="./tex/95d239357c7dfa2e8d1fd21ff6ed5c7b.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/>. [3]  

<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/2dGaussian.png?raw=1" width="550"/>

The <img src="./tex/7c91fa1fa7be856b248f729bd78b5f6f.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/> gaussian can be visualized as a 3D bell curve with the heights representing probability density.

<div id="image-table">
    <table>
        <tr>
            <td style="padding:10px">
                <img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/2d_gaussian3D_0.8.png?raw=1" width="400"/>
            </td>
            <td style="padding:10px">
                <img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/2d_gaussian_0.8.png?raw=1" width="350"/>
            </td>
        </tr>
    </table>
</div> 

Goes to [Appendix A](#Appendix-A) if you want to generate image on the left. 

Formally, multivariate Gaussian is expressed as [4]

<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/mul_var_gaussian.png?raw=1" width="400"/>

The `mean vector` <img src="./tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> is a 2d vector <img src="./tex/ff4961935f34c574c74a01d5a03dff44.svg?invert_in_darkmode&sanitize=true" align=middle width=54.65008394999998pt height=24.65753399999998pt/>, which are independent mean of each variable <img src="./tex/277fbbae7d4bc65b6aa601ea481bebcc.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> and <img src="./tex/95d239357c7dfa2e8d1fd21ff6ed5c7b.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/>.

The covariance matrix of <img src="./tex/7c91fa1fa7be856b248f729bd78b5f6f.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/> Gaussian is <img src="./tex/63812e72584465db6349f29f51a43a87.svg?invert_in_darkmode&sanitize=true" align=middle width=87.27955499999999pt height=47.6716218pt/>. The diagonal terms are independent variances of each variable, <img src="./tex/277fbbae7d4bc65b6aa601ea481bebcc.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> and <img src="./tex/95d239357c7dfa2e8d1fd21ff6ed5c7b.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/>. The offdiagonal terms represents correlations between the two variables. A correlation component represents how much one variable is related to another variable. 

A <img src="./tex/7c91fa1fa7be856b248f729bd78b5f6f.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/> Gaussian can be expressed as 
<p align="center"><img src="./tex/9c01d455cd47c359c5b2b27981436f64.svg?invert_in_darkmode&sanitize=true" align=middle width=325.08475395pt height=39.452455349999994pt/></p>

When we have an <img src="./tex/86c1d0ad53307b2439d4b4312ef18cd8.svg?invert_in_darkmode&sanitize=true" align=middle width=49.157390699999986pt height=22.465723500000017pt/> Gaussian, the covariance matrix <img src="./tex/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode&sanitize=true" align=middle width=11.87217899999999pt height=22.465723500000017pt/> is <img src="./tex/47e8490080cc632f4bd33525f9d638eb.svg?invert_in_darkmode&sanitize=true" align=middle width=29.999960099999985pt height=22.465723500000017pt/> and its <img src="./tex/aa20264597f5a63b51587e0581c48f2c.svg?invert_in_darkmode&sanitize=true" align=middle width=33.46496009999999pt height=24.65753399999998pt/> element is <img src="./tex/e35ef5b3fd11b9ef328b82be745d0ef9.svg?invert_in_darkmode&sanitize=true" align=middle width=117.61616789999998pt height=24.65753399999998pt/>. The <img src="./tex/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode&sanitize=true" align=middle width=11.87217899999999pt height=22.465723500000017pt/> is a symmetric matrix and stores the pairwise covariances of all the jointly modeled random variables.

Play around with the covariance matrix to see the correlations between the two Gaussians. 


```python
import pandas as pd
import seaborn as sns

mean, cov = [0., 0.], [(1., -0.6), (-0.6, 1.)]
data = np.random.multivariate_normal(mean, cov, 1000)
df = pd.DataFrame(data, columns=["x1", "x2"])
g = sns.jointplot("x1", "x2", data=df, kind="kde")

#(sns.jointplot("x1", "x2", data=df).plot_joint(sns.kdeplot))

g.plot_joint(plt.scatter, c="g", s=30, linewidth=1, marker="+")

#g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("<img src="./tex/8c76e0c69c5596634f9abb693bbf9438.svg?invert_in_darkmode&sanitize=true" align=middle width=17.614197149999992pt height=21.18721440000001pt/>", "<img src="./tex/1533fefb8348ed2119c7920bf5d7a8a5.svg?invert_in_darkmode&sanitize=true" align=middle width=17.614197149999992pt height=21.18721440000001pt/>");

g.ax_joint.legend_.remove()
plt.show()
```


![png](./img/codes_plot_output/output_29_0.png)


Another good MVN visualization is [Multivariante Gaussians and Mixtures of Gaussians (MoG)](https://www.cs.toronto.edu/~guerzhoy/411/lec/W08/MoG.html).

Besides the joint probalility, we are more interested to the conditional probability. If we cut a slice on the 3D bell curve or draw a line on the elipse contour, we got the conditional probability distribution $P(x_1 \vert \, x_2)$. The conditional distribution is also Gaussian. 

<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/2d_gaussian_conditional3D.png?raw=1" width="400"/>
      	    </td>
            <td style="padding:10px">
            	<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/2d_gaussian_conditional.png?raw=1" width="300"/>
            </td>
        </tr>
    </table>
</div> 

### C. Kernels 

We want to smooth the sampling functions by defining the covariance functions. Considering the fact that when two vectors are similar, their dot product output value is high. It is very clear to see this in the dot product equation <img src="./tex/d920595d94866be7b9fe83844a059fba.svg?invert_in_darkmode&sanitize=true" align=middle width=109.60221524999999pt height=22.831056599999986pt/>, where <img src="./tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> is the angle between two vectors. If an algorithm is defined solely in terms of inner products in input space then it can be lifted into feature space by replacing occurrences of those inner products by <img src="./tex/eee7a5f78ba55222c59c939ac23b417c.svg?invert_in_darkmode&sanitize=true" align=middle width=58.04797349999999pt height=24.7161288pt/>; we call <img src="./tex/5d01d8b29b1de37fa1d298968821c94f.svg?invert_in_darkmode&sanitize=true" align=middle width=45.60509744999999pt height=24.65753399999998pt/> a kernel function [1]. 

A popular covariance function (aka kernel function) is squared exponential kernal, also called the radial basis function (RBF) kernel or Gaussian kernel, defined as 

<p align="center"><img src="./tex/7cf00a5a1141aeaa2c63639b2cd3ea59.svg?invert_in_darkmode&sanitize=true" align=middle width=242.54299905pt height=40.11819404999999pt/></p>

Let's re-plot 20 independent Gaussian and connecting points in order by lines. Instead of generating 20 independent Gaussian before, we do the plot of a <img src="./tex/cd6667199156472c2d791d63cdfaf4c3.svg?invert_in_darkmode&sanitize=true" align=middle width=50.59584089999999pt height=22.465723500000017pt/> Gaussian with a identity convariance matrix. 


```python
n = 20 
m = 10

mean = np.zeros(n)
cov = np.eye(n)

f_prior = np.random.multivariate_normal(mean, cov, m).T

plt.clf()

#plt.plot(Xshow, f_prior, '-o')
Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

for i in range(m):
    plt.plot(Xshow, f_prior, '-o', linewidth=1)
    
plt.title('10 samples of the 20-D gaussian prior')
plt.show()
```


![png](./img/codes_plot_output/output_35_0.png)


We got exactly the same plot as expected. Now let's kernelizing our funcitons by use the RBF as our convariace. 


```python
# Define the kernel
def kernel(a, b):
    sqdist = np.sum(a**2,axis=1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    # np.sum( ,axis=1) means adding all elements columnly; .reshap(-1, 1) add one dimension to make (n,) become (n,1)
    return np.exp(-.5 * sqdist)
```


```python
n = 20  
m = 10

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

K_ = kernel(Xshow, Xshow)                  # k(x_star, x_star)        

mean = np.zeros(n)
cov = np.eye(n)

f_prior = np.random.multivariate_normal(mean, K_, m).T

plt.clf()

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

for i in range(m):
    plt.plot(Xshow, f_prior, '-o', linewidth=1)
    
plt.title('10 samples of the 20-D gaussian kernelized prior')
plt.show()
```


![png](./img/codes_plot_output/output_38_0.png)


We get much smoother lines and looks even more like functions. When the dimension of Gaussian gets larger, there is no need to connect points. When the dimension become infinity, there is a point represents any possible input. Let's plot `m=200` samples of `n=200`<img src="./tex/9a147fe833cc6eda5be947035d6cc8aa.svg?invert_in_darkmode&sanitize=true" align=middle width=26.851664399999994pt height=22.465723500000017pt/> Gaussian to get a feeling of functions with infinity parameters.  


```python
n = 200         
m = 200

Xshow = np.linspace(0, 1, n).reshape(-1,1)   

K_ = kernel(Xshow, Xshow)                    # k(x_star, x_star)        

mean = np.zeros(n)
cov = np.eye(n)

f_prior = np.random.multivariate_normal(mean, K_, m).T

plt.clf()
#plt.plot(Xshow, f_prior, '-o')
Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

plt.figure(figsize=(18,9))
for i in range(m):
    plt.plot(Xshow, f_prior, 'o', linewidth=1, markersize=2, markeredgewidth=1)
    
plt.title('200 samples of the 200-D gaussian kernelized prior')
#plt.axis([0, 1, -3, 3])
plt.show()
#plt.savefig('priorT.png', bbox_inches='tight', dpi=300)
```


    <Figure size 432x288 with 0 Axes>



![png](./img/codes_plot_output/output_40_1.png)


As we can see above, when we increase the dimension of Gaussian to infinity, we can sample all the possible points in our region of interest. 

A great visualization animation of points covariance of the "functions" [10].

<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/2points_covariance.gif?raw=1" width="500"/>
      	    </td>
            <td style="padding:10px">
            	<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/4points_covariance.gif?raw=1" width="500"/>
            </td>
        </tr>
    </table>
</div> 

---

Here we talk a little bit about **Parametric and Nonparametric model**. You can skip this section without compromising your Gaussian Process understandings. 

Parametric models assume that the data distribution can be modeled in terms of a set of finite number parameters. For regression, we have some data points, and we would like to make predictions of the value of <img src="./tex/0e241c321e18ed6141f9a47d8095bebd.svg?invert_in_darkmode&sanitize=true" align=middle width=62.56467194999998pt height=24.65753399999998pt/> with a specific <img src="./tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>. If we assume a linear regression model, <img src="./tex/eab7fea88470156cd8aa3e8c1a13158b.svg?invert_in_darkmode&sanitize=true" align=middle width=90.23571974999999pt height=22.831056599999986pt/>, we need to find the parameters <img src="./tex/edcbf8dd6dd9743cceeee21183bbc3b6.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> and <img src="./tex/f1fe0aebb1c952f09cdbfd83af41f50e.svg?invert_in_darkmode&sanitize=true" align=middle width=14.269439249999989pt height=22.831056599999986pt/> to define the line. In many cases, the linear model assumption isn’t hold, a polynomial model with more parameters, such as <img src="./tex/66e114b7f1f95026b004600d3d19c953.svg?invert_in_darkmode&sanitize=true" align=middle width=141.36579765pt height=26.76175259999998pt/> is needed. We use the training dataset <img src="./tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=22.465723500000017pt/> of <img src="./tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> observations, <img src="./tex/cad3b7fb49f21edeede8aeb09c8ede11.svg?invert_in_darkmode&sanitize=true" align=middle width=185.8496904pt height=24.65753399999998pt/> to train the model, i.e. mapping <img src="./tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> to <img src="./tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/> through parameters <img src="./tex/bacd6bdf70fdf933f86310421df0ab2b.svg?invert_in_darkmode&sanitize=true" align=middle width=107.37431924999999pt height=24.65753399999998pt/>. After the training process, we assume all the information of the data are captured by the feature parameters <img src="./tex/b35e24d8a08c0ab01195f2ad2a78fab7.svg?invert_in_darkmode&sanitize=true" align=middle width=12.785434199999989pt height=22.465723500000017pt/>, thus the prediction is independent of the training data <img src="./tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=22.465723500000017pt/>. It can be expressed as  <img src="./tex/91b69274abea319e9035079407c050a9.svg?invert_in_darkmode&sanitize=true" align=middle width=225.68921594999998pt height=24.65753399999998pt/>, in which <img src="./tex/4d00376b927ae39d0a206dc721cfc59f.svg?invert_in_darkmode&sanitize=true" align=middle width=14.783181599999988pt height=22.831056599999986pt/> is the prediction made at a unobserved point <img src="./tex/3c832332166b20f879ddebdf25829927.svg?invert_in_darkmode&sanitize=true" align=middle width=16.13018219999999pt height=14.15524440000002pt/>. 
Thus, conducting regression using the parametric model, the complexity or flexibility of model is limited by the parameter numbers. It’s natural to think to use a model that the number of parameters grows with the size of the dataset, and it’s a Bayesian non-parametric model. Bayesian non-parametric model do not imply that there are no parameters, but rather infinitely parameters. 

---

To generate correlated normally distributed random samples, one can first generate uncorrelated samples, and then multiply them
by a matrix *L* such that <img src="./tex/3a7e011fcce2798d8f8a0d1c26dc920c.svg?invert_in_darkmode&sanitize=true" align=middle width=69.7847139pt height=27.6567522pt/>, where *K* is the desired covariance matrix. *L* can be created, for example, by using 
the Cholesky decomposition of *K*.


```python
n = 20      
m = 10

Xshow = np.linspace(0, 1, n).reshape(-1,1)  # n number test points in the range of (0, 1)

K_ = kernel(Xshow, Xshow)                

L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))


f_prior = np.dot(L, np.random.normal(size=(n,m)))

plt.clf()
plt.plot(Xshow, f_prior, '-o')
plt.title('10 samples of the 20-D gaussian kernelized prior')
plt.show()
```


![png](./img/codes_plot_output/output_44_0.png)


## III. Math

First, again, going back to our task regression. There is a function <img src="./tex/47b0192f8f0819d64bce3612c46d15ea.svg?invert_in_darkmode&sanitize=true" align=middle width=7.56844769999999pt height=22.831056599999986pt/> we are trying to model given a set of data points <img src="./tex/d05b996d2c08252f77613c25205a0f04.svg?invert_in_darkmode&sanitize=true" align=middle width=14.29216634999999pt height=22.55708729999998pt/> (trainig data/existing observed data) from the unknow function <img src="./tex/47b0192f8f0819d64bce3612c46d15ea.svg?invert_in_darkmode&sanitize=true" align=middle width=7.56844769999999pt height=22.831056599999986pt/>. The traditional non-linear regression machine learning methods typically give one function that it considers to fit these observations the best. But, as shown at the begining, there can be more than one funcitons fit the observations equally well. 

Second, let's review what we got from MVN. We got the feeling that when the dimension of Gaussian is infinite, we can sample all the region of interest with random functions. These infinite random functions are MVN because it's our assumption (prior). More formally, the prior distribution of these infinite random functions are MVN. The prior distribution representing the kind out outputs <img src="./tex/47b0192f8f0819d64bce3612c46d15ea.svg?invert_in_darkmode&sanitize=true" align=middle width=7.56844769999999pt height=22.831056599999986pt/> that we expect to see over some inputs <img src="./tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> without even observing any data. 

When we have observation points, instead of infinite random functions, we only keep functions that are fit these points. Now we got our posterior, the current belief based on the existing observations. When we have more observation points, we use our previous posterior as our prior, use these new observations to update our posterior.  

This is **Gaussian process**. 

***A Gaussian process is a probability distribution over possible functions that fit a set of points.***

Because we have the probability distribution over all possible functions, we can caculate **the means as the function**, and caculate the variance to show how confidient when we make predictions using the function. 

Keep in mind, 
* The functions(posterior) updates with new observations. 
* The mean calcualted by the posterior distribution of the possible functions is the function used for regression. 

**Highly recommend to read Appendix A.1 and A.2 [3] before continue.** Basic math. 

The function is modeled by a multivarable Gaussian as 

<p align="center"><img src="./tex/719e9a6916bc1b80643e47b9e7c2f23a.svg?invert_in_darkmode&sanitize=true" align=middle width=154.94836005pt height=16.438356pt/></p>

where <img src="./tex/b1b4b3aa35f58e460cbd61a979593145.svg?invert_in_darkmode&sanitize=true" align=middle width=152.0642508pt height=24.65753399999998pt/>, <img src="./tex/d9dbac0b5d4059824ba8c99ee4a515a9.svg?invert_in_darkmode&sanitize=true" align=middle width=165.36337619999998pt height=24.65753399999998pt/> and <img src="./tex/644b2bf06bea8f12e9df49b87db4b61d.svg?invert_in_darkmode&sanitize=true" align=middle width=109.37198744999999pt height=24.65753399999998pt/>. <img src="./tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> is the mean function and it is common to use <img src="./tex/7f7cffcdf59b071067d26657f33d1c2a.svg?invert_in_darkmode&sanitize=true" align=middle width=67.33249049999998pt height=24.65753399999998pt/> as GPs are flexible enough to model the mean arbitrarily well. <img src="./tex/5c62da39aa7289df62d937cb24a31161.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=14.15524440000002pt/> is a positive definite *kernel function* or *covariance function*. Thus, a Gaussian process is a distribution over functions whose shape (smoothness, ...) is defined by <img src="./tex/558e1b6b0d61666c16dd87622253a301.svg?invert_in_darkmode&sanitize=true" align=middle width=14.817277199999989pt height=22.55708729999998pt/>. If points <img src="./tex/c416d0c6d8ab37f889334e2d1a9863c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.628015599999989pt height=14.611878600000017pt/> and <img src="./tex/796df3d6b2c0926fcde961fd14b100e7.svg?invert_in_darkmode&sanitize=true" align=middle width=16.08162434999999pt height=14.611878600000017pt/> are considered to be similar by the kernel the function values at these points, <img src="./tex/bb17decf13a9637a3f7cbda1e1e8cc63.svg?invert_in_darkmode&sanitize=true" align=middle width=38.05275869999999pt height=24.65753399999998pt/> and <img src="./tex/df64afdc323a31f0579a7423dffde0ca.svg?invert_in_darkmode&sanitize=true" align=middle width=39.50636579999999pt height=24.65753399999998pt/>, can be expected to be similar too. 

So, we have observations, and we have estimated functions <img src="./tex/47b0192f8f0819d64bce3612c46d15ea.svg?invert_in_darkmode&sanitize=true" align=middle width=7.56844769999999pt height=22.831056599999986pt/> with these observations. Now say we have some new points <img src="./tex/826b622228f5a0bd5f45ebf9e6765404.svg?invert_in_darkmode&sanitize=true" align=middle width=21.027360749999986pt height=22.55708729999998pt/> where we want to predict <img src="./tex/2eb55e8e14e6d3fe2a68fbbb8309823d.svg?invert_in_darkmode&sanitize=true" align=middle width=44.45210054999999pt height=24.65753399999998pt/>.

<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/mvn.png?raw=1" width="250"/> 

The joint distribution of <img src="./tex/47b0192f8f0819d64bce3612c46d15ea.svg?invert_in_darkmode&sanitize=true" align=middle width=7.56844769999999pt height=22.831056599999986pt/> and <img src="./tex/370055884e7a54d58c6a734d6a7b2a84.svg?invert_in_darkmode&sanitize=true" align=middle width=12.511420349999991pt height=22.831056599999986pt/> can be modeled as:

<p align="center"><img src="./tex/0cb4940c22af62831e65276a3af113bd.svg?invert_in_darkmode&sanitize=true" align=middle width=280.5386265pt height=39.452455349999994pt/></p>

where <img src="./tex/566b2cb05c0a99e03f7e1ba12945ce8a.svg?invert_in_darkmode&sanitize=true" align=middle width=94.88167094999999pt height=24.65753399999998pt/>, <img src="./tex/84f51d8b5ed141f9fda1a5cd161229db.svg?invert_in_darkmode&sanitize=true" align=middle width=109.99584749999998pt height=24.65753399999998pt/> and <img src="./tex/e198d4bc165f68b54a9d6e79f2f4affe.svg?invert_in_darkmode&sanitize=true" align=middle width=124.28810789999997pt height=24.65753399999998pt/>. And <img src="./tex/49ee6a80c122f4ff5148bcc61160568c.svg?invert_in_darkmode&sanitize=true" align=middle width=104.63843609999998pt height=47.6716218pt/>

This is modeling a joint distribution <img src="./tex/a57e2662285457919869713fb6eb66b1.svg?invert_in_darkmode&sanitize=true" align=middle width=102.75646919999998pt height=24.65753399999998pt/>, but we want the conditional distribution over <img src="./tex/370055884e7a54d58c6a734d6a7b2a84.svg?invert_in_darkmode&sanitize=true" align=middle width=12.511420349999991pt height=22.831056599999986pt/> only, which is <img src="./tex/c1a328f9cc7d846af9a80807d260088f.svg?invert_in_darkmode&sanitize=true" align=middle width=102.75646919999998pt height=24.65753399999998pt/>. The derivation process from the joint distribution <img src="./tex/a57e2662285457919869713fb6eb66b1.svg?invert_in_darkmode&sanitize=true" align=middle width=102.75646919999998pt height=24.65753399999998pt/> to the conditional <img src="./tex/c1a328f9cc7d846af9a80807d260088f.svg?invert_in_darkmode&sanitize=true" align=middle width=102.75646919999998pt height=24.65753399999998pt/> uses the **Marginal and conditional distributions of MVN** theorem [5].   

<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/mvn_theorem.png?raw=1" width="420"/> 

We got eqn. 2.19 [1] 
<p align="center"><img src="./tex/f4b3b11509dda9897b014202bf8a7df3.svg?invert_in_darkmode&sanitize=true" align=middle width=320.0444214pt height=18.7598829pt/></p>

It is realistic modelling situations that we do not have access to function values themselves, but only noisy versions thereof <img src="./tex/fd50d91508dfe8bf9a34b62d915f3f90.svg?invert_in_darkmode&sanitize=true" align=middle width=89.32825439999999pt height=24.65753399999998pt/>. Assuming additive independent identically distributed Gaussian noise with
variance <img src="./tex/0ffac17b7e0be52ed29050057d2f8813.svg?invert_in_darkmode&sanitize=true" align=middle width=17.519130749999988pt height=26.76175259999998pt/>, the prior on the noisy observations becomes <img src="./tex/483b1de0709e4c80f0688f7a551102e6.svg?invert_in_darkmode&sanitize=true" align=middle width=127.41040454999998pt height=26.76175259999998pt/>. The joint distribution of the observed target values and the function values at the test locations under the prior as [1]

<p align="center"><img src="./tex/860aa5e083826aed334c17131c32fa14.svg?invert_in_darkmode&sanitize=true" align=middle width=249.22769025pt height=39.452455349999994pt/></p>
Deriving the conditional distribution corresponding to eqn. 2.19 we get the predictive equations (eqn. 2.22, eqn. 2.23, and eqn. 2.24) [1] for Gaussian process regression as

<p align="center"><img src="./tex/3549998db32a2708908ef327f2b3a965.svg?invert_in_darkmode&sanitize=true" align=middle width=211.34687805000002pt height=19.726228499999998pt/></p>

where, 
<p align="center"><img src="./tex/fc320bd7465eb24927e2949902ac551d.svg?invert_in_darkmode&sanitize=true" align=middle width=284.2016694pt height=23.482771949999997pt/></p>

<p align="center"><img src="./tex/2046f09dc8dbbcef6eed2880b9c2f469.svg?invert_in_darkmode&sanitize=true" align=middle width=253.81778235pt height=20.95157625pt/></p>

## IV. Codes 

We do the regression example between -5 and 5. The observation data points (traing dataset) are generated from a uniform distribution between -5 and 5. This means any point value within the given interval [-5, 5] is equally likely to be drawn by uniform. The functions will be evaluated at `n` evenly spaced points between -5 and 5. We do this to show a continuous function for regression in our region of interest [-5, 5]. This is a simple example to do GP regression. It assumes a zero mean GP Prior. The code borrows heavily from Dr. Nando de Freitas’ Gaussian processes for nonlinear regression [lecture](https://youtu.be/4vGiHC35j9s) [6].

The algorithm executed follows 

The textbook [GPML](http://gaussianprocess.org/gpml/chapters/RW.pdf), P19. [1]
<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/gp1.png?raw=1" width="600"/> 

Dr. Nando de Freitas, [Introduction to Gaussian processes](https://youtu.be/4vGiHC35j9s). [6]
<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/gp.png?raw=1" width="500"/>


```python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()
x = np.arange(-5, 5, 0.1)

plt.plot(x, f(x))
plt.axis([-5, 5, -3, 3])
plt.show()
```


![png](./img/codes_plot_output/output_59_0.png)



```python
# Define the kernel
def kernel(a, b):
    kernelParameter_l = 0.1
    kernelParameter_sigma = 1.0
    sqdist = np.sum(a**2,axis=1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    # np.sum( ,axis=1) means adding all elements columnly; .reshap(-1, 1) add one dimension to make (n,) become (n,1)
    return kernelParameter_sigma*np.exp(-.5 * (1/kernelParameter_l) * sqdist)
```

We use a general Squared Exponential Kernel, also called Radial Basis Function Kernel or Gaussian Kernel:

<p align="center"><img src="./tex/334845bbe48024b4b5bec361bca3d0ad.svg?invert_in_darkmode&sanitize=true" align=middle width=320.7560268pt height=32.990165999999995pt/></p>

where <img src="./tex/127335a9fdbeac98b11620d6cfd8ecea.svg?invert_in_darkmode&sanitize=true" align=middle width=17.09297369999999pt height=14.15524440000002pt/> and <img src="./tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/> are hyperparameters. More information about the hyperparameters can be found after the codes. 


```python
# Sample some input points and noisy versions of the function evaluated at
# these points. 
N = 20         # number of existing observation points (training points).
n = 200        # number of test points.
s = 0.00005    # noise variance.

X = np.random.uniform(-5, 5, size=(N,1))     # N training points 
y = f(X) + s*np.random.randn(N)

K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))     # line 1 

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(X, Xtest))   # k_star = kernel(X, Xtest), calculating v := l\k_star
mu = np.dot(Lk.T, np.linalg.solve(L, y))    # \alpha = np.linalg.solve(L, y) 

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)                  # k(x_star, x_star)        
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)   
s = np.sqrt(s2)

# PLOTS:
plt.figure(1)
plt.clf()
plt.plot(X, y, 'k+', ms=18)
plt.plot(Xtest, f(Xtest), 'b-')
plt.gca().fill_between(Xtest.flat, mu-2*s, mu+2*s, color="#dddddd")
plt.plot(Xtest, mu, 'r--', lw=2)
#plt.savefig('predictive.png', bbox_inches='tight', dpi=300)
plt.title('Mean predictions plus 2 st.deviations')
plt.show()
#plt.axis([-5, 5, -3, 3])
```


![png](./img/codes_plot_output/output_62_0.png)



```python
# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,40)))  # size=(n, m), m shown how many posterior  
plt.figure(3)
plt.clf()
plt.figure(figsize=(18,9))
plt.plot(X, y, 'k+', markersize=20, markeredgewidth=3)
plt.plot(Xtest, mu, 'r--', linewidth=3)
plt.plot(Xtest, f_post, linewidth=0.8)
plt.title('40 samples from the GP posterior, mean prediction function and observation points')
plt.show()
#plt.axis([-5, 5, -3, 3])
#plt.savefig('post.png', bbox_inches='tight', dpi=600)
```


    <Figure size 432x288 with 0 Axes>



![png](./img/codes_plot_output/output_63_1.png)


We plotted `m=40` samples from the Gaussian Process posterior together with the mean function for prediction and the observation data points (training dataset). It's clear all posterior functions collapse at all observation points. 

The general RBF kernel:

<p align="center"><img src="./tex/334845bbe48024b4b5bec361bca3d0ad.svg?invert_in_darkmode&sanitize=true" align=middle width=320.7560268pt height=32.990165999999995pt/></p>

where <img src="./tex/127335a9fdbeac98b11620d6cfd8ecea.svg?invert_in_darkmode&sanitize=true" align=middle width=17.09297369999999pt height=14.15524440000002pt/> and <img src="./tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/> are hyperparameters. [7]
<img src="https://github.com/jwangjie/Gaussian-Process-be-comfortable-using-it/blob/master/img/hyperparameter.png?raw=1" width="750"/>

More complex kernel functions can be selected to depend on the specific tasks. More information about choosing the kernel/covariance function for a Gaussian process can be found in `The Kernel Cookbook` [8]. 

## V. GP Packages

There are several packages or frameworks available to conduct Gaussian Process Regression. In this section, I will summarize my initial impression after trying several of them written in Python. 

A lightweight one is [sklearn.gaussian_process](https://scikit-learn.org/stable/modules/gaussian_process.html), simple implementation like the example above can be quickly conducted. Just for gaining more implementation understandings of GP after the above simple implementation example. It's too vague for understanding GP theory purpose. 

GPR is computationally expensive in high dimensional spaces (features more than a few dozens) due to the fact it uses the whole samples/features to do the predictions. The more observations, the more computations are needed for predictions. A package that includes state-of-the-art algorithm implementations is preferred for efficient implementation of complex GPR tasks.

One of the most well-known GP frameworks is [GPy](https://sheffieldml.github.io/GPy/). GPy has been developed pretty maturely with well-documented explanations. GPy uses NumPy to perform all its computations. For tasks that don't require heavy computations and very up-to-date algorithm implementations, GPy is sufficient and the more stable. 

For bigger computation required GPR tasks, GPU acceleration are especially preferred. [GPflow](https://www.gpflow.org/) origins from GPy, and much of the interface is similar. GPflow leverages **TensorFlow** as its computational backend. More technical difference between GPy and GPflow frameworks is [here](https://gpflow.readthedocs.io/en/master/intro.html#what-s-the-difference-between-gpy-and-gpflow). 

[GPyTorch](https://gpytorch.ai/) is another framework that provides GPU acceleration through **PyTorch**. It contains very up-to-date GP algorithms. Similar to GPflow, GPyTorch provides automatic gradients. So complex models such as embedding deep NNs in GP models can be easier developed. 

After going through docs quickly and implementing basic GPR tutorials of [GPyTorch](https://github.com/jwangjie/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb) and [GPflow](https://github.com/jwangjie/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression_GPflow.ipynb), my impression is using GPyTorch is more automatic and GPflow has more controls. The impression may also come from the usage experience with TensorFlow and PyTorch. 

Check and run my ***modified*** GPR tutorials of 
* [GPyTorch](https://github.com/jwangjie/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb) 

* [GPflow](https://github.com/jwangjie/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression_GPflow.ipynb)

## VI. Summary

A Gaussian process (GP) is a probability distribution over possible functions that fit a set of points. [1] GPs are nonparametric models that model the function directly. Thus, GP provides a distribution (with uncertainty) for the prediction value rather than just one value as the prediction. In robot learning, quantifying uncertainty can be extremely valuable to achieve an efficient learning process. The areas with least certain should be explored next. This is the main idea behind Bayesian optimization. [9] Moreover, prior knowledge and specifications about the shape of the model can be added by selecting different kernel functions. [1] Priors can be specified based on criteria including if the model is smooth, if it is sparse, if it is able to change drastically, and if it need to be differentiable.

### Extra words

1. For simplicity and understanding reason, I ignore many math and technical talks. Read the first two chapters of the textbook `Gaussian Process for Machine Learning` [[1](#Reference)] serveral times to get a solid understanding of GPR. Such as **Gaussian process regression is a linear smoother. **

2. One of most tricky part in understanding GP is the mapping projection among **spaces**. From input space to latent (feature) space and back to output space. You can get some feeling about space by reading [`autoencoder`](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d).

## Reference

[1] C. E. Rasmussen and C. K. I. Williams, Gaussian processes for machine learning. MIT Press, 2006.

[2] R. Turner, “ML Tutorial: Gaussian Processes - YouTube,” 2017. [Online]. Available: https://www.youtube.com/watch?v=92-98SYOdlY&feature=emb_title.

[3] A. Ng, “Multivariate Gaussian Distribution - Stanford University | Coursera,” 2015. [Online]. Available: https://www.coursera.org/learn/machine-learning/lecture/Cf8DF/multivariate-gaussian-distribution.

[4] D. Lee, “Multivariate Gaussian Distribution - University of Pennsylvania | Coursera,” 2017. [Online]. Available: https://www.coursera.org/learn/robotics-learning/lecture/26CFf/1-3-1-multivariate-gaussian-distribution.

[5] F. Dai, Machine Learning Cheat Sheet: Classical equations and diagrams in machine learning. 2017.

[6] N. de Freitas, “Machine learning - Introduction to Gaussian processes - YouTube,” 2013. [Online]. Available: https://www.youtube.com/watch?v=4vGiHC35j9s&t=1424s.

[7] Y. Shi, “Gaussian Process, not quite for dummies,” 2019. [Online]. Available: https://yugeten.github.io/posts/2019/09/GP/.

[8] D. Duvenaud, “Kernel Cookbook,” 2014. [Online]. Available: https://www.cs.toronto.edu/~duvenaud/cookbook/.

[9] Y. Gal, “What my deep model doesn’t know.,” 2015. [Online]. Available: http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html.

[10] J. Hensman, "Gaussians." 2019. [Online]. Available: https://github.com/mlss-2019/slides/blob/master/gaussian_processes/presentation_links.md.

[11] Z. Dai, "GPSS2019 - Computationally efficient GPs" 2019. [Online]. Available: https://www.youtube.com/watch?list=PLZ_xn3EIbxZHoq8A3-2F4_rLyy61vkEpU&v=7mCfkIuNHYw.


## Appendix A

Visualizing 3D plots of a <img src="./tex/7c91fa1fa7be856b248f729bd78b5f6f.svg?invert_in_darkmode&sanitize=true" align=middle width=42.376631549999985pt height=22.465723500000017pt/> Gaussian by [Visualizing the bivariate Gaussian distribution](https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/). 


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[ 1. , 0.8], [0.8,  1.]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.2, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.2,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(30, -100)

ax.set_xlabel(r'<img src="./tex/277fbbae7d4bc65b6aa601ea481bebcc.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/>')
ax.set_ylabel(r'<img src="./tex/95d239357c7dfa2e8d1fd21ff6ed5c7b.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/>')
ax.set_zlabel(r'<img src="./tex/467f6c046e010c04cafe629aaca84961.svg?invert_in_darkmode&sanitize=true" align=middle width=66.46698464999999pt height=24.65753399999998pt/>')

plt.title('mean, cov = [0., 1.], [(1., 0.8), (0.8, 1.)]')
plt.savefig('2d_gaussian3D_0.8.png', dpi=600)
plt.show()
```


![png](/img/codes_plot_output/output_72_0.png)

