# machine-learning-1-week-4-fisher-discriminant-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 1 Week 4-Fisher Discriminant Solved](https://www.ankitcodinghub.com/product/machine-learning-1-week-4-fisher-discriminant-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98751&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;4&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (4 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 1 Week 4-Fisher Discriminant Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (4 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Exercise 1: Fisher Discriminant (10 + 10 + 10 P)

The objective function to find the Fisher Discriminant has the form max w⊤SBw

w w⊤SWw

where SB = (m2 − m1) (m2 − m1)⊤ is the between-class scatter matrix and SW is within-class scatter matrix, assumed to be positive definite. Because there are infinitely many solutions (multiplying w by a scalar doesn’t change the objective), we can extend the objective with a constraint, e.g. that enforces w⊤SW w = 1.

<ol>
<li>(a) &nbsp;Reformulate the problem above as an optimization problem with a quadratic objective and a quadratic constraint.</li>
<li>(b) &nbsp;Show using the method of Lagrange multipliers that the solution of the reformulated problem is also a solution of the generalized eigenvalue problem:
SBw = λSW w
</li>
<li>(c) &nbsp;Show that the solution of this optimization problem is equivalent (up to a scaling factor) to
w⋆ = S−1(m1 − m2) W

Exercise 2: Bounding the Error (10 + 10 P)

The direction learned by the Fisher discriminant is equivalent to that of an optimal classifier when the class- conditioned data densities are Gaussian with same covariance. In this particular setting, we can derive a bound on the classification error which gives us insight into the effect of the mean and covariance parameters on the error.
</li>
</ol>
Consider two data generating distributions P (x|ω1) = N (μ, Σ) and P (x|ω2) = N (−μ, Σ) with x ∈ Rd. Recall that the Bayes error rate is given by:

􏰆

P (error) =

<ol>
<li>(a) &nbsp;Show that the conditional error can be upper-bounded as:
P (error|x) ≤ 􏰇P (ω1|x)P (ω2|x)
</li>
<li>(b) &nbsp;Show that the Bayes error rate can then be upper-bounded by:
P (error) ≤ 􏰇P (ω1 )P (ω2 ) · exp 􏰃 − 21 μ⊤ Σ−1 μ􏰄 Exercise 3: Fisher Discriminant (10 + 10 P)

Consider the case of two classes ω1 and ω2 with associated data generating probabilities 􏰁􏰁−1􏰂 􏰁2 0􏰂􏰂 􏰁􏰁+1􏰂 􏰁2 0􏰂􏰂

p(x|ω1)=N −1 , 0 1 and p(x|ω2)=N +1 , 0 1
</li>
</ol>
<ol>
<li>(a) &nbsp;Find for this dataset the Fisher discriminant w (i.e. the projection y = w⊤x under which the ratio between
inter-class and intra-class variability is maximized).
</li>
<li>(b) &nbsp;Find a projection for which the ratio is minimized.
Exercise 4: Programming (30 P)

Download the programming files on ISIS and follow the instructions.
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
x

</div>
</div>
<div class="layoutArea">
<div class="column">
P (error|x) p(x) dx

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 4 (programming) [WiSe 2021/22] Machine Learning 1

</div>
</div>
<div class="layoutArea">
<div class="column">
Fisher Linear Discriminant

In this exercise, we apply Fisher Linear Discriminant as described in Chapter 3.8.2 of Duda et al. on the UCI Abalone dataset. A description of the dataset is given at the page https://archive.ics.uci.edu/ml/datasets/Abalone (https://archive.ics.uci.edu/ml/datasets/Abalone). The following two methods are provided for your convenience:

utils.Abalone.__init__(self) reads the Abalone data and instantiates two data matrices corresponding to: infant (I), non-infant (N). utils.Abalone.plot(self,w) produces a histogram of the data when projected onto a vector w , and where each class is shown in a different

color.

Sample code that makes use of these two methods is given below. It loads the data, looks at the shape of instantiated matrices, and plots the projection on the first dimension of the data representing the length of the abalone.

In [1]: %matplotlib inline

import utils,numpy # Load the data

<pre>abalone = utils.Abalone()
</pre>
<pre># Print dataset size for each class
</pre>
<pre>print(abalone.I.shape, abalone.N.shape)
</pre>
<pre># Project data on the first dimension
</pre>
<pre>w1 = numpy.array([1,0,0,0,0,0,0])
abalone.plot(w1,'projection on the first dimension (length)')
</pre>
<pre>(1342, 7) (2835, 7)
</pre>
Implementation (10 + 5 + 5 = 20 P)

Create a function w = fisher(X1,X2) that takes as input the data for two classes and returns the Fisher linear discriminant.

Create a function objective(X1,X2,w) that evaluates the objective defined in Equation 96 of Duda et al. for an arbitrary projection vector w .

Create a function z = phi(X) that returns a quadratic expansion for each data point x in the dataset. Such expansion consists of the vector x itself, to which we concatenate the vector of all pairwise products between elements of x . In other words, letting x = (x1, …, xd) denote the d-dimensional data point, the quadratic expansion for this data point is a d ⋅ (d + 3) / 2 dimensional vector given by

φ(x) = (xi)1 ≤ i ≤ d ∪ (xixj)1 ≤ i ≤ j ≤ d. For example, the quadratic expansion for d = 2 is (x1, x2, x21, x2, x1x2).

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
In [2]:

def fisher(X1,X2):

##### Replace by your code import solutions

return solutions.fisher(X1,X2) #####

def objective(X1,X2,w):

##### Replace by your code

import solutions

return solutions.objective(X1,X2,w) #####

def expand(X):

##### Replace by your code import solutions

return solutions.expand(X) #####

Analysis (5 + 5 = 10 P)

Printvalueoftheobjectivefunctionandthehistogramforseveralvaluesof w: w is a canonical coordinate vector for the first feature (length).

w is the difference between the mean vectors of the two classes.

w is the Fisher linear discriminant.

w is the Fisher linear discriminant (after quadratic expansion of the data). In [3]:

<pre>##### REPLACE BY YOUR CODE
</pre>
%matplotlib inline import solutions solutions.analysis() #####

</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>First dimension (length):  0.00048
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>          Means Linear:  0.00050
</pre>
<pre>                Fisher:  0.00057
</pre>
<pre>Fisher after expansion:  0.00077
</pre>
</div>
</div>
</div>
</div>
<div class="page" title="Page 5"></div>
<div class="page" title="Page 6"></div>
<div class="page" title="Page 7"></div>
