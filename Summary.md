<h1>Natural language processing - summary</h1>

__References__
- [Mastering NLP from Foundations to LLMs 2024 - By Lior Gazit, Meysam Ghaffari](https://learning.oreilly.com/library/view/mastering-nlp-from/9781804619186/)
- [Advance NLP Course - CMU 2024](https://phontron.com/class/anlp2024/)


__Table of contents__
- Overview



---

# Overview

NLP is a field of artificial intelligence (AI) focused on the interaction between computers and human languages. It involves using computational techniques to understand, interpret, and generate human language, making it possible for computers to understand and respond to human input naturally and meaningfully.

There are different approaches to solve a NLP task, inlcuding ruled-based, statistical, machine-learning, and deep-learning methods.

Preprocessing is an important step in nlp, which includes general preprocessing (e.g. removing white spaces, removing special characters, convert digit to word, ...), stemming (e.g. jumped -> jump). lammatization (e.g. ran->run), normalization (e.g. lowercase, ro removing punctuations), clearning (e.g. removing duplicate or irrelevant data and correcting errors or inconsistencies in the data). A pipeline for data preprocessing could be as follow:

raw text $\rightarrow$ remove encoding $\rightarrow$ lowe casing $\rightarrow$ digits to words $\rightarrow$ remove sspecial characters $\rightarrow$ spell correcting $\rightarrow$ stemming $\rightarrow$ lemmatizing

__Linear algebra__
- dot product
- Matrix transpose
- Symetric matrix
- Rectangular diagonal matrix
- Upper (or Lower) triangular matrix
- determinant: the determinant of a square matrix provides a notion of its impact on the volume of a d-dimensional object when multiplied by its co-ordinate vectors.
- Eigenvalues and eigenvectors: A vector x, belonging to a d × d matrix A, is an eigenvector if it satisfies the equation Ax = λx, where λ represents the eigenvalue associated with the matrix.

    In the case where A is a matrix that can be diagonalized, it can be deconstructed into a d × d invertible matrix, V, and a diagonal d × d matrix, Δ, such that: $A = V \Delta V^{-1}$. The columns of V encompass d eigenvectors, while the diagonal entries of Δ house the corresponding eigenvalues.

- Eigenvalue decomposition (also known as the eigen-decomposition or the diagonalization of a matrix): The goal of eigenvalue decomposition is to decompose a given matrix into a product of matrices that represent the eigenvectors and eigenvalues of the matrix.
- Value singular decomposition (SVD) is a mathematical technique that takes a rectangular matrix, A, and decomposes it into three matrices: $U$, $S$, and $V^{-1}$. The U matrix’s columns are known as the left singular vectors, while the rows of the transpose of the V matrix  are the right singular vectors. The S matrix, with singular values, is a diagonal matrix of the same size as A. SVD decomposes the original data into a co-ordinate system where the defining vectors are orthonormal (both orthogonal and normal).

__Probabilities__
- probability is a measure of the likelihood of an event occurring when an experiment is conducted.
- statistically independent events: P(A and B) = P(A)P(B)
- Complementary event: The complementary event to A, signified as A’, encompasses the probability of all potential outcomes in the sample space not included in A.
- Mutually exclusive: When two events have no shared outcomes, they are viewed as mutually exclusive.
- Independent: Two events are deemed independent when the occurrence of one doesn’t impact the occurrence of the other. $P(A \cap B) = P(A).P(B)$
- A discrete random variable refers to a variable that can assume a finite or countably infinite number of potential outcomes.
- The probability distribution of a discrete random variable assigns a certain likelihood to each potential outcome the variable could adopt.
- The probability density function (PDF) is a tool used to describe the distribution of a continuous random variable.
- Maximum likelihood is a statistical approach, that is used to estimate the parameters of a probability distribution. The objective is to identify the parameter values that maximize the likelihood of observing the data, essentially determining the parameters most likely to have generated the data.
- The maximum likelihood estimate (MLE) is the parameter vector value that offers the maximum value for the likelihood function across the parameter space.
- Bayesian estimation is a statistical approach that involves updating our beliefs or probabilities about a quantity of interest based on new data.


---

## Machine learning for NLP


### Data exploration

Some of the standard preprocessing steps are:
- visualization, enables visual exploration of data, getting insights into its distribution, patterns, and relationships with  scatter plots, bar charts, heatmaps, box plots, and correlation matrices.
- data cleaning, includes identify the errors, inconsistencies, and missing values and correct them, remove duplicates, and filling in missing values (dropping rows/columns, apply mean/mode/median, regression, multiple imputes, and k-NN).
- feature engineering, including scaling, normalization, dimensionality reduction, and feature selection, to identify pertinent features, transform existing ones, or generating novel features. Also dealing with outliers (removing, transforming, winsorizing, imputating, or using a robust methods (use median instead of mean)). 
- statistical analysis, including hypothesis testing, regression analysis, and time series analysis to understand data's characteristics.
- domain knwoledge, is a valuable asset in preprocessing that helps to extract insights and make informed decisions. It could help to recognize pertinent features, interpret results, and choose the most suitable algorithm for the task in hand.

__Feature selection__

The objective is to decrease the number of features without substantially compromising the model’s accuracy, resulting in enhanced performance, quicker training, and a more straightforward interpretation of the model.

Some of the techniques for featuer selection:
- filtering: employ statistical methods to rank features according to their correlation with the target variable (chi-squared, mutual information, and correlation coefficients), and select features based on predefined threshold.
- Chi-square test: to guage the dependence btw random variables. In feature selection, the chi-squared test evaluates the relationship between each feature and the target variable in the dataset. It determines significance based on whether a statistically significant difference exists between the observed and expected frequencies of the feature, assuming independence between the feature and target.
- Mutal information, as a metric to gauge te interdependence of two random variables. It quantifies the information a feature provides about the target variable.
- Correelation coefficient, serves as indicators of the strength and direction of the linear relationship between two variables. In the realm of feature selection, these coefficients prove useful in identifying features highly correlated with the target variable, thus serving as potentially valuable predictors.
- Wrapper methods, such as recursive feature elimination, use forward training and backward elimination to eliminate less informative features.
- Embedded methods such as LASSO and ridge regression, decision trees, and random forests, select features during the training process.
- Dimensionality reduction techniques, such as PCA, linear discriminant analysis (LDA), and t-SNE;  transform the features into a lower-dimensional space while retaining as much information as possible. 

Common ML models
Model underfitting and overfitting
Splitting data
Hyperparameter tuning
Ensemble models
Handling imbalanced data
Dealing with correlated data

__Feature engineering__


## Common machine learning methods
---

# Deep learning

DL can be supervised, unsupervise, or semi-supervised. One advantage of DL is its global capability, meaning it can process and model data of variety types, including text, image, and audio. The downside of DL is its performance is correlated to the size of the model, making it computationaly expensive to develop a high performance model. Furthermor, its performance is highly sensitive to the size of input data. 

Some of the advantage of a DL model:
- nonlinearity capability
- universal approaximation theorem states that with high enought hidden units, it can approximate virtually any function with a high degree of accuracy.
- ability to handle high dimensional data
- capability to recognize patterns and make prediction, given large enough dataset
- parallel processing
- learning from data, makes them highly effective as data size increases
- robusness, against noise in the input

With regards to NLP task, employ DL methods gives an edge due to the following reasons:
- ability to handle sequential data, whether using RNN models such as LSTM or GRUs, or transformers-based models.
- context understanding,
- semantic hashing, by employing word embedding (e.g. Word2Vec and GloVe), DL models can encode words while preserving their semantix meaning.
- end-to-end learning, eliminates some of the preporocessing steps and save time/computation
- peprformance, with the advance of transformers-based architectures (DeBERTA, GPT, ...) many NLP tasks such as text summarization, sentiment analysis, and question answering reach a pick performance.
- handling large vocabulary, employing various tokenization techniques, DL models could handle larage vocabulary and continuous text extreme.
- capability to learn hierarchical features, meaning lower layers learn simple things such as n-grams, while higher layers can represent complex concepts such as sentiments.

Typical computation steps in a DL model:
- weighted sum: each input $x$ is multiplied by a corresponding weight $w$m and summed with a bias term $b$.
- activation functin: a nonlinearity is introduced to the output of previousstep via an activation function such as ReLU or tanh.
- the weight and bias terms are learnable. During the forward pass a loss function computes loss based on the generated output and true target values. Then through the backward pass, the gradient of parameters w.r.t. loss is computated, and modification is applied by the optimization function. This iterative process continues until reaching a cap.

Some of the famouse activation functions
- sigmoid function: squashing the input into a range between 0 and 1. It has two drawbacks: 1. the vanishing gradients problem, and 2. the outputs are not zero-centered.
- tanh: squashes input into range between -1 and 1, which makes it zero centered. However, it still suffer from vanishing gradient problem.
- ReLU = $max(0,x)$: fixes the problem of vanishing gradient. But suffer from potential die issue.
- Leaky ReLU = $max(0..0x, x)$ fixed the _dying ReLU_ problem
- ELU
$\begin{equation}
\begin{cases}
x   \text{  if } x > 0 \\
\alpha(e^x - 1)   \text{ otherwise}
\end{cases}
\end{equation}$

$$\begin{equation}
\begin{cases}
\Delta_{0}=0,3\Delta_{1}+0,3\Delta_{0} \\ 
\Delta_{1}-5=0,2\Delta_{1}-0,2\Delta_{0}
\end{cases}
\end{equation}$$
 

## Classification with machine learning


---

## Classification with deep learning

---

## RAG

---

## LLM

---

## Frontier approaches!

---
---
# Advance NLP course - CMU 2024


