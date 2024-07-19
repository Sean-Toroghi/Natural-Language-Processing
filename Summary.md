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

## Overview
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
- ELU, adds a smoothness via $\alpha$ to return a non-zero value for negative input $x$

$$ELU  = \begin{equation}
\begin{cases}
x & x > 0 \\ 
\alpha (e^x -1) & \text{otherwise}
\end{cases}
\end{equation}$$

- softmax function: if the goal is to get a probability, we can employ softmax function: $f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$


__Challenges of training a neural network__
- local minima
- overfitting
- underfitting
- vanishing/exploding gradients
- computational resources
- lack of interpretability
- correctly format data
- zoo of model architectures

## Common NN architectures

- __feedForward neural netwrok (FNN)__ is the most straighforward type of a NN, in which information moves in one direction. 
- __MLP__ is a type of FNN with fully connected layers. It is also a feedforward network.
- __CNN__ consists of convolutional, pooling, and fully connected layers. 
- __RNN__ has connections that form directed cycles, allowing the network to use information from both inputs and previous outputs. 
- __LSTM__ is a variation of RNNs, which uses special units _memory cell_, in addition to standard units, that can maintain information in memory for long periods of time
- __Autoencoder (AE)__ is a type of neural network used to learn the efficient coding of input data. With a symmetrical architecture, it is designed to apply backpropagation and setting the target values to be equal to the inputs. Autoencoders are typically used for feature extraction, learning representations of data, and dimensionality reduction. They’re also used in generative models, noise removal, and recommendation systems.
- __Generative adversarial network (GAN)__: it consists of two parts, a generator and a discriminator. The generator creates data instances that aim to come from the same distribution as the training dataset. The discriminator’s goal is to distinguish between instances from the true distribution and instances from the generator. The generator and the discriminator are trained together, with the goal that the generator produces better instances as training progresses, whereas the discriminator becomes better at distinguishing true instances from generated ones.


# Language model

A language model is a statistical model in NLP that is designed to learn and understand the structure of human language. More specifically, it is a probabilistic model that is trained to estimate the likelihood of words when provided with a given word scenario.
During past five years a large number, and growing, number of language models are developed and introduced [ref](https://learning.oreilly.com/library/view/mastering-nlp-from/9781804619186/): 
![image](https://github.com/user-attachments/assets/6033886f-1733-44aa-8006-2c3913897174)



## Self-suprevised language models
__Maked language modeling__

By randomly masks some percentage of the input tokens and tasks the model with predicting the masked words based on the context provided by the unmasked words, the model is trained. BERT is an example of such a training approach.

__Autoregressive language modeling__ 

The model predicts the next word in a sentence given all the preceding words. It’s trained to maximize the likelihood of a word given its previous words in the sentence. GPT is an example of such models that employs this technique for its training.

## Transfer learning
Transfer learning is an ML technique where a pretrained model is reused as the starting point for a different but related problem. 

In transfer learning, a model is typically trained on a large-scale task, and then parts of the model are used as a starting point for another task.

## Quick over of transformers architecture
- model calculates the relevance of each word in the sequence to the current workd being processed.
- the input is a sequence of word embeddings, each of which split into Q,K,V using seperately learned transformations.
- the attenction for each word is computed as follow: $score(Q<K<V) = SoftMax \frac{QK_T}{\sqrt{d_k}}$
- attenction scores represent the weight given to each word’s value when producing the output for the current word.
- The output of the self-attention layer is a new sequence of vectors, where the output for each word is a weighted sum of all the input values, with the weights determined by the attention scores.
- To consider position of the word in the sequence as a part of input information, positional encoding is added to the input embeddings. 

## Tokenization
To make a language model more efficient, the input first need to be converted into a limited number of tokens, called tokenization. Some of the tokinzation algorithms are byte pair encoding (BPE), unigram language model (ULM), and WordPiece. They split words into smaller subword units.

## n-gram models
Simpler language models that use $n-1$ prior words to predict the $n^{th}$ word in a sequence. They are easy to implement and computationaly efficient, in the cost of performance. They lack capturing long-range deeondencies that is an important characteritic of a human language. As the result, their performance degtrades as the size of sequence increases. 

## Hidden Markov models
These models consider the “hidden” states that generate the observed data. In the context of language modeling, each word would be an observed state, and the “hidden” state would be some kind of linguistic feature that’s not directly observable (such as the part of speech of the word). However, like n-gram models, HMMs struggle to capture long-range dependencies between words.

## Recurrent neural network models
As a type of neural network approach, these models benefits from their internal state (memory) to process sequences of inputs and capturing long-range dependencies between words. However, they suffer from vanishing gradient problem.

## LSTM models
As a special type of RNN, LSTM addresses the long-range dependency requirement for modeling a language by adding series if gates that control the flow of information into and out of memory states in the network. 

## Gated recurrent unit (GRU) networks
As a variation of LSTM, GRU uses gates slightly different. Based on the task in hand they could perform better or worse than LSTM.

## Transformer-based model: BERT
Based on transformer model architecture, BERT is designed to pretrain deep bidirectional representations from the unlabeled text by joint conditioning on both left and right contexts in all layers. 

BERT tokenizer is a critical part of the model, which uses WordPiece tokenization. The steps BERT tokenizer takes are as following:
1. basic tokenization, which consists of breaking text into individdual words by spliting on whitespace and punctuation.
2. WordPeice tokenization: in this step, words are breaking down into subwords, until it finds a match in vocabulary of reaches character-level representation.
3. additing special tokens, including [cls] to the beginning and [sep] to the end of each sentence.
4. token to ID conversion is the last step, in which the tokens are mapped to integer ID corresponding to its index in the BERT vocabulary.

### BERT pretraining
During the pretraining stage, BERT was trained on large corpus of text, and was tasked to predict masked words and also distinguish whether two sentences come in order in the text. 

### BERT fine-tuning
After pretraining stage, BERT can be fined tuneed on a spesific task with a small size data (compare with the original dataset used for training it). 

Steps that are needed to be taken for fine-tuning BERT, and to larger extend any language model, are as follow:
1. preprocess input data to match the specific format that was originaly used to train the model. This includes using BERT tokenizer, adding special tokens, and pad/truncate sequences to match a uniform input size.
2. loading the pretrained model that matches the task in hand is the second step. Models differin size and language of the pretraining data.
3. adding a classification layer (classificaton head), a fully connected layer, is then added on top of the pretrained model, to make predictions for the text classification task. The input to this layer is the [cls] token and its outputs is the probability distribution over the classes.
4. fine-tune the model is the next step, which includes train the model on the specific task, using labeled data. A common approach is to update the weights of the pretrained BERT model and the newly added classification layer to minimize a loss function, typically the cross-entropy loss for classification tasks, with lower learning rate value and two to four epochs.
5. evaluating the fine-tuned model with eval set and fine-tune hyperparameters is th e next steps. Metrics such as F1-score, accuracy, recall, and precision are used for this evaluation.
6. Finally the fine-tuned model is used to make prediction.

__Note__: seq lenght fir BERT and its sisters is 512 tikens. In a case that requires longer input sequences, other models such as Longformer or BigBird are good choices.




## GPT-x
Generative pretrained transformer (GPT), is an autoregressive language model developed by OpenAI that uses DL techniques to generate human-like text. It comes in different version, the most up-to-dated one is 4 in 2024. GPT, unlike BERT, employs decoder part of the transformer architecture.

GPT, unlike BERT, processes input data sequentially from left to right and generates predictions for the next item in the sequence. 
### GPT pretraining
Similar to BERT, GPT was trained on a large corpus of text, and learned to predict the next word in a sentence. However, it only uses the left context to make the prediction.

### Fine-tuning GPT
The fine-tuning og GPT could be done based on a range of different tasks such as text completion, translation, summarization, question answering, and so on. One feature of GPT=x models is their capability of zero-shot, one-shot, and few-shot learning.

In the __zero-shot__ setting, the model is given a task without any prior examples. In the **one-shot** setting, it’s given one example, and in the **few-shot** setting, it’s given a few examples to learn from.

__NOTE__: GPT models have difficulty in learning tasks that requrie deep understanding of the words or common sense reasoning beyond what can be learned from text.

## Modeling pipeline

Developing a NLP-solution for a real-world problem requires to define a pipeline. The following diagram shows an example of a pipeline:

![image](https://github.com/user-attachments/assets/b845cd6d-1bc7-4292-848a-df244b33cd86)


---
# Large language models

LLMs are a subset of artificial intelligence (AI) models that can understand and generate human-like text.

Language models such as the newer versions of GPT models (3 and above) are considered as large language models. They are huge in size, and are trained on a very large dataset. Their performace correlated with their size and size of the data they trained on. LLMs are sessentially scaled-up versions of smaller language models. 

The improvement in performance of LLMs relies on diversity and size of the data they were trained on, which makes them capable of understanding context, identifying nuances, and generating coherent and contextually relevant responses. 

LLMs also are better in generalizing across different tasks, domain, or languages.

As the size of LLMs increases, they impose challenge of handling computational cost and also higher risk of overfitting. 

__challenges in developing LLMs__
- handling massive amounts of data,
- requiring vast computational resources,
- risk of introducing or perpetuating bias
- model robustness is not guaranteed. When dealing with ambiguous queries or out-of-distribution data, LLMs may not demonstrate consistant outputs.

## Different types of LLMs
LLMs are based on the “Transformer” architecture, are capable of processing sequenes, udnerstand long-range dependencies, speed up computation via parallelization, and come in variety of designs, each suitable for specific range of tasks.

## pre-training, fine-tuning, and RHLF
__Pre-training__:

At this stage, the model is trained on a large corpus of publicly available text from the internet, while it does not know specifics about which documents were in its training set or have access to any specific documents or sources.


__Fine-tuning__: the base model is further trained (fine-tuned) on custom datasets, which include demonstrations of correct behavior as well as comparisons to rank different responses.

__RHLF__: as a part of the fine-tuning process involves RLHF, where human AI trainers provide feedback on model outputs for a range of example inputs, and this feedback is used to improve the model’s responses. 

## RHLF
RHLF is an iterative process and follows these steps:
1. collect human feedback.
2. comparison data is created, where multiple model responses are ranked by quality by human.
3. The model is then fine-tuned using proximal policy optimization (PPO), a reinforcement learning algorithm.

   PPO attempts to improve the model’s responses based on human feedback, making small adjustments to the model’s parameters to increase the likelihood of better-rated responses and decrease the likelihood of worse-rated responses.


## PPO
PPO is a reinforcement learning algorithm used to optimize the π policy of an agent. PPO aims to optimize the policy to maximize the expected cumulative rewards. The goal of the agent is to learn a policy π that maximizes the expected sum of rewards.

__Policy__: the policy defines how the agent selects actions based on its current state.

__Reward model__: the reward model is a R(s, a) function, which assigns a reward value to every state-action pair (s, a).

__Objective of reinforcement learning__: $J(\pi) = \mathbb{E} \big(\sum R(s_t, a_t) \big)$ , where 
- $\mathbb{E}(.)$ is the expectation over trajectories (sequences of state-action pairs) generated by following policy π
- $s_t$ is the state at time $t$
- $a_t$ is the actiona taken at time $t$
- $R(s_t, a_t)$ is the reward received at time $t$

__Objective of PPO__: PPO modifies the objective via $r_t (\theta)$, which represents the ratio of the probabilities of the current policy to the old policy. Employ stochastiuc gradient ascent, the algorithm adjust its policy parameters to increase the objetive function.

__In the context of language model__
- states correspond  to the conversation history
- actions correspond to the model-generated messages
- rward correspond to the human feedback on the generated messages

## Implementing RHLF
Recent approach to utilize RHLF is to use it as a reward model, also lkknwon as preference model, to a text and return a scalar reward that reflexts human preference. There are two ways to implement this approach:
1. implement an end-to-end LLM
2. design an extra component that ranks different outputs of the LLM and erturns the best one.

The dataset for training the reward model is a set of prompt-generation pairs. Then human annotators assign score to each text. 

__Implementing RHLF__
1. design and train reward model
   
    There are several approaches for text ranking, among which is the Elo ranking system, which gnenerates a ranking of models and outputs relative to each other. The implementation of Elo ranking for RHLF in LMs is as follow:
    - initialization: all models starts with the same Elo rating (1000 or 1500)
    - comparison, for a given prompt, two models generate tow outputs (A and B). A human evaluator ranks the two outputs
    - the Elo rating is then updated after each evaluation.
2. apply reinforcement learning to optimize the original LM concerning the reward model.
3. Fine-tune:
    - Initially, the policy is an LM that accepts a prompt and produces a sequence of text (or merely probability distributions over text).
    - The action space of this policy is all the tokens aligning with the LM’s vocabulary (typically around 50 K tokens), and
    - the observation space is the distribution of possible input token sequences, which is also notably large in light of reinforcement learning’s prior uses (the dimension approximates the vocabulary size power (^) length of the input token sequence).
    - The reward function melds the preference model with a constraint on policy shift.

### Some open source RHLF 
__GPT-2__

OpenAI released the RHLF for GPT-2 [link](https://github.com/openai/lm-human-preferences) in 2019.

__TRL__

A crafted tool for fine0tuning a pre-trained LM using PPO within hugginface ecosystem [link](https://github.com/lvwerra/trl).

__TRLX__

An enhanced version of TRL, by CarperAI, equipped with production ready API supporting RHLF with PPO and implicit language Q-learning (ILQL) for deploying LLMs of up to 33 billion parameters [link](https://github.com/CarperAI/trlx). The newer version supports up to 200B parameters.

__Reinforcement Learning for Language Models (RL4LMs)__

The RL4LMs project addresses the challenge of training LLMs to align with human preference metrics [link](https://github.com/allenai/RL4LMs), by doing the following:
- Giving guidelines on when to use reinforcement learning and suggesting suitable NLP tasks/metrics via a continually updated benchmark called GRUE

- Introducing a new reinforcement learning algorithm, Natural Language Policy Optimization (NLPO), designed to handle large language action spaces and reward variance better
- Offering practical advice with high-quality implementations and hyperparameters of reinforcement learning, as well as other reinforcement learning algorithms, for training Transformers in the Hugging Face library

## LLaMA 
LLaMA is a LLM designed by Meta, with low computation power requirements. LLaMA takes a sequence of words as input to predict the next word and generate text.


## PaLM
PaLM is a desley-activated LLM. PaLM employs the conventional Transformer model architecture in a decoder-exclusive setup, which allows each timestep to attend only to itself and preceding timesteps. It shows high performance in the following areas:
- Multi-step reasoning tasks
- Imitation Game Benchmark (BIG-bench)
- Multilingual tasks
- Source code generation

PaLM employs some modifications to the original transformer design:
- employ SwiGLU activation for MLP intermediate activations
- parallel layers for each transformation block
- employ multi-query attention
- employ Rotary Position Embedding (RoPE) embeddings
- shared input-output embeddings
- no biases
- 256k-token SentencePiece vocabulary

---

# Utilize LLMs

## huggingface

Steps
1. install necessary packages: `pip install –upgrade transformers`
2. download a model, in this case microsoft DialoGPT

   ```python
   hf_model = "microsoft/DialoGPT-medium"
   max_length = 1000
   tokenizer = AutoTokenizer.from_pretrained(hf_model)
   model = AutoModelForCausalLM.from_pretrained(hf_model)
   ```
3. define the probpt
4. perform experiment
5. results

---

## RAG and LangChain

### RAG

Retrieval-Augmented Generation (RAG) is a development framework designed for seamless interaction with LLMs. RAG improves performance of LLM to be used for a specific task/field. It segments task specific documents, and when responding to a query, it find the most relevant segment to the query and adds it to the model retrieval process.

### LangChain

__Overview__

LangChain can connect an arbitraru LLM to a defined data sources, making it possible to customize LLM to a specific domain. When structuring answer to a query, LangChain makes the data source act as a reference. In addition to pointing the model to data scouce, LangChain processing scheme is quick and efficient. 

Steps
1. First step is to generate vector database from input database in text format. This is done by chunking the text into appropriate lengths and creating numerical text embeddings (embeding model could be different from LLM that will be used for prompting).
2. After storing embeddings in vector database, a search mechanism identifies the relevant data chunck to a user prompt.  This step is done by embedding the prompt and use a similarity algorithm such as cosine similarity to find the most similar chunks, and then retrieve texts associated with those chuncks.
3. The prompt concatenated with the texts retrieved in the step 2, is then sent to LLM.

__How LangChain works__
- The building block of LangChain is called components, a prompt template. Combining multiple components creates a chain.
- The next layer over chain is agents. Agents complement chains by adding extra information. Agents act as a reasoning mechanism, and add a prescribed logic to the chain for downstream processes.
- Long-term memory is another concept that LongChain employs to append additional data sources. This feature gives LangChain ability to refer to prior conversations and learn from them.

__LangChain pipeline__

LangChain pipeline consists of the following steps
- Load the text files
- Process the data so that it can be prepared for embedding
- Create the embeddings that would be stored in the vector database
- Create the vector database
- Perform a similarity search based on our in-house documents

---

# Advance approaches driven by LLM

---

## Frontier approaches!

---
---
# Advance NLP course - CMU 2024


