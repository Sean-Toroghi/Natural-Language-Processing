<h1>Large language models</h1>

# Overview
A language model approximates human language, and is built by training over a large body of text, thus imbuing it with various properties of langauge, including aspects of grammar (syntax) and meaning (semantics). During the training process, the language model predicts the next token and its parameters are updated if it gets it wrong. Modern language models are based on neural networks and most prominent architecture is the Transformers. 

Around 2019, researchers realized that increasing the size of the language model predictably improved performance, with no saturation point in sight. [Kaplan et al 2019](https://arxiv.org/abs/2001.08361) defined a law decribing the relationship between the size of a language model and dataset, and required computation. In another paper [Kaplan et al 2020](https://arxiv.org/pdf/2001.08361.pdf) showed the relationship between model performance and size of dataset, model, and training duration. They found that for a fixed compute budget, increasing the size of the dataset and the model in tandem improves the performance of the LM, but the dataset size needs to increase only by 1.8x for every 5.5x increase in model size to maintain optimal level of performance. [Hoffmann et al 2020](https://arxiv.org/pdf/2203.15556.pdf) showed that to optimize performance of a LLM at a fixed compute budget (called compute-optimal), the training data size needs to increase at the same proportion as the model size. __Takeaway__: even with limited resouces (compuation), increasing the size of dataset can improve performance.

__History of LLMs__
- In the mid-1960s, Joseph Weizenbaum released ELIZA, a chatbot program that applied pattern matching using regular expressions on the user’s input and selected response templates to generate an output. [paper](https://hackaday.com/wp-content/uploads/2024/02/WEIZENBAUM-1966-ELIZA-A-Computer-Program-For-the-Study-of-Natural-Language-Communication-Between-Man-And-Machine.pdf)
- Statistical approaches in NLP emerged in 2000s [book](https://www.amazon.com/Foundations-Statistical-Natural-Language-Processing/dp/0262133601)
- during 2010s the advent of deep learning impacts NLP, in the form of onstruct a task-specific architecture to solve each task. Some of the types of neural network architectures used include multi-layer perceptrons, convolutional neural networks, recurrent neural networks, and recursive neural network.
- In 2017, the Transformer architecture was invented, quickly followed by the invention of transfer learning and Transformer-based language models like BERT.
- GPT era:
  - GPT-1 - Showcased unsupervised pre-training on large scale data, followed by task-specific supervised fine-tuning.
  - GPT-2 - This version could solve several types of tasks in a zero-shot setting, without any task-specific fine-tuning. This marked the rise of prompting as a means to interact with a language model.
  - GPT-3 - Inspired by the scaling laws, this model is a hundred times larger than GPT-2 and popularized in-context/few-shot learning.
  - GPT4 - A key aspect of this release is the alignment training used to make the model more controllable and adhere to the principles and values of the model trainer.
 
__Prompting__




# Training 

# Tokenization

# Employ LLM for specific task

# Fine-tune LLM

# External tools for inference with LLMs

# RAG
