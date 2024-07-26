<h1>Large language models</h1>

# Overview
A language model approximates human language, and is built by training over a large body of text, thus imbuing it with various properties of langauge, including aspects of grammar (syntax) and meaning (semantics). During the training process, the language model predicts the next token and its parameters are updated if it gets it wrong. Modern language models are based on neural networks and most prominent architecture is the Transformers. 

Around 2019, researchers realized that increasing the size of the language model predictably improved performance, with no saturation point in sight. [Kaplan et al 2019](https://arxiv.org/abs/2001.08361) defined a law decribing the relationship between the size of a language model and dataset, and required computation. In another paper [Kaplan et al 2020](https://arxiv.org/pdf/2001.08361.pdf) showed the relationship between model performance and size of dataset, model, and training duration. They found that for a fixed compute budget, increasing the size of the dataset and the model in tandem improves the performance of the LM, but the dataset size needs to increase only by 1.8x for every 5.5x increase in model size to maintain optimal level of performance. [Hoffmann et al 2020](https://arxiv.org/pdf/2203.15556.pdf) showed that to optimize performance of a LLM at a fixed compute budget (called compute-optimal), the training data size needs to increase at the same proportion as the model size. __Takeaway__: even with limited resouces (compuation), increasing the size of dataset can improve performance.

__History of LLMs__
- In the mid-1960s, Joseph Weizenbaum released ELIZA, a chatbot program that applied pattern matching using regular expressions on the user’s input and selected response templates to generate an output. [paper](https://hackaday.com/wp-content/uploads/2024/02/WEIZENBAUM-1966-ELIZA-A-Computer-Program-For-the-Study-of-Natural-Language-Communication-Between-Man-And-Machine.pdf)
- 


# Training 

# Tokenization

# Employ LLM for specific task

# Fine-tune LLM

# External tools for inference with LLMs

# RAG
