<h1>Large language models</h1>

References
- [Building LLM powered applications](https://learning.oreilly.com/library/view/building-llm-powered/9781835462317)
- [Towards a Mechanistic Interpretation of Multi-Step Reasoning Capabilities of Language Models](https://arxiv.org/pdf/2310.14491v1)
- [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)
- []()
- 
# Overview

## Foundation model
A foundation model refers to a type of pre-trained generative AI model that is adaptable for various specific tasks via extensive training on vast and diverse datasets. They grasp general patterns and relationships withi the data providing them a foundation to understand across different domains. This cross-domain capability differentiates foundation models from standard natural language understanding algorithms. One characteristic of LLMs is their size. With transfer learning, foundation models can be used for a specific task with minimal additional traning (compare with their original traning phase). 

The language foundation models trained on unstructured data (such as natural language data) are called LLMs, such as GPT-x, ChatGPT, BERT, Megatron, and other models in ever evolving list of LLMs. In another word LLMs are type of foundation models that are specifically designed for NLP tasks. 


## Language models
A language model approximates human language, and is built by training over a large body of text, thus imbuing it with various properties of langauge, including aspects of grammar (syntax) and meaning (semantics). During the training process, the language model predicts the next token and its parameters are updated if it gets it wrong. Modern language models are based on neural networks and most prominent architecture is the Transformers. 

Around 2019, researchers realized that increasing the size of the language model predictably improved performance, with no saturation point in sight. [Kaplan et al 2019](https://arxiv.org/abs/2001.08361) defined a law decribing the relationship between the size of a language model and dataset, and required computation. In another paper [Kaplan et al 2020](https://arxiv.org/pdf/2001.08361.pdf) showed the relationship between model performance and size of dataset, model, and training duration. They found that for a fixed compute budget, increasing the size of the dataset and the model in tandem improves the performance of the LM, but the dataset size needs to increase only by 1.8x for every 5.5x increase in model size to maintain optimal level of performance. [Hoffmann et al 2020](https://arxiv.org/pdf/2203.15556.pdf) showed that to optimize performance of a LLM at a fixed compute budget (called compute-optimal), the training data size needs to increase at the same proportion as the model size. __Takeaway__: even with limited resouces (compuation), increasing the size of dataset can improve performance.

__Preprocessing overview__

Under the hood, LLMs are a particular type of artificial neural networks (ANNs). ANNs are, by definition, mathematical models that work with numerical data. To prepare text for LLMs we are required to perform the following two tasks:
1. Tokenization:  The goal of tokenization is to create a structured representation of the text that can be easily processed by machine learning models. At this stage, text is breaking down into smaller pieces called tokens. These tokens can be words, subwords, or characters depending on the tokenization algorithm.
2. Embedding: The goal of embedding is to transform tokens into a dense numerical verctor (embedding).  Embeddings are a way to represent words, subwords, or characters in a continuous vector space.

__Model overview__

A LLMs in its general form, consists of three types of layers:
1. input layer: the first layer that receives the input data.
2. hidden layers: layers sandwich between input and output layers, responsible to extract relevant patterns and representations from the data.
3. output layer: final layer of the model, that is customized based on the task in hand, such as classification. 


__Training process of a LLMs__
- data collection
- data preprocessing
- model architecture selection
- model initialization
- model pre-training
- fine-tuning: The output of this phase is called the supervised fine-tuned (SFT) model.
- reinforcement learning from human feedback (RLHF): iteratively optimizing the SFT model (by updating some of its parameters) with respect to the reward model.

__Customize LLMs__

There are three main approach to customize a LLM:
- extending non-parametric knowledge. Non-parametric knowledge doesn’t change the structure of the model, but rather, allows it to navigate through external documentation to be used as relevant context to answer the user’s query.
- Few-shot learning. The LLM is given a metaprompt with a small number of examples (typically between 3 and 5) of each new task it is asked to perform.
- Fine tuning: The fine-tuning process involves using smaller, task-specific datasets to customize the foundation models for particular applications.

## LLMs: History
- In the mid-1960s, Joseph Weizenbaum released ELIZA, a chatbot program that applied pattern matching using regular expressions on the user’s input and selected response templates to generate an output. [paper](https://hackaday.com/wp-content/uploads/2024/02/WEIZENBAUM-1966-ELIZA-A-Computer-Program-For-the-Study-of-Natural-Language-Communication-Between-Man-And-Machine.pdf)
- Statistical approaches in NLP emerged in 2000s [book](https://www.amazon.com/Foundations-Statistical-Natural-Language-Processing/dp/0262133601)
- during 2010s the advent of deep learning impacts NLP, in the form of onstruct a task-specific architecture to solve each task. Some of the types of neural network architectures used include multi-layer perceptrons, convolutional neural networks, recurrent neural networks, and recursive neural network.
- In 2017, the Transformer architecture was invented, quickly followed by the invention of transfer learning and Transformer-based language models like BERT.
- GPT era:
  - GPT-1 - Showcased unsupervised pre-training on large scale data, followed by task-specific supervised fine-tuning.
  - GPT-2 - This version could solve several types of tasks in a zero-shot setting, without any task-specific fine-tuning. This marked the rise of prompting as a means to interact with a language model.
  - GPT-3 - Inspired by the scaling laws, this model is a hundred times larger than GPT-2 and popularized in-context/few-shot learning.
  - GPT4 - A key aspect of this release is the alignment training used to make the model more controllable and adhere to the principles and values of the model trainer.




## Prompting

Prompting is the process by which we interact with a LLM. Ideal prompting: the best prefix of N tokens that when fed to an LLM, that leads it to generate the correct answer with the highest probability. __Note__: Language models are insensitive to word order. 

__Different types of prompting:__
- Zero-shot prompting
- Few-shot prompting
- Chain-of-Thought prompting
- Adversarial Prompting

---

# How LLMs changed software development

Nowadays developers can make API calls to a hosted version of an LLM, with the option of customizing it for their specific needs. An example is _copilot system_, such as Microsoft-Copilot and OpenAI ChatGPT both powered by GPT-4 model, which is a new category of software serve as an expert helper to users who want to perform a complex task. This concept allowsto embed and orchestrate LLMs within user applications, without a need to use programming language. Andrej Karpathy, the previous Director of AI at Tesla, tweeted in 2023 “English is the hottest new programming language.” 

## AI orchestrators to embed LLMs into applications

__Main components of AI orchestrators__

- model: the type of LLM that is picked to embed in customized application. There are two main categories of models: 1. proprietary LLMs owened by a company such as GPT-4 or Brad, and 2. open-source models, that are freely available such as Falcon LLM, developed by Abu Dhabi’s Technology Innovation Institute (TII), or LLaMA, developed by Meta.
- memory: LLM applications use a conversational interface, which requires the aiblity to refer back to earlier information within the conversation. Memory allows application to store and retrieve past interations, and additional non-parametric knowledge to be added to the model. All past conversations (embedded into VectorDB) are stored in the memory.

    VectorDB is a type of database that stores and retrieves information based on vectorized embeddings, the numerical representations that capture the meaning and context of text. VectorDB can be used to perform semantic search and retrieval based on the similarity of meanings rather than keywords. VectorDB can also help LLMs generate more relevant and coherent text by 
providing contextual understanding and enriching generation results. Some examples of VectorDBs are Chroma, Elasticsearch, Milvus, Pinecone, Qdrant, Weaviate, and Facebook AI Similarity Search (FAISS).
- Plug-insL are additional modules, integrated into the LLM to extend its functionality or adapt it to specific tasks. They act as add-ones, enhancing the capabilities of a LLM, beyound its core language generation or comprehension abilities.




# Training 

# Tokenization

# Employ LLM for specific task

# Fine-tune LLM

# External tools for inference with LLMs

# RAG
