<h1>Natural language processing - summary</h1>

__References__
- [Mastering NLP from Foundations to LLMs 2024 - By Lior Gazit, Meysam Ghaffari](https://learning.oreilly.com/library/view/mastering-nlp-from/9781804619186/)
- [Advance NLP Course - CMU 2024](https://phontron.com/class/anlp2024/)


__Table of contents__
- Overview



---

# Overview

NLP is a field of artificial intelligence (AI) focused on the interaction between computers and human languages. It involves using computational techniques to understand, interpret, and generate human language, making it possible for computers to understand and respond to human input naturally and meaningfully.

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
