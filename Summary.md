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
In the case where A is a matrix that can be diagonalized, it can be deconstructed into a d × d invertible matrix, V, and a diagonal d × d matrix, Δ, such that

<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" display="block"><mml:mi mathvariant="bold">A</mml:mi><mml:mi mathvariant="bold"> </mml:mi><mml:mo>=</mml:mo><mml:mi mathvariant="bold"> </mml:mi><mml:mi mathvariant="bold">V</mml:mi><mml:mi mathvariant="bold"> </mml:mi><mml:mi mathvariant="bold">Δ</mml:mi><mml:mi mathvariant="bold"> </mml:mi><mml:msup><mml:mrow><mml:mi mathvariant="bold">V</mml:mi></mml:mrow><mml:mrow><mml:mo>-</mml:mo><mml:mn>1</mml:mn></mml:mrow></mml:msup></mml:math>
