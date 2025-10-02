Word2Vec-from-Scratch-GPU

A "from scratch" implementation of the Word2Vec (Skip-Gram) algorithm in Python, accelerated with CuPy for GPU training on the enwiki9 Wikipedia corpus.


# Word2Vec Implementation from Scratch on a Large Corpus

This repository contains my implementation of the Word2Vec (Skip-Gram) algorithm, built from scratch in Python. The primary goal of this project was to gain a deep, foundational understanding of how word embeddings are trained, moving beyond the high-level abstractions of libraries like `gensim`.

The model is designed to run on a large text corpus (`enwiki9`) and is accelerated using CuPy to leverage an NVIDIA GPU, making the training process feasible. The notebook covers the entire pipeline:
* Efficient parsing of a large Wikipedia XML dump.
* Memory-efficient data streaming and vocabulary construction.
* The core Word2Vec training loop (forward and backward passes).
* Evaluation of the resulting embeddings using word similarity and analogy tasks.

## Results Showcase

After training on approximately 6.65 million pairs from the corpus, the model successfully learned strong semantic relationships.

### Word Similarity
The model correctly identifies similar concepts:
Words most similar to 'king':

queen: 0.8447

pope: 0.8165

duke: 0.7955

Words most similar to 'computer':

video: 0.7698

design: 0.7659

intel: 0.7543


### Word Analogies
It also solved several classic word analogy tasks:
Analogy: king - man + woman = ?

queen: 0.7345

Analogy: japan - tokyo + beijing = ?

china: 0.8357


## Built With

* [Python 3](https://www.python.org/)
* [CuPy](https://cupy.dev/) - For GPU-accelerated array computations (NumPy alternative).
* [NumPy](https://numpy.org/) - For CPU-based array manipulation and saving.
* [Jupyter Notebook](https://jupyter.org/) - For interactive development and documentation.

## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

* Python 3.x
* An NVIDIA GPU with a compatible CUDA Toolkit installed. This is required for the CuPy (GPU) version of the training code.
* The `enwiki9` corpus. You can run the first cell of the notebook to download it automatically, or download it manually from [mattmahoney.net](http://mattmahoney.net/dc/enwiki9.zip).

### Installation

1.  Clone the repo:
    ```sh
    git clone [https://github.com/your_username/Word2Vec-from-Scratch-GPU.git](https://github.com/your_username/Word2Vec-from-Scratch-GPU.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd Word2Vec-from-Scratch-GPU
    ```
3.  Install the required Python packages:
    ```sh
    pip install numpy nltk jupyter
    ```
4.  Install CuPy. This step is critical and depends on your CUDA version. The command below is for CUDA 12.x. Please check the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for the command that matches your system.
    ```sh
    pip install cupy-cuda12x
    ```

## Dataset

1. Download dataset from here https://mattmahoney.net/dc/enwik9.zip
2. Unzip the file and convert the extention to .xml

## Usage

1.  Place the `enwiki9.xml` file in the root of the project directory.
2.  Launch Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
3.  Open the `word2vec.ipynb` file and run the cells in order. The notebook is divided into sections for data processing, vocabulary building, training, and evaluation. You can choose between the GPU (CuPy) and CPU (NumPy) versions of the training cell.

## The Project Journey & Key Decisions

This project involved several challenges and pivots, which were crucial learning experiences.

### 1. The Data Processing Pipeline
* **Initial Plan:** My first approach was to use the `wikiextractor` library to parse the XML dump.
* **Problem:** I encountered a `re.error` indicating that the library was incompatible with the modern version of Python I was using.
* **Solution:** I decided to build a more robust, self-contained solution. I replaced `wikiextractor` with a custom, memory-efficient parser using Python's built-in `xml.etree.ElementTree.iterparse`. This avoided the dependency issue and gave me more control over the text cleaning process. A `ParseError` later revealed the XML file was truncated, which was handled by adding a `try...except` block to save the valid partial data.

### 2. The Tokenization Dependency
* **Initial Plan:** I started by using the standard `nltk` library for sentence and word tokenization.
* **Problem:** I ran into a persistent and unusual `LookupError` where `nltk` could not find its `punkt` data, even after multiple direct download attempts, manual file placement, and even a full environment reset.
* **Solution:** After exhausting all attempts to fix the library's environment, I made the decision to **remove the NLTK dependency entirely**. I replaced its tokenization functions with a simple and universal `re.findall(r'\b\w+\b', line)` approach. This made the project more resilient and dependency-free for this critical step.

### 3. Performance and GPU Acceleration
* **Problem:** After setting up the data pipeline, I calculated that training the model on the full corpus using NumPy (CPU) would take over a month.
* **Solution:** I chose to accelerate the training using a GPU. I evaluated two paths: a full rewrite in PyTorch or a minimal-change approach with CuPy. I selected **CuPy** because it allowed me to keep the low-level, "from scratch" logic of the NumPy implementation while simply swapping the backend to the GPU. This preserved the original goal of the project.

### 4. Training Duration
* **Problem:** Even with the GPU, training on the full ~1.2 billion pairs would take many days.
* **Solution:** For the purpose of this project and assignment, I decided to stop the training at **6.65 million pairs**. As the results show, this was more than sufficient to produce a high-quality model that learned complex semantic relationships, demonstrating the success of the implementation.

## License

Distributed under the MIT License. See `LICENSE` for more information.
