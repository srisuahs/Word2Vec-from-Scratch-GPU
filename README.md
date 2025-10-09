# Word2Vec from Scratch with GPU Acceleration

This repository contains a complete, from-scratch implementation of the Word2Vec (Skip-Gram) algorithm in Python. The primary goal of this project was to gain a deep, foundational understanding of how word embeddings are trained by building the entire pipeline manually, from data ingestion to model evaluation.

The model is designed to be self-contained and fully automated. It leverages **CuPy** for GPU acceleration, making the computationally intensive training process on the large `enwiki9` corpus feasible.

## Key Features

* **From-Scratch Implementation:** The core training loop, including the forward pass, softmax, backpropagation, and weight updates, is implemented manually.
* **GPU Acceleration:** Uses CuPy as a drop-in replacement for NumPy to run all matrix operations on an NVIDIA GPU.
* **Automated Data Pipeline:** The notebook automatically downloads, unzips, and prepares the `enwiki9` dataset.
* **Resilient Data Processing:** Employs a memory-efficient XML parser and a dependency-free tokenizer to handle large and potentially corrupted files robustly.
* **Comprehensive Evaluation:** Includes intrinsic evaluation methods for word similarity (cosine similarity) and word analogies (`king - man + woman = queen`).
* **Embedding Visualization:** Uses PCA and t-SNE to visualize the learned vector space and confirm semantic clustering.

## Results Showcase

After training, the model successfully learned strong semantic relationships.

### Word Analogies
The model solved several classic word analogy tasks, demonstrating that the vector space has learned consistent and meaningful structures.

Analogy: king - man + woman = ?

queen: 0.7345

Analogy: japan - tokyo + beijing = ?

china: 0.8357

### Embedding Visualization (t-SNE)
The t-SNE plot shows clear clustering of semantically related words, such as countries, numbers, and technology, providing strong visual confirmation of the model's learning.


## Technology Stack

* **Python 3**
* **Core Logic:** [CuPy](https://cupy.dev/) (for GPU) & [NumPy](https://numpy.org/) (for CPU/saving)
* **Visualization:** [Scikit-learn](https://scikit-learn.org/) (for PCA/t-SNE) & [Matplotlib](https://matplotlib.org/)
* **Environment:** [Jupyter Notebook](https://jupyter.org/)

## Getting Started: Data Flow and Execution

This repository is designed to be fully self-contained. The entire project can be run with just a clone of the repo and execution of the notebook.

### Data Flow
1.  **Download:** The notebook's first cell initiates a download of the `enwiki9.zip` file from `mattmahoney.net`.
2.  **Unzip & Rename:** The zip file is extracted. The resulting file, `enwiki9`, which contains XML data but has no extension, is automatically renamed to `enwiki9.xml`.
3.  **Parse & Clean:** The `enwiki9.xml` file is parsed in a memory-efficient stream. All Wikitext markup is cleaned using regular expressions, producing a `cleaned_corpus.txt` file.
4.  **Train:** The `cleaned_corpus.txt` is streamed to build a vocabulary and generate skip-gram pairs on-the-fly to train the model.
5.  **Save:** After training, the final word vectors are saved as `word_vectors.npy` and the vocabulary map as `vocabulary.json`.

### Installation & Usage

**Prerequisites:**
* Python 3.x
* An NVIDIA GPU with a compatible CUDA Toolkit installed.

**Instructions:**
1.  Clone the repository:
    ```sh
    git clone [https://github.com/srisuahs/Word2Vec-from-Scratch-GPU.git](https://github.com/srisuahs/Word2Vec-from-Scratch-GPU.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd Word2Vec-from-Scratch-GPU
    ```
3.  Launch Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
4.  Open `word2vec.ipynb` and follow one of the scenarios below.

---
### **Execution Scenarios**

This notebook supports different workflows depending on your goal.

#### Scenario A: Full Pipeline (Training from Scratch)
This is for running the entire process, from download to visualization. This is time-consuming.

* **Cells to Run:** Click **"Run All"** at the top of the notebook. The first cell will handle all installations and data downloads automatically.

#### Scenario B: Evaluation & Visualization Only (Using Pre-trained Vectors)
This is the recommended workflow for quickly exploring the project's results. This assumes the provided `word_vectors.npy` and `vocabulary.json` files are present in the repository.

1.  **Run Cell 1 ("Project Setup and Data Automation"):** This will install all required Python packages. You can ignore the data download messages if the files are already present.
2.  **SKIP** all the cells for data processing and model training (Cells under headings 1, 2, and 3).
3.  **Run all cells from the markdown header "4. Results and Analysis"** to the end of the notebook.
    * To run only the word analogy tests, execute the cells under section "4.2. Quantitative Analysis: Word Analogies".
    * To run only the visualizations, execute the cells under section "5. Visualizing the Embedding Space".

## The Project Journey & Key Decisions

This project involved several challenges that required iterative problem-solving.

1.  **Data Processing:** The initial plan to use the `wikiextractor` library failed due to Python version incompatibility. This was solved by building a more robust, self-contained parser using Python's built-in `xml` library, which also gracefully handled the discovery of a truncated (corrupted) XML file.

2.  **Tokenization:** The project initially used the `nltk` library, but a persistent and unresolvable `LookupError` in the execution environment forced a pivot. The solution was to **remove the NLTK dependency entirely** and replace its functions with a universal `re.findall()` approach, making the project more resilient.

3.  **GPU Acceleration:** Initial performance estimates on the CPU with NumPy predicted a training time of over a month. To make this feasible, I chose to accelerate the training with a GPU. I selected **CuPy** over PyTorch because it allowed me to keep the low-level, "from scratch" logic of the implementation while simply swapping the backend to the GPU.

4.  **Training Duration:** Even with a GPU, training on the full corpus was estimated to take many days. For this project, I made the practical decision to stop the training at **6.65 million pairs**. As the extensive evaluation results show, this was more than sufficient to produce a high-quality model that learned complex semantic relationships.

## License

Distributed under the MIT License. See `LICENSE` for more information.
