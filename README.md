# COMP90042 Project 2023: Automated Fact Checking For Climate Science Claims

This project is part of the COMP90042 Natural Language Processing course at the University of Melbourne, Semester 1, 2023.
![giphy](https://github.com/qvunguyen/automated_fact_checking_system/assets/125786884/7a936c57-ee06-4297-8fca-6a955e2d2236)

## [Data](https://drive.google.com/drive/folders/1ytu9cuuy72Xqp5WS2qzvflwKsqMoA96b?usp=drive_link)

[train-claims,dev-claims].json: JSON files for the labelled training and development set;

[test-claims-unlabelled].json: JSON file for the unlabelled test set; 

evidence.json: JSON file containing a large number of evidence passages (i.e. the “knowledge source”); 

dev-claims-baseline.json: JSON file containing predictions of a baseline system on the development set; 

eval.py: Python script to evaluate system performance (see “Evaluation” below for more details).

## Introduction

In the age of disinformation, reliable fact-checking methods are essential. This project aims to develop an automated fact-checking system for climate science claims. The system uses a two-step approach: evidence retrieval and claim classification.

## Methodologies

### Evidence Retrieval

Three models were experimented with for evidence retrieval:

1. **BM25**: A probabilistic information retrieval model that considers term frequency, inverse document frequency, and document length.
2. **Dense Retrieval**: Uses complex language models to comprehend text semantics and retrieve relevant documents.
3. **Ensemble Model**: Combines the strengths of both BM25 and Dense Retrieval.

### Claim Classification

For claim classification, traditional machine learning classifiers were tested, as well as a transformer-based model:

1. **Gaussian Naive Bayes**
2. **Logistic Regression**
3. **Support Vector Machines (SVMs)**
4. **Random Forest**
5. **DistilBERT**: A transformer-based model that excels in NLP tasks.

## Results

The system struggled in both evidence retrieval and claim categorisation, highlighting the complexity of automated fact-checking. It requires sophisticated models that can understand complicated language contexts and semantic linkages.

## Future Work

The results provide the groundwork for future study and the creation of more comprehensive and effective fact-checking systems to counteract disinformation. Future work will focus on refining phrase embeddings and the ranking mechanism for the evidence retrieval model and researching other architectures or attention processes for the claim classification model.

## How to Run

INSTRUCTION (Note: The code is intended to run on Goolge Colab while mounting to Google Drive)

The code is divided in 4 steps:

0. Environment Setup and Import library
1. Preprocessing and Data exploring
2. Build a retrieval system
3. Build classifier
4, System integrations

Please first execute "ONLY" the first two code cells under "Mount Google Drive" and "Create a new folder for project on Drive".
These 2 code cells will help you mount your Colab session to Drive and create a "project" folder in your "My Drive".
After that, please copy the "project-data" folder which contain the dataset of this project into the "project" folder we just created in the previous steps.

The path for the code and data set is set in the following code:

path = '/content/drive/MyDrive/project' # Path for code and project data
data_path = '/content/drive/MyDrive/project/project-data' # Path of project data

(IF you have your own path for code and data please specified it using the exact format as above)

After this you can run the rest of the code.

IF you want to evaluate the system on hidden datset, you can replace "data = test_data " to "data = your_data" under subsection "Predict lables for test data" in Step 4.
However, please load your dataset before doing so by following "Load dataset" subsection under Step1.

CAUTION: There is code snippet under subsection "Evaluate final system on dev dataset using eval.py" might cause you error if you do not follow the instruction closely. The code is intended to evaluate the final system using provided "eval.py". However, because the code is run on Colab instead of Terminal so the syntax is a little bit lengthy and require following closely with the instruction above. If the code in this subsection throws error, you can just comment it out and download the 'dev-claims-predictions.json' to the foler contain "eval.py" on your system. Then you can run the command line provided in project specification to evaluate the system performance.

## Dependencies

This project requires the following Python libraries:

- Google Colab: For mounting Google Drive and running the project in a Colab notebook.
- OS: For interacting with the operating system, particularly for directory and file operations.
- Transformers: For using transformer models, specifically DistilBERT.
- Rank-BM25: For using the BM25 algorithm for information retrieval.
- Tabulate: For creating neat tabular data for display.
- Imbalanced-Learn: For handling imbalanced datasets.
- Accelerate: For accelerating PyTorch operations.
- JSON, RE, Pickle, Collections, Random, Heapq: For various utility functions.
- NLTK: For natural language processing tasks.
- NumPy and Pandas: For numerical computations and data manipulation.
- Matplotlib: For data visualization.
- Scikit-learn: For machine learning tasks, similarity measurement, and preprocessing.
- PyTorch: For creating and training neural network models.
- TQDM: For creating progress bars.
- Concurrent.Futures: For creating parallel tasks.
- Rank-BM25: For using the BM25 algorithm for information retrieval.

To install these dependencies, use the following commands:

```bash
pip install google-colab
pip install os
pip install transformers
pip install rank-bm25
pip install tabulate
pip install -U imbalanced-learn
pip install --upgrade accelerate
pip install json
pip install re
pip install pickle
pip install collections
pip install random
pip install heapq
pip install nltk
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install torch
pip install tqdm
pip install concurrent.futures
pip install rank_bm25
```
## License

Copyright the University of Melbourne, 2023

## Contact

vunguyen.career@gmail.com
