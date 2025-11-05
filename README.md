# Hindi Fake News Detection

## Project Overview

This project implements a machine learning system to detect fake news in Hindi language using Natural Language Processing (NLP) techniques. The system classifies Hindi news articles as either True (0) or Fake (1).

## Project Structure

```
nlp_pro/
├── dataset-merged.csv                          # Original dataset
├── 1_Data_Exploration_and_Preprocessing.ipynb  # Notebook 1: EDA and preprocessing
├── 2_Model_Training_and_Evaluation.ipynb       # Notebook 2: Training and evaluation
├── requirements.txt                            # Python dependencies
└── archive/                                    # Old script files
```

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Notebooks

Open and run the notebooks in order:

1. **1_Data_Exploration_and_Preprocessing.ipynb**
   - Load and explore the dataset
   - Analyze class distribution
   - Preprocess text data
   - Save cleaned data

2. **2_Model_Training_and_Evaluation.ipynb**
   - Extract TF-IDF features
   - Train multiple models
   - Evaluate performance
   - Compare results

## Dataset Information

- **Source**: Hindi Fake News Detection Dataset (HFDND)
- **Total Records**: 17,124 Hindi news articles
- **Labels**: 0 (True News), 1 (Fake News)
- **Language**: Hindi (Devanagari script)

## Methodology

### 1. Data Exploration
- Load and analyze dataset
- Check for missing values
- Visualize class distribution
- Analyze text statistics

### 2. Text Preprocessing
- Remove URLs, numbers, and special characters
- Remove Hindi stopwords
- Apply stemming
- Normalize text

### 3. Feature Extraction
- Split data (80% train, 20% test)
- Apply TF-IDF vectorization
- Extract 5000 features

### 4. Model Training
- Multinomial Naive Bayes
- Logistic Regression
- Random Forest
- Support Vector Machine

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis
- Model comparison

## Expected Results

- **Accuracy**: 85-95%
- **Precision**: 85-93%
- **Recall**: 83-92%
- **F1-Score**: 84-92%

## Key Metrics Explanation

**Confusion Matrix:**
```
                Predicted
              True    Fake
Actual True   TN      FP
       Fake   FN      TP
```

- **Precision**: How many predicted fakes are actually fake
- **Recall**: How many actual fakes were detected
- **F1-Score**: Balance between Precision and Recall

## Technologies Used

- Python 3.x
- pandas, numpy (Data manipulation)
- scikit-learn (Machine Learning)
- matplotlib, seaborn (Visualization)

## Project Components

### Notebook 1: Data Exploration and Preprocessing
- Dataset loading and analysis
- Exploratory data analysis
- Text cleaning and preprocessing
- Stopword removal and stemming

### Notebook 2: Model Training and Evaluation
- TF-IDF feature extraction
- Multiple model training
- Performance evaluation
- Model comparison and selection

## How to Present

1. Show dataset statistics and visualizations
2. Explain preprocessing steps
3. Demonstrate TF-IDF feature extraction
4. Present trained models and their performance
5. Show confusion matrices and comparison charts
6. Demonstrate predictions on sample text

## License

This is an academic project for educational purposes.
