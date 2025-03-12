# **Text Classification and Part-of-Speech (POS) Tagging Challenge**  

## **Problem Statement**  
The objective of this project is to develop a robust algorithm capable of accurately classifying text snippets based on the perceived emotions of the speaker and to implement a Part-of-Speech (POS) tagging system using the Viterbi algorithm. The project is divided into four key parts, each focusing on different machine learning and deep learning techniques to address the text classification problem and enhance performance using rule-based, machine learning, and deep learning models.  

The task involves classifying each text snippet into one or more of the following six emotion categories:  
- Joy  
- Sadness  
- Fear  
- Anger  
- Surprise  
- Disgust  

Each emotion should be labeled as either present (1) or absent (0). The goal is to develop an optimized algorithm that maximizes classification accuracy while adhering to the specified architectural and training constraints.  

Additionally, the project requires implementing the Viterbi algorithm for POS tagging using Hidden Markov Models (HMM). The Viterbi algorithm should handle noisy data by adjusting emission probabilities dynamically and exploring multiple decoding paths to improve tagging accuracy.   

---

## **Part 1: Text Classification Using Regular Expressions and Machine Learning Techniques**  
The goal of Part 1 is to develop a text classification model using rule-based and feature-based approaches. The model should classify emotions based on extracted features such as regular expressions and n-grams (unigrams, bigrams, trigrams).  

### **Approach**  
1. **Feature Extraction**:  
   - Extract features using regular expressions and n-grams at the word and character levels.  
2. **Machine Learning Models**:  
   - Train and evaluate the following classifiers:  
     - Naive Bayes  
     - Logistic Regression  
     - Random Forest  
     - Support Vector Machine (SVM)  

### **Constraints**  
- The model should be implemented using `Scikit-learn`.  
- Regular expressions should be used for feature extraction.  
- Performance should be evaluated using standard metrics (accuracy, precision, recall, and F1-score).  

---

## **Part 2: Text Classification Using FFNN, RNN, and LSTM**  
Part 2 focuses on developing deep learning models to classify text snippets based on perceived emotions. The model should be built using Feedforward Neural Networks (FFNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory Networks (LSTM). A custom Word2Vec model should be trained from scratch to generate embeddings for the input text.  

### **Approach**  
1. **Embeddings**:  
   - Train a Word2Vec model with an embedding size of 100 dimensions.  
2. **Modeling**:  
   - Use FFNN with a maximum size of 64 units.  
   - Use RNN or LSTM layers with a maximum size of 64 units and a sequence length of up to 128 tokens.  
3. **Training Setup**:  
   - Use one of the following optimizers: Adam, AdamW, or SGD.  
   - Learning rate should be set to 0.001.  

### **Constraints**  
| Parameter | Value |  
|-----------|-------|  
| Maximum number of layers | 4 |  
| Maximum number of units per layer | 64 |  
| Maximum embedding size | 100 |  
| Maximum sequence length | 128 tokens |  
| Optimizer | Adam, AdamW, SGD |  
| Learning rate | 0.001 |  

---

## **Part 3: Text Classification Using Transformer and Pretrained Language Models**  
Part 3 requires the development of a transformer-based model for text classification. The model should be implemented using randomly initialized embeddings and transformer encoder layers. Additionally, a pre-trained language model such as BERT or RoBERTa should be fine-tuned for this task.  

### **Approach**  
1. **Base Model**:  
   - Train a transformer model using randomly initialized embeddings and encoder layers.  
   - Fine-tune a pre-trained model from one of the following options:  
     - BERT (`google-bert/bert-base-uncased`)  
     - RoBERTa (`FacebookAI/roberta-base`)  
2. **Preprocessing**:  
   - Preprocessing techniques are optional and can be applied to improve performance.  
3. **Training Setup**:  
   - The model should already be trained, and the deliverable should focus on inference only.  

### **Constraints**  
| Parameter | Value |  
|-----------|-------|  
| Pretrained Models | BERT, RoBERTa |  
| Library | PyTorch |  
| Maximum number of encoder layers | 12 |  

---

## **Part 4: Text Classification Using Transformer and Pretrained Language Models (Open Setup)**  
Part 4 involves building a transformer-based model for text classification without restrictions on the base model or embedding type. The objective is to develop the best-performing model using any transformer-based or pretrained language model.   

### **Approach**  
1. **Base Model**:  
   - Use any transformer-based model or pretrained language model.  
   - Combine multiple architectures if needed to improve performance.  
2. **Preprocessing**:  
   - Preprocessing techniques are optional.  
3. **Training Setup**:  
   - The model should already be trained, and the deliverable should focus on inference only.  

### **Constraints**  
- No restriction on the choice of base model or embeddings.  
- Preprocessing is optional.  

---

## **Part 5: Implementation of Viterbi Algorithm**  
Part 5 focuses on the implementation of the Viterbi algorithm for Part-of-Speech (POS) tagging using Hidden Markov Models (HMM). The model should be able to decode the most probable POS tags for each sentence and handle noisy data by dynamically adjusting emission probabilities and exploring multiple decoding paths.  

### **Approach**  
1. **HMM Setup**:  
   - Use the provided corpus to derive hidden states (POS tags) and observable states (words).  
2. **Viterbi Implementation**:  
   - Implement the Viterbi algorithm to decode the most probable POS tags for each sentence.  
3. **Noise Handling**:  
   - Handle noise in the test data by adjusting emission probabilities and exploring multiple decoding paths.  
4. **Performance Evaluation**:  
   - Compare the accuracy of the baseline Viterbi algorithm versus the noise-handled version on the provided datasets.  

### **Data Provided**  
- `train_data.txt` – 2000 sentences with tagged words for training.  
- `test_data.txt` – 400 sentences with correct tags.  
- `noisy_test_data.txt` – 400 sentences with noise introduced.  

### **Constraints**  
| Parameter | Value |  
|-----------|-------|  
| Libraries | Numpy, Collections |  

---

## **Deliverables**  
1. Source code for all parts of the project.  
2. Trained models for inference.  
3. Performance analysis and comparison of different models.  
4. Commented training code (can be uncommented for verification).  

---
