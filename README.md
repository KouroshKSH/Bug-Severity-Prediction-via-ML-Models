# Bug Severity Prediction Project Report
In this project, the classifier has to predict a severity label given a bug’s summary. The `summary` and `severity` columns are assigned to every unique bug (given their `bug_id`). The models were evaluated based on Macro Precision, as stated in the rules of the Kaggle competition. Finally, the model with the best performance was chosen as our final submission. 

## Team Members
- Ozan Parlayan
- Kourosh Sharifi
- Arya Hassibi 
- Kutluhan Aygüzel

---

# Problem Description 
Initially, we are given a dataset that contains 3 columns: 

1. The `bug_id` column: which is an integer number corresponding to a unique bug
2. The `summary` column: which is a string of text, describing the bug 
3. The `severity` column: which is a string, with 7 different classes 

Then, after training an ML model on this training dataset, we are expected to classify the bugs of the test dataset. In the end, we store the `bug_id` and the `severity` label in a CSV file as the final submission to Kaggle, where it will be evaluated and scored. 

> image here of dataset
![][image1]A screenshot of the first 5 items of the training dataset 

In regards to how we predict the severity of a given bug, we can use different ML models to accomplish this task. Every approach uses different techniques of data preprocessing, transformation, hyperparameter tuning, training and testing. More on this in the following pages. 

---

# Methods 

## Exploratory Data Analysis (EDA) 

### Class Distribution 

The tables below showcase the distribution of the bugs given their severity label. The table on the left uses a log scale for better representation. The pie chart below also showcases the percentage of each class. 

> 3 images here
![][image2]![][image3]![][image4]


### Severity vs. Bug ID 

Although not helpful, we also plotted the graph for severity classes and their corresponding bug IDs. We aimed to extract as much information as possible out of the dataset, and perhaps, to exploit any unintended bugs with the dataset. 

> image here
![][image5]3  

---

# Models 

We tried several approaches from baseline classifiers to deep learning models. We had the highest accuracy with our Logistic Regression (LR) model, which is a supervised classification model. Other models that we tried: 
- Support Vector Machines (SVM): which was employed without the Kernel Trick
- XGBoost: which displayed a macro accuracy of **94**% on train data, but failed to perform well on test data. 
- Light Gradient-Boosting Machines (LightGBM): perhaps our least optimized model since it had the lowest accuracy scores 
- Ensemble Learning via ScikitLearn in addition to early stopping mechanisms to avoid overfitting 
- Random Forests Classifiers (RFC) with Bayesian Optimization for hyperparameter fine-tuning 
- BERT: our attempt at using deep learning to analyze the bug summaries using an NLP approach to capture more information, yet we discarded it due to computation limitations. 

A detailed description of our different approaches and their step-by-step implementation can be found in the following sections. For this project, we purchased Google Colab Pro subscriptions to utilize the cloud GPUs that Google offers. Without them, it would have been impossible to even test some of the models. This cost us a total of 340TL. 

---

# Approaches 

## Logistic Regression 

Logistic regression was chosen as the classification algorithm due to its simplicity, efficiency, and interpretability. We employed grid search with cross-validation to find the optimal hyperparameters for the logistic regression model. 

### Data Preprocessing 

We started by loading the dataset using Pandas and preprocessing the bug descriptions. Text preprocessing involves converting text to lowercase, removing single characters, and replacing multiple spaces with a single space. Additionally, we concatenated the bug summary with the bug type to form the input text for the model. 

### Feature Engineering 

We used TF-IDF vectorization to convert the textual data into numerical features. This process transforms the text into a numerical representation, capturing the importance of each word in the bug descriptions. 

### Model Training and Evaluation 

The dataset was split into training and validation sets. We trained the logistic regression model on the training data and evaluated its performance on the validation set using macro precision as the evaluation metric. Additionally, we generated a classification report to assess the model's performance across different severity levels. 

> image here
![][image6]

### Model Testing 

Finally, we deployed the trained logistic regression model to predict the severity of bugs in the test dataset. The predictions were then saved into a submission file for evaluation on the Kaggle platform. 

### Plots 

> 3 images here
![][image7]  
ROC curves for individual classes 

![][image8]6  
Precision-Recall Curves for Individual Classes 

![][image9]Confusion Matrix of the logistic regression model 

## BERT 

Another approach was using BERT from HuggingFace, which was a model proposed in *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* by Devlin et al (link to paper). It’s a bidirectional transformer pre-trained using a combination of masked language modeling objective and next sentence prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia. 

### Data Preprocessing 

We started by loading the bug dataset using Pandas. To reduce computation time, we sampled a fraction of the data for initial training (10% at first). We encoded the severity labels using `LabelEncoder`. The dataset was split into training and validation sets using `train_test_split()`. 

### Model Building 

We used BERT for sequence classification. The tokenization of bug descriptions was performed using the BERT tokenizer. We converted the tokenized data into PyTorch Tensors for compatibility with BERT. The `BERTForSequenceClassification` model was employed for predicting bug severity. Training was conducted using the `Trainer` class from the Transformers library. 

### Training and Evaluation 

The training was conducted with a one-epoch limit for quick testing. Training arguments such as batch size, evaluation strategy, and logging directories were defined (can be found in the next section). The model was trained on the training dataset and evaluated on the validation dataset. Evaluation metrics such as macro precision were used to assess model performance. 


## SVM 

In comparison to LR, our SVM model was the 2nd best model we had in terms of performance, and it was quite efficient to run as well. We used a linear kernel as our kernel trick approach to facilitate the training. 

### Data Preprocessing 

The bug reports were loaded into Pandas DataFrames. The `’type'` column was checked and added with placeholder data if missing. Text data from bug summaries and types were combined into a single `'text'` column for analysis. Severity labels were encoded using `LabelEncoder` for model compatibility. 

### Model Training 

The whole dataset was split into training and validation sets for model evaluation. A Support Vector Machine (SVM) classifier with a linear kernel was used as the base model. A TF-IDF vectorizer was applied to convert text data into numerical features. Hyperparameter tuning was performed using gradient descent to find the optimal regularization parameter (C) for SVM. The best version of the model using grid-search is as below: 

`bootstrap=True,max_depth=40,max_features='auto',min_samples_leaf=1,min _samples_split=2, n_estimators=217` 

### Model Evaluation 

The precision score was chosen as the evaluation metric, computed using macro averaging. The model with the highest precision on the validation set was selected for final prediction. 


## XGBoost 

XGBoost is a popular gradient-boosting algorithm known for its efficiency and performance, which was another one of our approaches. During training, the model with the best hyperparameters gave us **94**% macro precision accuracy (Fİgure 2 of Appendix). Unfortunately, the model did not perform well on the actual test data, as it gave us only **24**% accuracy on Kaggle. More on this in the email that Kourosh Sharifi has sent. 

### Data Preprocessing 

The initial step involves importing necessary libraries and loading the dataset. Basic text preprocessing techniques were applied, including converting text to lowercase, removing punctuation, stopwords, and lemmatization. This ensures that the text data is clean and suitable for further processing. 

### Handling Imbalanced Data 

The dataset is checked for class imbalance, and oversampling of minority classes is performed using the `RandomOverSampler` technique from the imbalanced-learn library. This helps in addressing the imbalance issue and ensures that the models are trained on a balanced dataset. 

> image here
![][image10]

### Feature Extraction 

1. **TF-IDF Vectorization:** Text data is vectorized using the Term Frequency-Inverse Document Frequency (TF-IDF) approach. This technique converts text documents into numerical vectors, capturing the importance of words in each document relative to the entire corpus.   
2. **Word2Vec Vectorization:** This embedding technique is used to represent words in a continuous vector space. Word2Vec model is trained on the text corpus to generate dense word embeddings. 
3. **Doc2Vec Vectorization:** Similar to Word2Vec, Doc2Vec generates embeddings for entire documents rather than individual words. It learns to represent documents in a continuous vector space, capturing semantic similarities. 
4. **FastText Vectorization:** The FastText model was used to generate word embeddings with subword information, enabling better representation of out-of-vocabulary words. 

For Word2Vec, Doc2Vec and FastText, the vector size was set to **250**.

### Model Training 

The model was trained on the whole dataset using the combined feature vectors obtained from TF-IDF, Word2Vec, Doc2Vec, and FastText representations. With GPU acceleration enabled, this model took about 40-50 minutes per run to complete. 

The exact hyperparameters that we used (after some fine-tuning): 

> image here
![][image11]

---

# Results and Discussion 

This project analyzed the effectiveness of various ML models for bug severity prediction. Logistic regression achieved a macro-average precision of **45**% on the training data, indicating some ability to identify different severity levels. However, the weighted average precision of **82**% suggests a bias towards more frequent severity classes. This is further reflected in the test set accuracy of **54**% obtained from the Kaggle platform. While this demonstrates a basic classification capability, there's room for improvement through refined class divisions, and incorporating additional features. 

The SVM model, with hyperparameters optimized using gradient descent, achieved a macro-average precision of **55**% on the training data, surpassing the performance of logistic regression. However, its performance on the Kaggle test data fell short of the logistic regression model. This suggests potential overfitting on the training data. Future work with SVMs could focus on regularization techniques to improve generalizability and different kernels (e.g., polynomial kernels with higher degrees) 

The XGBoost approach and BERT also hold promise for bug severity prediction if done correctly. For XGBoost, feature importance analysis is needed to identify the most influential factors for classification and understand its major underperformance during testing. For BERT, further investigation is necessary regarding hyperparameter tuning, feature engineering, and potentially exploring alternative model architectures to decrease computation costs. To refine the forecast model, future work will delve deeper into the results, implications, and recommendations for all models with more emphasis on neural network-based approaches. 

---


# Appendix 
## Team Members and Responsibilities

1. Ozan Parlayan was responsible for the logistic regression and SVM implementations, which were our 2 best models. 
2. Kourosh Sharifi was responsible for the EDA, model implementation of XGBoost, LightGBM, Random Forest and BERT, and writing the final report for this project.
3. Kutluhan Aygüzel was responsible for research, data preprocessing, and writing the final report.
4. Arya Hassibi was responsible for BERT and ensemble models. 

## Figures 

> images needed
![][image12]  
Figure 1: Feature importance graph of the XGBoost model   
![][image13]  
Figure 2: The confusion matrix of the XGBoost model 
![][image14]  
Figure 3: The confusion matrix of the Random Forest model after Bayesian Optimization 21  
![][image15]Figure 4: The poor results of the Random Forest model 