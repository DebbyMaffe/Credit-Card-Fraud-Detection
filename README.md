# Credit-Card-Fraud-Detection
**Detecting Credit Card Fraud using Neural Networks** - University Group Project

## 01. INTRODUCTION
*Credit card fraud* is a serious concern, which involves unauthorized individuals exploiting someone else's credit card or account information for unauthorized purchases or cash advances.
Recognizing the effectiveness of *Neural Networks* in capturing intricate patterns and anomalies within extensive datasets, we have chosen to deploy Artificial Neural Networks (**ANNs**) and Convolutional Neural Networks (**CNNs**) to identify and prevent instances of fraudulent transactions.

**DATA SOURCE**

The dataset used for the project has been taken from *kaggle.com* at the following link: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.

It includes **284,807** transactions with **31** variables, where: 
* the variable **Time** contains the *seconds* elapsed between each transaction and the first transaction in the dataset.
* the feature **Amount** collects the *value* of each transaction.
* the *binary* variable **Class** represents our ***target***, revealing if a certain transaction was *genuine* (0) or a *fraud* (1).
* the features **V1, V2, .., V28** are numerical inputs result of a *PCA* transformation whose content couldn’t be displayed due to their confidential nature.

## 02. EXPLORATORY DATA ANALYSIS
The *creditcard* dataset, consisting entirely of *numerical* variables with **no missing** values, undergoes exploratory data analysis (**EDA**). Analyzing the distribution of the '**Amount**' data reveals that 75% of transactions are *below* €77.16, with the *highest* transaction amount recorded at €25,691.16, significantly *higher* than the average of €88.35.

The **wide spread** in transaction amounts suggests a *diverse* dataset, encompassing both *common* and *occasional* high-value transactions, impacting the overall average. Further analysis focuses on key features (class, amount, time). Examining genuine and fraudulent transactions reveals a highly **imbalanced** dataset, with only **0.173%** of observations being *fraudulent*.

Analyzing transaction **amounts**, *legitimate* transactions show 75% below €77.05, with the highest amount at €25,691.16, exceeding the average of €88.29. In contrast, *fraudulent* transactions have 75% below €105.89, and the maximum amount is €2,125.87, considerably higher than the average fraudulent amount of €122.21.

Exploring transaction **frequency** in a specific time frame indicates that *genuine* transactions are more frequent than fraudulent ones, consistent with the typical pattern in credit card datasets, where *legitimate* transactions significantly **outnumber** fraudulent ones.

## 03. DATA PREPROCESSING 
Since we are dealing with *binary* classification problems, let's perform **random oversampling** technique to address class *imbalance* within our datasets. The *RandomOverSampler()* method is designed to enhance the **balance** in class distribution by randomly duplicating samples from the *minority* class, which in our case is the *fraud* class. This process continues until a more balanced class distribution is achieved, enhancing the **robustness** of our classification models. After an **80/20** train-test *split*, **standardize** the preprocessed data. 

## 04. BUILDING MODEL
**ANNs**

Artificial Neural Networks (**ANNs**) are machine learning models inspired by the human brain. They consist of interconnected nodes (**neurons**) organized into input, hidden, and output layers, with connections represented by *weights*. ANNs operate in **parallel**, offering fast real-time solutions for complex problems.

Let's define an ANN model with an *input* layer of **30** neurons, one *hidden* layer with **16** neurons, and an *output* layer with **1** neuron. During **training**, the model is *compiled*, *fitted*, and *evaluated*. The **accuracy** plot shows consistent *improvement* in both training and validation, with the validation accuracy often surpassing the training accuracy in early epochs.

**Prediction** involves evaluating *probabilities* against specific thresholds (**0.5**, **0.8**, **0.9**). The classifier performs exceptionally well, achieving *almost perfect* classification for both classes. Considering the importance of **correctly** classifying *fraudulent* transactions, a **50%** threshold is deemed most appropriate, optimizing *sensitivity* and *specificity*.

The **ROC curve** of the model is positioned in the *top-left* corner, indicating **high** *sensitivity* and **low** *false positive* rates across thresholds. A perfect **AUC** of **1.0** demonstrates the classifier's ability to *perfectly* distinguish between fraudulent and genuine transactions, achieving ideal sensitivity and specificity.

**CNNs**

Convolutional Neural Networks (**CNNs**) are deep learning models effective for *structured* grid data, especially in tasks like image and video recognition. Inspired by visual processing mechanisms in animals, CNNs use neurons responding to stimuli within their receptive field, sharing *weights* through convolutional filters for translational invariance.

To create a CNN model, the input data is **reshaped** into a **3D** structure (width, height, channels). The model includes **two** *convolutional* layers (32 and 64 neurons) and **one** *output* layer (1 neuron). *Batch* normalization and *Dropout* regularization are applied for normalization and preventing overfitting.

During **training**, the accuracy plot shows **increasing** training *accuracy* and relatively **constant** validation *accuracy*, indicating the model is learning. Initial epochs reveal higher validation accuracy, suggesting better performance on unseen data. Loss plots confirm improvement in **minimizing** the chosen *loss* function over epochs.

**Prediction** involves evaluating *probabilities* against specific thresholds (**0.5**, **0.8**, **0.9**). The classifier achieves *nearly perfect* classification for both classes. An **80%** threshold is considered *optimal*, optimizing specificity, and sensitivity. However, a 50% threshold remains effective. 

The **ROC curve** of the model is positioned in the *top-left* corner, indicating **high** *sensitivity* and **low** *false positive* rates across thresholds, achieving perfect sensitivity and specificity (**AUC** of **1.0**).

## 05. CONCLUSION
Comparing *fraud detection* and *false alarm* metrics, the Convolutional Neural Network *(CNN)* **outperforms** the Artificial Neural Network *(ANN)* with a significantly lower average fraud not detection rate (0.4% vs. 1.98%). However, the *ANN* captures **fewer** *false* alarms with a rate of 0.32%, slightly better than the CNN (0.33%).

Considering the project's emphasis on **maximizing** fraud *detection* and **minimizing** *undetected* fraud, the **CNN** proves superior due to its higher sensitivity in identifying fraudulent transactions.
