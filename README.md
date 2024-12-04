# Author Profiling on Arabic Tweets using Deep Learning Techniques  

In today's world, where many people prefer to hide their identities, **Author Profiling** plays a crucial role in determining an author's characteristics based on their textual data. This project focuses on using Arabic tweets to identify the **gender**, **age**, and **language dialect** of authors.

---

## Programming Language  
- **Python 3**

---

## Dataset  
We are using the **Train** and **Test** datasets provided by the **Forum for Information Retrieval Evaluation (FIRE) 2019** for this research.

---

## Models  
Two models were developed and tested on the dataset to achieve better accuracy and efficiency:

### 1. Long Short-Term Memory (LSTM) Model  
- The first model is a standard **LSTM** network.
- The main file for this model is: `Train_test.py`.

### 2. Long Short-Term Memory (LSTM) + Features Model  
- The second model combines the **LSTM** network with additional features extracted from the tweets.
- These features include:
  - **Emoji Counter**
  - **Sentence Length**
  - And other relevant characteristics
- The feature extraction programs are located in the `Features` folder.
- The main file for this model is: `Train_test_feature.py`.

### Softmax Layer  
Both models utilize a **Softmax** layer for classification tasks.

---

## Accuracy  

### Model 1: LSTM  
**Accuracy on Test Data:**  
- **Gender**: 57.64%  
- **Age**: 27.50%  
- **Language Dialect**: 55.14%

### Model 2: LSTM + Features  
**Accuracy on Test Data:**  
- **Gender**: 66.24%  
- **Age**: 22.22%  
- **Language Dialect**: 80.28%

---

## Research Paper  
You can access our research paper titled:  
**"Gender, Age, and Dialect Recognition using Tweets in a Deep Learning Framework – Notebook for FIRE 2019."**

---

By leveraging deep learning techniques and feature engineering, this project demonstrates the potential of using social media text for accurate author profiling, which can be applied in various domains like forensics, marketing, and social media analytics.
