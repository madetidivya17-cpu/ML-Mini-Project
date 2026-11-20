# VIDEO DEMONSTRATION SCRIPT & PROTOTYPE STORYBOARD
## Fake News Detection Using Machine Learning Algorithms
**Course:** Machine Learning Mini-Project  
**Programme:** B.Tech – CSE (AI & ML), II Year, I Semester  
**Academic Year:** 2026–2027  

---

## 👥 Team Member Roles & Speaking Distribution

| S.No. | Student Name | Roll Number | Video Presentation Role | Segment |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Hema Varshini** | `25RBAIM0040` | **Project Leader:** Introduction, Aim, Problem Statement & Architecture | `00:00 - 00:50` |
| 2 | **Madhu Sri** | `25RBAIM0057` | **Data & NLP Lead:** Dataset Description, Text Preprocessing & TF-IDF Extraction | `00:50 - 01:40` |
| 3 | **Divya Madeti** | `25RBAIM0058` | **Prototype Lead:** Live Streamlit Web App Demonstration (Real & Fake Cases) | `01:40 - 02:50` |
| 4 | **Ch. J V V Sandeep** | `25RBAIM0061` | **Evaluation Lead:** Model Benchmark, Confusion Matrix, Conclusion & Future Scope | `02:50 - 03:50` |

---

## ⏱️ Minute-by-Minute Presentation Storyboard (Duration: ~3.5 to 4.0 Minutes)

### Segment 1: Introduction & Architecture (0:00 – 0:50)
- **Speaker:** Hema Varshini (`25RBAIM0040`)
- **Screen Display:** Title Slide / Streamlit App Header
- **Spoken Script:**
  > *"Respected faculty and evaluators, a very good day to all. I am Hema Varshini, along with my team members Madhu Sri, Divya Madeti, and Ch. J V V Sandeep from B.Tech Second Year, CSE (AI & ML).*  
  > *Today, we present our Machine Learning Mini-Project: **Fake News Detection using Machine Learning Algorithms**.*  
  > *In the digital era, misinformation spreads exponentially across social networks before manual fact-checkers can intervene. Our project objective is to design and implement an end-to-end NLP and Supervised Machine Learning system that automatically analyzes the linguistic patterns of news text and classifies it as either REAL or FAKE.*  
  > *Our architecture follows a 6-stage workflow: dataset curation, text normalization, TF-IDF n-gram feature extraction, multi-model training across 5 supervised algorithms, performance evaluation, and live interactive deployment.*  
  > *Now, Madhu Sri will explain the dataset and methodology."*

---

### Segment 2: Dataset, Preprocessing & Feature Extraction (0:50 – 01:40)
- **Speaker:** Madhu Sri (`25RBAIM0057`)
- **Screen Display:** Dataset Tab & Figure 1 (Class Distribution)
- **Spoken Script:**
  > *"Thank you, Hema. I am Madhu Sri. For our experimental evaluation, we curated a balanced benchmark dataset containing 1,196 news articles: 595 verified Real news articles and 601 Fake news articles.*  
  > *Raw online text is notoriously noisy. Our preprocessing pipeline normalizes all text by converting to lowercase, stripping HTTP/web URLs using regular expressions, removing non-alphabetic symbols, and filtering standard English stop-words.*  
  > *To transform unstructured text into numerical features, we employ **TF-IDF with unigrams and bigrams**, utilizing sublinear term-frequency scaling. This captures not only individual high-impact words but also critical contextual word pairs such as 'school holidays' or 'audit report'.*  
  > *We partitioned the data using an 80:20 stratified split into 956 training samples and 240 held-out test samples to rigorously prevent data leakage. Next, Divya will demonstrate our live interactive prototype."*

---

### Segment 3: Live Prototype Walkthrough (01:40 – 02:50)
- **Speaker:** Divya Madeti (`25RBAIM0058`)
- **Screen Display:** Live Streamlit Application (`app.py`) running in browser
- **Action 1:** Click on **"🚨 Fake Sample 1 (Finance)"** button
- **Spoken Script:**
  > *"Thank you, Madhu. I am Divya Madeti, and I will now demonstrate our working web prototype deployed with Streamlit.*  
  > *First, let us test a typical sensational fake news headline:*  
  > *'BREAKING: The finance minister is secretly hiding the truth about your bank savings. Share this before it gets deleted!'*  
  > *When we click 'Analyze News', the pipeline tokenizes the text, matches it against our fitted TF-IDF vocabulary, and queries our trained Logistic Regression classifier.*  
  > *(Points to red card on screen)*  
  > *As you can see, the system instantly flags it as **FAKE NEWS** with an **82.73% confidence score**, highlighting alarmist linguistic signals such as 'breaking', 'secretly', 'hiding', and 'deleted'.*  
  > *Now, let us test a formal, authentic news article:*  
  > *(Clicks '✅ Real Sample 1 (Clinical)')*  
  > *'The health ministry published results of a clinical study, according to an official statement.'*  
  > *(Points to green card on screen)*  
  > *The system immediately classifies this as **REAL NEWS** with **70.62% confidence**, identifying objective indicators such as 'ministry', 'clinical', 'study', and 'official statement'.*  
  > *Our prototype also supports batch classification where users can upload any CSV file to classify thousands of articles simultaneously. Now, Sandeep will present the experimental results."*

---

### Segment 4: Model Evaluation, Results & Conclusion (02:50 – 03:50)
- **Speaker:** Ch. J V V Sandeep (`25RBAIM0061`)
- **Screen Display:** Model Analytics Tab (Figures 3, 4, 5 & Comparison Table)
- **Spoken Script:**
  > *"Thank you, Divya. I am Sandeep, and I will discuss our comparative experimental results.*  
  > *We trained and benchmarked 5 distinct machine learning algorithms on our test set:*  
  > *1. Logistic Regression*  
  > *2. Multinomial Naive Bayes*  
  > *3. Linear Support Vector Machine (LinearSVC)*  
  > *4. Passive Aggressive Classifier, and*  
  > *5. Random Forest Ensemble.*  
  > *Our selected model, **Logistic Regression**, achieved an outstanding **94.58% Accuracy**, **92.86% Precision**, **96.69% Recall**, and an **F1-Score of 94.74%**.*  
  > *Examining the confusion matrix on 240 test articles, the model correctly classified 110 Real and 117 Fake articles, with only 9 false positives and 4 false negatives. Linear SVM and Random Forest also achieved strong performance, while Naive Bayes provided an extremely fast baseline.*  
  > *In conclusion, our mini-project successfully validates that machine learning and NLP can detect language signatures of online misinformation with high precision.*  
  > *In future iterations, we plan to scale to deep learning transformer architectures like BERT, incorporate social media metadata, and support multilingual Indian news. Thank you!"*

---

## 🎥 How to Record the Prototype Video Dump in 2 Minutes

### Method A: Built-in Windows Game Bar Screen Recorder (Easiest & Fastest)
1. Launch the Streamlit App:
   ```powershell
   streamlit run app.py
   ```
   *(Or open `prototype_demo.html` in your browser).*
2. Press <kbd>Windows Key</kbd> + <kbd>Alt</kbd> + <kbd>R</kbd> to immediately start recording your screen.
3. Follow the spoken script above (or click "Play Video Demo" in `prototype_demo.html`).
4. Press <kbd>Windows Key</kbd> + <kbd>Alt</kbd> + <kbd>R</kbd> again to stop recording.
5. Your completed `.mp4` video dump will be saved automatically in:
   `C:\Users\<Username>\Videos\Captures\`

### Method B: Using OBS Studio
1. Open OBS Studio -> Add Source -> **Window Capture** -> Select Browser (`app.py` or `prototype_demo.html`).
2. Click **Start Recording**.
3. Perform the live test cases and click **Stop Recording**.

---

## 🎯 Top Viva & Evaluation Defense Questions

1. **Why did you choose TF-IDF with Unigrams and Bigrams instead of simple Bag of Words?**
   - *Answer:* Simple Bag of Words only counts word frequencies regardless of how common words are across all documents. TF-IDF down-weights ubiquitous terms and up-weights discriminative terms. Including bigrams allows the model to capture 2-word phrases (e.g., `"school holidays"`, `"annual audit"`) that convey strong contextual meaning.

2. **Why select Logistic Regression as the final deployed model when Linear SVM and Random Forest gave similar accuracy?**
   - *Answer:* Logistic Regression provides direct probability estimates (`predict_proba`), making it ideal for displaying confidence scores to end-users. It is computationally lightweight, trains in milliseconds, and its learned coefficients provide full explainability.

3. **How did you prevent data leakage during feature extraction?**
   - *Answer:* We fitted `TfidfVectorizer.fit_transform()` strictly on the training set (80%), and only applied `transform()` to the test set and live inputs, ensuring the model never observed test vocabulary during training.

