"""
Streamlit Web Application Prototype
Fake News Detection using Machine Learning Algorithms
------------------------------------------------------
Course: Machine Learning Mini-Project
Authors:
  - Hema Varshini (25RBAIM0040)
  - Madhu Sri (25RBAIM0057)
  - Divya Madeti (25RBAIM0058)
  - Ch. J V V Sandeep (25RBAIM0061)

Programme: B.Tech – CSE (AI & ML), II Year, I Semester
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Fake News Detector | AI/ML Mini-Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #e0e8f5;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .team-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-fake {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 6px solid #e53e3e;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-real {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 6px solid #38a169;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .keyword-pill-fake {
        display: inline-block;
        background-color: #feb2b2;
        color: #9b2c2c;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 2px;
        font-weight: 600;
    }
    .keyword-pill-real {
        display: inline-block;
        background-color: #9ae6b4;
        color: #22543d;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 2px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Cleaning function
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Load model and vectorizer
@st.cache_resource
def load_artifacts():
    model_path = "models/best_model.pkl"
    vec_path = "models/tfidf_vectorizer.pkl"
    
    # If not found, try to run pipeline or load from root
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        import subprocess
        subprocess.run(["python", "fake_news_detection.py"], check=True)
        
    model = joblib.load(model_path)
    vec = joblib.load(vec_path)
    return model, vec

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error(f"Error loading models: {e}. Please run `python fake_news_detection.py` first.")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=70)
    st.title("Project Information")
    st.caption("Machine Learning Mini-Project (2026–2027)")
    
    st.markdown("---")
    st.subheader("👥 Team Members")
    st.markdown("""
    - **Hema Varshini** `(25RBAIM0040)`
    - **Madhu Sri** `(25RBAIM0057)`
    - **Divya Madeti** `(25RBAIM0058)`
    - **Ch. J V V Sandeep** `(25RBAIM0061)`
    """)
    st.markdown("**Programme:** B.Tech – CSE (AI & ML)")
    st.markdown("**Year/Sem:** II Year, I Semester")

    st.markdown("---")
    st.subheader("⚙️ Active Model Specs")
    st.markdown("""
    - **Classifier:** Logistic Regression
    - **Feature Extraction:** TF-IDF (1–2 Grams)
    - **Vocabulary:** 1,041 n-grams
    - **Test Accuracy:** **94.58%**
    - **F1-Score:** **94.74%**
    - **Train/Test Split:** 80% / 20%
    """)

    st.markdown("---")
    st.caption("Developed with Python, scikit-learn & Streamlit.")


# ---------------------------------------------------------
# MAIN BANNER HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛡️ Fake News Detection System</h1>
    <p>Automated Natural Language Processing & Supervised Machine Learning Classifier for Online Misinformation Verification</p>
    <div style="margin-top: 12px;">
        <span class="team-badge">Hema Varshini</span>
        <span class="team-badge">Madhu Sri</span>
        <span class="team-badge">Divya Madeti</span>
        <span class="team-badge">Ch. J V V Sandeep</span>
    </div>
</div>
""", unsafe_allow_html=True)


# TABS NAVIGATION
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live News Classifier",
    "📊 Model Performance & Metrics",
    "📁 Batch CSV Classification",
    "📚 Dataset & Methodology"
])


# ---------------------------------------------------------
# TAB 1: LIVE NEWS CLASSIFIER
# ---------------------------------------------------------
with tab1:
    st.subheader("Interactive News Article Verification")
    st.write("Enter or paste any news headline or complete article text below to verify whether it exhibits language patterns of **REAL** or **FAKE** news.")

    # Sample news buttons
    st.markdown("**Or test with benchmark examples:**")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    sample_text = ""
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    with col_btn1:
        if st.button("🚨 Fake Sample 1 (Finance)"):
            st.session_state["input_text"] = "BREAKING: The finance minister is secretly hiding the truth about your bank savings. Share this before it gets deleted!"
    with col_btn2:
        if st.button("🚨 Fake Sample 2 (Health)"):
            st.session_state["input_text"] = "Secret truth exposed: Doctors reveal hidden truth about vaccines that mainstream media refuses to broadcast! Forward urgently!"
    with col_btn3:
        if st.button("✅ Real Sample 1 (Clinical)"):
            st.session_state["input_text"] = "The health ministry published results of a clinical study, according to an official statement."
    with col_btn4:
        if st.button("✅ Real Sample 2 (Weather)"):
            st.session_state["input_text"] = "The meteorological department published the annual rainfall monsoon report indicating normal precipitation across southern states."

    user_input = st.text_area(
        "Enter News Text / Headline:",
        value=st.session_state["input_text"],
        height=140,
        placeholder="Paste article text or headline here..."
    )

    col_act1, col_act2 = st.columns([1, 4])
    with col_act1:
        predict_btn = st.button("🔍 Analyze News", type="primary", use_container_width=True)
    with col_act2:
        clear_btn = st.button("🧹 Clear Input", on_click=lambda: st.session_state.update({"input_text": ""}))

    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter a news article or headline before analyzing.")
        else:
            cleaned = clean_text(user_input)
            features = vectorizer.transform([cleaned])
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]

            prob_fake = prob[1] * 100
            prob_real = prob[0] * 100

            # Find top influential words present in this article
            feature_names = vectorizer.get_feature_names_out()
            coefs = model.coef_[0]
            non_zeros = features.nonzero()[1]
            word_impacts = [(feature_names[i], coefs[i]) for i in non_zeros]
            word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

            if pred == 1:
                # FAKE
                st.markdown(f"""
                <div class="result-fake">
                    <h2 style="color: #c53030; margin-top:0;">🚨 FAKE / SUSPICIOUS NEWS DETECTED</h2>
                    <p style="font-size: 1.1rem; color: #2d3748;">
                        Our machine learning model classified this text as <strong>FAKE NEWS</strong> with 
                        <strong style="color: #c53030;">{prob_fake:.2f}% confidence</strong>.
                    </p>
                    <p style="color: #4a5568; margin-bottom: 8px;">
                        <em>Linguistic indicator: The article uses sensational vocabulary, urgent call-to-actions, or emotional triggers typical of misinformation.</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # REAL
                st.markdown(f"""
                <div class="result-real">
                    <h2 style="color: #276749; margin-top:0;">✅ VERIFIED / GENUINE NEWS ARTICLE</h2>
                    <p style="font-size: 1.1rem; color: #2d3748;">
                        Our machine learning model classified this text as <strong>REAL NEWS</strong> with 
                        <strong style="color: #276749;">{prob_real:.2f}% confidence</strong>.
                    </p>
                    <p style="color: #4a5568; margin-bottom: 8px;">
                        <em>Linguistic indicator: The article adopts an objective, neutral reporting style referencing verifiable entities and formal reporting terms.</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities Progress Bars
            st.markdown("### 📊 Prediction Confidence Breakdown")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.write(f"**Real News Probability:** {prob_real:.1f}%")
                st.progress(float(prob_real) / 100.0)
            with col_p2:
                st.write(f"**Fake News Probability:** {prob_fake:.1f}%")
                st.progress(float(prob_fake) / 100.0)

            # Influential detected keywords
            if word_impacts:
                st.markdown("### 🔑 Key Vocabulary Signals Detected")
                fake_cues = [w for w, c in word_impacts if c > 0][:8]
                real_cues = [w for w, c in word_impacts if c < 0][:8]

                if fake_cues:
                    st.write("**Signals pushing towards FAKE:**")
                    pills = "".join([f'<span class="keyword-pill-fake">{w}</span>' for w in fake_cues])
                    st.markdown(pills, unsafe_allow_html=True)

                if real_cues:
                    st.write("**Signals pushing towards REAL:**")
                    pills = "".join([f'<span class="keyword-pill-real">{w}</span>' for w in real_cues])
                    st.markdown(pills, unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & METRICS
# ---------------------------------------------------------
with tab2:
    st.subheader("Model Evaluation & Experimental Results")
    st.write("Performance metrics evaluated on an unseen held-out test set (20% = 240 articles, 80:20 stratified split).")

    # Metric Cards for Best Model
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">94.58%</div>
            <div class="metric-lbl">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">92.86%</div>
            <div class="metric-lbl">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">96.69%</div>
            <div class="metric-lbl">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">94.74%</div>
            <div class="metric-lbl">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Comparison of 5 Machine Learning Algorithms")
    
    metrics_data = {
        "Classifier": [
            "Logistic Regression (Selected)",
            "Multinomial Naive Bayes",
            "Linear SVM (LinearSVC)",
            "Passive Aggressive Classifier",
            "Random Forest Classifier"
        ],
        "Accuracy (%)": [94.58, 97.08, 95.83, 96.25, 96.25],
        "Precision (%)": [92.86, 96.72, 95.12, 95.90, 93.08],
        "Recall (%)": [96.69, 97.52, 96.69, 96.69, 100.00],
        "F1-Score (%)": [94.74, 97.12, 95.90, 96.30, 96.41],
        "Primary Advantage": [
            "Highly interpretable, fast training, optimal calibration",
            "Extremely fast probabilistic baseline for NLP",
            "Max-margin boundary effective on high-dimensional text",
            "Online incremental learning suitable for live streams",
            "Ensemble robustness against individual word noise"
        ]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Visual Performance Plots")
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("outputs/fig3_model_comparison.png"):
            st.image("outputs/fig3_model_comparison.png", caption="Figure 3: Performance Comparison Across 5 ML Classifiers", use_container_width=True)
    with col_img2:
        if os.path.exists("outputs/fig4_confusion_matrix.png"):
            st.image("outputs/fig4_confusion_matrix.png", caption="Figure 4: Confusion Matrix (Logistic Regression on 240 Test Articles)", use_container_width=True)

    col_img3, col_img4 = st.columns(2)
    with col_img3:
        if os.path.exists("outputs/fig5_roc_curves.png"):
            st.image("outputs/fig5_roc_curves.png", caption="Figure 5: Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
    with col_img4:
        if os.path.exists("outputs/fig6_top_features.png"):
            st.image("outputs/fig6_top_features.png", caption="Figure 6: Top Indicative TF-IDF Features (Real vs Fake)", use_container_width=True)


# ---------------------------------------------------------
# TAB 3: BATCH CSV CLASSIFICATION
# ---------------------------------------------------------
with tab3:
    st.subheader("Batch News Article Classification")
    st.write("Upload a `.csv` or `.txt` file containing news articles to classify multiple texts at once.")

    uploaded_file = st.file_uploader("Upload CSV file (Must contain a 'text' column)", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                lines = [line.decode("utf-8").strip() for line in uploaded_file.readlines() if line.strip()]
                df_upload = pd.DataFrame({"text": lines})

            if "text" not in df_upload.columns:
                st.error("Uploaded file must have a column named `text`. Available columns: " + ", ".join(df_upload.columns))
            else:
                st.success(f"Loaded {len(df_upload)} articles from `{uploaded_file.name}`.")
                
                with st.spinner("Classifying articles..."):
                    df_upload["clean"] = df_upload["text"].apply(clean_text)
                    feats = vectorizer.transform(df_upload["clean"])
                    preds = model.predict(feats)
                    probs = model.predict_proba(feats)

                    df_upload["Prediction"] = ["FAKE" if p == 1 else "REAL" for p in preds]
                    df_upload["Fake_Confidence (%)"] = (probs[:, 1] * 100).round(2)
                    df_upload["Real_Confidence (%)"] = (probs[:, 0] * 100).round(2)

                # Summary statistics
                fake_tot = (df_upload["Prediction"] == "FAKE").sum()
                real_tot = (df_upload["Prediction"] == "REAL").sum()
                
                bcol1, bcol2, bcol3 = st.columns(3)
                bcol1.metric("Total Articles", len(df_upload))
                bcol2.metric("Detected Fake", fake_tot, delta=f"{fake_tot/len(df_upload)*100:.1f}%", delta_color="inverse")
                bcol3.metric("Detected Real", real_tot, delta=f"{real_tot/len(df_upload)*100:.1f}%")

                st.dataframe(df_upload[["text", "Prediction", "Fake_Confidence (%)", "Real_Confidence (%)"]].head(50), use_container_width=True)

                # Export CSV
                csv_out = df_upload[["text", "Prediction", "Fake_Confidence (%)", "Real_Confidence (%)"]].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Classified Results CSV",
                    data=csv_out,
                    file_name="classified_news_results.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ---------------------------------------------------------
# TAB 4: DATASET & METHODOLOGY
# ---------------------------------------------------------
with tab4:
    st.subheader("Dataset Details & System Methodology")
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("""
        ### 📦 Project Dataset Profile
        - **File:** `fake_news_sample.csv`
        - **Total Records:** **1,196** unique articles
        - **Class Distribution:**
          - **REAL News (0):** 595 articles (49.75%)
          - **FAKE News (1):** 601 articles (50.25%)
        - **Balance Ratio:** Almost perfectly balanced (1 : 1.01)
        - **Train / Test Split:** 80% Train (956) / 20% Test (240) Stratified
        """)
        if os.path.exists("outputs/fig1_class_distribution.png"):
            st.image("outputs/fig1_class_distribution.png", caption="Figure 1: Balanced Class Distribution", use_container_width=True)

    with col_d2:
        st.markdown("""
        ### 🔬 Preprocessing & Feature Pipeline
        1. **Text Normalization:**
           - Conversion to lower case
           - Removal of URLs (`http\\S+|www\\.\\S+`)
           - Stripping digits, punctuation, and non-ASCII symbols
           - Elimination of extra spaces
        2. **TF-IDF Feature Representation:**
           - **N-gram Range:** Unigrams & Bigrams `(1, 2)`
           - **Stop-words:** Standard English stop-words removed
           - **Document Frequency Filters:** `max_df = 0.9`, `min_df = 2`
           - **Term Weighting:** Sublinear TF scaling `(1 + log(tf))`
        3. **Classification Decision Function:**
           $$\\hat{y} = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$
        """)

    st.markdown("---")
    st.subheader("🌐 Standard Public Datasets for Scaled Research")
    st.write("For broader academic benchmarking, the exact same pipeline can be trained on these open-source datasets:")

    st.markdown("""
    | Dataset Name | Source / Description | Link |
    | :--- | :--- | :--- |
    | **ISOT Fake News Dataset** | University of Victoria (44,898 articles from Reuters and fake sites) | [ISOT Dataset Portal](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php) |
    | **Kaggle Fake News Challenge** | 20,800 labeled news articles from Kaggle competition | [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data) |
    | **WELFake Dataset** | 72,134 news articles aggregating 4 popular datasets (IEEE 2021) | [Kaggle WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
    | **LIAR Dataset** | 12,836 labeled short statements from PolitiFact (ACL 2017) | [LIAR Benchmark on Hugging Face](https://huggingface.co/datasets/liar) |
    """)

    # Dataset preview
    try:
        sample_df = pd.read_csv("dataset/fake_news_sample.csv")
        st.markdown("### 🔍 Dataset Sample Preview")
        st.dataframe(sample_df.head(15), use_container_width=True)
    except Exception:
        pass

