"""
Test Suite for Fake News Detection System
Corresponds to Section 11 (Testing) & Table 9 of the Mini-Project Report
------------------------------------------------------------------------
Test Cases:
  TC-01: Load fake_news_sample.csv
  TC-02: Check class balance
  TC-03: Clean text containing URL and symbols
  TC-04: Train-test split (80:20 stratified)
  TC-05: Fit TF-IDF on training data
  TC-06: Train all five models without error
  TC-07: Predict sensational fake-style text -> FAKE
  TC-08: Predict formal factual text -> REAL
"""

import os
import re
import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from fake_news_detection import clean_text, predict_news


class TestFakeNewsDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = "dataset/fake_news_sample.csv"
        if not os.path.exists(cls.dataset_path):
            cls.dataset_path = "fake_news_sample.csv"
        cls.df = pd.read_csv(cls.dataset_path).dropna().drop_duplicates()

    def test_tc01_load_dataset(self):
        """TC-01: Load fake_news_sample.csv (Expected: 1196 rows, 'text' & 'label' columns)"""
        print("\n[TC-01] Testing Dataset Loading...")
        self.assertIn("text", self.df.columns)
        self.assertIn("label", self.df.columns)
        self.assertEqual(len(self.df), 1196, f"Expected 1196 rows, found {len(self.df)}")
        print("  -> PASS: 1196 rows loaded successfully.")

    def test_tc02_class_balance(self):
        """TC-02: Check class balance (Expected: 595 real, 601 fake)"""
        print("\n[TC-02] Testing Class Balance...")
        real_count = (self.df["label"] == 0).sum()
        fake_count = (self.df["label"] == 1).sum()
        self.assertEqual(real_count, 595, f"Expected 595 real, got {real_count}")
        self.assertEqual(fake_count, 601, f"Expected 601 fake, got {fake_count}")
        print(f"  -> PASS: Real = {real_count}, Fake = {fake_count}.")

    def test_tc03_clean_text(self):
        """TC-03: Clean text containing URL and symbols (Expected: URLs & symbols removed)"""
        print("\n[TC-03] Testing Text Pre-processing & Cleaning...")
        dirty_text = "BREAKING: Check http://fake-news.xyz/alert.html for #123 secret truth!!!"
        cleaned = clean_text(dirty_text)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("123", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertTrue(cleaned.islower())
        print(f"  -> PASS: Cleaned result -> '{cleaned}'")

    def test_tc04_train_test_split(self):
        """TC-04: Train-test split (Expected: 80:20 stratified split -> 956 train / 240 test)"""
        print("\n[TC-04] Testing Train-Test Split...")
        Xtr, Xte, ytr, yte = train_test_split(
            self.df["text"], self.df["label"], test_size=0.2, stratify=self.df["label"], random_state=42
        )
        self.assertEqual(len(Xtr), 956)
        self.assertEqual(len(Xte), 240)
        self.assertEqual((ytr == 0).sum(), 476)
        self.assertEqual((ytr == 1).sum(), 480)
        print(f"  -> PASS: 956 train (476 real, 480 fake) / 240 test (119 real, 121 fake).")

    def test_tc05_tfidf_feature_extraction(self):
        """TC-05: Fit TF-IDF on training data (Expected: Feature matrix created)"""
        print("\n[TC-05] Testing TF-IDF Feature Extraction...")
        clean_train = self.df["text"].apply(clean_text)
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=2, sublinear_tf=True)
        feats = vec.fit_transform(clean_train)
        self.assertGreater(feats.shape[1], 1000)
        print(f"  -> PASS: Feature matrix created with {feats.shape[1]} n-gram features.")

    def test_tc06_train_all_models(self):
        """TC-06: Train all five models without error"""
        print("\n[TC-06] Testing Training of All 5 ML Models...")
        clean_train = self.df["text"].apply(clean_text)
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=2, sublinear_tf=True)
        Xtr_v = vec.fit_transform(clean_train)
        ytr = self.df["label"]

        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Multinomial Naive Bayes", MultinomialNB()),
            ("Linear SVM", LinearSVC(random_state=42)),
            ("Passive Aggressive", PassiveAggressiveClassifier(max_iter=1000, random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))
        ]

        for name, m in models:
            m.fit(Xtr_v, ytr)
            print(f"    - {name} trained successfully.")
        print("  -> PASS: All 5 models trained without exceptions.")

    def test_tc07_predict_fake_news(self):
        """TC-07: Predict sensational fake-style text -> FAKE"""
        print("\n[TC-07] Testing Sensational Fake News Prediction...")
        fake_input = "BREAKING: The finance minister is secretly hiding the truth about your bank savings. Share this before it gets deleted!"
        res = predict_news(fake_input)
        self.assertEqual(res["label"], "FAKE")
        print(f"  -> PASS: Predicted: {res['label']} (Confidence: {res['confidence']*100:.2f}%)")

    def test_tc08_predict_real_news(self):
        """TC-08: Predict formal factual text -> REAL"""
        print("\n[TC-08] Testing Formal Factual Real News Prediction...")
        real_input = "The health ministry published results of a clinical study, according to an official statement."
        res = predict_news(real_input)
        self.assertEqual(res["label"], "REAL")
        print(f"  -> PASS: Predicted: {res['label']} (Confidence: {res['confidence']*100:.2f}%)")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFakeNewsDetection)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

