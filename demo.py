"""
Demo: Fitness FAQ IR — Answers-Only Display

Instructions:
- Ensure `fitness_dataset.json` (SQuAD-like) is in the project root
- Install dependencies: pip install pandas scikit-learn nltk streamlit
- Run UI: streamlit run demo.py
- Run CLI: python demo.py cli

This app reuses the retrieval pipeline (TF-IDF + cosine similarity) but only
displays the Answer for each retrieved item in both UI and CLI.
"""

import os
import sys
import json
from typing import List, Tuple

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:
    nltk = None


# --------------------------- Config ---------------------------------
DATASET_PATH = "fitness_dataset.json"
TFIDF_MAX_FEATURES = 5000


# --------------------------- Utilities -------------------------------

def ensure_nltk_resources() -> None:
    global nltk
    if nltk is None:
        try:
            import nltk as _nltk  # type: ignore
            nltk = _nltk
        except Exception:
            print("NLTK not available. Falling back to simple tokenization.")
            return
    for res in ["punkt", "wordnet", "omw-1.4", "stopwords"]:
        try:
            nltk.data.find(res)  # type: ignore
        except Exception:
            try:
                nltk.download(res)  # type: ignore
            except Exception as e:
                print(f"Could not download NLTK resource {res}: {e}")


# --------------------------- Preprocessing ---------------------------

class Preprocessor:
    def __init__(self) -> None:
        self.use_nltk = nltk is not None
        if self.use_nltk:
            ensure_nltk_resources()
            try:
                self.stopwords = set(stopwords.words("english"))  # type: ignore
                self.lemmatizer = WordNetLemmatizer()  # type: ignore
            except Exception:
                self.use_nltk = False
        if not self.use_nltk:
            self.stopwords = set([
                "the", "a", "an", "in", "on", "and", "is", "are", "of",
                "for", "to", "with", "that", "this", "it", "as", "be"
            ])
            self.lemmatizer = None

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
        if self.use_nltk:
            try:
                tokens = word_tokenize(text)  # type: ignore
                tokens = [t for t in tokens if t not in self.stopwords and t.isalpha()]
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]  # type: ignore
                return " ".join(tokens)
            except Exception:
                tokens = [t for t in text.split() if t not in self.stopwords]
                return " ".join(tokens)
        else:
            tokens = [t for t in text.split() if t not in self.stopwords]
            return " ".join(tokens)


# --------------------------- IR Model --------------------------------

class FAQIR:
    def __init__(self, questions: List[str], answers: List[str], preprocessor: Preprocessor) -> None:
        self.raw_questions = questions
        self.raw_answers = answers
        self.preprocessor = preprocessor

        print("Preprocessing questions...")
        self.processed_questions = [self.preprocessor.preprocess(q) for q in self.raw_questions]

        print("Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        if not query:
            return []
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        results = [
            (int(i), float(scores[i]), self.raw_questions[i], self.raw_answers[i])
            for i in ranked_idx if scores[i] > 0
        ]
        return results


# --------------------------- Loading Data -----------------------------

def load_fitness_data(json_path: str) -> Tuple[List[str], List[str]]:
    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}. Update DATASET_PATH.")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload.get("data", [])
    questions: List[str] = []
    answers: List[str] = []
    for entry in data:
        for para in entry.get("paragraphs", []):
            for qa in para.get("qas", []):
                q_text = str(qa.get("question", ""))
                ans_list = qa.get("answers", [])
                a_text = ""
                if isinstance(ans_list, list) and len(ans_list) > 0:
                    a_text = str(ans_list[0].get("text", ""))
                questions.append(q_text)
                answers.append(a_text)
    cleaned = [(q, a) for q, a in zip(questions, answers) if (q.strip() or a.strip())]
    if not cleaned:
        raise ValueError("No Q&A pairs parsed from fitness_dataset.json")
    questions, answers = zip(*cleaned)
    return list(questions), list(answers)


# --------------------------- Streamlit UI -----------------------------

def run_streamlit_app(ir_system: FAQIR) -> None:
    try:
        import streamlit as st
    except Exception:
        print("Streamlit is not installed. pip install streamlit")
        return

    st.set_page_config(page_title="Fitness FAQ IR — Answers Only", layout="wide")

    st.title("Fitness FAQ — Answers Only")
    st.markdown(
        "Enter a query. Results will display only the Answer for each retrieved item."
    )

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Top N results", 1, 10, 5)

    query = st.text_input("Type your query here", value="whey protein isolate")

    if st.button("Search") and query:
        results = ir_system.search(query, top_n=top_k)
        if not results:
            st.warning("No matching answers found.")
        else:
            for _, _, _q, a_text in results:
                # Display only the answer
                st.write(a_text)
                st.markdown("---")


# --------------------------- Main ------------------------------------

def main() -> None:
    pre = Preprocessor()
    questions, answers = load_fitness_data(DATASET_PATH)
    ir_system = FAQIR(questions, answers, pre)

    if "STREAMLIT_RUN" not in os.environ and (len(sys.argv) > 1 and sys.argv[1] == "cli"):
        print("CLI demo (answers only). Type queries (empty to exit).")
        while True:
            try:
                q = input("Query> ")
            except EOFError:
                break
            if not q:
                break
            res = ir_system.search(q, top_n=5)
            if not res:
                print("No results.\n")
                continue
            for _idx, _score, _q_text, a_text in res:
                # Print only the answer
                print(f"{a_text}\n")
        return

    run_streamlit_app(ir_system)


if __name__ == "__main__":
    main()


