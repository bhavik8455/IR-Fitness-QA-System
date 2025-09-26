"""
Fitness FAQ IR Retrieval System

Instructions:
- Ensure `fitness_dataset.json` is present in the project root (SQuAD-like format)
- Install dependencies: pip install pandas scikit-learn nltk streamlit
- Run UI: streamlit run fitness.py
- Run CLI demo: python fitness.py cli

This script builds a TF-IDF index over Fitness Q&A pairs, provides cosine-similarity
retrieval, a simple evaluation method, and a Streamlit UI to query the system.
It demonstrates core IR concepts: preprocessing, indexing, query processing,
ranking, similarity, and evaluation.

Expected dataset format (SQuAD-like):
{
  "data": [
    { "paragraphs": [
        { "qas": [ { "id": "...", "question": "...", "answers": [{"text": "..."}, ...] }, ... ] },
        ...
    ] }
  ]
}
"""

import os
import sys
import json
from typing import List, Tuple

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional preprocessing via NLTK
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:
    nltk = None


# --------------------------- Config ---------------------------------
DATASET_PATH = "fitness_dataset.json"  # change if needed
TFIDF_MAX_FEATURES = 5000


# --------------------------- Utilities -------------------------------

def ensure_nltk_resources() -> None:
    """Download necessary NLTK resources if they are missing."""
    global nltk
    if nltk is None:
        try:
            import nltk as _nltk  # type: ignore
            nltk = _nltk
        except Exception:
            print("NLTK not available. Falling back to simple tokenization.")
            return

    resources = ["punkt", "wordnet", "omw-1.4", "stopwords"]
    for res in resources:
        try:
            nltk.data.find(res)  # type: ignore
        except Exception:
            try:
                nltk.download(res)  # type: ignore
            except Exception as e:
                print(f"Could not download NLTK resource {res}: {e}")


# --------------------------- Preprocessing ---------------------------

class Preprocessor:
    """Tokenize, remove stopwords, and lemmatize text for IR.

    Uses NLTK if available; otherwise falls back to simple rules.
    """

    def __init__(self) -> None:
        self.use_nltk = nltk is not None
        if self.use_nltk:
            ensure_nltk_resources()
            try:
                self.stopwords = set(stopwords.words("english"))  # type: ignore
                self.lemmatizer = WordNetLemmatizer()  # type: ignore
            except Exception:
                # Fallback if corpus isn't accessible
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
                # Safety fallback
                tokens = [t for t in text.split() if t not in self.stopwords]
                return " ".join(tokens)
        else:
            tokens = [t for t in text.split() if t not in self.stopwords]
            return " ".join(tokens)


# --------------------------- IR Model --------------------------------

class FAQIR:
    """Information Retrieval system for Q&A data using TF-IDF + cosine similarity."""

    def __init__(self, questions: List[str], answers: List[str], preprocessor: Preprocessor) -> None:
        self.raw_questions = questions
        self.raw_answers = answers
        self.preprocessor = preprocessor

        # Preprocess for indexing
        print("Preprocessing questions...")
        self.processed_questions = [self.preprocessor.preprocess(q) for q in self.raw_questions]

        # Build TF-IDF index
        print("Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        """Search the indexed Q&A and return top_n results.

        Returns a list of tuples: (index, score, question, answer)
        """
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

    def explain(self, query: str, doc_idx: int, top_k_terms: int = 5) -> List[Tuple[str, float]]:
        """Return contribution of top terms from the vectorizer for a specific document.

        Gives a rough explanation: top TF-IDF features overlapping between query and doc.
        """
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query]).toarray().flatten()
        doc_vec = self.tfidf_matrix[doc_idx].toarray().flatten()
        overlap = query_vec * doc_vec
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_indices = np.argsort(overlap)[::-1][:top_k_terms]
        terms_scores = [(feature_names[i], float(overlap[i])) for i in top_indices if overlap[i] > 0]
        return terms_scores


# --------------------------- Evaluation -------------------------------

def simple_evaluate(ir_system: FAQIR, sample_queries: List[Tuple[str, List[int]]], k: int = 5):
    """
    Very small evaluation harness.

    sample_queries: list of tuples (query_text, relevant_indices)
    relevant_indices: list of ground-truth indices (0-based) relevant to the query

    Computes precision@k for each sample and returns average precision.
    """
    precisions: List[float] = []
    for q, relevant in sample_queries:
        results = ir_system.search(q, top_n=k)
        retrieved_indices = [r[0] for r in results]
        if len(retrieved_indices) == 0:
            precisions.append(0.0)
            continue
        num_relevant_retrieved = sum(1 for idx in retrieved_indices if idx in relevant)
        prec = num_relevant_retrieved / min(k, len(retrieved_indices))
        precisions.append(prec)
    avg_prec = float(np.mean(precisions)) if precisions else 0.0
    return {
        f"average_precision_at_{k}": avg_prec,
        "detailed": precisions,
    }


# --------------------------- Loading Data -----------------------------

def load_fitness_data(json_path: str) -> Tuple[List[str], List[str]]:
    """Load SQuAD-like fitness JSON and return parallel lists of questions and answers."""
    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}. Please update DATASET_PATH at the top of the script.")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload.get("data", [])
    questions: List[str] = []
    answers: List[str] = []

    for entry in data:
        paragraphs = entry.get("paragraphs", [])
        for para in paragraphs:
            qas = para.get("qas", [])
            for qa in qas:
                q_text = str(qa.get("question", ""))
                ans_list = qa.get("answers", [])
                a_text = ""
                if isinstance(ans_list, list) and len(ans_list) > 0:
                    # take first provided answer text
                    a_text = str(ans_list[0].get("text", ""))
                questions.append(q_text)
                answers.append(a_text)

    # Remove empty pairs if any
    cleaned = [(q, a) for q, a in zip(questions, answers) if isinstance(q, str) and isinstance(a, str) and (q.strip() or a.strip())]
    if not cleaned:
        raise ValueError("No Q&A pairs could be parsed from fitness_dataset.json")
    questions, answers = zip(*cleaned)
    return list(questions), list(answers)


# --------------------------- Streamlit UI -----------------------------

def run_streamlit_app(ir_system: FAQIR) -> None:
    try:
        import streamlit as st
    except Exception:
        print("Streamlit is not installed. Install it with: pip install streamlit")
        return

    st.set_page_config(page_title="Fitness FAQ IR", layout="wide")

    st.title("Fitness FAQ — IR Retrieval Demo")
    st.markdown(
        (
            "Enter a query (a word like `whey` or a full question). The system uses TF-IDF + cosine\n"
            "similarity to retrieve the most relevant Q&A. This demo shows core IR building blocks:\n"
            " preprocessing, indexing, query processing, ranking, and simple evaluation."
        )
    )

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Top N results", 1, 10, 5)
        show_explain = st.checkbox("Show explanation (overlapping terms)", value=True)
        sample_eval = st.checkbox("Run example evaluation", value=False)

    query = st.text_input("Type your query here", value="whey protein isolate")

    if st.button("Search") and query:
        results = ir_system.search(query, top_n=top_k)
        if not results:
            st.warning("No matching FAQs found. Try a different query or check preprocessing settings.")
        else:
            for rank, (idx, score, q_text, a_text) in enumerate(results, start=1):
                st.subheader(f"{rank}. {q_text} (score: {score:.4f})")
                st.write(a_text)
                if show_explain:
                    terms = ir_system.explain(query, idx, top_k_terms=5)
                    if terms:
                        st.markdown("**Top overlapping terms (query vs doc):**")
                        st.write(pd.DataFrame(terms, columns=["term", "overlap_score"]))

    st.markdown("---")
    st.subheader("Index statistics")
    st.write(f"Number of Q&A indexed: {len(ir_system.raw_questions)}")
    st.write(f"TF-IDF vocabulary size: {len(ir_system.vectorizer.get_feature_names_out())}")

    if sample_eval:
        st.markdown("### Example evaluation")
        sample_queries = [
            ("whey isolate vs concentrate", find_relevant_indices_for_keyword(ir_system, "whey", top_n=10)),
            ("breakfast before jogging", find_relevant_indices_for_keyword(ir_system, "breakfast", top_n=10)),
        ]
        eval_result = simple_evaluate(ir_system, sample_queries, k=top_k)
        st.write(eval_result)


# --------------------------- Helper for sample eval ------------------

def find_relevant_indices_for_keyword(ir_system: FAQIR, keyword: str, top_n: int = 10) -> List[int]:
    """Heuristic to construct demo ground-truth: any item whose question contains the keyword."""
    keyword = keyword.lower()
    indices = [i for i, q in enumerate(ir_system.raw_questions) if keyword in q.lower()]
    if not indices:
        results = ir_system.search(keyword, top_n=top_n)
        indices = [r[0] for r in results]
    return indices


# --------------------------- Main ------------------------------------

def main() -> None:
    pre = Preprocessor()
    questions, answers = load_fitness_data(DATASET_PATH)
    ir_system = FAQIR(questions, answers, pre)

    # CLI demo mode: python fitness.py cli
    if "STREAMLIT_RUN" not in os.environ and (len(sys.argv) > 1 and sys.argv[1] == "cli"):
        print("CLI demo mode. Type queries (empty to exit).")
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
            for idx, score, q_text, a_text in res:
                print(f"[{idx}] ({score:.4f}) {q_text}\n-> {a_text}\n")
        return

    # Otherwise run Streamlit UI (works when launched via `streamlit run`)
    run_streamlit_app(ir_system)


if __name__ == "__main__":
    main()


