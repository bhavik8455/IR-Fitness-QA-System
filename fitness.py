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
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

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
    Includes fitness-specific synonym mapping for better matching.
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
        
        # Fitness-specific synonym mapping for basic fitness concepts
        self.synonyms = {
            "fitness": ["physical fitness", "health", "wellness", "exercise"],
            "exercise": ["workout", "training", "physical activity", "fitness"],
            "cardio": ["cardiovascular", "aerobic", "endurance", "heart"],
            "strength": ["muscular strength", "power", "force", "weight"],
            "endurance": ["stamina", "cardiovascular endurance", "muscular endurance"],
            "flexibility": ["stretching", "range of motion", "mobility"],
            "body": ["body composition", "physical", "physique"],
            "muscle": ["muscles", "muscular", "muscle group"],
            "aerobic": ["cardio", "cardiovascular", "endurance"],
            "anaerobic": ["strength", "power", "intensity"],
            "hiit": ["high intensity interval training", "interval training"],
            "health": ["wellness", "fitness", "well-being"],
            "training": ["exercise", "workout", "practice"],
            "physical": ["fitness", "body", "bodily"]
        }

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
        
        # Apply synonym expansion
        text = self._expand_synonyms(text)
        
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
    
    def _expand_synonyms(self, text: str) -> str:
        """Expand synonyms in the text to improve matching."""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check if word has synonyms
            word_synonyms = []
            for key, synonyms in self.synonyms.items():
                if word in synonyms:
                    word_synonyms.extend(synonyms)
                    break
                elif word == key:
                    word_synonyms.extend(synonyms)
                    break
            
            # Add original word and its synonyms
            expanded_words.append(word)
            expanded_words.extend(word_synonyms)
        
        return " ".join(expanded_words)


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

    def search(self, query: str, top_n: int = 5, fallback_threshold: float = 0.0) -> List[Tuple[int, float, str, str]]:
        """Search the indexed Q&A and return top_n results.

        Args:
            query: Search query string
            top_n: Maximum number of results to return
            fallback_threshold: Minimum score threshold for results (0.0 = return all, including zero scores)

        Returns a list of tuples: (index, score, question, answer)
        """
        if not query:
            return []
        
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        
        # Use fallback_threshold instead of hardcoded > 0
        results = [
            (int(i), float(scores[i]), self.raw_questions[i], self.raw_answers[i])
            for i in ranked_idx if scores[i] > fallback_threshold
        ]
        
        # If no results found and fallback_threshold is 0, try fuzzy matching
        if not results and fallback_threshold == 0.0:
            return self._fuzzy_search(query, top_n)
        
        return results
    
    def _fuzzy_search(self, query: str, top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        """Fallback fuzzy search when TF-IDF finds no matches."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Calculate fuzzy similarity for each document
        doc_scores = []
        for i, (question, answer) in enumerate(zip(self.raw_questions, self.raw_answers)):
            # Combine question and answer for matching
            doc_text = f"{question} {answer}".lower()
            doc_words = set(doc_text.split())
            
            # Calculate word overlap score
            word_overlap = len(query_words.intersection(doc_words))
            if word_overlap == 0:
                continue
                
            # Calculate string similarity for partial matches
            similarity = max(
                SequenceMatcher(None, query_lower, question.lower()).ratio(),
                SequenceMatcher(None, query_lower, answer.lower()).ratio()
            )
            
            # Combined score: word overlap + string similarity
            combined_score = (word_overlap / len(query_words)) * 0.7 + similarity * 0.3
            doc_scores.append((i, combined_score, question, answer))
        
        # Sort by combined score and return top results
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_n]
    
    def suggest_similar_keywords(self, query: str, top_n: int = 5) -> List[str]:
        """Suggest similar keywords from the vocabulary when query has no matches."""
        if not query:
            return []
        
        query_words = set(query.lower().split())
        vocabulary = self.vectorizer.get_feature_names_out()
        
        suggestions = []
        for word in query_words:
            # Find similar words in vocabulary
            for vocab_word in vocabulary:
                similarity = SequenceMatcher(None, word, vocab_word).ratio()
                if similarity > 0.6:  # Threshold for similarity
                    suggestions.append((vocab_word, similarity))
        
        # Sort by similarity and return unique suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        unique_suggestions = []
        seen = set()
        for word, score in suggestions:
            if word not in seen:
                unique_suggestions.append(word)
                seen.add(word)
                if len(unique_suggestions) >= top_n:
                    break
        
        return unique_suggestions

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
    
    def expand_query_with_answer(self, original_query: str, answer_text: str, max_keywords: int = 10) -> str:
        """Expand the original query with relevant keywords extracted from the answer text.
        
        Args:
            original_query: The user's original search query
            answer_text: The answer text to extract keywords from
            max_keywords: Maximum number of keywords to add to the query
            
        Returns:
            Expanded query string with original query + relevant keywords
        """
        if not answer_text or not original_query:
            return original_query
            
        # Preprocess the answer text to extract meaningful terms
        processed_answer = self.preprocessor.preprocess(answer_text)
        
        # Get TF-IDF scores for the answer text
        answer_vec = self.vectorizer.transform([processed_answer]).toarray().flatten()
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Get top terms from the answer (excluding very common words)
        # Sort by TF-IDF score and get top terms
        top_indices = np.argsort(answer_vec)[::-1]
        
        # Extract meaningful keywords (filter out very short words and common terms)
        expanded_keywords = []
        original_words = set(original_query.lower().split())
        
        for idx in top_indices:
            if answer_vec[idx] > 0:  # Only include terms that appear in the answer
                term = feature_names[idx]
                # Filter criteria: meaningful length, not already in query, not too common
                if (len(term) > 2 and 
                    term not in original_words and 
                    answer_vec[idx] > 0.01):  # Threshold to avoid very common terms
                    expanded_keywords.append(term)
                    if len(expanded_keywords) >= max_keywords:
                        break
        
        # Combine original query with expanded keywords
        if expanded_keywords:
            expanded_query = f"{original_query} {' '.join(expanded_keywords)}"
        else:
            expanded_query = original_query
            
        return expanded_query


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
    """Load fitness Q&A JSON and return parallel lists of questions and answers."""
    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}. Please update DATASET_PATH at the top of the script.")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions: List[str] = []
    answers: List[str] = []

    # Handle both old SQuAD format and new simple format
    if isinstance(data, dict) and "data" in data:
        # Old SQuAD-like format
        for entry in data["data"]:
            paragraphs = entry.get("paragraphs", [])
            for para in paragraphs:
                qas = para.get("qas", [])
                for qa in qas:
                    q_text = str(qa.get("question", ""))
                    ans_list = qa.get("answers", [])
                    a_text = ""
                    if isinstance(ans_list, list) and len(ans_list) > 0:
                        a_text = str(ans_list[0].get("text", ""))
                    questions.append(q_text)
                    answers.append(a_text)
    elif isinstance(data, list):
        # New simple format: [{"question": "...", "answer": "..."}, ...]
        for item in data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                questions.append(str(item["question"]))
                answers.append(str(item["answer"]))
    else:
        raise ValueError("Unsupported JSON format in fitness_dataset.json")

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

    st.title("Fitness FAQ — IR Retrieval Demo with Relevance Feedback")
    st.markdown(
        (
            "Enter a query (a word like `whey` or a full question). The system uses TF-IDF + cosine\n"
            "similarity to retrieve the most relevant Q&A. Click 'Relevant' on any answer to expand your\n"
            "query with keywords from that answer for better subsequent searches."
        )
    )

    # Initialize session state for feedback tracking
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    if 'expanded_queries' not in st.session_state:
        st.session_state.expanded_queries = {}
    if 'current_query' not in st.session_state:
        st.session_state.current_query = "whey protein isolate"
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = []
    if 'last_search_query' not in st.session_state:
        st.session_state.last_search_query = ""

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Top N results", 1, 10, 5)
        show_explain = st.checkbox("Show explanation (overlapping terms)", value=True)
        sample_eval = st.checkbox("Run example evaluation", value=False)
        
        st.markdown("---")
        st.subheader("Feedback History")
        if st.session_state.feedback_history:
            for i, feedback in enumerate(st.session_state.feedback_history[-5:]):  # Show last 5
                st.text(f"{i+1}. Query: {feedback['query'][:30]}...")
                st.text(f"   Answer: {feedback['answer'][:30]}...")
                st.text(f"   Feedback: {'✅ Relevant' if feedback['relevant'] else '❌ Not Relevant'}")
                st.text("")
        else:
            st.text("No feedback provided yet")

    # Query input with expanded query display
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Type your query here", value=st.session_state.current_query)
    with col2:
        if st.button("Clear History"):
            st.session_state.feedback_history = []
            st.session_state.expanded_queries = {}
            st.session_state.current_query = ""
            st.session_state.last_search_results = []
            st.session_state.last_search_query = ""
            st.rerun()

    if st.button("Search") and query:
        # Update current query in session state
        st.session_state.current_query = query
        
        # Debug information
        st.write(f"🔍 Searching for: '{query}'")
        st.write(f"📊 Top K: {top_k}")
        
        # First try normal search with threshold
        results = ir_system.search(query, top_n=top_k, fallback_threshold=0.01)
        
        # Debug: show number of results found
        st.write(f"🔍 Initial search found: {len(results)} results")
        
        if not results:
            # Try with fuzzy search (threshold = 0.0)
            st.info("🔍 No exact matches found. Trying fuzzy search...")
            results = ir_system.search(query, top_n=top_k, fallback_threshold=0.0)
            
            # Debug: show fuzzy search results
            st.write(f"🔍 Fuzzy search found: {len(results)} results")
            
            if not results:
                # Still no results - show suggestions
                st.warning("❌ No matching FAQs found for your query.")
                
                # Show keyword suggestions
                suggestions = ir_system.suggest_similar_keywords(query, top_n=5)
                if suggestions:
                    st.markdown("**💡 Did you mean one of these keywords?**")
                    suggestion_cols = st.columns(len(suggestions))
                    for i, suggestion in enumerate(suggestions):
                        with suggestion_cols[i]:
                            if st.button(f"🔍 {suggestion}", key=f"suggest_{suggestion}"):
                                st.session_state.current_query = suggestion
                                st.rerun()
                
                # Show some random FAQs as suggestions
                st.markdown("**📚 Here are some popular fitness topics:**")
                import random
                sample_indices = random.sample(range(len(ir_system.raw_questions)), min(5, len(ir_system.raw_questions)))
                for i, idx in enumerate(sample_indices):
                    with st.expander(f"💪 {ir_system.raw_questions[idx][:50]}..."):
                        st.write(ir_system.raw_answers[idx])
                # Store empty results
                st.session_state.last_search_results = []
                st.session_state.last_search_query = query
            else:
                st.info("🔍 Found some related results using fuzzy matching:")
                # Store results in session state
                st.session_state.last_search_results = results
                st.session_state.last_search_query = query
        else:
            st.success(f"✅ Found {len(results)} results")
            # Store results in session state
            st.session_state.last_search_results = results
            st.session_state.last_search_query = query
            
        # Debug: show final results count
        st.write(f"🔍 Final results to display: {len(results)}")
            
        # Display results if any found
        if results:
            for rank, (idx, score, q_text, a_text) in enumerate(results, start=1):
                st.subheader(f"{rank}. {q_text} (score: {score:.4f})")
                st.write(a_text)
                
                # Feedback buttons
                feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 3])
                
                with feedback_col1:
                    if st.button(f"✅ Relevant", key=f"relevant_{rank}_{idx}"):
                        # Expand query with keywords from this answer
                        expanded_query = ir_system.expand_query_with_answer(query, a_text)
                        
                        # Store feedback in session state
                        feedback_entry = {
                            'query': query,
                            'answer': a_text,
                            'relevant': True,
                            'expanded_query': expanded_query
                        }
                        st.session_state.feedback_history.append(feedback_entry)
                        st.session_state.expanded_queries[query] = expanded_query
                        st.session_state.current_query = expanded_query
                        
                        st.success(f"✅ Query expanded! New query: {expanded_query}")
                        st.rerun()
                
                with feedback_col2:
                    if st.button(f"❌ Not Relevant", key=f"not_relevant_{rank}_{idx}"):
                        # Store negative feedback
                        feedback_entry = {
                            'query': query,
                            'answer': a_text,
                            'relevant': False,
                            'expanded_query': None
                        }
                        st.session_state.feedback_history.append(feedback_entry)
                        st.success("❌ Feedback recorded. This helps improve future searches.")
                        st.rerun()
                
                with feedback_col3:
                    st.write("")  # Empty space for alignment
                
                # Show expanded query if available
                if query in st.session_state.expanded_queries:
                    expanded = st.session_state.expanded_queries[query]
                    if expanded != query:
                        st.info(f"🔍 **Expanded Query:** {expanded}")
                
                if show_explain:
                    terms = ir_system.explain(query, idx, top_k_terms=5)
                    if terms:
                        st.markdown("**Top overlapping terms (query vs doc):**")
                        st.write(pd.DataFrame(terms, columns=["term", "overlap_score"]))

    # Display stored results if they exist (for when user clicks feedback buttons)
    elif st.session_state.last_search_results:
        st.markdown("---")
        st.subheader(f"📋 Previous Search Results for: '{st.session_state.last_search_query}'")
        
        for rank, (idx, score, q_text, a_text) in enumerate(st.session_state.last_search_results, start=1):
            st.subheader(f"{rank}. {q_text} (score: {score:.4f})")
            st.write(a_text)
            
            # Feedback buttons
            feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 3])
            
            with feedback_col1:
                if st.button(f"✅ Relevant", key=f"relevant_{rank}_{idx}"):
                    # Expand query with keywords from this answer
                    expanded_query = ir_system.expand_query_with_answer(st.session_state.last_search_query, a_text)
                    
                    # Store feedback in session state
                    feedback_entry = {
                        'query': st.session_state.last_search_query,
                        'answer': a_text,
                        'relevant': True,
                        'expanded_query': expanded_query
                    }
                    st.session_state.feedback_history.append(feedback_entry)
                    st.session_state.expanded_queries[st.session_state.last_search_query] = expanded_query
                    st.session_state.current_query = expanded_query
                    
                    st.success(f"✅ Query expanded! New query: {expanded_query}")
                    st.rerun()
            
            with feedback_col2:
                if st.button(f"❌ Not Relevant", key=f"not_relevant_{rank}_{idx}"):
                    # Store negative feedback
                    feedback_entry = {
                        'query': st.session_state.last_search_query,
                        'answer': a_text,
                        'relevant': False,
                        'expanded_query': None
                    }
                    st.session_state.feedback_history.append(feedback_entry)
                    st.success("❌ Feedback recorded. This helps improve future searches.")
                    st.rerun()
            
            with feedback_col3:
                st.write("")  # Empty space for alignment
            
            # Show expanded query if available
            if st.session_state.last_search_query in st.session_state.expanded_queries:
                expanded = st.session_state.expanded_queries[st.session_state.last_search_query]
                if expanded != st.session_state.last_search_query:
                    st.info(f"🔍 **Expanded Query:** {expanded}")
            
            if show_explain:
                terms = ir_system.explain(st.session_state.last_search_query, idx, top_k_terms=5)
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
            
            # Try normal search first
            res = ir_system.search(q, top_n=5, fallback_threshold=0.01)
            if not res:
                print("No exact matches found. Trying fuzzy search...")
                res = ir_system.search(q, top_n=5, fallback_threshold=0.0)
                
            if not res:
                print("No results found.")
                # Show keyword suggestions
                suggestions = ir_system.suggest_similar_keywords(q, top_n=3)
                if suggestions:
                    print(f"Did you mean: {', '.join(suggestions)}?")
                print()
                continue
                
            print(f"Found {len(res)} results:\n")
            for idx, score, q_text, a_text in res:
                print(f"[{idx}] ({score:.4f}) {q_text}\n-> {a_text}\n")
        return

    # Otherwise run Streamlit UI (works when launched via `streamlit run`)
    run_streamlit_app(ir_system)


if __name__ == "__main__":
    main()


