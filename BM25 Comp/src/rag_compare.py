import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return WORD_RE.findall(text) + CJK_RE.findall(text)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, target: str) -> float:
    p_tokens = tokenize(normalize_text(prediction))
    t_tokens = tokenize(normalize_text(target))
    if not p_tokens or not t_tokens:
        return 1.0 if p_tokens == t_tokens else 0.0

    p_count = Counter(p_tokens)
    t_count = Counter(t_tokens)
    common = sum((p_count & t_count).values())
    if common == 0:
        return 0.0

    precision = common / len(p_tokens)
    recall = common / len(t_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class SearchResult:
    doc_id: str
    score: float


class BM25Retriever:
    def __init__(self, documents: List[Dict[str, str]], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.doc_term_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.doc_ids: List[str] = []

        df = Counter()
        for doc in documents:
            tokens = tokenize(doc["text"])
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            self.doc_ids.append(doc["id"])
            for term in tf.keys():
                df[term] += 1

        self.num_docs = len(documents)
        self.avg_doc_len = (sum(self.doc_lengths) / self.num_docs) if self.num_docs else 0.0
        self.idf = {
            term: math.log(1.0 + (self.num_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        q_terms = Counter(tokenize(query))
        scores = np.zeros(self.num_docs, dtype=np.float32)

        for idx, tf in enumerate(self.doc_term_freqs):
            dl = self.doc_lengths[idx]
            score = 0.0
            for term, qf in q_terms.items():
                if term not in self.idf:
                    continue
                term_freq = tf.get(term, 0)
                if term_freq == 0:
                    continue
                denom = term_freq + self.k1 * (1.0 - self.b + self.b * dl / (self.avg_doc_len + 1e-12))
                score += self.idf[term] * ((term_freq * (self.k1 + 1.0)) / denom) * qf
            scores[idx] = score

        top_indices = np.argsort(-scores)[:top_k]
        return [SearchResult(doc_id=self.doc_ids[i], score=float(scores[i])) for i in top_indices]


class TfidfVectorRetriever:
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(2, 4),
        )
        self.doc_matrix = self.vectorizer.fit_transform([doc["text"] for doc in documents])

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self.doc_matrix.T).toarray()[0]
        top_indices = np.argsort(-scores)[:top_k]
        return [SearchResult(doc_id=self.doc_ids[i], score=float(scores[i])) for i in top_indices]


class DenseVectorRetriever:
    def __init__(self, documents: List[Dict[str, str]], model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers` "
                "or switch --vector_backend to tfidf."
            ) from exc

        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(
            [doc["text"] for doc in documents],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        q_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        scores = self.doc_embeddings @ q_embedding
        top_indices = np.argsort(-scores)[:top_k]
        return [SearchResult(doc_id=self.doc_ids[i], score=float(scores[i])) for i in top_indices]


class HybridRRF:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever,
        rrf_k: int = 60,
        rank_window: int = 50,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.rrf_k = rrf_k
        self.rank_window = rank_window
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        sparse = self.bm25.search(query, top_k=self.rank_window)
        dense = self.vector.search(query, top_k=self.rank_window)

        fused_scores = defaultdict(float)

        for rank, result in enumerate(sparse, start=1):
            fused_scores[result.doc_id] += self.bm25_weight / (self.rrf_k + rank)

        for rank, result in enumerate(dense, start=1):
            fused_scores[result.doc_id] += self.vector_weight / (self.rrf_k + rank)

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [SearchResult(doc_id=doc_id, score=score) for doc_id, score in ranked]


def split_sentences(text: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


def generate_extractive_answer(question: str, contexts: Iterable[str]) -> str:
    contexts = list(contexts)
    q_tokens = set(tokenize(question))
    best_sentence = ""
    best_score = -1.0

    for ctx in contexts:
        for sentence in split_sentences(ctx):
            s_tokens = set(tokenize(sentence))
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens)
            score = 0.7 * (overlap / (len(q_tokens) + 1e-12)) + 0.3 * (overlap / (len(s_tokens) + 1e-12))
            if score > best_score:
                best_score = score
                best_sentence = sentence

    if best_sentence:
        return best_sentence[:220]

    return contexts[0][:220] if contexts else "I do not know."


def evaluate(
    name: str,
    retriever,
    docs_by_id: Dict[str, Dict[str, str]],
    qa_items: List[Dict[str, object]],
    top_k: int,
    context_k: int,
) -> Dict[str, object]:
    hit_total = 0.0
    recall_total = 0.0
    mrr_total = 0.0
    em_total = 0.0
    f1_total = 0.0

    per_question = []

    for item in qa_items:
        qid = item["id"]
        question = item["question"]
        answers = item["answers"]
        gold_doc_ids = set(item["gold_doc_ids"])

        ranked = retriever.search(question, top_k=top_k)
        ranked_ids = [r.doc_id for r in ranked]

        hit = 1.0 if any(doc_id in gold_doc_ids for doc_id in ranked_ids) else 0.0
        hit_total += hit

        recall = len(gold_doc_ids.intersection(ranked_ids)) / max(1, len(gold_doc_ids))
        recall_total += recall

        rr = 0.0
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in gold_doc_ids:
                rr = 1.0 / rank
                break
        mrr_total += rr

        contexts = [docs_by_id[doc_id]["text"] for doc_id in ranked_ids[:context_k] if doc_id in docs_by_id]
        prediction = generate_extractive_answer(question, contexts)

        em = max(1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0 for gold in answers)
        f1 = max(token_f1(prediction, gold) for gold in answers)
        em_total += em
        f1_total += f1

        per_question.append(
            {
                "qid": qid,
                "question": question,
                "top_docs": ranked_ids,
                "gold_docs": sorted(gold_doc_ids),
                "prediction": prediction,
                "answers": answers,
                "hit": hit,
                "recall": recall,
                "mrr": rr,
                "em": em,
                "f1": f1,
            }
        )

    n = len(qa_items)
    return {
        "pipeline": name,
        "metrics": {
            "Hit@k": hit_total / n,
            "Recall@k": recall_total / n,
            "MRR@k": mrr_total / n,
            "EM": em_total / n,
            "F1": f1_total / n,
        },
        "per_question": per_question,
    }


def print_summary(results: List[Dict[str, object]]) -> None:
    headers = ["Pipeline", "Hit@k", "Recall@k", "MRR@k", "EM", "F1"]
    row_fmt = "| {0:28} | {1:7} | {2:8} | {3:6} | {4:5} | {5:5} |"

    print("\n=== Retrieval + RAG Comparison ===")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|")

    for res in results:
        m = res["metrics"]
        print(
            row_fmt.format(
                res["pipeline"],
                f"{m['Hit@k']:.3f}",
                f"{m['Recall@k']:.3f}",
                f"{m['MRR@k']:.3f}",
                f"{m['EM']:.3f}",
                f"{m['F1']:.3f}",
            )
        )


def build_vector_retriever(
    backend: str,
    documents: List[Dict[str, str]],
    embedding_model: str,
):
    if backend == "tfidf":
        return TfidfVectorRetriever(documents), "tfidf"

    if backend == "sbert":
        return DenseVectorRetriever(documents, model_name=embedding_model), "sbert"

    if backend == "auto":
        try:
            retriever = DenseVectorRetriever(documents, model_name=embedding_model)
            return retriever, "sbert"
        except Exception as exc:
            print(
                f"[WARN] Dense retriever init failed ({exc}). Falling back to TF-IDF vector retriever.",
                file=sys.stderr,
            )
            return TfidfVectorRetriever(documents), "tfidf_fallback"

    raise ValueError(f"Unknown vector backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BM25, vector RAG, and BM25+vector hybrid RAG")
    parser.add_argument("--dataset", type=Path, default=Path("data/sample_rag_qa.json"))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--context_k", type=int, default=3)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--rrf_rank_window", type=int, default=50)
    parser.add_argument("--bm25_weight", type=float, default=0.5)
    parser.add_argument("--vector_weight", type=float, default=1.5)
    parser.add_argument("--vector_backend", choices=["auto", "tfidf", "sbert"], default="tfidf")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--output", type=Path, default=Path("results/comparison_results.json"))
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    documents = payload["documents"]
    qa_items = payload["qa"]
    docs_by_id = {doc["id"]: doc for doc in documents}

    bm25 = BM25Retriever(documents=documents, k1=1.2, b=0.75)
    vector, vector_backend_used = build_vector_retriever(
        backend=args.vector_backend,
        documents=documents,
        embedding_model=args.embedding_model,
    )

    hybrid = HybridRRF(
        bm25_retriever=bm25,
        vector_retriever=vector,
        rrf_k=args.rrf_k,
        rank_window=args.rrf_rank_window,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
    )

    results = [
        evaluate("bm25_rag", bm25, docs_by_id, qa_items, top_k=args.top_k, context_k=args.context_k),
        evaluate(f"vector_rag_{vector_backend_used}", vector, docs_by_id, qa_items, top_k=args.top_k, context_k=args.context_k),
        evaluate(
            f"hybrid_rrf_rag_{vector_backend_used}",
            hybrid,
            docs_by_id,
            qa_items,
            top_k=args.top_k,
            context_k=args.context_k,
        ),
    ]

    print_summary(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "top_k": args.top_k,
                    "context_k": args.context_k,
                    "rrf_k": args.rrf_k,
                    "rrf_rank_window": args.rrf_rank_window,
                    "bm25_weight": args.bm25_weight,
                    "vector_weight": args.vector_weight,
                    "vector_backend_requested": args.vector_backend,
                    "vector_backend_used": vector_backend_used,
                    "embedding_model": args.embedding_model,
                    "dataset": str(args.dataset),
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved detailed results to: {args.output}")


if __name__ == "__main__":
    main()
