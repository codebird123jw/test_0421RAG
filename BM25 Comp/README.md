# BM25 vs 傳統向量 RAG 比較（含 BM25+RAG 混合流程）

這個資料夾已經實作三條可直接執行的 pipeline：

- `bm25_rag`：BM25 檢索 + 簡化 RAG 生成
- `vector_rag_tfidf`：傳統向量空間（TF-IDF 字元 n-gram + cosine）+ 簡化 RAG 生成
- `hybrid_rrf_rag_tfidf`：BM25 + 向量檢索，以 RRF 融合後再做 RAG

## 專案結構

- `src/rag_compare.py`：主程式（檢索、融合、評估）
- `data/sample_rag_qa.json`：示範資料（documents + QA + gold evidence）
- `results/comparison_results.json`：執行後的詳細結果
- `research_notes.md`：上網整理的理論與可行實作方法
- `requirements.txt`：最小依賴

## 執行方式

```bash
python src/rag_compare.py
```

可調整參數：

```bash
python src/rag_compare.py \
  --top_k 5 \
  --context_k 3 \
  --rrf_k 60 \
  --bm25_weight 0.5 \
  --vector_weight 1.5 \
  --vector_backend tfidf
```

## 目前一次執行結果（本機）

| Pipeline | Hit@k | Recall@k | MRR@k | EM | F1 |
|---|---:|---:|---:|---:|---:|
| bm25_rag | 0.667 | 0.667 | 0.461 | 0.000 | 0.040 |
| vector_rag_tfidf | 0.750 | 0.750 | 0.621 | 0.000 | 0.041 |
| hybrid_rrf_rag_tfidf | 0.750 | 0.750 | 0.621 | 0.000 | 0.041 |

說明：
- 這裡的 RAG 生成器是離線的 extractive baseline（不是雲端 LLM），所以 EM/F1 偏低屬正常。
- 本範例重點在檢索路徑比較與融合策略驗證。

## 若要改成 Dense Embedding 向量 RAG

程式已支援：

```bash
python src/rag_compare.py --vector_backend sbert \
  --embedding_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

若環境無法連線 HuggingFace，建議先維持 `--vector_backend tfidf`。

## 研究與方法來源

請看 `research_notes.md`，包含：
- RAG / DPR 論文
- Lucene BM25 公式與預設參數
- RRF 原始論文
- Hybrid 檢索官方文件與可落地做法
