# BM25 與向量 RAG：上網整理與實作路線

## 1) 核心背景（論文 / 官方文件）

- **RAG 定義**：RAG 把參數記憶（seq2seq）和非參數記憶（外部檢索索引）結合，先檢索再生成。來源：Lewis et al., 2020。
  - https://arxiv.org/abs/2005.11401
- **Dense Retriever 代表**：DPR 使用雙編碼器（question encoder + passage encoder）做向量檢索。來源：Karpukhin et al., 2020。
  - https://arxiv.org/abs/2004.04906
- **BM25 常用設定與公式**：Lucene `BM25Similarity` 預設 `k1=1.2, b=0.75`，IDF 形式為 `log(1 + (docCount - docFreq + 0.5)/(docFreq + 0.5))`。
  - https://lucene.apache.org/core/9_9_1/core/org/apache/lucene/search/similarities/BM25Similarity.html
- **Hybrid / 融合檢索**：Weaviate 文件指出可並行執行向量檢索與 BM25，並用融合策略（`relativeScoreFusion`、`rankedFusion`）合併分數。
  - https://docs.weaviate.io/weaviate/concepts/search/hybrid-search
- **RRF（Reciprocal Rank Fusion）**：原始論文公式 `RRFscore(d)=sum(1/(k+r(d)))`，文中使用 `k=60`。
  - https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

## 2) 可行實作方法

### 方法 A：本地最小可重現（本 repo 目前採用）
- 檢索器：
  - BM25（自實作）
  - 向量檢索（TF-IDF 向量空間 + cosine）
  - Hybrid（RRF 融合 BM25 與向量排名）
- 生成器：
  - 先用 extractive 生成（離線可跑）
  - 後續可替換任意 LLM API
- 優點：快速、可重現、零外部服務依賴
- 缺點：向量檢索不是神經 embedding，語義能力有限

### 方法 B：Elasticsearch / OpenSearch 生產化
- 用 BM25 + kNN（dense vector）建立單一檢索服務
- 以 RRF 或 weighted fusion 做混合排序
- 優點：可擴展、方便監控與部署
- 缺點：維運成本較高

### 方法 C：向量資料庫 Hybrid（如 Weaviate）
- 直接使用 Hybrid API，透過 `alpha` 調整向量與關鍵字權重
- 可選 `relativeScoreFusion`（保留原始分數差異）
- 優點：混合檢索配置快
- 缺點：需依賴資料庫平台生態

### 方法 D：神經檢索（DPR/e5/bge）+ FAISS + BM25
- 向量端採用 sentence-transformers / bge embedding
- 稀疏端保留 BM25
- 融合端可用 RRF 或 learning-to-rank
- 優點：語義效果更好
- 缺點：模型下載、GPU/CPU 成本與延遲較高

## 3) 評估建議

- 檢索：`Recall@k`, `Hit@k`, `MRR@k`
- 端到端 QA：`Exact Match`, `token-F1`
- 做法：同一組 query / gold evidence 下，分別跑 BM25、向量、Hybrid 三條 pipeline 比較。
