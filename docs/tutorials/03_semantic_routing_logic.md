# Tutorial: Mastering Semantic Routing

Semantic Routing is the "Brain" of your RAG pipeline. It ensures that queries are handled by the most capable tool, rather than forcing everything through a generic vector search.

## 🔀 The Scenario: Multi-Modal Queries
A production system often receives three distinct types of requests:
1.  **Informational**: "How does the RRF algorithm work?"
2.  **Computational**: "What is 25% of the 2024 marketing budget?"
3.  **Conversational**: "Hello, can you help me with a technical question?"

---

## 🛠️ Step-by-Step: Testing the Router

### 1. View Configured Routes
Go to the **Semantic Router** view in the dashboard. You will see a list of pre-configured routes:
- `calculator`: For arithmetic.
- `sql_agent`: For database-style lookups.
- `vector_db`: For semantic knowledge.
- `conversational`: For small talk.

### 2. Run a Test Route
Enter a query like: *"Is it possible to calculate the ROI of this pipeline?"*
- **Observe**: The router embeds the query and compares it to the "intent centroids" of each route.
- **Trace**: Check the **Confidence Bar**. If it's above 0.8, the router locks onto the target.

### 3. Handling Fallbacks
What if you ask: *"What is the meaning of life?"*
- **Observe**: The similarity might be low across all specific routes.
- **The System Fix**: The router triggers a **Fallback to Hybrid**, ensuring the standard RAG pipeline is used as a safety net.

## 🎓 Why This Matters
By using the router, you save significant LLM costs. A calculator route doesn't even need an LLM call — it can be handled by a Python script, resulting in 0ms latency and 0 cost for your most common queries.

## 🚀 Pro Tip
You can add new routes by modifying the `SEMANTIC_ROUTES` list in [core/router.py](../../core/router.py). Adding just 5 example utterances is enough to create a new, high-precision route.
