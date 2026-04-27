# Tutorial: Context Recovery (Parent-Child Auto-Merging)

This tutorial explains how to solve the "Small Chunk Context Loss" problem using hierarchical retrieval.

## 🧩 The Granularity Paradox
- **Small chunks (100 tokens)** are excellent for **retrieval**. They have a high signal-to-noise ratio and align well with short queries.
- **Large chunks (1000 tokens)** are excellent for **generation**. They provide the LLM with enough context to give a nuanced answer.

How do we get both? **Parent-Child Auto-Merging**.

---

## 🛠️ Step-by-Step: Implementing the Hierarchy

### 1. Ingest with Parent-Child Strategy
In the **Ingest Documents** view, select the **Parent-Child** strategy.
- **Process**: The system creates a large "Parent" chunk and many small "Child" chunks that belong to it.

### 2. The Retrieval Stage
Go to the **Query Explorer** and enable the **Auto-Merge** toggle.
- When you run a query, the system searches for the small **Child Chunks**.
- Look at the **Retrieved Chunks** list. You might see a badge: `↑ parent merged`.

### 3. Verification
- **Trace the Stage**: In the **Pipeline Trace**, you will see the `auto_merge` stage. 
- It identifies if multiple child chunks belong to the same parent.
- If they do, it discards the children and fetches the full, original parent context for the LLM.

## 🎓 Summary
Auto-merging gives your system "X-ray vision." It can find the exact sentence the user wants (Child) and then zoom out to see the entire paragraph or section (Parent) before answering. This prevents the LLM from giving "truncated" or incomplete answers.
