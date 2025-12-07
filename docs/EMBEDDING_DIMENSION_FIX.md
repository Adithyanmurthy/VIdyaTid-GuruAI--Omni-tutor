# Embedding Dimension Mismatch Fix

## Problem

You're seeing this error:
```
Error during retrieval: Collection expecting embedding with dimension of 384, got 768
```

This happens because:
- **Old collection**: Created with local embeddings (384 dimensions)
- **New system**: Using Cloudflare BGE embeddings (768 dimensions)

## Solution Options

### Option 1: Quick Fix (Recommended for Testing)

**Recreate empty collection with correct dimensions:**

```bash
python fix_embedding_dimensions.py
```

This will:
- Delete the old collection
- Create a new empty collection with correct dimensions
- You'll need to re-index content afterward

**Pros:**
- Fast (takes seconds)
- Clean start

**Cons:**
- Loses existing embeddings
- Need to re-index content

### Option 2: Full Migration (Recommended for Production)

**Migrate existing data to Cloudflare embeddings:**

```bash
python migrate_embeddings_to_cloudflare.py
```

This will:
- Backup existing documents
- Re-embed all documents using Cloudflare AI
- Preserve all your indexed content

**Pros:**
- Keeps all existing data
- No need to re-index

**Cons:**
- Takes longer (several minutes for large collections)
- Uses Cloudflare AI API calls

### Option 3: Disable Cloudflare Embeddings

**Use local embeddings instead:**

Edit your `.env` file:
```env
USE_CLOUDFLARE_AI=false
```

Or remove the Cloudflare credentials temporarily.

**Pros:**
- No migration needed
- Works with existing collection

**Cons:**
- Slower embeddings
- Can't use Cloudflare chat model

## After Fixing

### If you used Option 1 (Quick Fix):

Re-index your NCERT content:

```bash
# Index NCERT textbooks
python setup_ncert_pipeline.py

# Index diagrams (optional)
python setup_diagram_indexing_final.py

# Index previous papers (optional)
python setup_previous_papers_complete.py
```

### If you used Option 2 (Full Migration):

You're done! The system should work immediately.

### If you used Option 3 (Disable Cloudflare):

Restart your app:
```bash
python app.py
```

## Verification

Test the system:

```bash
# Test Cloudflare AI
python test_cloudflare_ai.py

# Test RAG system
python -c "from services.rag_system import RAGSystem; rag = RAGSystem(); print(rag.get_stats())"
```

## Understanding the Issue

### Embedding Dimensions

Different embedding models produce vectors of different sizes:

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 (local) | 384 | Fast | Good |
| BGE-Base-en-v1.5 (Cloudflare) | 768 | Very Fast | Better |

### Why This Matters

- ChromaDB stores vectors with a fixed dimension
- All vectors in a collection must have the same dimension
- Mixing dimensions causes the error you saw

### The Fix

You need to either:
1. Use consistent embeddings (all local OR all Cloudflare)
2. Recreate the collection when switching embedding models

## Preventing This in the Future

When switching embedding models:
1. Always check the dimension compatibility
2. Plan for migration or re-indexing
3. Test with a small dataset first

## Need Help?

If you encounter issues:

1. Check your `.env` file has correct Cloudflare credentials
2. Verify Cloudflare AI is working: `python test_cloudflare_ai.py`
3. Check ChromaDB path exists: `ls -la chroma_db/`
4. Review logs in `logs/` directory

## Quick Reference

```bash
# Check current collection status
python -c "import chromadb; from config import Config; c = chromadb.PersistentClient(path=str(Config.CHROMA_DB_PATH)); coll = c.get_collection(Config.CHROMA_COLLECTION_NAME); print(f'Documents: {coll.count()}')"

# Delete collection manually
python -c "import chromadb; from config import Config; c = chromadb.PersistentClient(path=str(Config.CHROMA_DB_PATH)); c.delete_collection(Config.CHROMA_COLLECTION_NAME); print('Deleted')"

# Check Cloudflare AI status
python -c "from services.cloudflare_ai import is_cloudflare_ai_enabled; print(f'Cloudflare AI: {is_cloudflare_ai_enabled()}')"
```
