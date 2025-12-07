# Indexing NCERT Content with Cloudflare Embeddings

## ✓ Fixed Issues

1. **Embedding dimension mismatch** - ChromaDB collection recreated for 768d embeddings
2. **NCERT processor** - Now uses Cloudflare AI when enabled
3. **NLTK tokenizer** - Downloads punkt_tab automatically
4. **Environment loading** - Fixed dotenv loading order

## Ready to Index!

Your system is now configured to use **Cloudflare BGE embeddings (768 dimensions)** for indexing.

### Run the Indexing

```bash
python setup_ncert_pipeline.py
```

When prompted "Do you want to process the PDF files now? (y/n):", type **y**

### What Will Happen

The script will:
1. ✓ Check dependencies
2. ✓ Create necessary directories
3. ✓ Find your PDF files (137 PDFs detected)
4. ✓ Process each PDF:
   - Extract text from pages
   - Split into ~500-word chunks
   - Generate embeddings using **Cloudflare AI** (768d)
   - Store in ChromaDB
5. ✓ Test with a sample query

### Processing Time

- **With Cloudflare AI**: ~5-10 minutes for 137 PDFs
- Each document is embedded using Cloudflare's fast API
- Progress is logged for each batch

### Monitoring Progress

Watch the logs for:
```
INFO - Using Cloudflare AI for embeddings (768d)
INFO - Processing Biology_11_Ch01.pdf...
INFO - Stored batch 1 (100 chunks)
INFO - Completed processing Biology_11_Ch01.pdf: 45 chunks created
```

### After Indexing

Your vector store will contain:
- ~3000-4000 chunks from NCERT textbooks
- Each chunk with 768-dimensional embeddings
- Metadata: subject, class, chapter, page number

### Verify Success

After indexing completes, test the system:

```bash
# Test RAG system
python -c "from services.rag_system import RAGSystem; rag = RAGSystem(); print(rag.get_stats())"

# Start the app
python app.py
```

Then visit http://127.0.0.1:5001 and ask a question!

## Comparison: Before vs After

| Aspect | Before (Local) | After (Cloudflare) |
|--------|---------------|-------------------|
| Embedding Model | all-MiniLM-L6-v2 | BGE-Base-en-v1.5 |
| Dimensions | 384 | 768 |
| Embedding Speed | ~100ms/doc | ~50ms/doc |
| Quality | Good | Better |
| Chat Response | 10-30s | 2-3s |
| Requires Internet | No | Yes |

## Troubleshooting

### If you see "Using local embeddings (384d)":

Check your `.env` file has:
```env
USE_CLOUDFLARE_AI=true
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_api_token
```

### If Cloudflare API fails:

The system will log errors but continue with zero vectors. Check:
1. Internet connection
2. Cloudflare credentials are valid
3. API rate limits not exceeded

### If you want to use local embeddings instead:

1. Edit `.env`: `USE_CLOUDFLARE_AI=false`
2. Run: `python auto_fix_embeddings.py`
3. Re-index: `python setup_ncert_pipeline.py`

## Next Steps

1. **Run indexing**: `python setup_ncert_pipeline.py`
2. **Wait for completion**: ~5-10 minutes
3. **Start app**: `python app.py`
4. **Test queries**: Visit http://127.0.0.1:5001

Enjoy your fast, Cloudflare-powered AI tutor! 🚀
