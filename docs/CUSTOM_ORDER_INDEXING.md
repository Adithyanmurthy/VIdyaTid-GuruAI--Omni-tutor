# Custom Order NCERT Indexing

## What's Different

The new script `setup_ncert_custom_order.py` processes PDFs in this specific order:

1. **Chemistry** (all chapters)
2. **Physics** (all chapters)
3. **Mathematics** (all chapters)
4. **Biology Class 11** (all chapters)
5. **Biology Class 12** (all chapters)

## Why This Order?

- Process JEE/NEET priority subjects first (Chemistry, Physics)
- Mathematics next
- Biology last (split by class)

## How to Use

### Step 1: Stop Current Process

If the old script is still running, press **Ctrl+C** to stop it.

### Step 2: Fresh Start (Already Done)

✓ Collection has been reset and is ready for indexing

### Step 3: Run Custom Order Script

```bash
python setup_ncert_custom_order.py
```

### Step 4: Confirm

When prompted:
```
Proceed with indexing? (yes/no):
```

Type **yes** and press Enter.

## What You'll See

The script will show clear section headers:

```
======================================================================
📚 CHEMISTRY
======================================================================
[1/137] Processing Chemistry_11_Part1_Ch01.pdf...
✓ Completed: 25 chunks

[2/137] Processing Chemistry_11_Part1_Ch02.pdf...
✓ Completed: 30 chunks

...

======================================================================
⚛️  PHYSICS
======================================================================
[35/137] Processing Physics_11_Part1_Ch01.pdf...
✓ Completed: 28 chunks

...

======================================================================
🔢 MATHEMATICS
======================================================================
[70/137] Processing Mathematics_11_Ch01.pdf...
✓ Completed: 32 chunks

...

======================================================================
🧬 BIOLOGY CLASS 11
======================================================================
[105/137] Processing Biology_11_Ch01.pdf...
✓ Completed: 11 chunks

...

======================================================================
🧬 BIOLOGY CLASS 12
======================================================================
[125/137] Processing Biology_12_Ch01.pdf...
✓ Completed: 15 chunks
```

## Processing Time

- **Total time**: ~10-15 minutes for all 137 PDFs
- **Chemistry**: ~3-4 minutes
- **Physics**: ~3-4 minutes
- **Mathematics**: ~2-3 minutes
- **Biology 11**: ~2-3 minutes
- **Biology 12**: ~2-3 minutes

## Features

✓ Clear subject section headers
✓ Progress counter [current/total]
✓ Cloudflare AI embeddings (768d)
✓ Error handling (continues on failure)
✓ Test queries at the end
✓ Can be interrupted and resumed

## After Completion

You'll see:
```
======================================================================
INDEXING COMPLETE!
======================================================================

📊 Summary:
   Total PDFs processed: 137
   Total chunks created: ~3500
   Embedding model: Cloudflare BGE (768d)

📦 Vector Store:
   Collection: ncert_content
   Total documents: ~3500

✓ Ready to use! Start your app with: python app.py
```

## Interrupting and Resuming

If you need to stop:
- Press **Ctrl+C**
- Progress is saved automatically
- Run the script again to continue (it will skip already processed PDFs)

## Comparison: Old vs New Script

| Feature | Old Script | New Script |
|---------|-----------|------------|
| Order | Alphabetical | Custom (Chem→Phy→Math→Bio11→Bio12) |
| Progress | Basic | Detailed with sections |
| Subject headers | No | Yes (with emojis) |
| Counter | No | Yes [current/total] |
| Test queries | 1 | 3 different subjects |

## Ready to Start!

Run this command:
```bash
python setup_ncert_custom_order.py
```

Type **yes** when prompted, and watch your content get indexed in the perfect order! 🚀
