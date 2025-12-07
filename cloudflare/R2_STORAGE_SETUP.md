# Cloudflare R2 Storage Setup Guide

## Overview

Cloudflare R2 is object storage compatible with Amazon S3 API, but without egress fees. This guide covers setting up R2 for VidyaTid file storage.

## What to Store in R2

- **NCERT Content**: PDF files and extracted text
- **Diagrams**: All educational diagrams and images
- **Previous Year Papers**: JEE/NEET question papers
- **User Uploads**: Student-uploaded images for doubt solving
- **Backups**: Database backups and exports

## Prerequisites

- Cloudflare account
- Wrangler CLI installed (`npm install -g wrangler`)
- Logged in to Wrangler (`wrangler login`)

## Step 1: Create R2 Bucket

### Using Wrangler CLI

```bash
# Create production bucket
wrangler r2 bucket create guruai-storage

# Create staging bucket (optional)
wrangler r2 bucket create guruai-storage-staging
```

### Using Cloudflare Dashboard

1. Go to https://dash.cloudflare.com
2. Navigate to **R2** in the sidebar
3. Click **Create bucket**
4. Enter bucket name: `guruai-storage`
5. Click **Create bucket**

## Step 2: Generate R2 API Tokens

### Create API Token

1. In R2 dashboard, click **Manage R2 API Tokens**
2. Click **Create API token**
3. Configure token:
   - **Token name**: `guruai-r2-access`
   - **Permissions**: Object Read & Write
   - **Bucket**: `guruai-storage` (or All buckets)
   - **TTL**: Never expire (or set expiration)
4. Click **Create API Token**
5. **Save the credentials**:
   - Access Key ID
   - Secret Access Key
   - Endpoint URL

**Important**: Save these credentials immediately - you won't be able to see them again!

## Step 3: Configure Environment Variables

Add to your `.env` file:

```env
# Cloudflare R2 Configuration
CLOUDFLARE_R2_BUCKET=guruai-storage
CLOUDFLARE_R2_ACCESS_KEY=your_access_key_id_here
CLOUDFLARE_R2_SECRET_KEY=your_secret_access_key_here
CLOUDFLARE_R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com
CLOUDFLARE_R2_PUBLIC_URL=https://storage.vidyatid.com
```

## Step 4: Upload Existing Files

### Using Python Script

Create `cloudflare/upload_to_r2.py`:

```python
"""
Upload existing files to Cloudflare R2
"""
import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure R2 client (S3-compatible)
s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('CLOUDFLARE_R2_ENDPOINT'),
    aws_access_key_id=os.getenv('CLOUDFLARE_R2_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('CLOUDFLARE_R2_SECRET_KEY'),
    region_name='auto'
)

bucket_name = os.getenv('CLOUDFLARE_R2_BUCKET')

def upload_directory(local_path, r2_prefix):
    """Upload a directory to R2"""
    local_path = Path(local_path)
    
    for file_path in local_path.rglob('*'):
        if file_path.is_file():
            # Calculate relative path
            relative_path = file_path.relative_to(local_path)
            r2_key = f"{r2_prefix}/{relative_path}".replace('\\', '/')
            
            print(f"Uploading {file_path} to {r2_key}...")
            
            try:
                s3.upload_file(
                    str(file_path),
                    bucket_name,
                    r2_key,
                    ExtraArgs={'ContentType': get_content_type(file_path)}
                )
                print(f"  ✓ Uploaded")
            except Exception as e:
                print(f"  ✗ Failed: {e}")

def get_content_type(file_path):
    """Get content type based on file extension"""
    ext = file_path.suffix.lower()
    content_types = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.txt': 'text/plain',
        '.json': 'application/json',
        '.html': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript'
    }
    return content_types.get(ext, 'application/octet-stream')

def main():
    print("=" * 70)
    print("VidyaTid - Upload Files to Cloudflare R2")
    print("=" * 70)
    
    # Upload NCERT content
    print("\n1. Uploading NCERT PDFs...")
    upload_directory('ncert_content/pdfs', 'ncert/pdfs')
    
    print("\n2. Uploading NCERT extracted text...")
    upload_directory('ncert_content/extracted', 'ncert/extracted')
    
    print("\n3. Uploading diagrams...")
    upload_directory('diagrams', 'diagrams')
    
    print("\n4. Uploading previous papers...")
    upload_directory('previous_papers', 'previous-papers')
    
    print("\n" + "=" * 70)
    print("✅ Upload complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()
```

Run the upload script:

```bash
pip install boto3
python cloudflare/upload_to_r2.py
```

### Using Wrangler CLI

```bash
# Upload a single file
wrangler r2 object put guruai-storage/ncert/pdfs/Biology_11_Ch01.pdf --file=ncert_content/pdfs/Biology_11_Ch01.pdf

# Upload directory (requires scripting)
# See Python script above for batch uploads
```

## Step 5: Configure CDN URLs

### Option A: R2 Public Buckets (Simple)

1. In R2 dashboard, select your bucket
2. Click **Settings**
3. Enable **Public access**
4. Note the public URL: `https://pub-xxxxx.r2.dev`

### Option B: Custom Domain (Recommended)

1. In R2 dashboard, select your bucket
2. Click **Settings** → **Custom Domains**
3. Click **Connect Domain**
4. Enter domain: `storage.vidyatid.com`
5. Follow DNS configuration instructions
6. Wait for SSL certificate provisioning

**Benefits of custom domain:**
- Professional URLs
- Better caching control
- Custom SSL certificates
- No Cloudflare branding

## Step 6: Update Application Code

### Create R2 Service

Create `services/r2_storage.py`:

```python
"""
Cloudflare R2 storage service
"""
import boto3
import logging
from typing import Optional, BinaryIO
from config import Config

logger = logging.getLogger(__name__)

class R2Storage:
    """Cloudflare R2 storage client"""
    
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=Config.CLOUDFLARE_R2_ENDPOINT,
            aws_access_key_id=Config.CLOUDFLARE_R2_ACCESS_KEY,
            aws_secret_access_key=Config.CLOUDFLARE_R2_SECRET_KEY,
            region_name='auto'
        )
        self.bucket = Config.CLOUDFLARE_R2_BUCKET
        self.public_url = Config.CLOUDFLARE_R2_PUBLIC_URL
    
    def upload_file(self, file_path: str, key: str, content_type: str = None) -> str:
        """Upload a file to R2"""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.client.upload_file(file_path, self.bucket, key, ExtraArgs=extra_args)
            return f"{self.public_url}/{key}"
        except Exception as e:
            logger.error(f"Failed to upload {key}: {e}")
            raise
    
    def upload_fileobj(self, file_obj: BinaryIO, key: str, content_type: str = None) -> str:
        """Upload a file object to R2"""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.client.upload_fileobj(file_obj, self.bucket, key, ExtraArgs=extra_args)
            return f"{self.public_url}/{key}"
        except Exception as e:
            logger.error(f"Failed to upload {key}: {e}")
            raise
    
    def download_file(self, key: str, file_path: str):
        """Download a file from R2"""
        try:
            self.client.download_file(self.bucket, key, file_path)
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            raise
    
    def get_url(self, key: str) -> str:
        """Get public URL for a file"""
        return f"{self.public_url}/{key}"
    
    def delete_file(self, key: str):
        """Delete a file from R2"""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            raise
    
    def list_files(self, prefix: str = '') -> list:
        """List files in R2 with optional prefix"""
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise

# Global instance
_r2_storage = None

def get_r2_storage() -> R2Storage:
    """Get or create R2Storage instance"""
    global _r2_storage
    if _r2_storage is None:
        _r2_storage = R2Storage()
    return _r2_storage
```

### Update File Access Paths

Update your application to use R2 URLs:

```python
# Before (local file system)
diagram_path = f"diagrams/Biology/class11_ch1/diagram1.png"

# After (R2 storage)
from services.r2_storage import get_r2_storage
r2 = get_r2_storage()
diagram_url = r2.get_url("diagrams/Biology/class11_ch1/diagram1.png")
```

## Step 7: Configure Caching

### Cache-Control Headers

Set appropriate cache headers when uploading:

```python
s3.upload_file(
    file_path,
    bucket_name,
    key,
    ExtraArgs={
        'ContentType': content_type,
        'CacheControl': 'public, max-age=31536000',  # 1 year for static content
        'Metadata': {
            'uploaded-by': 'guruai',
            'upload-date': datetime.utcnow().isoformat()
        }
    }
)
```

### Cloudflare Cache Rules

1. Go to Cloudflare Dashboard
2. Select your domain
3. Navigate to **Rules** → **Page Rules**
4. Create rule for `storage.vidyatid.com/*`:
   - Cache Level: Cache Everything
   - Edge Cache TTL: 1 month
   - Browser Cache TTL: 1 month

## Step 8: Test R2 Integration

Create `cloudflare/test_r2.py`:

```python
"""
Test R2 storage integration
"""
from services.r2_storage import get_r2_storage
import tempfile
import os

def test_r2():
    print("Testing Cloudflare R2 Storage...")
    
    r2 = get_r2_storage()
    
    # Test upload
    print("\n1. Testing upload...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content for R2")
        temp_path = f.name
    
    try:
        url = r2.upload_file(temp_path, 'test/test-file.txt', 'text/plain')
        print(f"   ✓ Uploaded: {url}")
    finally:
        os.unlink(temp_path)
    
    # Test list
    print("\n2. Testing list...")
    files = r2.list_files('test/')
    print(f"   ✓ Found {len(files)} files in test/")
    
    # Test download
    print("\n3. Testing download...")
    download_path = tempfile.mktemp(suffix='.txt')
    try:
        r2.download_file('test/test-file.txt', download_path)
        with open(download_path, 'r') as f:
            content = f.read()
        print(f"   ✓ Downloaded: {content}")
    finally:
        if os.path.exists(download_path):
            os.unlink(download_path)
    
    # Test delete
    print("\n4. Testing delete...")
    r2.delete_file('test/test-file.txt')
    print("   ✓ Deleted")
    
    print("\n✅ All R2 tests passed!")

if __name__ == '__main__':
    test_r2()
```

## Cost Estimation

### Free Tier
- **Storage**: 10 GB
- **Class A operations**: 1 million/month (writes, lists)
- **Class B operations**: 10 million/month (reads)
- **Egress**: FREE (no egress fees!)

### Paid Tier (if needed)
- **Storage**: $0.015/GB/month
- **Class A operations**: $4.50 per million
- **Class B operations**: $0.36 per million

### Example Costs for VidyaTid

Assuming:
- 5 GB of content (NCERT + diagrams)
- 100,000 file reads/day = 3M/month
- 1,000 file writes/day = 30k/month

**Monthly Cost:**
- Storage: FREE (under 10 GB)
- Reads: FREE (under 10M)
- Writes: FREE (under 1M)
- Egress: FREE (always)

**Total: $0/month** (within free tier)

## Best Practices

### 1. Organize Files Hierarchically

```
guruai-storage/
├── ncert/
│   ├── pdfs/
│   │   ├── Biology_11_Ch01.pdf
│   │   └── ...
│   └── extracted/
│       ├── Biology_11_Ch01_extracted.txt
│       └── ...
├── diagrams/
│   ├── Biology/
│   │   ├── class11_ch1/
│   │   └── ...
│   └── ...
├── previous-papers/
│   ├── JEE_MAIN/
│   └── NEET/
└── user-uploads/
    ├── 2024/
    │   ├── 01/
    │   └── ...
    └── ...
```

### 2. Use Appropriate Content Types

Always set correct Content-Type headers:
- PDFs: `application/pdf`
- Images: `image/png`, `image/jpeg`
- Text: `text/plain`
- JSON: `application/json`

### 3. Implement Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upload_with_retry(file_path, key):
    return r2.upload_file(file_path, key)
```

### 4. Monitor Usage

Check R2 usage in Cloudflare Dashboard:
- Storage used
- Operations count
- Bandwidth (though it's free!)

### 5. Implement Lifecycle Policies

For user uploads, implement cleanup:
- Delete files older than 90 days
- Move old files to cheaper storage
- Compress large files

## Troubleshooting

### Issue: "Access Denied"

**Solution**: Verify API token permissions
```bash
# Test with wrangler
wrangler r2 object list guruai-storage
```

### Issue: "Endpoint not found"

**Solution**: Check endpoint URL format
```
Correct: https://xxxxx.r2.cloudflarestorage.com
Wrong: https://r2.cloudflarestorage.com/xxxxx
```

### Issue: "Slow uploads"

**Solution**: Use multipart uploads for large files
```python
# For files > 100MB, use multipart upload
s3.upload_file(
    file_path,
    bucket,
    key,
    Config=boto3.s3.transfer.TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # 100MB
        multipart_chunksize=10 * 1024 * 1024    # 10MB chunks
    )
)
```

## Next Steps

After R2 setup:
1. ✅ Update application to use R2 URLs
2. ✅ Test file uploads and downloads
3. ✅ Configure CDN caching
4. ✅ Set up monitoring
5. ✅ Implement backup strategy

## Resources

- **R2 Documentation**: https://developers.cloudflare.com/r2/
- **S3 API Compatibility**: https://developers.cloudflare.com/r2/api/s3/
- **Pricing**: https://developers.cloudflare.com/r2/pricing/
- **boto3 Documentation**: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
