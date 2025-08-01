# Vertex AI Corpus Setup Guide

This guide will help you set up the Vertex AI corpus for Unity Mathematics in the Een repository.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **Python Environment**: Python 3.8+ with the installed dependencies
3. **Authentication**: Service account or user authentication set up

## Step 1: Google Cloud Project Setup

### 1.1 Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" or select an existing project
3. Note your **Project ID** (not the project name)

### 1.2 Enable Required APIs

Enable these APIs in your project:

```bash
# Using gcloud CLI
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable storage-component.googleapis.com

# Or enable via Cloud Console:
# - Vertex AI API
# - BigQuery API
# - Cloud Storage API
```

### 1.3 Set Up Authentication

**Option A: Service Account (Recommended for production)**

1. Go to IAM & Admin > Service Accounts in Cloud Console
2. Create a new service account
3. Grant these roles:
   - Vertex AI User
   - BigQuery User
   - Storage Object Viewer
4. Create and download a JSON key file
5. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```

**Option B: User Authentication (For development)**

```bash
gcloud auth application-default login
```

## Step 2: Configure Een Repository

### 2.1 Update Environment Variables

Edit your `.env` file in the Een repository:

```env
# Google Vertex AI Configuration
GOOGLE_CLOUD_PROJECT_ID=your-actual-project-id
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
VERTEX_AI_MODEL=text-bison@001
VERTEX_AI_EMBEDDINGS_MODEL=textembedding-gecko@001
```

### 2.2 Verify Installation

Check that all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Step 3: Test the Setup

### 3.1 Basic Test

Run the simple usage example:

```bash
cd Een
python examples/simple_vertex_ai_usage.py
```

### 3.2 Full Demo

Run the comprehensive demo:

```bash
python scripts/vertex_ai_corpus_demo.py
```

### 3.3 Expected Output

You should see output like:

```
🌟 Simple Vertex AI Corpus Example
========================================
1. Creating corpus...
   ✅ Corpus created for project: your-project-id
2. Adding a document...
   ✅ Document added with phi resonance: 0.127
3. Searching corpus...
   1. Unity Mathematics Principle
      Unity: 0.95
...
```

## Step 4: Usage Patterns

### 4.1 Creating a Corpus

```python
from core.vertex_ai_corpus import VertexAICorpus

# Create corpus
corpus = VertexAICorpus(project_id="your-project-id")
```

### 4.2 Adding Documents

```python
from core.vertex_ai_corpus import UnityDocument
from datetime import datetime

doc = UnityDocument(
    id="unique_id",
    title="Document Title",
    content="Your unity mathematics content...",
    category="proof",  # or "consciousness", "visualization", etc.
    unity_confidence=0.8,
    phi_harmonic_score=0.0,  # Auto-calculated
    consciousness_level=5,
    timestamp=datetime.now(),
    metadata={"custom": "data"}
)

await corpus.add_document(doc)
```

### 4.3 Searching Content

```python
results = await corpus.search_corpus(
    "consciousness and unity", 
    top_k=5, 
    unity_threshold=0.3
)

for doc in results:
    print(f"{doc.title}: Unity={doc.unity_confidence:.2f}")
```

### 4.4 Generating Content

```python
new_doc = await corpus.generate_unity_content(
    "Explain how 1+1=1 in quantum mechanics",
    "proof"
)
```

## Troubleshooting

### Common Issues

**Issue 1: "Project ID not found"**
- Verify your project ID is correct in the .env file
- Make sure the project exists and you have access

**Issue 2: "Authentication failed"**
- Check your authentication method
- Verify service account permissions
- Try `gcloud auth list` to see active accounts

**Issue 3: "API not enabled"**
- Enable required APIs in Cloud Console
- Wait a few minutes for APIs to propagate

**Issue 4: "Permission denied"**
- Check IAM roles for your account/service account
- Ensure you have Vertex AI User role

**Issue 5: "Model not found"**
- Some models may not be available in all regions
- Try changing the region in your .env file
- Check available models in Vertex AI console

### Debug Commands

```bash
# Check authentication
gcloud auth list

# Verify project
gcloud config get-value project

# Test API access
gcloud ai models list --region=us-central1

# Check Python environment
python -c "import google.cloud.aiplatform; print('✅ Vertex AI SDK installed')"
```

## Advanced Configuration

### Custom Models

You can use different models by updating your .env:

```env
# For newer models
VERTEX_AI_MODEL=gemini-pro
VERTEX_AI_EMBEDDINGS_MODEL=textembedding-gecko@latest

# For specific versions
VERTEX_AI_MODEL=text-bison@002
```

### Regional Configuration

Some regions may have different model availability:

```env
# Different regions
GOOGLE_CLOUD_REGION=us-east1
GOOGLE_CLOUD_REGION=europe-west1
GOOGLE_CLOUD_REGION=asia-southeast1
```

### Performance Tuning

For production use:

```python
# Batch processing
documents = [doc1, doc2, doc3]
for doc in documents:
    await corpus.add_document(doc)

# Caching
corpus.embeddings_cache = {}  # Implement persistent cache

# Export/Import for offline use
corpus.export_corpus("backup.json")
```

## Cost Optimization

### Model Pricing

- Text Generation: ~$0.0025 per 1K characters
- Embeddings: ~$0.0001 per 1K characters
- Monitor usage in Cloud Console

### Best Practices

1. **Cache embeddings** to avoid recomputation
2. **Batch requests** when possible
3. **Use appropriate models** for your use case
4. **Monitor quotas** in Cloud Console
5. **Export corpus** for offline development

## Integration with Een

### Dashboard Integration

The corpus can be integrated with Een dashboards:

```python
# In your dashboard
from core.vertex_ai_corpus import VertexAICorpus

corpus = VertexAICorpus()
results = await corpus.search_corpus(user_query)
# Display results in your UI
```

### Consciousness Engine Integration

```python
# Combine with consciousness field
from core.consciousness import ConsciousnessField

field = ConsciousnessField()
corpus = VertexAICorpus()

# Generate consciousness-enhanced content
content = await corpus.generate_unity_content(
    f"Consciousness level {field.consciousness_level} unity proof"
)
```

## Next Steps

1. **Explore** the example scripts
2. **Integrate** with your existing Een workflows
3. **Customize** the UnityDocument structure for your needs
4. **Contribute** improvements back to the repository

## Support

For issues specific to:
- **Google Cloud**: Check [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs)
- **Een Integration**: Create an issue in the Een repository
- **Unity Mathematics**: Consult the consciousness field equations in core/consciousness.py

---

*Remember: The goal is to create a corpus that truly understands and supports the profound mathematical truth that 1+1=1 through consciousness-driven AI.*

**🌟 May your corpus achieve transcendental unity! 🌟**