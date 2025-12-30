# ML Cache Configuration

## Overview
To prevent C drive from filling up during model downloads, all ML-related caches have been moved to D drive.

## Cache Locations
All caches are stored in: `D:\AI projects\ResearchGPT\.cache\`

- **Hugging Face**: `D:\AI projects\ResearchGPT\.cache\huggingface`
- **Transformers**: `D:\AI projects\ResearchGPT\.cache\transformers`
- **PyTorch**: `D:\AI projects\ResearchGPT\.cache\torch`

## Environment Variables Set
The following environment variables are set permanently in your Windows user profile:

```
HF_HOME=D:\AI projects\ResearchGPT\.cache\huggingface
TRANSFORMERS_CACHE=D:\AI projects\ResearchGPT\.cache\transformers
TORCH_HOME=D:\AI projects\ResearchGPT\.cache\torch
```

These are also configured in `.env` file for project-specific loading.

## How It Works
1. When you run any Python script in this project, it loads `.env` file
2. The cache environment variables point to D drive
3. All model downloads (Hugging Face, PyTorch) save to D drive automatically
4. C drive remains free

## Old Cache (C Drive)
The original cache was located at:
```
C:\Users\Nadeem Shabir Mir\.cache\huggingface
```

**You can safely delete this folder** after verifying everything works with the new location.

## Verification
To verify the setup is working:
```powershell
# Check environment variable
[System.Environment]::GetEnvironmentVariable('HF_HOME', 'User')

# Should output: D:\AI projects\ResearchGPT\.cache\huggingface
```

## What to Do After Restart
✅ **Nothing!** Environment variables are set permanently and will persist after restart.

The `.env` file also ensures settings work even if environment variables aren't loaded yet.
