cat > data/DATASETS_INFO.md << 'EOF'
# Deepfake Datasets Information

## Primary Datasets for This Project

### 1. Celeb-DF (Recommended for beginners)
- **Size:** ~15 GB
- **Content:** 590 real videos, 5,639 deepfake videos
- **Quality:** High-quality celebrity faces
- **Download:** https://github.com/yuezunli/celeb-deepfakeforensics
- **Access:** Requires form submission for academic use

### 2. FaceForensics++ (Comprehensive benchmark)
- **Size:** ~38 GB (compressed version)
- **Content:** 1,000 original videos, 4,000 manipulated videos
- **Methods:** Includes multiple deepfake generation methods
- **Download:** https://github.com/ondyari/FaceForensics
- **Access:** Requires registration and approval

### 3. Sample Dataset (What we'll create today)
- **Size:** ~100 MB
- **Content:** Small test videos for development
- **Purpose:** Immediate testing while waiting for real datasets

## Dataset Download Priority
1. Start with sample data (today)
2. Request access to Celeb-DF (background process)
3. Download Celeb-DF when approved
4. Optionally add FaceForensics++ for more comprehensive training
EOF