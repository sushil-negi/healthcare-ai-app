# Large Training Data Files

## Files Not Included in Git Repository

The following large training data files are not included in this Git repository due to size constraints:

- `combined_healthcare_training_data.json` (611MB)
- `combined_healthcare_training_data.jsonl` (519MB)

## How to Obtain Training Data

### Option 1: Generate Training Data
```bash
# Run the data generation script to create training data
python scripts/healthcare_data_generator.py

# Combine datasets
python scripts/combine_all_datasets.py
```

### Option 2: Download from External Storage
```bash
# If using cloud storage (example with AWS S3)
aws s3 cp s3://your-bucket/combined_healthcare_training_data.json data/
aws s3 cp s3://your-bucket/combined_healthcare_training_data.jsonl data/

# If using Git LFS (after setting up LFS)
git lfs pull
```

### Option 3: Use Sample Data for Development
For development and testing, you can use the smaller sample files:
- `test_healthcare_training.json` (4.5KB) - Small test dataset
- `healthcare_training_data.json` (144KB) - Medium development dataset

## Setting Up Git LFS (Optional)

If you want to version control large files:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.json"
git lfs track "data/*.jsonl"

# Add and commit
git add .gitattributes
git add data/combined_healthcare_training_data.json
git commit -m "Add large training data with Git LFS"
```

## Security Note

Large training data files may contain sensitive healthcare information. Ensure:
- Files are stored securely with appropriate access controls
- HIPAA compliance is maintained for any real healthcare data
- Use synthetic/anonymized data for development when possible