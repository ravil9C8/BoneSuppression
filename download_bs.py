import kagglehub

# Download latest version
path = kagglehub.dataset_download("hmchuong/xray-bone-shadow-supression")

print("Path to dataset files:", path)