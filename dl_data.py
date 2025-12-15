import kagglehub

# Download latest version
path = kagglehub.dataset_download("tuannguyenvananh/iwslt15-englishvietnamese", path="data/archive")

print("Path to dataset files:", path)