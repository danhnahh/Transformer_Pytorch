import kagglehub

# Download latest version
path = kagglehub.dataset_download("tuannguyenvananh/iwslt15-englishvietnamese", path="data/archive", unzip=True, force=True)

print("Path to dataset files:", path)