import pkg_resources

packages = [
    "ultralytics",
    "opencv-python",
    "numpy",
    "pandas",
    "supervision",
    "torch",
    "torchvision",
    "pickle5"
]

req_lines = []

for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        req_lines.append(f"{pkg}=={version}")
    except pkg_resources.DistributionNotFound:
        req_lines.append(f"# {pkg} NOT INSTALLED")

# Write to requirements.txt
with open("requirements.txt", "w") as f:
    f.write("\n".join(req_lines))

print("requirements.txt created!")
