import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
"""
I wanna try to do image distribution analysis, from what I see, it detects the difference between google earth screenshots and real photos
"""
# Define the folder containing your images
image_folder = r'D:\mploi\Documents\Albatros\albatros\smoke_dataset_V1\images'

# Step 1: Load and Preprocess Color Images
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
images = []
image_names = []  # To store the image names

for img_path in image_paths:
    img = cv2.imread(img_path)  # Load color image
    img = cv2.resize(img, (512, 512))  # Resize images to a larger resolution (adjust as needed)
    images.append(img)
    image_names.append(os.path.basename(img_path))  # Extract and store the image name

# Convert the list of images to a numpy array
image_data = np.array(images)

# Step 2: Flatten and Standardize the Data
image_data_flattened = image_data.reshape(image_data.shape[0], -1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_data_flattened)

# Step 3: Perform PCA
n_components = 50  # Number of principal components to retain (adjust as needed)
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)

# Calculate Z-scores for PC1 and PC2
z_scores_pc1 = (pca_result[:, 0] - np.mean(pca_result[:, 0])) / np.std(pca_result[:, 0])
z_scores_pc2 = (pca_result[:, 1] - np.mean(pca_result[:, 1])) / np.std(pca_result[:, 1])

# Print the Z-scores for PC1 and PC2
for i, (z1, z2) in enumerate(zip(z_scores_pc1, z_scores_pc2)):
    if i + 1 <= 2:
        print(f"Z-scores for PC{i + 1}: PC1={z1:.4f}, PC2={z2:.4f}")

# Print the explained variance for PC1 and PC2
explained_variance_ratio = pca.explained_variance_ratio_
for i, variance in enumerate(explained_variance_ratio[:2]):
    print(f"Explained Variance for PC{i + 1}: {variance:.4f}")



# Annotate outliers with image names using Z-scores (adjust the Z-score threshold as needed)
z_score_threshold = 2.0  # Adjust this threshold based on your data
outliers = np.where(np.abs(z_scores_pc1) > z_score_threshold)[0]

"""
# Plot the data in a 2D subspace (PC1 vs. PC2)
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], marker='o', alpha=0.5, s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: PC1 vs. PC2')

for i in outliers:
    plt.annotate(f'Outlier: {image_names[i]}', (pca_result[i, 0], pca_result[i, 1]), color='red', fontsize=12)
    print(image_names[i])
plt.grid()
plt.show()
"""

num_sample_images = min(5, len(outliers))  # Display up to 5 sample images
plt.figure(figsize=(12, 6))
plt.suptitle('Sample Outlier Images', fontsize=16)

for i, outlier_idx in enumerate(outliers[:num_sample_images]):
    plt.subplot(1, num_sample_images, i + 1)
    outlier_image = images[outlier_idx]
    plt.imshow(cv2.cvtColor(outlier_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Outlier: {image_names[outlier_idx]}', fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()
# Step 5: Save the PCA results (if needed)
# np.save('pca_results.npy', pca_result)

# Optionally, you can also save the PCA components for future use:
# np.save('pca_components.npy', pca.components_)
