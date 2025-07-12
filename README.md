# 🧠 DCIC Smart Labeler (Colab Edition)

This repository presents a complete implementation of a **data-centric image classification labeler** submitted to the [DCIC Challenge 2025](https://codalab.lisn.upsaclay.fr/competitions/17039). The objective is to intelligently label a large image dataset with minimal annotations using **CLIP**, **UMAP**, **Active Learning**, and **Label Propagation**, resulting in low KL divergence scores while staying under a fixed annotation budget.

📌 Project Overview

- ✅ **Challenge**: Provide soft labels for image datasets with minimal annotations.
- ✅ **Goal**: Minimize KL Divergence between predicted labels and ground truth.
- ✅ **Our Score**: Achieved **KL Divergence ≈ 0.99** on local validation.


## 🧠 How the Smart Labeler Works

The `smart_labeler_colab.py` script is a full labeling pipeline built around:

### 1. **CLIP Feature Extraction**  
`self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)`

- Loads OpenAI’s CLIP model to convert each image into a **512-dimensional semantic embedding**.
- These embeddings reflect how "similar" images are in meaning or appearance.

### 2. **Loading the Dataset**
```python
def _load_dataset(self, dataset_name):
    json_path = os.path.join(FLAGS.data_root, dataset_name, f"{dataset_name}-slice{FLAGS.v_fold}.json")
    with open(json_path, 'r') as f:
        return json.load(f)
```
--> Loads the DCIC-provided .json dataset for the specified fold.
--> The dataset contains image paths and placeholders for soft labels.

### 3. **Dummy Label Injection (Simulated Oracle)**
```python
def _inject_dummy_soft_labels(self, data):
    inject_count = len(data['images']) // 2
```
--> The algorithm simulates initial labeled data by injecting random soft labels for 50% of the dataset.
--> Soft labels are probability distributions across classes (not hard labels).
--> Ensures diversity of training examples to begin label propagation.

### 4. **Feature Extraction via CLIP**
```python
def _extract_features(self, paths):
    image = self.preprocess(Image.open(path).convert("RGB"))
```
--> For every image, uses CLIP to extract feature vectors.
--> If an image cannot be read, a zero vector is substituted.

### 5. **Dimensionality Reduction (UMAP)**
```python
def _reduce_dimensionality(self, features):
    return umap.UMAP(...).fit_transform(features)
```
--> Reduces the 512-D feature space to 2D or 3D (default is 2D).
--> Used later for both visualization and k-NN label propagation.

### 6. **Label Initialization**
```python
def _initialize_labels(self, soft_gts, n_classes):
    labels = np.zeros((len(soft_gts), n_classes))
```
--> Initializes a matrix of label probabilities.
--> If a soft label is present, it’s used; otherwise, uniform probabilities are assumed.

### 7. **Entropy-Based Active Learning**
```python
def _compute_entropy(self, probs):
    return entropy(probs, axis=1)
```
--> Computes entropy of predicted labels: higher entropy = more uncertain.
--> Uncertain samples are queried in small batches (active_learning_batch), simulating annotation.

### 8. **Label Propagation via k-NN**
```python
def _propagate_labels(self, features, labels, confident_idxs):
    nbrs = NearestNeighbors(n_neighbors=5).fit(features)
```
--> Propagates confident labels to neighbors in UMAP space.
--> If neighbor consensus exceeds threshold, that label is adopted.

### 9. **Output JSON Generation**
```python
def _save_results(self, dataset_name, labels):
    json.dump(labels, f)
```
A valid DCIC-format JSON is created with:

--> images: paths and soft_gt
--> categories
--> budget and weighted_budget (auto-added downstream)
--> name, v_fold

### 10. **KL Divergence Evaluation**
```python
def _calculate_kl_divergence(self, predicted, gt):
    return np.mean(np.sum(gt * np.log(gt / predicted), axis=1))
```
--> KL divergence is computed between predicted and ground-truth soft labels.
--> Tracks performance across confident samples.

### 11. **Visualization (Optional but Insightful)**
```pyhton
def _visualize_embeddings(self, embeddings, labels, dataset_name):
def _visualize_confidence(self, confidences, dataset_name):
```

--> Generates UMAP scatter plots and confidence histograms.
--> Saved as PNGs named after each dataset.
