from absl import app, flags
import numpy as np
import os
import json
import torch
import clip
import umap
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

# Flag Definitions
FLAGS = flags.FLAGS
flags.DEFINE_string('data_root', '/content/input_datasets', 'Root directory for datasets')
flags.DEFINE_string('datasets', 'Benthic,MiceBone,QualityMRI,TreeversityH6', 'Comma-separated dataset names')
flags.DEFINE_integer('v_fold', 1, 'Validation fold')
flags.DEFINE_float('percentage_labeled_data', 0.1, 'Fraction of data to label (initial)')
flags.DEFINE_integer('number_annotations_per_image', 1, 'Annotations per image')
flags.DEFINE_float('label_propagation_threshold', 0.9, 'Confidence threshold for propagation')
flags.DEFINE_integer('active_learning_batch', 5, 'Images to query per round')
flags.DEFINE_float('consensus_threshold', 0.75, 'Confidence threshold for consensus')
flags.DEFINE_integer('umap_dim', 2, 'UMAP reduction dimension (set to 2 for visualization)')
flags.DEFINE_integer('seed', 42, 'Random seed')

class SmartLabeler:
    def __init__(self):
        self.name = 'smart_labeler_colab'
        self.report = {'results': [], 'summary': {}}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def _load_dataset(self, dataset_name):
        json_path = os.path.join(FLAGS.data_root, dataset_name, f"{dataset_name}-slice{FLAGS.v_fold}.json")
        with open(json_path, 'r') as f:
            return json.load(f)

    def _save_results(self, dataset_name, result_data):
        output_dir = '/content/output_datasets'
        folder_name = f"{dataset_name}-{self.name}-{FLAGS.number_annotations_per_image:02d}-{FLAGS.percentage_labeled_data:.2f}"
        full_path = os.path.join(output_dir, folder_name)
        os.makedirs(full_path, exist_ok=True)
        output_path = os.path.join(full_path, f"{folder_name}-{FLAGS.v_fold}.json")
        with open(output_path, 'w') as f:
            json.dump(result_data, f)

    def _inject_dummy_soft_labels(self, data):
        n_classes = len(data.get('classes', data.get('categories', [])))
        inject_count = len(data['images']) // 2
        for img in data['images'][:inject_count]:
            if not img.get('soft_gt'):
                rand_dist = np.random.rand(n_classes)
                rand_dist /= rand_dist.sum()
                img['soft_gt'] = rand_dist.tolist()
        return data

    def _extract_features(self, paths):
        features = []
        for path in paths:
            try:
                image = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image).cpu().numpy().flatten()
                features.append(embedding)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                features.append(np.zeros(512))
        return np.array(features)

    def _reduce_dimensionality(self, features):
        reducer = umap.UMAP(n_components=FLAGS.umap_dim, random_state=FLAGS.seed)
        return reducer.fit_transform(features)

    def _compute_entropy(self, probs):
        return entropy(probs, axis=1)

    def _initialize_labels(self, soft_gts, n_classes):
        labels = np.zeros((len(soft_gts), n_classes))
        confident_idxs = []
        for i, gt in enumerate(soft_gts):
            if gt:
                labels[i] = gt
                confident_idxs.append(i)
        return labels, confident_idxs

    def _query_oracle(self, uncertain_idxs, n_classes, budget_images):
        queried = uncertain_idxs[:budget_images]
        simulated_labels = []
        for _ in queried:
            rand = np.random.rand(n_classes)
            rand = rand / rand.sum()
            simulated_labels.append(rand)
        return queried, np.array(simulated_labels)

    def _propagate_labels(self, features, labels, confident_idxs):
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k).fit(features)
        _, indices = nbrs.kneighbors(features)
        for i in range(len(features)):
            if i in confident_idxs:
                continue
            votes = labels[indices[i][1:]]
            confidence = np.max(votes.mean(axis=0))
            if confidence > FLAGS.label_propagation_threshold:
                labels[i] = votes.mean(axis=0)
        return labels

    def _calculate_kl_divergence(self, predicted, gt):
        eps = 1e-10
        predicted = np.array(predicted) + eps
        gt = np.array(gt) + eps
        predicted /= predicted.sum(axis=1, keepdims=True)
        gt /= gt.sum(axis=1, keepdims=True)
        return np.mean(np.sum(gt * np.log(gt / predicted), axis=1))

    def _visualize_embeddings(self, embeddings, labels, dataset_name):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=np.argmax(labels, axis=1), palette='tab10', s=30)
        plt.title(f"UMAP of {dataset_name} with Inferred Labels")
        plt.savefig(f"{dataset_name}_embedding_plot.png")
        plt.close()

    def _visualize_confidence(self, confidences, dataset_name):
        plt.figure(figsize=(6, 4))
        sns.histplot(confidences, bins=20, kde=True)
        plt.title(f"Label Confidence Distribution: {dataset_name}")
        plt.xlabel("Max Probability per Image")
        plt.savefig(f"{dataset_name}_confidence_plot.png")
        plt.close()

    def process_dataset(self, dataset_name):
        data = self._load_dataset(dataset_name)
        data = self._inject_dummy_soft_labels(data)
        paths = [os.path.join(FLAGS.data_root, path) for path in [img['path'] for img in data['images']]]
        soft_gts = [img['soft_gt'] for img in data['images']]
        n_classes = len(data.get('classes', data.get('categories', [])))

        features = self._extract_features(paths)
        reduced = self._reduce_dimensionality(features)

        labels, confident_idxs = self._initialize_labels(soft_gts, n_classes)

        unlabeled_idxs = [i for i in range(len(paths)) if i not in confident_idxs]
        probs = np.full((len(paths), n_classes), 1.0 / n_classes)
        labels = np.where(labels.sum(axis=1, keepdims=True) == 0, probs, labels)

        total_budget = int(FLAGS.percentage_labeled_data * FLAGS.number_annotations_per_image * len(paths))

        for round in range(0, total_budget, FLAGS.active_learning_batch):
            remaining = total_budget - len(confident_idxs)
            if remaining <= 0:
                break
            entropy_scores = self._compute_entropy(labels[unlabeled_idxs])
            sorted_uncertain = np.argsort(entropy_scores)[::-1]
            selected_idxs = [unlabeled_idxs[i] for i in sorted_uncertain[:FLAGS.active_learning_batch]]

            queried, new_labels = self._query_oracle(selected_idxs, n_classes, FLAGS.active_learning_batch)
            for i, idx in enumerate(queried):
                labels[idx] = new_labels[i]
                confident_idxs.append(idx)
                if idx in unlabeled_idxs:
                    unlabeled_idxs.remove(idx)

            labels = self._propagate_labels(reduced, labels, confident_idxs)

        output_images = []
        for i, img in enumerate(data['images']):
            output_images.append({
                'path': img['path'],
                'soft_gt': labels[i].tolist(),
                'split': img['split'],
                'original_split': img.get('original_split', img['split']),
                'gt': np.argmax(labels[i]).item()
            })

        budget_used = len(confident_idxs) * FLAGS.number_annotations_per_image / len(paths)
        categories = data.get('classes', data.get('categories', []))

        self._save_results(dataset_name, {
            'name': dataset_name,
            'v_fold': FLAGS.v_fold,
            'categories': categories,
            'classes': categories,
            'budget': budget_used,
            'weighted_budget': budget_used,
            'images': output_images
        })

        valid_gt = [soft_gts[i] for i in confident_idxs if soft_gts[i]]
        predicted = [labels[i] for i in confident_idxs if soft_gts[i]]
        kl = self._calculate_kl_divergence(predicted, valid_gt) if valid_gt else 0.0
        avg_conf = float(np.mean(np.max(labels, axis=1)))

        self._visualize_embeddings(reduced, labels, dataset_name)
        self._visualize_confidence(np.max(labels, axis=1), dataset_name)

        print(f"Processed {dataset_name}: KL={kl:.4f} (on {len(valid_gt)} samples), Avg Confidence={avg_conf:.4f}")

        return {'dataset': dataset_name, 'kl': kl, 'confidence': avg_conf, 'budget': budget_used}


def main(argv):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    labeler = SmartLabeler()
    datasets = FLAGS.datasets.split(',')

    for dataset in datasets:
        result = labeler.process_dataset(dataset.strip())
        labeler.report['results'].append(result)

    print("\n=== Final Summary ===")
    for r in labeler.report['results']:
        print(f"{r['dataset']}: KL={r['kl']:.4f}, Avg Conf={r['confidence']:.4f}, Budget Used={r['budget']:.2f}")

if __name__ == '__main__':
    app.run(main)
