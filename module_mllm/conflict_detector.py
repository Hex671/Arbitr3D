import numpy as np
from typing import List, Dict
from datatypes import PointProbabilityField
from sklearn.cluster import DBSCAN

class ConflictDetector:
    def __init__(self, vote_threshold: float = 0.7, knn_threshold: float = 0.7, min_cluster_size: int = 50):
        self.vote_threshold = vote_threshold
        self.knn_threshold = knn_threshold
        self.min_cluster_size = min_cluster_size

    def extract_conflict_regions(self, prob_field: PointProbabilityField) -> List[Dict]:
        """
        找出无法确定语义的争议点，通过 KNN 投票初筛，最后通过聚类提取出若干个“争议 3D 簇”
        """
        # 1. 找出争议点：最大获票率小于 vote_threshold (70%) 的点
        max_probs = np.max(prob_field.prob_matrix, axis=1)
        conflict_indices = np.where(max_probs < self.vote_threshold)[0]

        if len(conflict_indices) == 0:
            return []

        # Get coordinates of conflict points
        all_points = np.asarray(prob_field.point_cloud.points)
        conflict_points = all_points[conflict_indices]

        # 2. 对各争议点单独执行一次 KNN 判断
        from sklearn.neighbors import NearestNeighbors

        # 构建全点云的 KNN 树
        knn = NearestNeighbors(n_neighbors=15, algorithm='auto').fit(all_points)

        # 查找所有争议点的邻居
        distances, indices = knn.kneighbors(conflict_points)

        # 记录仍然无法确定的争议点
        remaining_conflict_indices = []

        # 获取所有点的当前最优语义和置信度
        current_labels = np.argmax(prob_field.prob_matrix, axis=1)
        current_max_probs = np.max(prob_field.prob_matrix, axis=1)

        # 识别 background 和 unlabeled 的列索引，KNN 投票时排除这些"幽灵"标签
        num_classes = prob_field.prob_matrix.shape[1]
        bg_label_idx = num_classes - 1      # background 是最后一列
        unlabeled_label_idx = num_classes - 2  # unlabeled 是倒数第二列

        for i, neighbor_indices in enumerate(indices):
            # 【关键修复】只统计「高置信度且非 background/unlabeled」的邻居
            # 排除：(1) 置信度 < vote_threshold 的争议邻居
            #       (2) argmax 为 background 或 unlabeled 的兜底邻居
            neighbor_labels = current_labels[neighbor_indices]
            neighbor_confs = current_max_probs[neighbor_indices]

            qualified_mask = (
                (neighbor_confs >= self.vote_threshold) &
                (neighbor_labels != bg_label_idx) &
                (neighbor_labels != unlabeled_label_idx)
            )
            qualified_labels = neighbor_labels[qualified_mask]

            # 如果没有合格邻居，直接归入残余争议点
            if len(qualified_labels) == 0:
                remaining_conflict_indices.append(conflict_indices[i])
                continue

            # 在合格邻居中做多数投票
            unique_labels, counts = np.unique(qualified_labels, return_counts=True)
            max_count_idx = np.argmax(counts)
            most_frequent_label = unique_labels[max_count_idx]
            max_ratio = counts[max_count_idx] / len(qualified_labels)

            if max_ratio >= self.knn_threshold:
                # 如果周围 70% 的点都是同一个语义，则把该争议点确定为该语义
                # 更新概率矩阵，将该类别的概率设为 0.99
                num_classes = prob_field.prob_matrix.shape[1]
                new_probs = np.ones(num_classes, dtype=np.float32) * (0.01 / max(1, num_classes - 1))
                new_probs[most_frequent_label] = 0.99
                prob_field.prob_matrix[conflict_indices[i]] = new_probs
            else:
                # 否则，保留为争议点
                remaining_conflict_indices.append(conflict_indices[i])

        remaining_conflict_indices = np.array(remaining_conflict_indices)

        if len(remaining_conflict_indices) == 0:
            return []

        remaining_conflict_points = all_points[remaining_conflict_indices]

        # 3. 对剩余的争议点进行聚类
        # eps depends on the scale of the point cloud, assuming normalized to unit sphere
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(remaining_conflict_points)
        labels = clustering.labels_

        conflict_clusters = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue # Noise

            cluster_mask = labels == label
            cluster_indices = remaining_conflict_indices[cluster_mask]

            if len(cluster_indices) >= self.min_cluster_size:
                # Calculate average probabilities for this cluster
                avg_probs = np.mean(prob_field.prob_matrix[cluster_indices], axis=0)

                conflict_clusters.append({
                    'cluster_id': label,
                    'point_indices': cluster_indices.tolist(),
                    'avg_probs': avg_probs.tolist()
                })

        return conflict_clusters
