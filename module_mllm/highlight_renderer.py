import cv2
import numpy as np
from typing import List, Dict
from datatypes import RenderOutput

class HighlightRenderer:
    def render_conflict_highlights(
        self,
        render_output: RenderOutput,
        conflict_clusters: List[Dict]
    ) -> List[Dict]:
        """
        找到能看清争议簇的最佳视角，返回拼贴图像及对应的视角索引列表。
        """
        highlighted_data = []

        for cluster in conflict_clusters:
            indices = set(cluster['point_indices'])

            view_scores = []
            for view_idx, index_map in enumerate(render_output.index_maps):
                visible_indices = index_map[index_map != -1]
                visible_cluster_points = [idx for idx in visible_indices if idx in indices]
                view_scores.append((view_idx, len(visible_cluster_points)))

            view_scores.sort(key=lambda x: x[1], reverse=True)
            top_views = [v[0] for v in view_scores[:4] if v[1] > 0]

            if not top_views:
                highlighted_data.append({'image': np.zeros((800, 800, 3), dtype=np.uint8), 'views': []})
                continue

            view_images = []
            for view_idx in top_views:
                img = render_output.images[view_idx].copy()
                index_map = render_output.index_maps[view_idx]

                cluster_mask = np.zeros(index_map.shape[:2], dtype=bool)
                for r in range(index_map.shape[0]):
                    for c in range(index_map.shape[1]):
                        if index_map[r, c, 0] in indices:
                            cluster_mask[r, c] = True

                highlight_color = [255, 0, 0]
                overlay = img.copy()
                overlay[cluster_mask] = highlight_color
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

                contours, _ = cv2.findContours(cluster_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

                cv2.putText(img, f"View {view_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                view_images.append(img)

            h, w = view_images[0].shape[:2]
            collage = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

            for i, img in enumerate(view_images):
                row = i // 2
                col = i % 2
                collage[row*h:(row+1)*h, col*w:(col+1)*w] = img

            highlighted_data.append({'image': collage, 'views': top_views})

        return highlighted_data
