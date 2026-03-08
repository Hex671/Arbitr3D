import os
import sys
import yaml
import numpy as np
import open3d as o3d

# 将根目录加入环境变量，防止模块导入失败
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datatypes import Entity3D
from data_prep.dataloader import PointCloudDataset
from module_render.camera_poses import generate_sphere_cameras
from module_render.renderer import MultiViewRenderer
from module_2d_fm.glip_extractor import GLIPExtractor
from module_2d_fm.sam_segmenter import SAMSegmenter
from module_3d_lift.projector import Projector2DTo3D
from module_3d_lift.merger import MacroMerger
from module_3d_lift.conflict_resolver import ConflictResolver
from module_mllm.prior_compressor import PriorCompressor
from module_mllm.key_view_selector import KeyViewSelector
from module_mllm.som_renderer import SoMRenderer
from module_mllm.mllm_arbitrator import MLLMArbitrator
from evaluation.visualizer import Visualizer3D

class Arbitr3DPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 懒加载初始化，避免过早占用显存
        self.dataloader = PointCloudDataset(self.config['data']['base_path'])
        self.renderer = MultiViewRenderer(tuple(self.config['render']['resolution']))
        
        self.glip = GLIPExtractor(self.config['model']['glip_weight_path'])
        self.sam = SAMSegmenter(self.config['model']['sam_weight_path'], self.config['model']['sam_model_type'])
        
        self.projector = Projector2DTo3D()
        self.merger = MacroMerger(self.config['algorithm']['iou_threshold'])
        self.resolver = ConflictResolver()
        
        self.prior_compressor = PriorCompressor()
        self.view_selector = KeyViewSelector(target_views=4)
        self.som_renderer = SoMRenderer()
        
        api_key = os.environ.get(self.config['mllm']['api_key_env'], "YOUR_API_KEY")
        base_url = self.config['mllm'].get('base_url', None)
        self.arbitrator = MLLMArbitrator(api_key=api_key, base_url=base_url, model_name=self.config['mllm']['model_name'])
        
        self.visualizer = Visualizer3D()

    def run(self, point_cloud_filename: str, prompt_classes: str):
        print(f"--- Processing {point_cloud_filename} ---")
        
        # 1. 数据加载与预处理
        pcd = self.dataloader.load_point_cloud(point_cloud_filename)
        norm_pcd = self.dataloader.normalize_pc(pcd)
        pc_xyz = np.asarray(norm_pcd.points)
        pc_colors = self.dataloader.extract_color(norm_pcd)
        
        # 2. 多视角渲染
        print("Rendering multi-views...")
        cam_positions, cam_rotations = generate_sphere_cameras(self.config['render']['num_cameras'])
        images, index_maps = self.renderer.render(pc_xyz, pc_colors, cam_positions, cam_rotations)
        
        # 3. 2D 基础模型提取
        print("Extracting 2D features via GLIP & SAM...")
        glip_detections = self.glip.get_bboxes_and_scores(images, prompt_classes)
        masks_across_views = self.sam.generate_masks_from_bboxes(images, glip_detections)
        
        # 4. 3D 反投影、融合与硬隔离
        print("Lifting 2D masks to 3D entities and resolving conflicts...")
        local_clusters = self.projector.back_project_2d_to_3d(masks_across_views, index_maps, glip_detections)
        macro_entities = self.merger.macro_merge(local_clusters)
        resolved_entities = self.resolver.resolve_overlaps(macro_entities)
        final_entities = self.resolver.graph_cut_smoothing(resolved_entities, norm_pcd)
        
        print(f"Obtained {len(final_entities)} perfect 3D entities.")
        
        # 5. MLLM 仲裁
        print("Generating Set-of-Mark images and querying MLLM...")
        prior_dict = self.prior_compressor.aggregate_glip_scores(final_entities)
        key_views = self.view_selector.greedy_view_selection(final_entities, index_maps)
        som_image = self.som_renderer.render_som_collage(images, index_maps, final_entities, key_views)
        
        prompt = self.arbitrator.construct_prompt(prior_dict)
        response = self.arbitrator.call_mllm_api(som_image, prompt)
        
        semantic_mapping = self.arbitrator.parse_mllm_json(response)
        print(f"MLLM Mapping Result: {semantic_mapping}")
        
        # 6. 语义回传
        print("Binding semantics back to 3D entities...")
        num_points = pc_xyz.shape[0]
        final_labels = np.ones(num_points, dtype=np.int32) * -1 # -1 for unassigned
        
        # 给每个类别分配一个唯一的 ID
        class_to_id = {}
        class_counter = 0
        
        for entity in final_entities:
            eid_str = str(entity.entity_id)
            if eid_str in semantic_mapping:
                semantic_class = semantic_mapping[eid_str]
                if semantic_class not in class_to_id:
                    class_to_id[semantic_class] = class_counter
                    class_counter += 1
                cls_id = class_to_id[semantic_class]
                
                for pt in entity.point_indices:
                    final_labels[pt] = cls_id
                    
        # 7. 可视化
        # self.visualizer.visualize_3d_result(norm_pcd, final_labels)
        
        print("Processing finished.")
        return norm_pcd, final_labels, class_to_id

if __name__ == "__main__":
    # Demo execution
    pipeline = Arbitr3DPipeline("config/config.yaml")
    # pcd, labels, cls_map = pipeline.run("test.ply", "chair back, chair seat, chair leg, chair arm")
    print("Pipeline initialized successfully.")
