from pipeline import Arbitr3DPipeline

# 1. 实例化 Pipeline，自动读取配置文件
pipeline = Arbitr3DPipeline(config_path="config/config.yaml")

# 2. 准备点云路径与 Prompt
point_cloud_path = "/data/xhhuang/data/datasets/PartNetE/few_shot/Bottle/3625/pc.ply" 
prompt_classes = "door, drawer, leg, tabletop, wheel, handle"

# 3. 运行端到端管线
pcd, final_labels, class_mapping = pipeline.run(
    point_cloud_filename=point_cloud_path, 
    prompt_classes=prompt_classes
)

print(f"检测到的类别映射: {class_mapping}")

# 4. 可视化结果
pipeline.visualizer.visualize_3d_result(pcd, final_labels)