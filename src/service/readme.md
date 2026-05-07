# 服务说明

## 核心文件

核心服务文件是：

```bash
/home/dct/work/sam-3d-objects/src/service/sam3d_service.py
```

启动入口

```bash
/home/dct/work/sam-3d-objects/src/service/run_sam3d_service.py
```

类目前已经具备三个关键接口：

`load_model()`：加载 SAM3D 模型权重；`recon()`：完整重建主流程；`release()`：释放模型引用和 CUDA cache。`recon()` 接口需要 `rgb_path`、`depth_path`、`mask_path`、`k_input`、`out_dir`、`run_sam3d` 等参数，并会输出点云、mesh、metrics、timings 等路径和指标。

所以封装了 4 个接口：

```
GET  /health
POST /load_model
POST /release
POST /reconstruct
```

其中 /reconstruct 会调用的

```py
reconstructor.recon(...)
```

`/load_model` 会显式加载模型；`/release` 会显式释放模型。类里 `release()` 已经会断开模型引用、清理 `Inference/load_image` 引用、GC，并尝试清理 CUDA cache，所以直接复用了这个能力。


也保留了的“初始化不加载模型”的设计。服务启动时默认只实例化 `SAM3DReconstructor`，不强制占显存；真正请求 `/reconstruct` 且 `run_sam3d=True` 时，你的类会自动触发模型加载。需要启动时预加载的话，设置：

```bash
export SAM3D_PRELOAD_MODEL=1
```

## 如何启动

启动前建议显式指定：

```bash
cd /home/dct/work/sam-3d-objects

export SAM3D_PROJECT_ROOT=/home/dct/work/sam-3d-objects
export SAM3D_ROOT=/home/dct/work/sam-3d-objects/libs/SAM3D
export SAM3D_CONFIG=/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml
export SAM3D_SERVICE_PORT=58860
export SAM3D_OUTPUT_ROOT=/home/dct/work/sam-3d-objects/Outputs/service_outputs
export SAM3D_COMPILE_MODEL=0

python src/service/run_sam3d_service.py
```

启动成功后会监听：

`0.0.0.0:58860`


### 健康检查

另开一个终端：

```bash
curl http://127.0.0.1:58860/health
```

正常情况下返回

```json
{
  "model_loaded": false,
  "output_root": "/home/dct/work/sam-3d-objects/Outputs/service_outputs",
  "project_root": "/home/dct/work/sam-3d-objects",
  "sam3d_config": "/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml",
  "sam3d_root": "/home/dct/work/sam-3d-objects/libs/SAM3D",
  "service_port": 58860,
  "status": "healthy"
}
```

### 显示加载模型

```bash
curl -X POST http://127.0.0.1:58860/load_model \
  -H "Content-Type: application/json" \
  -d '{"force_reload": false}'
```

加载成功后再访问：

```bash
curl http://127.0.0.1:58860/health
```

model_loaded 应该变成 true。


### 发起重建请求：本地路径方式

```bash
curl -X POST http://127.0.0.1:58860/reconstruct \
  -H "Content-Type: application/json" \
  -d '{
    "rgb_path": "/home/dct/work/sam-3d-objects/test_data/images/77.png",
    "depth_path": "/home/dct/work/sam-3d-objects/test_data/depth/77.png",
    "mask_path": "/home/dct/work/sam-3d-objects/test_data/mask/77.png",
    "k_input": [
      384.897491455078, 0.0, 326.897308349609,
      0.0, 384.471374511719, 240.425384521484,
      0.0, 0.0, 1.0
    ],
    "out_dir": "/home/dct/work/sam-3d-objects/Outputs/service_outputs/test_case_001",
    "run_sam3d": true,
    "output_prefix": "sam3d_metric_general",
    "raw_sam_name": "sam3d_raw_v8.ply",
    "anchor_name": "anchor_metric_visible_surface_v8.ply",
    "save_metrics_npy": true,
    "save_metrics_json": true,
    "seed": 42,
    "release_after_recon": false
  }'
```

返回结构是：

```json

{
  "task_id": "task-...",
  "code": 0,
  "message": "ok",
  "result": {
    "out_dir": "...",
    "anchor_pcd_path": "...",
    "raw_sam_pcd_path": "...",
    "metric_full_pcd_path": "...",
    "metric_visible_pcd_path": "...",
    "metric_mesh_ply_path": "...",
    "metric_mesh_obj_path": "...",
    "metrics_npy_path": "...",
    "metrics_json_path": "...",
    "timings_json_path": "...",
    "metrics": {},
    "timings": {},
    "elapsed_seconds": 0.0,
    "release_after_recon": false
  }
}
```

### 释放 GPU

手动释放GPU

```bash
curl -X POST http://127.0.0.1:58860/release \
  -H "Content-Type: application/json" \
  -d '{"clear_imports": false, "reset_timings": false}'
```

* 或者在每次重建后自动释放 

```json
{
  "release_after_recon": true
}
```


默认设置为 false，原因是服务场景下一般会希望模型常驻，下一次请求更快；需要最轻量显存状态时，再设成 true 或调用 /release。
