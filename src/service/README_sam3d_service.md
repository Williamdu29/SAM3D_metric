# SAM3D Reconstruction Service

## Files

Put these files under:

```bash
/home/dct/work/sam-3d-objects/src/service
```

- `sam3d_service.py`: Flask service implementation
- `run_sam3d_service.py`: service entrypoint
- `client_example.py`: request example
- `__init__.py`: package marker

## Start service

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

## Health check

```bash
curl http://127.0.0.1:58860/health
```

## Run example client

```bash
python src/service/client_example.py
```
