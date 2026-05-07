# SAM3D-Objects 模型

SAM 3D Objects 的论文摘要明确说它是一个从单张图像预测 geometry、texture、layout 的生成式 3D 重建模型，并强调它针对自然图像中的遮挡、杂乱背景、小物体和非常规姿态做了优化

## SAM3D 解决了什么问题？

传统 3D 重建通常需要多视角照片、深度传感器、NeRF 优化，或者人工建模。SAM3D 的目标更激进：只给一张图 + 一个目标 mask，就生成这个目标的完整 3D 资产。这里的“完整”不是只恢复可见表面，而是要猜出被遮挡、背面、侧面等不可见区域。官方 GitHub 描述它可以把 masked objects 转成带有 pose、shape、texture、layout 的 3D 模型，并能处理自然场景中的遮挡、小物体和复杂场景

```
输入：
  I: 单张 RGB/RGBA 图像
  M: 目标物体的二值分割 mask
  seed: 随机种子，用于可复现生成

输出：
  3D Gaussian Splat / mesh
  包含：3D 点/高斯的位置、颜色/纹理、尺度、旋转、透明度，以及物体在图像场景中的布局'
```


## 输入数据在模型内部的第一步：从图像和 mask 变成条件特征

SAM3D Objects 的基础输入是 图像 + segmentation mask。DeepWiki 对代码流程的总结显示，图像通常是 PNG/RGBA，mask 是灰度或二值 PNG；单物体模式读一个 mask，多物体模式读多个 mask，每个 mask 对应一个要重建的对象

```
I, M
↓
根据 M 找到目标区域
↓
裁剪 / 归一化 / padding / resize
↓
把图像内容与 mask 对齐成模型可吃的张量
↓
送入条件编码器 condition embedder / vision encoder
↓
得到视觉条件 token
```

## 整体架构：粗到细的两阶段生成

SAM3D Objects 不是一个简单的 “CNN 直接回归 mesh”。它更像一个 条件 3D 生成模型，走的是 coarse-to-fine，两阶段生成：

```
图像 I + mask M
↓
条件编码器：得到图像/物体条件 token
↓
阶段 1：Sparse Structure Generator
      生成粗 3D 结构 O + 全局布局 R, t, s
↓
阶段 2：Sparse Latent / SLaT Generator
      在粗结构上细化几何和外观纹理
↓
Decoder
      输出 3D Gaussian Splat / mesh / texture
```

Fast-SAM3D 论文对原始 SAM3D 的结构有一个很清楚的概括：SAM3D 接收图像 I 和 object mask M，重建物体的 3D shape、texture 和 layout 参数 (R, t, s)；pipeline 是两阶段 coarse-to-fine，先预测粗结构和全局布局，再细化几何细节并合成纹理

## 阶段 0：Condition Embedding，图像和 mask 怎么变成 token？

假设输入是一张图片 $I ∈ R^{H×W×3}$，mask 是 $M ∈ {0,1}^{H×W}$。

模型会先做预处理：

```
1. 根据 mask 找目标 bounding box
2. 扩大一点上下文区域
3. crop 出目标附近图像
4. resize 到模型固定输入尺度
5. mask 同步 resize
6. 图像归一化
7. 图像和 mask 一起作为条件输入
```

然后使用视觉编码器把 2D 输入转成 token。Fast-SAM3D 对 SAM3D 的描述中提到，condition embedding 会把 (I, M) 编码成 visual tokens，并使用预训练视觉编码器，例如 DINOv2

```
image patches:
  p1, p2, p3, ..., pn

mask-aware encoding:
  每个 patch 不只是有 RGB 语义，
  还知道自己是否属于目标物体、是否在目标边界附近、是否是背景上下文。

输出：
  C = {c1, c2, ..., cn}
  C 是一组视觉条件 token。
```

这些 token 后面会作为条件，控制 3D 生成模型不要乱生成，而是生成“这张图片里的这个物体”


## 阶段 1：Sparse Structure Generator，先生成粗 3D 占据结构


第一阶段的任务是：不要急着生成纹理，先决定物体在 3D 空间里大概长什么样、占哪些体素、朝向是什么、在图像里的尺度和位置是什么。

Fast-SAM3D 对原模型的描述是：Sparse Structure，简称 SS generator，会通过迭代去噪生成一个粗结构 latent，也就是类似 voxel 的表示 `O`，并预测全局 layout 参数 $(R, t, s)$。

这里的 `O` 可以理解为稀疏 3D 体素结构：

```
O = {v1, v2, ..., vk}

每个 vi 是一个被认为“属于物体”的 3D voxel / sparse coordinate。
```


为什么用 sparse structure，而不是完整 3D grid？

因为完整 3D grid 太贵。假设用 128×128×128 的体素网格，就是 200 多万个 cell，大多数是空的。物体真正占据的地方只是一小部分。所以 SAM3D 选择稀疏表示，只处理可能有物体的 3D 位置。

这一阶段会产生两个关键结果：


```
O: 物体粗 3D 结构
R: rotation，物体朝向
t: translation，物体相对相机的位置
s: scale，物体尺度
```

“这张图里的这个物体，大概是一个椅子/玩具/杯子/鞋子，它在 3D 空间里应该占据哪些位置？它是正对镜头还是斜着？它离相机多远？大概多大？”


## 阶段 1 的计算方式：从噪声到粗结构的迭代去噪

SAM3D 的生成过程不是一次前向回归，而是类似 diffusion/flow matching 的 iterative denoising。Fast-SAM3D 的 profiling 显示，SAM3D 的延迟主要来自双阶段迭代去噪：structure generator 和 texture/SLaT generator。

```py
# 条件 token
C = condition_encoder(I, M)

# 初始化粗结构噪声
O_t = random_sparse_structure_noise()

for t in denoising_steps:
    delta = sparse_structure_transformer(O_t, C, t)
    O_t = update(O_t, delta)

O = O_t
R, trans, scale = layout_head(O, C)
```


这里的 transformer 会不断看两类信息：

当前噪声结构 `O_t`
图像/mask 条件 token `C`

然后预测下一步如何把噪声变得更像目标物体的 3D 结构。

直觉上，如果图片里是椅子，第一阶段不需要马上知道木纹颜色，但需要知道：

```
椅背在哪？
坐面在哪？
四条腿大概在哪？
被遮挡的后腿可能在哪里？
整体朝向是什么？
```

## 阶段 2：Sparse Latent / SLaT Generator，细化几何和纹理

第一阶段只给出“物体大体占据哪里”。第二阶段要做更难的事：在这个结构上补充 细节几何、纹理、颜色、局部形状特征。

Fast-SAM3D 对原始 SAM3D 的第二阶段描述是：Sparse Latent，简称 SLaT generator，会在 (I, M, O) 条件下做迭代去噪，细化 appearance-related signals，例如 texture/color，以及 fine-grained geometry。

```
输入：
  C: 图像和 mask 条件 token
  O: 阶段 1 生成的稀疏 3D 结构
  z_noise: 每个稀疏 3D 位置上的 latent 噪声

输出：
  Z: 每个稀疏 3D 位置上的结构化 latent 特征
```

伪代码

```py
C = condition_encoder(I, M)
O = sparse_structure_generator(C)

Z_t = random_sparse_latent_noise(shape=O)

for t in denoising_steps:
    delta = slat_transformer(Z_t, O, C, t)
    Z_t = update(Z_t, delta)

Z = Z_t
```

这里 Z 就是 structured latent。它不是最终 mesh，也不是最终 Gaussian Splat，而是一个中间 3D latent field。它在每个稀疏 3D 坐标上存放高维特征，比如:

```
这个位置属于表面吗？
局部法线/几何趋势是什么？
这里应该是什么颜色？
这里是金属、木头、布料还是塑料？
这个位置是否需要透明度？
```


阶段 2 比阶段 1 更细，因为它既要看 3D 结构，又要回看原图纹理。例如一只鞋的粗结构只是鞋形，但第二阶段要补出鞋带、鞋底颜色、logo 区域、材质等。


## Decoder：从 SLaT latent 变成 mesh 和 Gaussian Splat

第二阶段输出 `Z` 后，模型还需要 decoder 把 latent 解码成可用 3D 资产。

SAM3D Objects 的实际输出核心是 Gaussian Splat。DeepWiki 总结说，`output["gs"]` 是一个 Gaussian Splat 对象，包含 3D representation 的 point positions、colors、scales、rotations 和 opacity values，并可导出或进一步处理。

一个 3D Gaussian 通常可以写成：

```
g_i = {
  μ_i: 3D 中心位置
  Σ_i: 3D 协方差/尺度/方向
  c_i: 颜色或球谐颜色参数
  α_i: opacity
}
```

也就是说，每个高斯不是一个普通点，而是一个带方向、尺度、颜色和透明度的“小椭球”。很多个高斯叠加起来，就能表示复杂表面和纹理。

所以 decoder 大致做的是：

```
Z
↓
Gaussian decoder
↓
{ μ_i, scale_i, rotation_i, color_i, opacity_i } for i = 1...N
↓
3D Gaussian Splat
```

## Layout：R、t、s 怎么用？

SAM3D 不只是生成一个“孤立物体坐标系”里的 3D 模型，它还要把物体放回原图对应的相机空间中。因此它会估计 layout：

```
R: rotation，物体旋转
t: translation，物体位置
s: scale，物体尺度
```

Fast-SAM3D 对 SAM3D 的描述中明确提到原模型会重建 shape、texture 和 layout parameters `(R, t, s)`

```
# object canonical coordinates
x_obj

# transform to camera/image-relative scene coordinates
x_scene = s * R @ x_obj + t
```

这一步很重要。否则模型只能生成一个居中的物体，不知道它在原图里的真实位置、朝向和大小。多物体场景里，每个物体都先独立重建，然后靠各自的 layout 对齐到同一场景。DeepWiki 也说明，多物体工作流是对每个 mask 独立跑 inference，再用 make_scene(*outputs) 组合成 unified scene representation。

## 多物体输入时，SAM3D 怎么算？

多物体并不是一次性把所有物体混在一起生成，而更像：

```py
image = load_image(...)
masks = load_masks(...)

outputs = []
for mask in masks:
    out = inference(image, mask, seed=42)
    outputs.append(out)

scene = make_scene(*outputs)
```

也就是说，同一张图片 I，每个 mask M_i 单独作为目标条件:

```
(I, M_1) -> object 1 Gaussian Splat
(I, M_2) -> object 2 Gaussian Splat
(I, M_3) -> object 3 Gaussian Splat
...
```

然后再组合为场景。DeepWiki 对 multi-object workflow 的描述正是：加载单张图像、加载所有 masks、对每个 mask 执行 inference、收集 outputs、组合 unified scene。

这也解释了为什么 mask 质量特别重要：mask 错了，模型就会重建错对象，或者把背景也混入 3D 资产。

## 为什么 SAM3D 能从单张图“猜出背面”？



这是单图 3D 重建最核心的问题：背面根本看不见，模型怎么知道？

答案是：它不是几何测量出来的，而是 生成式先验 + 图像条件约束。

模型在训练中见过大量物体的 3D 形状、纹理、姿态标注。论文摘要中提到，Meta 使用 human- and model-in-the-loop pipeline 来标注 object shape、texture 和 pose，构建大规模 visually grounded 3D reconstruction 数据，并采用 synthetic pretraining + real-world alignment 的多阶段训练框架



所以推理时，模型会结合

```
可见轮廓：mask 形状
可见纹理：RGB 像素
上下文：物体所在场景
类别先验：它像椅子、杯子、玩具还是鞋
3D 形状先验：这种物体通常有哪些背面结构
布局先验：它相对相机通常怎么摆放
```

例如看到一个正面杯子，模型不知道背面每个像素真实是什么，但它知道“杯子通常是圆柱形、有连续杯壁、杯柄可能延伸到背面”。所以它生成的是 合理的 3D 补全，不是从单图中物理唯一推导出来的真实背面。


## 用一个具体例子串起来：输入一张椅子图，内部发生什么？

假设输入是一张儿童房照片，mask 选中一把椅子。

### 第一步：加载

```py
image = load_image("image.png")      # RGBA image tensor
mask = load_single_mask(folder, 14)  # binary mask
output = inference(image, mask, seed=42)
```

### 第二步：预处理

```
原图 H×W×4
mask H×W
↓
根据 mask 裁剪椅子附近区域
↓
resize / normalize / pad
↓
得到模型输入 tensor
```

### 第三步：条件编码

```
椅子的 RGB 外观 + mask 轮廓 + 周围上下文
↓
vision encoder / condition embedder
↓
visual tokens C
```

这些 token 里包含：这是椅子、椅背在上方、坐垫在中间、腿在下方、左侧有遮挡等信息


### 第四步：粗结构生成

```
noise → sparse structure generator → 粗 3D voxel-like structure O
```

模型先生成椅子的粗 3D 骨架：

```
椅背：一大片竖直结构
坐面：水平结构
椅腿：几个细长结构
整体姿态：相对相机斜着
```

同时估计：

```
R: 椅子朝向
t: 椅子位置
s: 椅子大小
```


### 第五步：SLaT 细化


```
O + visual tokens C + latent noise
↓
SLaT generator 迭代去噪
↓
细化后的 3D latent Z
```

现在模型开始补细节：

```
椅腿粗细
椅背边缘
坐垫厚度
被遮挡后腿的合理位置
颜色和材质
```



### 第六步：解码

```
Z
↓
Gaussian decoder / mesh decoder
↓
3D Gaussian Splat / mesh
```

输出的 Gaussian Splat 中，每个高斯都有位置、颜色、尺度、旋转、透明度等属性。

### 第七步：导出和渲染

```py
output["gs"].save_ply("splat.ply")
```

如果是多物体，则多个输出再 make_scene() 合成同一场景。DeepWiki 的数据流总结也列出：image loading → mask loading → inference → GS extraction → scene composition → rendering/export。








