# clip_server

> 本文介绍如何用 Triton 在多 GPU 环境下部署高性能 CLIP 推理服务。

CLIP 是一个多模态模型。它能将图像和文本映射到同一个向量空间中，由此可以产生诸多应用。比如，通过计算图片与文本的相似性，可以用近似最近邻 (ANN) 从相册中检索与给定 query 语义相近的图片。此外，CLIP 的 Vision Encoder 可以作为特征提取器使用，用于生成的图像 Embedding。如果在 Vision Encoder 后加一个 fc 层，并且冻住骨干网络仅对 fc 层做训练，通常可以得到一个效果不错的图像分类器。

本文涉及的内容包括：

- 用 `transformers` 库运行 `openai/clip-vit-base-patch32` 的简单示例
- 在 `titanic` 数据集上训练一个 MLP 并导出成 ONNX 格式
- 介绍如何安装预装了 Triton 的 Nvidia 官方 Docker 镜像 `&` 启动容器
- 介绍如何将 MLP 的 ONNX 模型配置到 Triton 模型仓库中
- 写了一个简单的 [客户端](utils.py#L108) 用于获取 Triton 的推理结果
- 介绍 Triton 的 Python Backend，其通常用于模型预处理和后处理
- 用 Model Ensemble 组装 Python Backend 和 ONNX 组成完整的推理服务

✨ 注意：运行以下代码依赖 [utils.py](/utils.py) 文件和 [mlp.py](/mlp.py) 文件。

### 一、简单的 CLIP 模型

#### 1）CLIP 模型介绍

为了训练 CLIP，OpenAI 收集了 4 亿对图文数据进行训练。训练目标是让图片的特征向量与对应文本的特征向量在向量空间中靠得更近。训练采用多模态对比学习的方法。在一个 batch 中，对于每张图片，它的目标是找到当前 batch 中与之最匹配的文本，最大化与匹配文本的相似度（正样本），并同时最小化与其他文本的相似度（负样本）。

CLIP 训练了两个独立的编码器：

- **图像编码器**：通常使用 ResNet 或 Vision Transformer (ViT)。
- **文本编码器**：基于 Transformer 结构。

OpenAI 尝试了多种编码器，得出一个很直觉的结论：模型的效果与参数量呈现正相关。基本上使用参数越大的编码器，效果就越好。

#### 2）用 CLIP 计算图文相似性分数

用 `transformers` 库加载 `openai/clip-vit-base-patch32`。并用一张猫的图片与两句话进行对比：

- a photo of a cat
- a photo of a dog

使用 CLIP 模型，计算猫的图片与每句话的相似性分数，取分数最高的句子作为图片的分类标签。验证模型能否有效区分猫和狗。

> **Note:** 值得注意的是 `a photo of {item}` 是一种 <a href="https://github.com/openai/CLIP/blob/main/> notebooks/Prompt_Engineering_for_ImageNet.ipynb" target="_blank">Prompt Engineer</a> 方法。除了前面这个，OpenAI 还用了很多其他标签。比如：
>
> ```
> a bad photo of a {}.
> a photo of many {}.
> a sculpture of a {}.
> a photo of the hard to see {}.
> a low resolution photo of the {}.
> a rendering of a {}.
> graffiti of a {}.
> ```


### 二、ONNX 模型导出

本节我们在 <a href="https://www.kaggle.com/datasets/heptapod/titanic/data" target="_blank">titanic</a> 数据集上训练一个 MLP 模型，并将它导出成 ONNX 格式。

> **Note:** 需要特别注意，用 `torch.onnx.export` 导出模型时，一定要将所有可变维度都配置上。比如，如果 `batch_size`, `sequence_length` 这些维度都是可变的，一定要全部配上，否则后期写 Triton 的 `config.pbtxt` 配置的时候会出问题。


### 三、搭建 Triton 推理服务

本节介绍如何下载、安装、配置、启动 Triton 服务，以及如何用客户端获取 Triton 的推理结果。


### 四、Python Backend 入门

ONNX 只接受张量作为输入。如果希望给 Triton 提供图像或者文本，那就需要对输入做预处理，处理成张量后再传给 ONNX.

在 Triton 中做图像、文本预处理，需要用到 <a href="https://github.com/triton-inference-server/python_backend" target="_blank">Python Backend</a>.


### 五、模型集成 Model Ensembles

前面所有小节都是准备工作，本节终于进入正题：**如何用 Triton 搭建 CLIP 图文 Embedding 推理服务**。

步骤大致如下：

1. **拆解模型**：将模型预处理部分和主体部分的代码分开
3. **预处理模型**：
    - **新建模型**：新建一个模型文件 `your_preprocess_model/1/model.py`
    - **编写预处理代码**：将预处理过程写入 `model.py`，需要按文档要求实现 `TritonPythonModel` 的三个函数：`initialize`, `execute` 和 `finalize`。其中 `initialize` 只在模型初始化时执行一次，`execute` 负责处理请求，`finalize` 在模型卸载时调用
    - **配置**：写配置文 `your_preprocess_model/config.pbtxt`
    - **测试**：用客户端测试预处理模型
4. **主体模型**：
    - **新建模型**：在模型仓库下，新建一个模型目录 `your_onnx_model/1/`，并将 ONNX 模型放入
    - **模型格式转换**：将主体模型转成 ONNX 格式，这样可以加速推理
    - **配置**：写配置文件 `your_onnx_model/config.pbtxt`
    - **测试**：用客户端测试主体模型
5. **模型集成**：
    - **新建模型**：新建一个模型目录 `your_ensemble_model/1/`，只需要这个目录结构，不用往里放文件
    - **配置**：写配置文件 `your_ensemble_model/config.pbtxt`
    - **测试**：用客户端测试集成模型


### 参考资料

#### 1）GitHub

- [openai/CLIP](https://github.com/openai/CLIP)
- [triton-inference-server/server](https://github.com/triton-inference-server/server)
- [triton-inference-server/python_backend](https://github.com/triton-inference-server/python_backend)
- [triton-inference-server/tutorials](https://github.com/triton-inference-server/tutorials)
- [triton: model_configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)
- [onnx/onnx](https://github.com/onnx/onnx)

#### 2）视频资源

- [CLIP 论文逐段精读](https://www.bilibili.com/video/BV1SL4y1s7LQ)
- [CLIP 改进工作串讲（上）](https://www.bilibili.com/video/BV1FV4y1p7Lm)
- [CLIP 改进工作串讲（下）](https://www.bilibili.com/video/BV1gg411U7n4)

#### 3）其他资源

- [OpenAI CLIP 官网](https://openai.com/index/clip/)
- [Hugging Face CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip)
- [arXiv: *Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020)
