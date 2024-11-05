# FineZip & PPT Projects

## FineZip Overview
**FineZip** is a novel approach to lossless text compression using Large Language Models (LLMs). Building on previous work like LLMZip, FineZip pushes the boundaries of text compression by integrating both **online memorization** and **dynamic context size** techniques. These innovations lead to significant improvements in compression speed while maintaining competitive compression ratios compared to both traditional methods (e.g., gzip, bzip2) and neural network-based methods (e.g., NNCP, LLMZip). FineZip compresses text 54 times faster than LLMZip with a minor loss in compression performance. FineZip with Arithmetic coding also improves on LLMZip's AC approach by adding batch encoding and decoding.

### Main Contributions:
1. **FineZip** combines "online" memorization using parameter-efficient fine-tuning (PEFT) and "offline" pre-trained LLMs for text compression, enabling faster compression without sacrificing too much performance.
2. A **dynamic context window** allows batching of compression steps, significantly improving the compression speed.
3. **Quantization techniques** further optimize performance by reducing memory requirements, allowing larger batch sizes and faster compression times.
4. **Arithmetic Coding LLM Compression** allows batched encoding and decoding with Arithmetic Coding for LLM compression.

## PPT Project Overview
PPT (PowerPoint Presentation Tool) 是一个基于 Next.js 开发的在线演示文稿工具。

### 主要功能：
- 在线创建和编辑演示文稿
- 实时预览
- 支持多种幻灯片布局
- 响应式设计，支持多设备访问

### 技术栈：
- Next.js 14
- TypeScript
- Tailwind CSS
- [其他使用的技术]

## 安装说明

### FineZip 安装


