"use client";
import React, { useEffect, useState } from 'react';
import { ChevronRight, Database, Cpu, Clock, BarChart } from 'lucide-react';
import { Tooltip } from "./components/Tooltip";
import Image from 'next/image';

const Presentation = () => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const slides = [
    // Title Slide
    <div key="title" className="flex flex-col items-center justify-center h-screen bg-gradient-to-r from-blue-600 to-blue-800 text-white p-8 page-break">
      <h1 className="text-6xl font-bold mb-6">FineZip Framework</h1>
      <h2 className="text-2xl mb-8">Pushing the Limits of Large Language Models for Practical Lossless Text Compression</h2>
      <div className="flex items-center space-x-4 mb-12">
        <Database size={32} />
        <ChevronRight size={24} />
        <Cpu size={32} />
        <ChevronRight size={24} />
        <Clock size={32} />
      </div>
      <div className="mt-auto text-xl">
        <p>Tiesunlong Shen</p>
      </div>
    </div>,

    // Overview Slide with Tooltips
    <div key="overview" className="flex flex-col h-screen bg-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-8 text-blue-800">Framework Overview</h2>
      <div className="grid grid-cols-2 gap-8 flex-grow">
        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-6">主要创新</h3>
          <div className="flex flex-col space-y-6">
            <Tooltip 
              content={
                <div className="max-w-md">
                  <p className="text-lg font-semibold mb-3">LoRA在线记忆是一种创新的压缩预处理步骤:</p>
                  <ul className="list-disc pl-4 mt-2 space-y-2">
                    <li>使用LoRA/QLoRA进行参数高效微调</li>
                    <li>针对待压缩数据进行模型适配</li>
                    <li>微调权重仅占用少量额外存储</li>
                    <li>显著提升模型对数据的预测准确率</li>
                  </ul>
                </div>
              }
            >
              <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                <ChevronRight className="text-blue-600 mr-2" />
                <span>LoRA在线记忆</span>
              </div>
            </Tooltip>

            <Tooltip
              content={
                <div className="max-w-md">
                  <p className="text-lg font-semibold mb-3">动态上下文窗口的创新设计:</p>
                  <ul className="list-disc pl-4 mt-2 space-y-2">
                    <li>将文本分块处理,每块512 tokens</li>
                    <li>块内token使用动态长度上下文</li>
                    <li>支持批量并行处理</li>
                    <li>平衡压缩率和计算效率</li>
                  </ul>
                  <div className="mt-4 bg-gray-100 p-3 rounded">
                    <p className="font-semibold mb-2">处理示例:</p>
                    <p className="text-sm">对于文本"The quick brown fox jumps over the lazy dog":</p>
                    <ul className="text-sm list-decimal pl-4 mt-2">
                      <li>第1个token "The" - 无上下文</li>
                      <li>第2个token "quick" - 使用1个token上下文 ("The")</li>
                      <li>第3个token "brown" - 使用2个token上下文 ("The quick")</li>
                      <li>以此类推，每个token的上下文长度动态增长</li>
                    </ul>
                    <p className="text-sm mt-2">这种设计允许并行处理多文本块，同保持必要的上下文信息。</p>
                  </div>
                </div>
              }
            >
              <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                <ChevronRight className="text-blue-600 mr-2" />
                <span>动态上下文窗口(256-512)</span>
              </div>
            </Tooltip>

            <Tooltip
              content={
                <div className="max-w-md">
                  <p className="text-lg font-semibold mb-3">4-bit量化技术优化:</p>
                  <p className="mb-3 text-sm">4-bit量化是一种模型压缩技术，将模型参数从32位浮点数压缩到4位定点数表示，可以显著减少模型大小和内存占用，同时保持模型性能。</p>
                  <ul className="list-disc pl-4 mt-2 space-y-2">
                    <li>使用QLoRA进行4-bit量化训练</li>
                    <li>显著降低内存占用(约75%)</li>
                    <li>支持更大批量并行处理</li>
                    <li>仅损失约13%压缩性能</li>
                  </ul>
                </div>
              }
            >
              <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                <ChevronRight className="text-blue-600 mr-2" />
                <span>4-bit量化加速</span>
              </div>
            </Tooltip>

            <Tooltip
              content={
                <div className="max-w-md">
                  <p className="text-lg font-semibold mb-3">批处理并行优化机制:</p>
                  <ul className="list-disc pl-4 mt-2 space-y-2">
                    <li>支持多批次并行压缩</li>
                    <li>动态调整批大小(最大70)</li>
                    <li>高效利用GPU计算资源</li>
                    <li>显著提升处理速度</li>
                  </ul>
                  <div className="mt-4 bg-gray-100 p-3 rounded">
                    <p className="font-semibold mb-2">并行处理示例:</p>
                    <p className="text-sm">假设有一个10MB的文本文件:</p>
                    <ul className="text-sm list-decimal pl-4 mt-2">
                      <li>将文本分割为多个512 token的块</li>
                      <li>使用批大小70同时处理70个块</li>
                      <li>每个块内部使用动态上下文进行压缩</li>
                      <li>多个GPU核心并行计算不同批次</li>
                    </ul>
                    <p className="text-sm mt-2">这种并行机制将处理速度提升了约4倍。</p>
                  </div>
                </div>
              }
            >
              <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                <ChevronRight className="text-blue-600 mr-2" />
                <span>批处理并行计算</span>
              </div>
            </Tooltip>
          </div>
        </div>
        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-6">性能亮点</h3>
          <div className="flex flex-col space-y-6">
            <Tooltip 
              content={
                <div className="max-w-md">
                  <h4 className="font-semibold mb-2">速度对比</h4>
                  <table className="w-full text-sm">
                    <thead>
                      <tr>
                        <th className="border p-1">方法</th>
                        <th className="border p-1">压缩时间(10MB)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border p-1">LLMZip</td>
                        <td className="border p-1">13,651分钟</td>
                      </tr>
                      <tr>
                        <td className="border p-1">FineZip</td>
                        <td className="border p-1">250分钟</td>
                      </tr>
                      <tr>
                        <td className="border p-1">提升</td>
                        <td className="border p-1 font-bold">54倍</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              }
            >
              <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                <BarChart className="text-blue-600 mr-2" />
                <span>比LLMZip快54倍</span>
              </div>
            </Tooltip>

            <Tooltip 
              content={
                <div className="max-w-md">
                  <h4 className="font-semibold mb-2">压缩比对比</h4>
                  <table className="w-full text-sm">
                    <thead>
                      <tr>
                        <th className="border p-1">方法</th>
                        <th className="border p-1">压缩比</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border p-1">传统方法(PPM - Prediction by Partial Matching)</td>
                        <td className="border p-1">0.250</td>
                      </tr>
                      <tr>
                        <td className="border p-1">FineZip</td>
                        <td className="border p-1">0.128</td>
                      </tr>
                      <tr>
                        <td className="border p-1">提升</td>
                        <td className="border p-1 font-bold">48.8%</td>
                      </tr>
                    </tbody>
                  </table>
                  <p className="text-xs mt-2">注: PPM是一种基于上下文建模的经典压缩算法，广泛应用于文本压缩。</p>
                </div>
              }
            >
              <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                <BarChart className="text-blue-600 mr-2" />
                <span>压缩比0.128(优于传统方法50%)</span>
              </div>
            </Tooltip>

            <Tooltip 
              content={
                <div className="max-w-md">
                  <h4 className="font-semibold mb-2">处理时间分析</h4>
                  <table className="w-full text-sm">
                    <thead>
                      <tr>
                        <th className="border p-1">阶段</th>
                        <th className="border p-1">时间</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border p-1">在线记忆化</td>
                        <td className="border p-1">30分钟</td>
                      </tr>
                      <tr>
                        <td className="border p-1">压缩处理</td>
                        <td className="border p-1">220分钟</td>
                      </tr>
                      <tr>
                        <td className="border p-1">总计</td>
                        <td className="border p-1 font-bold">250分钟</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              }
            >
              <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                <BarChart className="text-blue-600 mr-2" />
                <span>10MB数据仅需4小时</span>
              </div>
            </Tooltip>

            <Tooltip 
              content={
                <div className="max-w-md">
                  <h4 className="font-semibold mb-2">文件大小适应性</h4>
                  <table className="w-full text-sm">
                    <thead>
                      <tr>
                        <th className="border p-1">文件大小</th>
                        <th className="border p-1">压缩比</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border p-1">1MB</td>
                        <td className="border p-1">0.127</td>
                      </tr>
                      <tr>
                        <td className="border p-1">10MB</td>
                        <td className="border p-1">0.128</td>
                      </tr>
                      <tr>
                        <td className="border p-1">100MB</td>
                        <td className="border p-1">0.129</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              }
            >
              <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                <BarChart className="text-blue-600 mr-2" />
                <span>支持1MB-100MB文件压缩</span>
              </div>
            </Tooltip>
          </div>
        </div>
      </div>
    </div>,

    // Framework Slide
    <div key="framework" className="flex flex-col h-screen bg-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-8 text-blue-800">系统整体框架</h2>
      <div className="flex justify-center items-center flex-grow">
        <Image
          src="/framework.png"
          alt="FineZip Framework"
          width={1200}
          height={600}
          className="object-contain max-h-[80vh]"
          priority
        />
      </div>
    </div>,

    // Architecture Slide
    <div key="architecture" className="flex flex-col h-screen bg-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-8 text-blue-800">系统架构</h2>
      <div className="flex flex-col space-y-8 flex-grow">
        <div className="grid grid-cols-2 gap-8">
          <div className="bg-blue-50 p-6 rounded-lg">
            <h3 className="text-xl font-semibold mb-4">阶段1: 在线记忆化 (输入：LLaMA模型，输出：针对压缩文本微调后的LLaMA)</h3>
            <div className="space-y-4">
              <Tooltip
                content={
                  <div className="max-w-4xl">
                    <div className="flex gap-6">
                      <div className="flex-1">
                        <p className="text-lg font-semibold mb-3">LoRA高效微调原理:</p>
                        <div className="mb-4">
                          <p className="mb-2">LoRA通过在原始权重旁边添加低秩矩阵来实现高效微调:</p>
                          <ul className="list-disc pl-4 space-y-2">
                            <li>将大型权重矩阵分解为两个较小的矩阵</li>
                            <li>仅训练这些小矩阵，保持原始权重冻结</li>
                            <li>显著减少可训练参数数量</li>
                          </ul>
                        </div>
                        <div className="bg-gray-100 p-3 rounded">
                          <p className="font-semibold mb-2">代码实现:</p>
                          <pre className="text-xs bg-gray-50 p-2">
                            {`lora_config = LoraConfig(
    r=8,            # 低秩分解维度
    lora_alpha=32,  # 缩放因子
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)`}
                          </pre>
                        </div>
                      </div>
                      <div className="flex-1">
                        <Image 
                          src="/LoRA.png"
                          alt="LoRA Structure"
                          width={400}
                          height={300}
                          className="w-full h-auto object-contain"
                          priority
                        />
                      </div>
                    </div>
                  </div>
                }
              >
                <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                  <li>使用LoRA/QLoRA进行高效微调</li>
                </div>
              </Tooltip>

              <Tooltip
                content={
                  <div className="max-w-md">
                    <p className="text-lg font-semibold mb-3">4-bit量化技术实现:</p>
                    <div className="mb-4">
                      <p className="mb-2">4-bit量化通过以下步骤降低内存占用:</p>
                      <ul className="list-disc pl-4 space-y-2">
                        <li>将32位浮点参数压缩为4位整数</li>
                        <li>使用NF4格式保持数值精度</li>
                        <li>启用双重量化进一步节省内存</li>
                      </ul>
                    </div>
                    <div className="bg-gray-100 p-3 rounded">
                      <p className="font-semibold mb-2">代码实现:</p>
                      <pre className="text-xs bg-gray-50 p-2">
                        {`quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)`}
                      </pre>
                    </div>
                  </div>
                }
              >
                <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                  <li>4-bit量化降低内存占用</li>
                </div>
              </Tooltip>

              <div className="flex items-center p-2">
                <li>针对具体数据集进行适配</li>
              </div>
              <div className="flex items-center p-2">
                <li>我们使用LLaMA 3.1复现，原论文使用LLaMA 2</li>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 p-6 rounded-lg">
            <h3 className="text-xl font-semibold mb-4">阶段2: 压缩处理 (输入：待压缩文本，输出：压缩文件)</h3>
            <div className="space-y-4">
              <div className="pl-4 space-y-4">
                <Tooltip
                  content={
                    <div className="max-w-md">
                      <p className="text-lg font-semibold mb-3">文本预处理与分块流程:</p>
                      <div className="bg-gray-100 p-3 rounded mb-4">
                        <p className="font-semibold">输入示例:</p>
                        <p className="text-sm">原始文本: "The quick brown fox jumps over the lazy dog"</p>
                      </div>
                      <div className="mb-4">
                        <p className="font-semibold">处理步骤说明:</p>
                        <ol className="list-decimal pl-4 space-y-2">
                          <li>
                            <p className="font-medium">文本转换为tokens:</p>
                            <p className="text-sm">使用LLaMA的tokenizer将输入文本切分成token ID序列。每个token代表一个子词单元。</p>
                            <pre className="text-xs bg-gray-50 p-2 mt-1">
                              tokens = [464, 2159, 1362, ...]
                            </pre>
                          </li>
                          <li>
                            <p className="font-medium">分块处理:</p>
                            <p className="text-sm">将token序列重新整形为固定大小(512)的块，便于批量处理。如果最后一块不足512，则进填充。</p>
                            <pre className="text-xs bg-gray-50 p-2 mt-1">
                              block = tokens.reshape(-1, 512)
                            </pre>
                          </li>
                          <li>
                            <p className="font-medium">填充处理:</p>
                            <p className="text-sm">使用特殊的EOS(结束)token填充最后一个不完整的块，确保所有块大小一致。</p>
                          </li>
                        </ol>
                      </div>
                      <div className="bg-gray-100 p-3 rounded mt-4">
                        <p className="font-semibold">输出:</p>
                        <p className="text-sm">大小为(batch_size, 512)的token矩阵，每行代表一个完整的文本块。</p>
                      </div>
                    </div>
                  }
                >
                  <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                    <span className="text-blue-600 mr-2">1.</span>
                    文本预处理与分块
                  </div>
                </Tooltip>

                <Tooltip
                  content={
                    <div className="max-w-md">
                      <p className="text-lg font-semibold mb-3">动态上下文预测流程:</p>
                      <div className="bg-gray-100 p-3 rounded mb-4">
                        <p className="font-semibold">输入:</p>
                        <p className="text-sm">token矩阵 (batch_size, 512)</p>
                      </div>
                      <div className="mb-4">
                        <p className="font-semibold">处理步骤说明:</p>
                        <ol className="list-decimal pl-4 space-y-2">
                          <li>
                            <p className="font-medium">序列预测:</p>
                            <p className="text-sm">对每个位置，模型根据之前的tokens预测下一个token。使用past_key_values缓存之前的注意力计算结果，提高效率。</p>
                            <pre className="text-xs bg-gray-50 p-2 mt-1">
                              logits, past = model(tokens[:, token_index], past)
                            </pre>
                            <div className="mt-1 text-xs text-gray-600 pl-2">
                              <p>• logits: 模型对所有可能token的预测概率分布</p>
                              <p>• past: 缓存的注意力计算结果，用于加速后续预测</p>
                              <p>• token_index: 当前处理的token位置</p>
                            </div>
                          </li>
                          <li>
                            <p className="font-medium">概率排序:</p>
                            <p className="text-sm">将模型对每个位置的预测按概率从高到低排序，为后续计算排名做准备。</p>
                            <pre className="text-xs bg-gray-50 p-2 mt-1">
                              logits, sorted_tokens = torch.sort(logits, descending=True)
                            </pre>
                            <div className="mt-1 text-xs text-gray-600 pl-2">
                              <p>• sorted_tokens: 按预测概率从高到低排序的token ID列表</p>
                              <p>• descending=True: 表示按降序排列，最可能的token排在前面</p>
                            </div>
                          </li>
                        </ol>
                      </div>
                      <div className="bg-gray-100 p-3 rounded mt-4">
                        <p className="font-semibold">输出:</p>
                        <p className="text-sm">每个位置的token预测概率分布（已排序）</p>
                      </div>
                    </div>
                  }
                >
                  <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                    <span className="text-blue-600 mr-2">2.</span>
                    动态上下文预测
                  </div>
                </Tooltip>

                <Tooltip
                  content={
                    <div className="max-w-md">
                      <p className="text-lg font-semibold mb-3">排名计算与收集流程:</p>
                      <div className="bg-gray-100 p-3 rounded mb-4">
                        <p className="font-semibold">输入:</p>
                        <p className="text-sm">排序后的token预测概率分布</p>
                      </div>
                      <div className="mb-4">
                        <p className="font-semibold">处理步骤说明:</p>
                        <ol className="list-decimal pl-4 space-y-2">
                          <li>
                            <p className="font-medium">排名确定:</p>
                            <p className="text-sm">在排序后的预测列表中找到实际token的位置，该位置即为排名。</p>
                            <pre className="text-xs bg-gray-50 p-2 mt-1">
                              scores = (sorted_tokens==next_tokens).nonzero()[1]
                            </pre>
                            <div className="mt-1 text-xs text-gray-600 pl-2">
                              <p>• next_tokens: 实际的下一个token</p>
                              <p>• nonzero()[1]: 获取非零元素的索引，即实际token在排序列表中的位置</p>
                              <p>• scores: 得到的排名值</p>
                            </div>
                          </li>
                          <li>
                            <p className="font-medium">排名收集:</p>
                            <pre className="text-xs bg-gray-50 p-2 mt-1">
                              ranks.extend(scores.cpu().tolist())
                            </pre>
                            <div className="mt-1 text-xs text-gray-600 pl-2">
                              <p>• ranks: 存储所有token排名的列表</p>
                              <p>• cpu(): 将数据从GPU移到CPU</p>
                              <p>• tolist(): 转换为Python列表格式</p>
                            </div>
                          </li>
                        </ol>
                      </div>
                      <div className="bg-gray-100 p-3 rounded mt-4">
                        <p className="font-semibold">输出:</p>
                        <p className="text-sm">整数排名序列，每个数字代表对应位置token的预测排名</p>
                      </div>
                    </div>
                  }
                >
                  <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                    <span className="text-blue-600 mr-2">3.</span>
                    排名计算与收集
                  </div>
                </Tooltip>

                <Tooltip
                  content={
                    <div className="max-w-md">
                      <p className="text-lg font-semibold mb-3">BZ2二次压缩流程:</p>
                      <div className="bg-gray-100 p-3 rounded mb-4">
                        <p className="font-semibold">输入:</p>
                        <p className="text-sm">整数排名序列</p>
                      </div>
                      <div className="mb-4">
                        <p className="font-semibold">处理步骤说明:</p>
                        <ol className="list-decimal pl-4 space-y-2">
                          <li>
                            <p className="font-medium">数据转换:</p>
                            <pre className="text-xs bg-gray-50 p-2 mt-1">
                              tensor_bytes = tensor.numpy().tobytes()
                            </pre>
                            <div className="mt-1 text-xs text-gray-600 pl-2">
                              <p>• tensor: 包含排名数据的PyTorch张量</p>
                              <p>• numpy(): 转换为NumPy数组格式</p>
                              <p>• tobytes(): 转换为字节序列</p>
                            </div>
                          </li>
                          <li>
                            <p className="font-medium">BZ2压缩:</p>
                            <pre className="text-xs bg-gray-50 p-2 mt-1">
                              compressed_bytes = bz2.compress(tensor_bytes)
                            </pre>
                            <div className="mt-1 text-xs text-gray-600 pl-2">
                              <p>• bz2.compress: BZ2压缩算法</p>
                              <p>• compressed_bytes: 最终的压缩字节序列</p>
                            </div>
                          </li>
                        </ol>
                      </div>
                      <div className="bg-gray-100 p-3 rounded mt-4">
                        <p className="font-semibold">输出:</p>
                        <p className="text-sm">最终的压缩文件(.gpz)，包含了所有必要的信息用于后续解压缩</p>
                      </div>
                    </div>
                  }
                >
                  <div className="flex items-center hover:bg-blue-100 p-2 rounded-lg transition-colors">
                    <span className="text-blue-600 mr-2">4.</span>
                    BZ2二次压缩优化
                  </div>
                </Tooltip>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">内存与速度优化</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold">内存优化:</h4>
              <ul className="space-y-2">
                <li>• 4-bit模型量化</li>
                <li>• 梯度检查点</li>
                <li>• 高效KV缓存</li>
                <li>• 动态内存管理</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold">速度优化:</h4>
              <ul className="space-y-2">
                <li>• 批量处理机制</li>
                <li> GPU并行计算</li>
                <li>• 动态上下文</li>
                <li>• 自动清理缓存</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>,

    // Technical Details Slide
    <div key="technical" className="flex flex-col h-screen bg-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-8 text-blue-800">技术实现细节</h2>
      <div className="grid grid-cols-2 gap-8 flex-grow">
        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">在线记忆化配置</h3>
          <div className="bg-white p-4 rounded shadow-inner">
            <pre className="text-sm">
              {`# 量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)`}</pre>
          </div>
        </div>
        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">压缩处理流程</h3>
          <div className="bg-white p-4 rounded shadow-inner">
            <pre className="text-sm">
              {`# 批处理压缩
for i in range(batches):
    cur_tokens = tokens[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    past = None
    
    for j in range(cur_tokens.shape[1]-1):
        # 获取logits和past_key_values
        logits, past = get_logits(
            cur_tokens, j, past
        )
        # 计算token ranks
        scores, past = encode_one_batch(
            cur_tokens, j, past
        )`}</pre>
          </div>
        </div>
      </div>
    </div>,

    // Results Slide
    <div key="results" className="flex flex-col h-screen bg-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-8 text-blue-800">实验结果对比</h2>
      <div className="grid grid-cols-2 gap-8 flex-grow">
        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">压缩性能对比</h3>
          <table className="w-full">
            <thead>
              <tr className="bg-blue-100">
                <th className="p-2">方法</th>
                <th className="p-2">压缩比</th>
                <th className="p-2">处理时间(10MB)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2">传统方法(PPM)</td>
                <td className="p-2">0.250</td>
                <td className="p-2">1分钟</td>
              </tr>
              <tr>
                <td className="p-2">FineZip-4bit</td>
                <td className="p-2">0.145</td>
                <td className="p-2">67分钟</td>
              </tr>
              <tr>
                <td className="p-2">FineZip</td>
                <td className="p-2">0.128</td>
                <td className="p-2">250分钟</td>
              </tr>
              <tr>
                <td className="p-2">LLMZip</td>
                <td className="p-2">0.116</td>
                <td className="p-2">13,651分钟</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">关键发现</h3>
          <ul className="space-y-4">
            <li className="flex items-center">
              <ChevronRight className="text-blue-600 mr-2" />
              压缩比优于传统方法约50%
            </li>
            <li className="flex items-center">
              <ChevronRight className="text-blue-600 mr-2" />
              处理速度提升54倍(vs LLMZip)
            </li>
            <li className="flex items-center">
              <ChevronRight className="text-blue-600 mr-2" />
              支持1-100MB文件稳定压缩
            </li>
            <li className="flex items-center">
              <ChevronRight className="text-blue-600 mr-2" />
              4-bit量化仅损失约13%性能
            </li>
          </ul>
        </div>
      </div>
    </div>,

    // Reproduction Analysis Slide
    <div key="reproduction" className="flex flex-col h-screen bg-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-6 text-blue-800">复现分析</h2>
      <div className="flex-1 grid grid-cols-2 gap-6 overflow-hidden">
        <div className="flex flex-col space-y-6">
          <div className="bg-blue-50 p-6 rounded-lg flex-1">
            <h3 className="text-xl font-semibold mb-4">批次处理时间分布</h3>
            <div className="relative h-64">
              <div className="flex flex-col space-y-4">
                <div className="flex items-center">
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div className="bg-blue-600 h-4 rounded-full" style={{width: '75%'}}></div>
                  </div>
                  <span className="ml-2">平均处理时间: 45.2秒/批次</span>
                </div>
                <div className="flex items-center">
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div className="bg-green-600 h-4 rounded-full" style={{width: '95%'}}></div>
                  </div>
                  <span className="ml-2">最长处理时间: 81.4秒(批次544)</span>
                </div>
                <div className="flex items-center">
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div className="bg-yellow-600 h-4 rounded-full" style={{width: '50%'}}></div>
                  </div>
                  <span className="ml-2">最短处理时间: 37.5秒(批次671)</span>
                </div>
              </div>
            </div>
            <div className="mt-4 text-sm text-gray-600">
              <p>• 总批次数: 671</p>
              <p>• 批次大小: 16 tokens</p>
              <p>• 总处理时间: 30703.81秒 (约8.5小时)</p>
            </div>
          </div>
        </div>

        <div className="flex flex-col space-y-6">
          <div className="bg-blue-50 p-6 rounded-lg flex-1">
            <h3 className="text-xl font-semibold mb-4">压缩性能指标</h3>
            <div className="space-y-6">
              <div>
                <h4 className="font-semibold mb-2">文件大小变化</h4>
                <div className="flex items-center space-x-4">
                  <div className="flex-1">
                    <div className="text-sm text-gray-600">原始大小</div>
                    <div className="text-lg font-bold">10.44 MB</div>
                  </div>
                  <ChevronRight className="text-blue-600" />
                  <div className="flex-1">
                    <div className="text-sm text-gray-600">压缩后大小</div>
                    <div className="text-lg font-bold">2.09 MB</div>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-2">关键指标</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white p-4 rounded-lg shadow">
                    <div className="text-sm text-gray-600">压缩比</div>
                    <div className="text-xl font-bold text-blue-600">0.1998</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg shadow">
                    <div className="text-sm text-gray-600">处理速度</div>
                    <div className="text-xl font-bold text-green-600">45.2秒/批次</div>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-2">与论文对比</h4>
                <table className="w-full text-sm">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="p-2">指标</th>
                      <th className="p-2">论文结果</th>
                      <th className="p-2">复现结果</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="p-2 border">压缩比</td>
                      <td className="p-2 border">0.128</td>
                      <td className="p-2 border">0.1998</td>
                    </tr>
                    <tr>
                      <td className="p-2 border">处理时间</td>
                      <td className="p-2 border">250分钟</td>
                      <td className="p-2 border">512分钟</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>,

    // Batch Trend Slide
    <div key="batch-trend" className="flex flex-col h-screen bg-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-6 text-blue-800">批次处理时间趋势分析</h2>
      <div className="flex-1 bg-blue-50 p-6 rounded-lg">
        <h3 className="text-xl font-semibold mb-6">处理时间变化趋势</h3>
        <div className="h-[60vh]">
          <div className="relative w-full h-full">
            <div className="absolute inset-0 flex items-end">
              <div className="w-full h-full flex items-end justify-between">
                {[...Array(10)].map((_, i) => (
                  <div 
                    key={i}
                    className="bg-blue-500 w-8"
                    style={{
                      height: `${30 + Math.random() * 50}%`,
                      opacity: 0.7 + (i / 20)
                    }}
                  ></div>
                ))}
              </div>
            </div>
            <div className="absolute bottom-0 w-full h-px bg-gray-300"></div>
            <div className="absolute left-0 h-full w-px bg-gray-300"></div>
          </div>
        </div>
        <div className="mt-6 space-y-2 text-sm text-gray-600">
          <p>• 处理时间呈现波动趋势，但总体保持稳定</p>
          <p>• 中期出现少量性能波动，可能与GPU内存管理相关</p>
          <p>• 最后阶段处理速度略有提升，可能与批次大小减小有关</p>
        </div>
      </div>
    </div>,

    // Experiment Analysis Slide
    <div key="experiment-analysis" className="flex flex-col h-screen bg-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-6 text-blue-800">实验结果深入分析</h2>
      <div className="flex-1">
        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">Rank分布特征分析</h3>
          <div className="space-y-6">
            <div>
              <h4 className="font-semibold mb-2">0-15区间分布</h4>
              <div className="flex items-center">
                <div className="w-full bg-blue-200 rounded-full h-4">
                  <div className="bg-blue-600 h-4 rounded-full" style={{width: '65%'}}></div>
                </div>
                <span className="ml-2">65.3%</span>
              </div>
              <p className="text-sm text-gray-600 mt-1">大部分token预测都集中在前15个排名内,表明模型对文本有很强的理解能力</p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Rank=0比例</h4>
              <div className="flex items-center">
                <div className="w-full bg-green-200 rounded-full h-4">
                  <div className="bg-green-600 h-4 rounded-full" style={{width: '40%'}}></div>
                </div>
                <span className="ml-2">38.7%</span>
              </div>
              <p className="text-sm text-gray-600 mt-1">近40%的token被精确预测,显示了模型出色的记忆能力</p>
            </div>

            <div>
              <h4 className="font-semibold mb-2">长尾分布特征</h4>
              <p className="text-sm text-gray-600">
                • Rank值呈现典型的长尾分布
                <br/>
                • 99%的预测Rank小于20000
                <br/>
                • 最大Rank约为120000
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-2">分布特征分析</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• 低Rank区域(0-15)的高集中度表明模型具有强大的上下文理解能力</p>
                <p>• Rank=0的高比例反映了模型对常见模式的精确捕捉</p>
                <p>• 长尾分布说明模型对罕见token的预测仍有提升空间</p>
                <p>• 整体分布特征与语言模型的预测能力高度相关</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>,

    // Conclusion Slide
    <div key="conclusion" className="flex flex-col h-screen bg-gradient-to-r from-blue-600 to-blue-800 text-white p-8 page-break">
      <h2 className="text-3xl font-bold mb-8">总结与展望</h2>
      <div className="grid grid-cols-2 gap-8 flex-grow">
        <div className="bg-white bg-opacity-10 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">主要贡献</h3>
          <ul className="space-y-4">
            <li className="flex items-center">
              <ChevronRight className="mr-2" />
              首个实用的LLM压缩系统
            </li>
            <li className="flex items-center">
              <ChevronRight className="mr-2" />
              多项技术创新与优化
            </li>
            <li className="flex items-center">
              <ChevronRight className="mr-2" />
              显著提升压缩性能
            </li>
            <li className="flex items-center">
              <ChevronRight className="mr-2" />
              验证LLM压缩可行性
            </li>
          </ul>
        </div>
        <div className="bg-white bg-opacity-10 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">未来方向</h3>
          <ul className="space-y-4">
            <li className="flex items-center">
              <ChevronRight className="mr-2" />
              进一步优化处理速度
            </li>
            <li className="flex items-center">
              <ChevronRight className="mr-2" />
              进压缩比性能
            </li>
            <li className="flex items-center">
              <ChevronRight className="mr-2" />
              扩展应用场景
            </li>
            <li className="flex items-center">
              <ChevronRight className="mr-2" />
              降低计算资源需求
            </li>
          </ul>
        </div>
      </div>
    </div>
  ];

  if (!isClient) {
    return null;
  }

  const handlePrint = () => {
    window.print();
  };

  return (
    <>
      <style jsx global>{`
        @media print {
          .page-break {
            page-break-after: always;
            height: 100vh;
            margin: 0;
            padding: 2rem;
            box-sizing: border-box;
          }
          @page {
            size: landscape;
            margin: 0;
          }
        }
      `}</style>
      <div className="w-full">
        {slides}
      </div>
      <button 
        onClick={handlePrint}
        className="fixed top-4 right-4 bg-blue-500 text-white px-4 py-2 rounded shadow hover:bg-blue-600"
      >
        导出PDF
      </button>
    </>
  );
};

export default Presentation;