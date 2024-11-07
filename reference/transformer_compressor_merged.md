# transformer_compressor

## 元数据
- 页数: N/A
- 创建时间: N/A

## 文档内容
# Efficient Contextformer: Spatio-Channel Window Attention for Fast Context Modeling in Learned Image Compression

# A. Burakhan Koyuncu, Panqi Jia, Atanas Boev, Elena Alshina, and Eckehard Steinbach, Fellow, IEEE

# Abstract

Entropy estimation is essential for the performance of learned image compression. It has been demonstrated that a transformer-based entropy model is of critical importance for achieving a high compression ratio, however, at the expense of a significant computational effort. In this work, we introduce the Efficient Contextformer (eContextformer) – a computationally efficient transformer-based autoregressive context model for learned image compression. The eContextformer efficiently fuses the patch-wise, checkered, and channel-wise grouping techniques for parallel context modeling, and introduces a shifted window spatio-channel attention mechanism. We explore better training strategies and architectural designs and introduce additional complexity optimizations. During decoding, the proposed optimization techniques dynamically scale the attention span and cache the previous attention computations, drastically reducing the model and runtime complexity. Compared to the non-parallel approach, our proposal has ∼145x lower model complexity and ∼210x faster decoding speed, and achieves higher average bit savings on Kodak, CLIC2020, and Tecnick datasets. Additionally, the low complexity of our context model enables online rate-distortion algorithms, which further improve the compression performance. We achieve up to 17% bitrate savings over the intra coding of Versatile Video Coding (VVC) Test Model (VTM) 16.2 and surpass various learning-based compression models. The performance of the backward adaptation is a major factor in the efficiency of the LIC framework, and methods to improve its performance have been actively investigated. Various context model architectures have been proposed – 2D masked convolutions using local dependencies; channel-wise autoregressive mechanisms exploiting channel dependencies; non-local simplified attention-based models capturing long-range spatial dependencies; sophisticated transformer-based models leveraging a content-adaptive context modeling.

# Index Terms

Learned image compression, efficient context modeling, transformers.

# I. INTRODUCTION

ONLINE media consumption generates an ever-growing demand for higher-quality and lower-bitrate content and learned image compression (LIC) algorithms. Such demand drives advancements in both classical and learned image compression algorithms. Using an autoregressive model requires recursive access to previously processed data, which results in slow decoding time and inefficient utilization of NPU/GPU. To remedy this, some researchers proposed to optimize the decoding process by using wavefront parallel processing (WPP) inspired by a similar approach used in classical codecs. While using WPP significantly increases parallelism, it still requires a large number of autoregressive steps for processing large images. A more efficient approach is to split the latent elements into groups and code each group separately. The latent elements might be split into, e.g., spatial patches, channel segments, or using a checkered pattern. A combination of the channel-wise and the checkered grouping has also been studied. Such parallelization approaches can reduce decoding time by 2-3 orders of magnitude at the cost of higher model complexity.

Manuscript received 30 June 2023; revised 3 November 2023 and 18 December 2023; accepted 21 February 2024. Date of publication 29 February 2024; date of current version 12 August 2024. This article was recommended by Associate Editor L. Yu. (Corresponding author: A. Burakhan Koyuncu.)

A. Burakhan Koyuncu is with the School of Computation, Information and Technology, Department of Computer Engineering, Chair of Media Technology, Technical University of Munich, 80333 Munich, Germany, and also with the Huawei Munich Research Center, 80992 Munich, Germany (e-mail: burakhan.koyuncu@tum.de).

Panqi Jia is with the Department of Electrical-Electronic-Communication Engineering, Chair of Multimedia Communications and Signal Processing, Friedrich-Alexander University, 91058 Erlangen, Germany, and also with the Huawei Munich Research Center, 80992 Munich, Germany.

Atanas Boev and Elena Alshina are with the Huawei Munich Research Center, 80992 Munich, Germany.

Eckehard Steinbach is with the School of Computation, Information and Technology, Department of Computer Engineering, Chair of Media Technology, Technical University of Munich, 80333 Munich, Germany.

Color versions of one or more figures in this article are available at https://doi.org/10.1109/TCSVT.2024.3371686.

Digital Object Identifier 10.1109/TCSVT.2024.3371686

© 2024 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/.


# KOYUNCU et al.: EFFICIENT CONTEXTFORMER: SPATIO-CHANNEL WINDOW ATTENTION

and up to a 3% performance drop. For instance, the patch-wise attention module, a point-wise multi-layer perceptron (MLP) model [14] uses a recurrent neural network (RNN) to share information between patches. Other works [16], [22], [26] use a channel-wise model and implement additional convolutional layers to combine decoded channel segments. Our previous work proposed a transformer-based entropy model with spatio-channel attention (Contextformer) [24]. In the same work, we presented a framework using our entropy model, which outperforms contemporary LIC frameworks [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [16], [17], [23]. However, the model complexity of Contextformer makes it unsuitable for real-time operation. In this work, we propose a fast and low-complexity version of the Contextformer, which we call Efficient Contextformer (eContextformer). We extend the window attention of [29] to spatio-channel window attention in order to achieve a high-performance and low-complexity context model. Additionally, we use the checkered grouping to increase parallelization and reduce autoregressive steps in context modeling. By exploiting the properties of eContextformer, we also propose algorithmic optimizations to reduce complexity and runtime even further.

In terms of PSNR, our model surpasses VTM 16.2 [30] (intra)1 on Kodak [31], CLIC2020 [32] (Professional and Mobile), and Tecnick [33] datasets, providing average bitrate savings of 10.9%, 12.3%, 6.9%, and 13.2%, respectively. Our optimized implementation requires 145x less multiply-accumulate (MAC) operations and takes 210x less decoding time than the earlier Contextformer. It also contains significantly fewer model parameters compared to other channel-wise autoregressive and transformer-based prior-art context models [16], [17], [26]. Furthermore, due to its manageable complexity, eContextformer can be used with an online rate-distortion optimization (oRDO) similar to the one in [34]. With oRDO enabled, the performance of our model reaches up to 17% average bitrate saving over VTM 16.2 [30].

In the next section we overview of the related work and introduce the necessary terminology. In Section III, we introduce our proposal for an efficient and high-performance transformer-based context model. In Section IV, we describe the experiments and the experimental results, and in Section V, we present the conclusions.

# II. RELATED WORK

# A. Transformers in Computer Vision

The underlying principle of the transformer network [35] can be summarized as a learned projection of sequential vector embeddings x ∈ RS×de into sub-representations of query Q ∈ RS×de, key K ∈ RS×de and value V ∈ RS×de, where S and de denote the sequence length and the embedding size, respectively. Subsequently, a scaled dot-product calculates the attention, which weighs the interaction between the query and key-value pairs. The Q, K and V are split into h groups, known as heads, which enables parallel computation and greater attention granularity. The separate attention of each head is finally combined by a learned projection W. After the attention is computed, the output is processed to produce the final representation.

The entropy of ˆ is minimized by learning its probability distribution. Entropy modeling is built using two methods, backward and forward adaptation [27]. The forward adaption uses a hyperprior estimator, i.e., a second autoencoder with its own analysis transform ha and synthesis transform hs. The hyperprior creates a side, separately encoded channel ˆ. A factorized density model [3] learns local histograms to estimate the probability mass pˆz(ˆ|ψ f) with model parameters ψ f. The backward adaptation uses a context model gcm, which estimates the entropy of the current latent element ˆiy using the previously coded elements ˆ

# IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 8, AUGUST 2024

# Fig. 1

Illustration of the context modeling process, where the symbol probability of the current latent variable estimated by aggregating the information of the latent variables. The previously decoded latent elements not joining to context modeling and yet to be coded elements are depicted as and , respectively. The illustrated context models are (a) the model with 2D masked convolutions [5], [11], (b) the model with 3D masked convolutions [6], [10], (c) channel-wise autoregressive model [16], and (d–e) Contextformer with sfo and cfo coding mode [24], respectively.

# Fig. 2

Illustration of different parallelization techniques for the context modeling in (a) patch-wise grouping [14], (b) checkered grouping [15], (c) channel-wise grouping [16], and (d-e) combination of checkered and channel-wise grouping with sfo and cfo coding, respectively. All latent elements within the same group (depicted with the same color) are coded simultaneously, while the context model aggregates the information from the previously coded groups. For instance, [14], [15] use 2D masked convolutions in the context model, and [16] applies multiple CNNs to channel-wise concatenated groups. The context model of [22] combines the techniques of [15] and [16] and can be illustrated as in (d). Our proposed model (eContextformer), as well as the experimental model (pContextformer), use the parallelization techniques depicted in (d–e). However, our models employ spatio-channel attention in context modeling and do not require additional networks for channel-wise concatenation.

Fashion. The entropy parameters network ge uses the outputs spatially local relations in the latent space. In order to capture a variety of spatial dependencies, [7], [8] implemented a conditional distribution p ˆy( ˆ|ˆ). The lossy LIC framework can be formulated as:

yˆ = Q(ga (x; φa)),

xˆ = gs ( ˆ; φs)),y

z ˆ = Q(ha ( ˆ; θa)),y

p ˆi ( ˆi y y|ˆ) ← ge(gc( ˆ

# KOYUNCU et al.: EFFICIENT CONTEXTFORMER: SPATIO-CHANNEL WINDOW ATTENTION

non-overlapping patches. Inside each patch, a serial context model is applied while the inter-patch relations are aggregated with an RNN. The authors of [15] proposed grouping of the latent variables according to a checkered pattern. They use a 2D masked convolution with a checkered pattern in context modeling. Both [14] and [15] outperform their serial baseline in high bitrates but are inferior to them in low bitrates. The model in [16] is another example of a parallel context model, which uses channel-wise grouping. He et al. [22] combined checkered grouping with channel-wise context modeling. Liu et al. [26] improved the architecture in [22] with transformer-based transforms. Both models [22], [26] reached significantly high rate-distortion performance. Wang et al. [45] proposed a checkered grouping with a pattern for the channel dimension in addition to the spatial ones. The model approximates channel-wise grouping with a significantly lower model complexity but it reaches a lower rate-distortion performance.

# III. EFFICIENT CONTEXTFORMER

# A. Contextformer

In a previous work, we proposed a high-performance transformer-based context model (a.k.a Contextformer) [24]. Compared to the prior transformer-based context modeling method [17], Contextformer introduces additional autoregression for the channel dimension by splitting the latent tensor into several channel segments Ncs. This enables adaptively exploiting both spatial and channel-wise dependencies in the latent space with a spatio-channel attention mechanism. Unlike [16], each segment is processed by the same model without requiring additional complex architectures for aggregating channel segments. Contextformer architecture is based on ViT [39], but the patch-wise global attention mechanism is replaced with a sliding window attention similar to the one used in [41]. The receptive field traverses the spatial dimensions and grows in the channel dimension. The proposed attention mechanism has two different coding modes – spatial-first-order (sfo) and channel-first-order (cfo). The coding modes differ in the order of the autoregressive steps within the sliding window, resulting in a prioritization of the correlations in the primarily processed dimension (see Figs. 2d and 2e).

# D. Online Rate-Distortion Optimization

Training an LIC encoder on a large dataset results in a globally optimal but locally sub-optimal network parameters (φ, θ, ψ). For a given input image x, a more optimal set of parameters might exist. This problem is known as the amortization gap [34], [48], [49]. Producing optimal network parameters for each x is not feasible during encoding due to the signaling overhead and computational complexity involved. In [34] and [48], the authors proposed an online rate-distortion optimization (oRDO), which adapts y and z during encoding. The oRDO algorithms first initialize the latent variables by passing the input image x through the encoder without tracking the gradients of the transforms. Then, the network parameters are kept frozen, and the latent variables are iteratively optimized using (3a). Finally, the optimized variables ( ˆopt , ˆopt ) are encoded into the bitstream. oRDO does not influence the complexity or the runtime of the decoder, but it significantly increases the encoder complexity. Therefore, oRDO is suitable for LIC frameworks with low complexity entropy models [4], [5], [11].

# B. Exploring Parallelization Techniques

We analyzed the effect of state-of-the-art parallelization techniques on the Contextformer. We observed a few issues that can be improved: (1) overly large kernel size of the attention mechanism; (2) a discrepancy between train and test time behavior; (3) a large number of autoregressive steps. For an input image ˆ ∈ RH ×W ×3 with a latent representation H × 16 W × C (with height H, width W and the number of channels C), the complexity C of the proposed attention mechanism expressed in number of MAC operations is:

Cattn = Nwin (2N2de + 4N de2)Nwin = H W Ncs, 256s2N = K2 Ncs

# IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 8, AUGUST 2024

where the spatial kernel size K and the number of channel segments Ncs defines the sequence length N inside a spatio-channel window. The window operation is performed Nwin times with a stride of s=1 over ˆ.y

The Contextformer is trained with a large spatial kernel (K =16) on 256×256 image crops. Therefore, the model learns a global attention mechanism. During inference, it uses sliding window attention to reduce computational complexity. Compared to global attention used in [12], the sliding window attention provides lower latency. However, the Contextformer does not learn the sliding behavior, which creates a discrepancy between training and inference results. This problem could be fixed by training with sliding window attention and either increasing the image crops and keeping the kernel size the same, or by decreasing the kernel size and keeping the crop size intact. Training with larger crop size does not affect the complexity and still increases the compression performance. In this case, direct comparison with prior art trained on small crops would be unfair and would make the effects of parallelization less noticeable. Training with a smaller kernel provides biquadratic complexity reduction for each window. Moreover, the Contextformer requires Nwin autoregressive steps, which cannot be efficiently implemented on a GPU/NPU. Based on those observations, we built a set of parallel Contextformers (pContextformer), which fuse the patch-wise context [14] and checkerboard context model [15] and need eight autoregressive steps for Ncs =4.

We trained the set of models with 256×256 image crops for ∼65% of the number of iterations used for training in [24]. We varied the coding method (sfo or cfo), the kernel size K, and the stride during training st and inference si. The stride defines whether the windows are processed in overlapping or separately. To achieve the overlapping window process, we set the stride to 2 K. For K =16, the model could learn only global attention, which could be replaced with a local one during the inference. Following the methodology of JPEG-AI [50], we measured the performance of the codec in Bjøntegaard Delta rate (BD-Rate) [51] over VTM 16.2 [30], and the complexity of each context model in kilo MAC operations per pixel (kMAC/px). For simplicity, we measured the complexity of a single pass on the encoder side, where the whole latent variable ˆ is processed at once with the given causality mask.y We present the results in Fig. 3 and Tables I and II) and make the following observations:

1. When Trained With Global Attention (s=K =16): the spatial-first coding is better than channel-first coding at high bitrates but worse at low bitrates. At low bitrates, the spatial-first coding cannot efficiently exploit spatial dependencies due to the usage of non-overlapping windows on the sparse latent space. Also, spatial-first coding benefits more from the overlapping window attention applied at inference time (s<K).
2. Trained With Overlapping Attention Windows (st <K): the spatial-first coding outperforms channel-first coding. Moreover, using overlapping windows at inference time helps the models with a small kernel size (K =8) to reach performance close to the ones with a larger kernel (K =16).

# TABLE I

RATE SAVINGS OVER VTM 16.2 [30] (INTRA) AND COMPLEXITY OF VARIOUS PCONTEXTFORMERS WITH K =16 COMPARED TO [24], SHOWING THE EFFECT OF CODING MODE AND USING OVERLAPPED WINDOWS DURING INFERENCE

# TABLE II

PERFORMANCE AND COMPLEXITY OF VARIOUS PCONTEXTFORMERS W.R.T. KERNEL SIZE, CODING MODE, AND USING OVERLAPPED WINDOWS DURING INFERENCE AND TRAINING

In General: pContextformer models can provide more than 100x complexity reduction for a 3% performance drop. Theoretically, more efficient models are possible with using overlapping window attention at training and inference time. However, the simple overlapping window attention is still sub-optimal since it increases the complexity four-fold.

# C. Proposed Model

Based on previous experiments we propose the Efficient Contextformer (eContextformer) as an improved version of our previous architecture [24]. It uses the same compression framework as [8] and [24]. The analysis transform ga comprises four 3×3 convolutional layers with a stride of 2, GDN activation function [52], and a single residual attention module without the non-local mechanism [8], [11], [53], same as in our previous work. The synthesis transform gs closely resembles ga and implements deconvolutional layers with inverse GDN (iGDN). In order to expand receptive fields and reduce quantization error, gs utilizes residual blocks [6] in its first layer, and an attention module. The entropy model integrates a hyperprior network, universal quantization, and estimates p ˆyi ( ˆi |ˆ) with y z a Gaussian Mixture Model (GMM) [11] with km =3 mixtures. In contrast to [24], we use 5×5 convolutions in lieu of the 3×3 ones in the hyperprior in order to align our architecture with the recent studies [16], [22].

Our experiments show that the Contextformer [24] can support the checkered and patch-wise parallelization techniques discussed in Section III-A but require a more efficient.

In [24], we misclassified the attention module as being a non-local one. It uses is the same architecture as in [53], but without the non-local module. A similar architecture is proposed in [11].

# KOYUNCU et al.: EFFICIENT CONTEXTFORMER: SPATIO-CHANNEL WINDOW ATTENTION

Fig. 4. Illustration of our compression framework utilizing with the eContextformer with window and shifted-window spatio-channel attention. The segment generator splits the latent into Ncs channel segments for further processing. Following our previous work [24], the output of hyperdecoder is not segmented but repeated along channel dimension to include more channel-wise local neighbors for the entropy modeling.

To achieve implementation for overlapping window attention, we replaced the ViT-based transformer in Contextformer with a Swin-based one and added spatio-channel attention. Fig. 4 describes the overall architecture of our compression framework. For context modeling, the segment generator rearranges the latent *ŷ ∈ R16H×16W×pcs into a spatio-channel segmented representation ˆ ∈ RC. A linear layer converts ˆ to t cs × 16 tensor embedding with the last dimension of d*e. The embedding representation passes through L Swin transformer layers where window and shifted-window spatio-channel attention (W-SCA and SW-SCA) are applied in an alternating fashion.

Window attention is computed on the intermediate outputs with RNw × (NcsK2) × pcs, where the first and second dimension represents the number of windows Nw = H W2 processed in 256K parallel and the sequential data within a window of K × K, respectively. We masked each window according to the group coding order (see Fig. 2) to ensure coding causality, where each group uses all the previously coded groups for the context modeling. Following [29], we replaced absolute positional encodings with relative ones to introduce more efficient permutation variance. We employ multi-head attention with *h* heads to yield better parallelization and build independent relations between different parts of the spatio-channel segments.

More detailed description of our compression framework with the eContextformer is provided in Table IX.

# Table III

# IMPROVEMENTS OF PROPOSED METHOD OVER THE CONTEXFORMER [24]

|Proposed Method|Proposed Method|Improvements|
|---|---|
|eContextformer|W-SCA|Improved efficiency|
|SW-SCA|Reduced complexity|Better parallelization|

For instance, in the spatial-first-order (sfo) setting, two autoregressive steps are required per iteration, while the computation of half of the attention map is unnecessary for each first step. To remedy this, we rearranged the latent tensors into a form *ˆ ∈ R2Ncs × 16 r* H×32W×pcs that is more suitable for efficient processing. We set the window size to K × 2 to preserve the spatial attention span, and applied the window and shifted attention as illustrated in Fig. 5.

We refer to the proposed rearrangement operation as Efficient coding Group Rearrangement (EGR). Furthermore, we code the first group *ˆ1 only using the hyperprior instead of using a start token for context modeling of ˆ*1. This reduced the total autoregressive steps for using the context model to 2Ncs − 1.

# IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 8, AUGUST 2024

Fig. 5. Illustration of the optimized processing steps of eContextformer. From left to right, the latent tensor (a) is first split into channel segments (b) and reordered according to group coding order (c). Finally, the transformer layers with window and shifted-window spatio-channel attention (d-e) are applied on the reordered tensor, sequentially.

i.e., a total complexity reduction of 13% for Ncs =4. We refer to this optimization as Skipping First coding Group (SFG).

Note that transformers are sequence-to-sequence models and compute attention between every query and key-value pair. Let Q(1 ≤ n), K (1 ≤ n), and V (1 ≤ n) be all queries, keys, and values, and Attn( Q(1 ≤ n), K (1 ≤ n), V (1 ≤ n)) be the attention computed up to coding step n. Since context modeling is an autoregressive task, during decoding, one can omit to compute the attention between the previously coded queries and key-value pairs and compute simply the attention Attn( Q(n), K (1 ≤ n), V (1 ≤ n)) between the current query and the cached key-value pairs. For efficient implementation, we also adopted the key-value caching according to the white paper [54] and the published work [55].

The combined implementation of the EGR and key-value caching reveals an emergent property of the proposed attention mechanism. At its foundation, the attention mechanism resembles the shifted window attention of [29] for each coding step; ergo, ˆi r. However, the attention span expands with each coding step in spatial and channel dimensions. Therefore, our optimized attention mechanism can also be called as an expanding (S)W-SCA (see Table III). Additionally, the caching efficiently reduces the complexity from O(N2) to O(N) for the growing attention span.

# IV. EXPERIMENTS

# A. Experimental Setup

# 1) Training:

We configured eContextformer using the following parameters {N =192, M=192, Ncs =4, K =8, L=8, de=8M, dmlp=4de}. Here, N, M, Ncs, and de stand for the intermediate layer size of the encoder and decoder, the bottleneck size, the number of channel segments, and the embedding size. L denotes the total number of transformer layers, where half utilize W-SCA and the other half has SW-SCA. Those parameters were selected to be identical to the ones used in Contextformer [24], in order to ensure a fair comparison. Since our initial experiments showed deteriorating performance for the c f o coding method for window attention, we continued with the s f o version. K defines the size of the attention span in the spatial dimension, which results in K × 2K window for the proposed optimization algorithms (see Section III-D). Following [24], [56], we trained eContextformer for 120 epochs (∼1.2M iterations) on 256×256 random image crops with a batch size of 16 from the Vimeo-90K dataset [57], and used ADAM optimizer [58] with the initial learning rate of 10−4. In order to cover a range of bitrates, we trained various models with λ ∈ {0.002, 0.004, 0.007, 0.014, 0.026, 0.034, 0.058} with mean-squared-error (MSE) as the distortion metric D(·). To evaluate the perceptual quality of our models, we finetuned them with MS-SSIM [59] as the distortion metric for ∼500K iterations. For high bitrate models (λ5,6,7), we increased the bottleneck size to 312 according to the common practice [5], [11], [16]. Empirically, we observed better results with a lower number of heads for the highest rate model, so we used 6 heads for this model. Additionally, we investigated the effect of the training dataset and crop size by finetuning the models with 256×256 and 384×384 image crops from the COCO 2017 dataset [60] for ∼600K iterations.

# 2) Evaluation:

We analyzed the performance of the eContextformer on the Kodak image dataset [31], CLIC2020 [32] (Professional and Mobile) test dataset, and Tecnick [33] dataset. We compared its performance to various serial and parallel context models: the 2D context models (Minnen et al. [5], Cheng et al. [11] and Fu et al. [46]), the multi-scale 2D context model (Cui et al. [8]), the 3D context models (Chen et al. [6] and Tang et al. [44]), the context model with a simplified attention (Guo et al. [23]), the transformer-based context model with spatio-channel attention (Contextformer [24]), the channel-wise autoregressive context model (Minnen and Singh [16]), the checkerboard context model (He et al. [15]), transformer-based checkerboard context model (Qian et al. [17]), the channel-wise and checkerboard context model (He et al. [22]), the transformer-based channel-wise and checkerboard context model (Liu et al. [26]). We tested the small (S) and large (L) models of Liu et al. [26], which differ in the number of intermediate layer channels. Additionally, we experimented with the

# KOYUNCU et al.: EFFICIENT CONTEXTFORMER: SPATIO-CHANNEL WINDOW ATTENTION

Fig. 6. The rate-distortion performance in terms of (a) PSNR and (b) MS-SSIM on Kodak dataset [31] showing the performance of our model compared to various learning-based and classical codecs. We also include the performance of our model combined with oRDO.

Fig. 7. Experimental study of the effects of different training datasets and crop sizes on the performance, showing the rate-distortion performance on (a) Kodak [31] and (b) Tecnick [33] datasets.

low-complexity asymmetric frameworks of Wang et al. [45], Fu et al. [46] and Yang and Mandt [47]. We used the model (LL) from [45] with a large synthesis and large analysis transforms since it has a higher rate-distortion performance. We also included the results of the framework from [47] using an oRDO called Stochastic Gumbel Annealing (SGA) [48] that applies iterative optimization for 3000 steps. If the source was present, we executed the inference algorithms of all those methods; otherwise, we obtained the results from relevant publications. We also used classical image coding frameworks such as BPG [20] and VTM 16.2 [30] (intra) for comparison.

Moreover, we measured the number of parameters of entropy models with the summary functions of PyTorch [61] or Tensorflow [62] (depending on the published implementation). By following the recent standardization activity JPEG-AI [50], we computed the model complexity in kMAC/px with the ptflops package [63]. In case of missing hooks for attention calculation, we integrated them with the help of the official code repository of the Swin transformer [29]. For the runtime measurements, we used DeepSpeed [64] and excluded arithmetic coding time for a fair comparison since each framework uses a different implementation of the arithmetic codec. All the tests, including the runtime measurements, are done on a machine with a single NVIDIA Titan RTX and Intel Core i9-10980XE. We used PyTorch 1.10.2 [61] with CUDA Toolkit 11.4 [65].

# B. Model Performance

Fig. 6 shows the rate-distortion performance of the eContextformer trained on the Vimeo-90K with 256×256 image crops. In terms of PSNR, our model qualitatively outperforms the VTM 16.2 [30] for all rate points under test and achieves 5.3% bitrate savings compared to it (see Fig. 6a). Our compression framework shares the same analysis, synthesis transforms, and hyperprior with the model with multi-scale 2D context model [8] and Contextformer [24]. Compared to [8], our model saves 8.5% more bits, while it provides 1.7% lower performance than Contextformer due to the parallelization of context modeling. The eContextformer achieves competitive performance to parallelized models employing channel-wise autoregression with an increased number of model parameters, such as Minnen and Singh [16] and He et al. [22]. When comparing with VTM 16.2 [30], the former model gives 1.6% loss in BD-Rate performance, and the latter gives 6.7% gain.

In Fig. 6b, we also evaluated the perceptually optimized models compared to the prior art. In terms of MS-SSIM [59], the eContextformer saves on average 48.3% bitrate compared to VTM 16.2 [30], which is performance-wise 0.8% better than He et al. [22] and 1.1% worse than the Contextformer [24].

# C. Effect of Training With Larger Image Crops

Although eContextformer yields a performance close to Contextformer on the Kodak dataset [31], it underperforms on larger resolution test datasets such as Tecnick [33] (see Fig. 7). The Vimeo-90K dataset [57] we initially used for our training has a resolution of 448 × 256. In order to avoid a bias towards horizontal images, we used 256 × 256 image crops, ergo 16 × 16 latent variable, for training similar to the state-of-the-art [56]. However, the attention window size of K =8 combined with low latent resolution limits learning an

# IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 8, AUGUST 2024

Fig. 8. The rate-distortion performance in terms of PSNR on (a) Kodak [31], (b) CLIC2020 [32] (Professional and Mobile), and (c) Tecnick [33] datasets, showing the performance of our model w/ and w/o finetuning compared to various learning-based and classical codecs. We also include the performance of our models combined with oRDO.

sizes achieve similar performance on the Kodak dataset [31], which has about 0.4M pixels. On the contrary, larger crop sizes help the models to reach more than two times in performance on the Tecnick dataset [33], which has ∼1.5M pixels per image.Fig. 8 and Table IV show the rate-distortion performance of the finetuned eContextformer on Kodak [31], CLIC2020 [32] (Professional and Mobile), and Tecnick [33] datasets. Our finetuned models reach a compression performance on par with the state-of-the-art LIC frameworks, providing over 5% bitrate saving over the initial training and achieving average savings of 10.9%, 12.3%, 6.9%, and 13.2% over VTM 16.2 [30] on those datasets, respectively.

efficient context model. In order to achieve high rate-distortion gain, the recent studies [14], [16], [17], [22], [46] experimented with higher resolution datasets, such as ImageNet [66], COCO 2017 [60], and DIV2K [67], and larger crop sizes up to 512×512. Following those studies, we finetuned eContextformer with 256×256 and 384×384 image crops from COCO 2017 [60]. The models finetuned with different crop.

# D. Model and Runtime Complexity

During encoding and decoding, each of the EGR and SFG methods decreases the complexity of context modeling by 10-13%, whereas the combination of them with the caching of key-value pairs provides an 84% complexity reduction in total.We also compared the efficiency of caching to the single pass, where the whole latent variable ˆ is processed on the encodery side at once with the given causality mask. The caching is also

# KOYUNCU et al.: EFFICIENT CONTEXTFORMER: SPATIO-CHANNEL WINDOW ATTENTION

# TABLE VI

# NUMBER OF PARAMETERS AND ENTROPY MODEL COMPLEXITY OF OUR MODEL COMPARED TO VARIOUS LEARNING-BASED CODECS

# TABLE VII

# ENCODING AND DECODING TIME OF OUR MODEL COMPARED TO VARIOUS LEARNING-BASED AND CLASSICAL CODECS

and a significantly lower total number of parameters in the entropy model compared to channel-wise autoregressive models [16].

Table VII presents the encoding and decoding runtime complexity of our model, some of the prior learning-based models and VTM 16.2 [30]. The proposed optimizations speed up the encoding and decoding up to 3x, proving a 210x improvement over the optimized version of the Contextformer [24]. Furthermore, we observed that coding time scales better for the optimized eContextformer compared to the one without the proposed optimizations. The 4K images have 21x more pixels than the Kodak images, while the relative encoding and decoding time of the optimized models for those images increase only 14x w.r.t. the ones on the Kodak dataset. Moreover, our optimized model provides competitive runtime performance to the Minnen&Singh [16] and both small and large models of Liu et al. [26] (S/L).

We also observed that the memory usage in [17] increases significantly with the image resolution due to the expensive global attention mechanism in their context model. Similarly, in [26], the residual transformer-based layers in synthesis and analysis transforms, and the context model combined with the channel-wise grouping put a heavy load on memory. Therefore, we could not test Liu et al. [26] (L) and Qian et al. [17] on 4K images on our GPU with 24 GB memory.

# E. Online Rate-Distortion Optimization

Since using eContextformer in our framework results in a significantly lower encoder complexity, we could afford to incorporate an oRDO technique, which further improves the compression efficiency. We took the oRDO algorithm from [34], replaced the noisy quantization with the straight-through estimator described in [68], and used an exponential decay for scheduling the learning rate, which results in a simplified oRDO with faster convergence. The learning rate at n-th iteration can be defined in closed-form as:

αn = α0γn,

where α0 and γ are the initial learning rate and the decay rate, respectively. Fig. 9a illustrates the immediate learning rate αn w.r.t. oRDO steps for different combinations of (α0, γ).

more efficient than the single pass since only the required parts of the attention map are calculated. We obtained the optimal parameters by a search using Tree-Structured Parzen Estimator (TPE) [69] with the objective function (3a) and the constraints of αn >10−7, α0 ∈ [0.02, 0.08], and γ ∈ [0.5, 0.7]. To omit over-fitting, we used 20 full-resolution images randomly sampled from the COCO 2017 dataset [60] for the search. Figs. 9b and 9c show the results of TPE [69] for models trained with λ1 and λ4, respectively. We observed that the higher bitrate models (λ>3) generally perform better with a higher initial learning rate compared to the ones trained for lower bitrates (λ<4). This suggests that the optimization of less sparse latent space requires a larger step size at each iteration. We set α0=0.062 and γ =0.72 for λ<4 and α0=0.062 and γ =0.72 for λ>3, which results in 26 iteration steps for all models.

# IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 8, AUGUST 2024

# Fig. 9

Illustration of (a) the learning rate decay used for the oRDO w.r.t. optimization iteration for different initial learning rate α0 and decay rate γ. The results of the TPE for our model (b) with λ1 and (c) with λ4 for different combinations of (α0, γ). The size of each square symbolizes the required number of oRDO iteration steps.

# TABLE VIII

# ABLATION STUDY FOR ECONTEXTFORMER ON KODAK DATASET [31]

Figs. 6 and 8 show the rate-distortion performance of the eContextformer with oRDO using the optimal parameter setting. Compared to VTM 16.2 [30], the oRDO increases the rate-distortion performance of the finetuned models up to 4%, providing 14.7%, 14.1%, 10.4%, and 17.0% average bitrate savings on Kodak [31], CLIC2020 [32] (Professional and Mobile), and Tecnick [33] datasets, respectively. The encoding runtime complexity is proportional to the number of optimization steps used. For the selected number of steps, the total encoding runtime complexity of our model is still lower than the VTM 16.2 [30]. Furthermore, we observed that the oRDO increases the performance of our initial models (without the finetuning) by up to 7%, which indicates those models are trained sub-optimally.

# F. Ablation Study

Table VIII summarizes the ablation study we conducted with Contextformer [24] and eContextformer on the Kodak dataset [31]. Notably, the proposed architecture shares the same synthesis and analysis transforms with [8] and [24] and differs in the entropy modeling part. We observed that increasing the number of mixtures km of GMM [11] provides 1.6-1.9% performance gain for the Contextformer [24] and eContextformer, while the context model with spatio-channel attention increases the performance by more than 6%. Compared to the straight-forward parallelization (pContextformer in Section III-B), the combination of proposed (S)W-SCA and complexity optimization techniques allows eContextformer to reach a better performance-complexity trade-off. The finetuning with larger resolution images is helpful for both eContextformer and Contextformer [24]. However, we observed that the sliding window attention is less stable while training with a higher resolution images, which might explain the lower performance gain for the Contextformer [24]. Moreover, finetuning reduces the discrepancy between the training and testing on different resolution images. Therefore, the performance contribution of oRDO to the finetuned model is lower than the one without the finetuning.

Notably, the oRDO does not impact the entropy model complexity of the decoder since it is only applied during encoding. Each oRDO iteration requires one forward and one backward pass through the entropy model, where the backward pass approximately has two times more computations [64], [70]. Therefore, the encoder-side entropy model complexity with n oRDO iterations can be estimated as 3n times the encoder-side complexity without the oRDO. Moreover, for all models, the decoder-side entropy model complexity is slightly lower (∼9 kMAC/px) since the hyperprior’s analysis transform (ha) is not used during decoding.

# G. Visual Quality

Fig. 10 allows for a visual quality comparison between our models with oRDO applied, and VTM 16.2 [30]. The figure shows enlarged crops from two images from the Kodak dataset – kodim07 and kodim23. We compressed the images using each algorithm and targeting the same bitrate. In the figure, VTM 16.2 [30] exhibit noticeable artifacts such as smear and aliasing. On the other hand, our models offer superior visual quality compared to VTM 16.2 [30], and are better at preserving contours and high-frequency details. According to our experience, the MSE-optimized model excels in producing.

# KOYUNCU et al.: EFFICIENT CONTEXTFORMER: SPATIO-CHANNEL WINDOW ATTENTION

Fig. 10. Visual comparison of reconstructed images kodim23 (left) and kodim07 (right) from the Kodak image dataset. The images are compressed for the target bpp of 0.07 and 0.21, respectively. Both of our models, optimized for MSE and MS-SSIM, produce visually and objectively higher quality images for a lower bit-stream size than VTM 16.2 [30].

# TABLE IX

THE ARCHITECTURE OF THE PROPOSED MODEL USING THE ECONTEXTFORMER. EACH ROW DESCRIBES ONE LAYER OR COMPONENT OF THE MODEL. “CONV: Kc2. SIMILARLY, “DECONV” STANDS FOR TRANSPOSED CONVOLUTIONS. “DENSE” LAYERS ARE SPECIFIED BY THEIR OUTPUT “S” OF ×Kc×N S2” IS A CONVOLUTIONAL LAYER WITH KERNEL SIZE OF Kc×Kc, NUMBER OF N OUTPUT CHANNELS WITH A STRIDE DIMENSION, WHEREAS D1 = 2M + de, AND D2 = (4km M)/Ncs

sharper edges, while the MS-SSIM optimized one is better at preserving texture grain.

# V. CONCLUSION

This work introduces eContextformer – an efficient and fast upgrade to the Contextformer. We conduct extensive experimentation to reach a fast and low-complexity context model while presenting state-of-the-art results. Notably, the algorithmic optimizations we provide further reduce the complexity by 84%. Aiming to close the amortization gap, we also experimented with an encoder-side iterative algorithm.

It further improves the rate-distortion performance and still has lower complexity than the state-of-art video compression standard. Undoubtedly, there are more advanced compression algorithms yet to be discovered which employ better non-linear transforms and provide more energy-compacted latent space.

This work focuses on providing an efficient context model architecture, and defer such an improved transforms to future work.

# REFERENCES

1. S. Ma, X. Zhang, C. Jia, Z. Zhao, S. Wang, and S. Wang, “Image and video compression with neural networks: A review,” IEEE Trans. Circuits Syst. Video Technol., vol. 30, no. 6, pp. 1683–1698, Apr. 2019.
2. R. Birman, Y. Segal, and O. Hadar, “Overview of research in the field of video compression using deep neural networks,” Multimedia Tools Appl., vol. 79, nos. 17–18, pp. 11699–11722, May 2020.
3. J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-end optimized image compression,” in Proc. 5th Int. Conf. Learn. Represent. ICLR, 2017, pp. 1–27.
4. J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” in Proc. Int. Conf. Learn. Represent., 2018, pp. 1–47.
5. D. Minnen, J. Ballé, and G. Toderici, “Joint autoregressive and hierarchical priors for learned image compression,” in Proc. NeurIPS, 2018, pp. 1–10.
6. T. Chen, H. Liu, Z. Ma, Q. Shen, X. Cao, and Y. Wang, “End-to-end learnt image compression via non-local attention optimization and improved context modeling,” IEEE Trans. Image Process., vol. 30, pp. 3179–3191, 2021.
7. J. Zhou, S. Wen, A. Nakagawa, K. Kazui, and Z. Tan, “Multi-scale and context-adaptive entropy model for image compression,” 2019, arXiv:1910.07844.
8. Z. Cui, J. Wang, S. Gao, T. Guo, Y. Feng, and B. Bai, “Asymmetric gained deep image compression with continuous rate adaptation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 10532–10541.
9. J. Lee, S. Cho, and S.-K. Beack, “Context-adaptive entropy model for end-to-end optimized image compression,” in Proc. Int. Conf. Learn. Represent., 2019. [Online]. Available: https://openreview.net/forum?id=HyxKIiAqYQ
10. F. Mentzer, E. Agustsson, M. Tschannen, R. Timofte, and L. V. Gool, “Conditional probability models for deep image compression,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Jun. 2018, pp. 4394–4402.
11. Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized Gaussian mixture likelihoods and attention modules,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 7939–7948.
12. Y. Qian et al., “Learning accurate entropy model with global reference for image compression,” in Proc. Int. Conf. Learn. Represent., 2021. [Online]. Available: https://openreview.net/forum?id=cTbIjyrUVwJ
13. M. Li, K. Ma, J. You, D. Zhang, and W. Zuo, “Efficient and effective context-based convolutional entropy modeling for image compression,” IEEE Trans. Image Process., vol. 29, pp. 5900–5911, 2020.
14. A. B. Koyuncu, K. Cui, A. Boev, and E. Steinbach, “Parallelized context modeling for faster image coding,” in Proc. Int. Conf. Vis. Commun. Image Process. (VCIP), Dec. 2021, pp. 1–5.
15. D. He, Y. Zheng, B. Sun, Y. Wang, and H. Qin, “Checkerboard context model for efficient learned image compression,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2021, pp. 14771–14780.

# IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 34, NO. 8, AUGUST 2024

1. D. Minnen and S. Singh, “Channel-wise autoregressive entropy models for learned image compression,” in Proc. IEEE Int. Conf. Image Process. (ICIP), Oct. 2020, pp. 3339–3343.
2. Y. Qian, X. Sun, M. Lin, Z. Tan, and R. Jin, “Entroformer: A transformer-based entropy model for learned image compression,” in Proc. Int. Conf. Learn. Represent., 2022. [Online]. Available: https://openreview.net/forum?id=VrjOFfcnSV8
3. G. K. Wallace, “The JPEG still picture compression standard,” IEEE Trans. Consum. Electron., vol. 38, no. 1, pp. 18–34, Jun. 1992.
4. A. Skodras, C. Christopoulos, and T. Ebrahimi, “The JPEG 2000 still image compression standard,” IEEE Signal Process. Mag., vol. 18, no. 5, pp. 36–58, Sep. 2001.
5. F. Bellard. (2015). BPG Image Format. Accessed: Dec. 1, 2022. [Online]. Available: https://bellard.org/bpg
6. V. Sze, M. Budagavi, and G. J. Sullivan, “High efficiency video coding (HEVC),” in Integrated Circuit and Systems, Algorithms and Architectures, vol. 39. Cham, Switzerland: Springer, 2014, p. 40.
7. D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang, “ELIC: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 5718–5727.
8. Z. Guo, Z. Zhang, R. Feng, and Z. Chen, “Causal contextual prediction for learned image compression,” IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 4, pp. 2329–2341, Apr. 2022.
9. A. B. Koyuncu, H. Gao, A. Boev, G. Gaikov, E. Alshina, and E. Steinbach, “Contextformer: A transformer with spatio-channel attention for context modeling in learned image compression,” in Proc. Eur. Conf. Comput. Vis., Tel Aviv, Israel. Springer, Oct. 2022, pp. 447–463.
10. Versatile Video Coding, Standard Rec. ITU-T H.266 and ISO/IEC 23090-3, Aug. 2020.
11. J. Liu, H. Sun, and J. Katto, “Learned image compression with mixed transformer-CNN architectures,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023, pp. 14388–14397.
12. J. Ballé et al., “Nonlinear transform coding,” IEEE J. Sel. Topics Signal Process., vol. 15, no. 2, pp. 339–353, Feb. 2021.
13. C. C. Chi et al., “Parallel scalability and efficiency of HEVC parallelization approaches,” IEEE Trans. Circuits Syst. Video Technol., vol. 22, no. 12, pp. 1827–1838, Dec. 2012.
14. Z. Liu et al., “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 10012–10022.
15. JVE Team. Versatile Video Coding (VVC) Reference Software: Vvc Test Model (VTM). Accessed: Dec. 1, 2022. [Online]. Available: https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
16. R. Franzen, “Kodak lossless true color image suite,” 1999. [Online]. Available: http://r0k.us/graphics/kodak/
17. G. Toderici et al., “Workshop and challenge on learned image compression (CLIC 2020),” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020.
18. N. Asuni and A. Giachetti, “TESTIMAGES: A large-scale archive for testing visual devices and basic image processing algorithms,” in Proc. STAG, 2014, pp. 63–70.
19. J. Campos, S. Meierhans, A. Djelouah, and C. Schroers, “Content adaptive optimization for neural image compression,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops, 2019. [Online]. Available: https://openaccess.thecvf.com/content_CVPRW_2019/papers/CLIC%202019/Campos_Content_Adaptive_Optimization_for_Neural_Image_Compression_CVPRW_2019_paper.pdf
20. A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inform. Process. Syst. (NIPS), 2017, pp. 5998–6008.
21. A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” in Advances in Neural Information Processing Systems, vol. 25, F. Pereira, C. Burges, L. Bottou, and K. Weinberger, Eds. Red Hook, NY, USA: Curran Associates, 2012. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
22. M. M. Naseer, K. Ranasinghe, S. H. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Intriguing properties of vision transformers,” in Proc. Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 23296–23308.
23. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in Proc. Eur. Conf. Comput. Vis. Cham, Switzerland: Springer, 2020, pp. 213–229.
24. A. Dosovitskiy et al., “An image is worth 16×16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Represent., 2020, pp. 1–12.
25. Y. Jiang, S. Chang, and Z. Wang, “TransGAN: Two transformers can make one strong GAN,” 2021, arXiv:2102.07074.
26. P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2021, pp. 12873–12883.
27. N. Parmar et al., “Image transformer,” in Proc. Int. Conf. Mach. Learn., 2018, pp. 4055–4064.
28. J. Rissanen and G. G. Langdon, “Arithmetic coding,” IBM J. Res. Develop., vol. 23, no. 2, pp. 149–162, 1979.
29. Z. Tang, H. Wang, X. Yi, Y. Zhang, S. Kwong, and C.-C. J. Kuo, “Joint graph attention and asymmetric convolutional neural network for deep image compression,” IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 1, pp. 421–433, Jan. 2023.
30. G.-H. Wang, J. Li, B. Li, and Y. Lu, “EVC: Towards real-time neural image compression with mask decay,” 2023, arXiv:2302.05071.
31. H. Fu, F. Liang, J. Liang, B. Li, G. Zhang, and J. Han, “Asymmetric learned image compression with multi-scale residual block, importance scaling, and post-quantization filtering,” IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 8, pp. 4309–4321, Aug. 2023.
32. Y. Yang and S. Mandt, “Computationally-efficient neural image compression with shallow decoders,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 530–540.
33. Y. Yang, R. Bamler, and S. Mandt, “Improving inference for neural image compression,” in Proc. Adv. Neural Inf. Process. Syst., vol. 33, 2020, pp. 573–584.
34. C. Gao, T. Xu, D. He, Y. Wang, and H. Qin, “Flexible neural image compression via code editing,” in Proc. Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 12184–12196.
35. J. Ascenso, E. Alshina, and T. Ebrahimi, “The JPEG AI standard: Providing efficient human and machine visual data consumption,” IEEE MultimediaMag., vol. 30, no. 1, pp. 100–111, Jan. 2023.
36. G. Bjontegaard, Calculation of Average PSNR Differences Between RD-curves, document VCEG-M33, 2001. Simoncelli, “Density modeling of images using a generalized normalization transformation,” in Proc. Int. Conf. Learn. Represent., 2016. [Online]. Available: http://arxiv.org/abs/1511.06281
37. Y. Zhang, K. Li, K. Li, B. Zhong, and Y. Fu, “Residual non-local attention networks for image restoration,” in Proc. Int. Conf. Learn. Represent., 2019, pp. 1–18. [Online]. Available: https://openreview.net/forum?id=HkeGhoA5FX
38. A. Matton and A. Lam. (2020). Making Pytorch Transformer Twice As Fast on Sequence Generation. Accessed: Dec. 1, 2022. [Online]. Available: https://scale.com/blog/Pytorch-improvements
39. Y. Yan et al., “FastSeq: Make sequence generation faster,” in Proc. 59th Annu. Meeting Assoc. Comput. Linguistics 11th Int. Joint Conf. Natural Lang. Process., Syst. Demonstrations, 2021, pp. 218–226.
40. J. Bégaint, F. Racapé, S. Feltman, and A. Pushparaja, “CompressAI: A Pytorch library and evaluation platform for end-to-end compression research,” 2020, arXiv:2011.03029.
41. T. Xue, B. Chen, J. Wu, D. Wei, and W. T. Freeman, “Video enhancement with task-oriented flow,” Int. J. Comput. Vis., vol. 127, no. 8, pp. 1106–1125, Aug. 2019.
42. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 2014, arXiv:1412.6980.
43. Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale structural similarity for image quality assessment,” in Proc. 37th Asilomar Conf. Signals, Syst. Comput., 2003, pp. 1398–1402.
44. T.-Y. Lin et al., “Microsoft COCO: Common objects in context,” in Proc. Comput. Vis. (ECCV) 13th Eur. Conf., Zurich, Switzerland. Cham, Switzerland: Springer, Sep. 2014, pp. 740–755.
45. A. Paszke et al., “Pytorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems, vol. 32. Red Hook, NY, USA: Curran Associates, 2019, pp. 8024–8035. [Online]. Available: http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
46. M. Abadi et al. (2015). TensorFlow: Large-scale Machine Learning on Heterogeneous Systems. [Online]. Available: https://www.tensorflow.org/
47. V. Sovrasov. (2023). PTflops: A Flops Counting Tool for Neural Networks in Pytorch Framework. [Online]. Available: https://github.com/sovrasov/flops-counter.Pytorch
48. J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He, “DeepSpeed: System optimizations enable training deep learning models with over 100 billion parameters,” in Proc. 26th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, Aug. 2020, pp. 3505–3506.
49. NVIDIA, P. Vingelmann, and F. H. Fitzek. (2021). Cuda, Release: 11.4. [Online]. Available: https://developer.nvidia.com/cuda-toolkit

# KOYUNCU et al.: EFFICIENT CONTEXTFORMER: SPATIO-CHANNEL WINDOW ATTENTION

# References

1. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2009, pp. 248–255.
2. E. Agustsson and R. Timofte, “NTIRE 2017 challenge on single image super-resolution: Dataset and study,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops, 2017, pp. 126–135.
3. Y. Bengio, N. Léonard, and A. Courville, “Estimating or propagating gradients through stochastic neurons for conditional computation,” 2013, arXiv:1308.3432.
4. J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, “Algorithms for hyper-parameter optimization,” in Proc. Adv. Neural Inf. Process. Syst., vol. 24, J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Q. Weinberger, Eds. Curran Associates, 2011, pp. 2546–2555. [Online]
5. J. Kaplan. (2019). Notes on Contemporary Machine Learning for Physicists. [Online]

# Authors

# A. Burakhan Koyuncu

received the B.Sc. and M.Sc. (Hons.) degrees in electrical and computer engineering from the Technical University of Munich (TUM) in 2016 and 2019, respectively. He is currently pursuing the joint Ph.D. degree with the Chair of Media Technology, TUM, and the Audiovisual Technology Laboratory, Huawei Munich Research Center, Huawei Technologies, Munich. During the bachelor’s, he worked on methods for neuromorphic architectures and bio-inspired artificial neural networks, and during the master’s, he researched deep learning-based control algorithms for driving simulators. Since 2021, he has been actively involved in the development of the JPEG AI standard that was issued by ISO/IEC. His research interests include deep learning-based image and video processing, compression, and coding.

# Atanas Boev

received the Ph.D. degree from Tampere University of Technology, Finland, in 2012. In 2013, he was a Visiting Researcher with Holografika KFT, Hungary, involved on development and implementation of lightfield rendering algorithms. In 2014, he was a Post-Doctoral Researcher with Tampere University of Technology, working on modern signal processing methods for lightfield displays. He is currently a Principal Engineer with the Audiovisual Technology Laboratory, Huawei Munich Research Center, Huawei Technologies, Munich, working on AI video compression, color perception, and HDR tone mapping. He is a Researcher with expertise in video compression, lightfield displays, and human stereopsis.

# Elena Alshina

received the Ph.D. degree in mathematical modeling from Moscow State University in 1998. In 2006, she joined Samsung Electronics and was part of the team that submitted the top-performing response for HEVC/H.265 Call for Proposals in 2010. Since that time, she has been an active participant in international video codec standardization. She was chairing multiple core experiments and AhGs in JCTVC, JVET, and MPEG. Since 2018, she has been a Chief Video Scientist with Huawei Munich Research Center, Huawei Technologies, Munich, and the Director of Media Codec and Audiovisual Technology Labs. She has authored several scientific books, more than 100 papers in various journals and international conferences, and more than 200 patents. Her current research interests include AI-based image and video coding, signal processing, and computer vision. She is currently the Co-Chair of JVET, the Exploration on Neural Network-Based Video Coding and JPEG AI Standardization Project Co-Chair, and the Co-Editor. She (together with Alexander Alshin) received the Gold medal from Russian Academy of Science in 2003 for series of research papers on computer science.

# Panqi Jia

received the master’s degree in electrical engineering from Leibniz University Hannover (LUH), Hannover, Germany, in 2019. He is currently pursuing the joint Ph.D. degree with Huawei Technologies, which has collaborated with the Chair of Multimedia Communications and Signal Processing, Friedrich Alexander University (FAU). During the bachelor’s degree, he worked on the application of ultra-wideband radar. During the master’s degree, he researched Car to X (C2X) communication systems and stereo matching algorithms in the vehicle communication area. His research interests include methods for image and video compression and deep learning.

# Eckehard Steinbach

(Fellow, IEEE) received the Ph.D. degree in electrical engineering from the University of Erlangen-Nuremberg, Germany, in 1999. In February 2002, he joined the Department of Computer Engineering, Technical University of Munich (TUM), where he is currently a Professor of media technology. His research interests include visual-haptic information processing and communication, telepresence and teleoperation, and networked and interactive multimedia systems.

