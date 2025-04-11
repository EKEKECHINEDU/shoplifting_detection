# Shoplifting Detection and Video Surveillance  

<br>
<br>
<p align="center">
  <img src="GitHubtopicpicimages/trends-in-shoplifting.jpg" 
       width="350">
</p>
<br>

<p align="center">
  <a href="#introduction"> Introduction </a> •
  <a href="#approach"> Approaches & Models </a> •
  <a href="#performance"> Performance </a> •
  <a href="#trends"> Trends </a> •
  <a href="#limitation"> Limitations </a> •
  <a href="#solution"> Solutions </a> •
  <a href="#discussion"> Discussion </a> •
  <a href="#conclusion"> Conclusion </a> •
  <a href="#reference"> References</a> •
</p>

<a id = 'introduction'></a>
## Introduction

Retailing, ranging from small stalls to large malls, is a common global practice but is often threatened by shoplifting, with studies showing one in every eleven shoppers engages in theft. Traditional surveillance methods like security patrols and CCTV monitoring are labor-intensive and not always effective, especially in busy retail environments where vast video data makes real-time monitoring difficult. To address this, there is a growing interest in automated video surveillance powered by machine learning and artificial intelligence, which can efficiently detect suspicious behavior and monitor multiple feeds simultaneously. This study explores emerging trends, challenges, and future directions in automated shoplifting detection and event classification.


<a id = 'approach'></a>

## 📌 Current Approaches and Mathematical Models for Shoplifting Detection

### Convolutional Neural Networks (CNNs)
CNNs are used for extracting spatial features from video frames. They apply convolutional filters to capture patterns like edges, textures, and objects.

**Mathematical Representation:**

```math
y = f(W * x + b)
```

Where:
- \( y \) is the output feature map  
- \( W \) are learnable filters (weights)  
- \( x \) is the input image  
- \( b \) is the bias  
- \( f(.) \) is a non-linear activation function (e.g., ReLU)

✅ *Pretrained models such as InceptionV3, ResNet-50, VGG-16, or MobileNetV3 are often used for feature extraction via transfer learning.*

---

### 3D Convolutional Neural Networks (3D-CNNs)
3D-CNNs capture spatial and short-term motion features by analyzing video as a volume of frames.

**Mathematical Representation:**

```math
y^{(l)} = f\left(\sum_{i} W_i^{(l)} * x_i^{(l-1)} + b^{(l)}\right)
```

Where:
- \( $y^{(l)}$ \) is the output of the 3D convolutional layer  
- \( W_i^{(l)} \) are 3D filters (with depth, height, and width: \( d_t, d_h, d_w \))  
- \( x_i^{(l-1)} \) is the input video tensor  
- \( f \) is typically a ReLU activation

---

### Recurrent Neural Networks (RNNs)
RNNs model temporal dependencies in sequential data. Bidirectional LSTM (BiLSTM) variants capture both past and future context in sequences of features.

**Mathematical Representation:**

```math
\overrightarrow{h_t} = \text{LSTM}(x_t, \overrightarrow{h_{t-1}}), \quad
\overleftarrow{h_t} = \text{LSTM}(x_t, \overleftarrow{h_{t+1}})
```

**Final Output:**

```math
h_t = [\overrightarrow{h_t}, \overleftarrow{h_t}]
```

✅ *CNNs are typically used first to extract features, which are then passed through RNNs like LSTM or GRU for temporal modeling.*

---

### CNN-LSTM Hybrid Models
These models combine CNN for spatial feature extraction and LSTM for modeling temporal patterns.

**Mathematical Representation (LSTM update):**

```math
h_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
```

Where:
- \( h_t \) is the hidden state at time \( t \)  
- \( x_t \) is the CNN-extracted feature input  
- \( \sigma \) is an activation function (often sigmoid)

---

### Vision Transformers (ViTs)
ViTs split video frames into patches and use self-attention mechanisms to capture global and local relationships.

**Mathematical Representation:**

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Where:
- \( Q \) = Queries  
- \( K \) = Keys  
- \( V \) = Values  
- \( d_k \) = dimension of the key vectors

✅ *ViTs can model complex spatiotemporal dependencies efficiently.*

---

### Two-Stream Networks
Two-stream networks use separate pipelines for spatial (RGB frames) and temporal (optical flow) information, and then fuse the outputs.

**Mathematical Representation:**

```math
F = \alpha F_{spatial} + (1 - \alpha) F_{temporal}
```

Where:
- \( F_{spatial} \) is the output of the spatial stream  
- \( F_{temporal} \) is the output of the temporal stream  
- \( \alpha \in [0, 1] \) is the weighting factor

---

### Hybrid Architectures for Anomaly Detection
Modern hybrid architectures combine CNNs and RNNs (e.g., GRU, BiLSTM) to extract both spatial and motion features for robust anomaly detection.

✅ *Example models:*
- **CNN-GRU** for combining frame-level features and temporal modeling  
- **CNN-BiLSTM** as used in recent works, achieving high accuracy (~81%) for shoplifting detection

<a id = 'performance'></a>
## Performance

**3D-CNN**: Well-suited for analyzing high-resolution CCTV footage where both spatial and temporal context is important.

**CNN-LSTM**: Ideal for real-time alert systems due to its ability to extract spatial features (via CNN) and learn time-based patterns (via LSTM).

**Vision Transformer (ViT)**: Effective in complex scenarios such as occlusion, where traditional convolutional models might struggle to maintain context.

<a id = 'trends'></a>
## Trends

- Hybrid Models
- Transfer Learning
- Anomaly Detection Framing

<a id = 'limitation'></a>
## Limitations

- Data Limitation & Bias 
- Environmental and Operational Variability
- Interpretability
- Computational Demands

<a id = 'solution'></a>
## Solutions

**Data Augmentation and New Datasets**: Collecting and publicly releasing real-world, multi-angle surveillance data can improve model generalizability.
**Explainable AI (XAI)**: Incorporating explainability tools to identify which features drive classification decisions can mitigate bias.
**Optimized Hybrid Models**: Fine-tuning hyperparameters (e.g., batch size, learning rate) in hybrid CNN–RNN architectures improves real-time performance and accuracy.
**Anomaly Detection Frameworks**: Using unsupervised anomaly detection techniques to learn normal shopping behaviour and flag deviations.

<a id ='discussion'></a>
## Discussion

AI, machine learning, and deep learning have greatly improved shoplifting detection by automating surveillance, reducing human error, and enhancing security. Traditional systems rely on human monitoring, which can be inconsistent. AI-driven models, trained on diverse datasets, can identify suspicious behaviors like loitering or item removal, while deep learning techniques such as CNNs and RNNs can analyze video footage with higher precision. However, challenges include bias in training data, privacy concerns, and the high cost of deployment. Addressing these issues requires careful data curation, balancing privacy with security, and exploring affordable AI solutions for smaller retailers.

<a id = 'conclusion'></a>
## Conclusion

AI, machine learning, and deep learning have significantly enhanced shoplifting detection by improving surveillance accuracy, automating security processes, and reducing human error, enabling proactive loss prevention for retailers. However, challenges such as data bias, ethical issues, privacy concerns, and financial constraints must be addressed through responsible deployment, including diverse training datasets, clear privacy policies, and affordable solutions for all retailers. Future research should focus on refining datasets, enhancing model interpretability, and developing lightweight models for real-time applications.

<a id = 'reference'></a>
## References

- Gim, U.J.; Lee, J.J.; Kim, J.H.; Park, Y.H.; Nasridinov, A. An Automatic Shoplifting Detection from Surveillance Videos. In Proceedings of the AAAI Conference on Artificial Intelligence, New York, NY USA, 7–12 2020; Apress: Berkeley, CA, USA, 2020; Volume 34, pp. 13795–13796.
- Chemere, D.S. Real-time Shoplifting Detection from Surveillance Video. Master Thesis, Addis Ababa University, Addis Ababa, Ethiopia, 2018; p. 94.
- Kirichenko, L.; Radivilova, T.; Sydorenko, B.; Yakovlev, S. Detection of Shoplifting on Video Using a Hybrid Network. Computation 2022, 10(11), 199.
- Ansari, M.A.; Singh, D.K. ESAR, An Expert Shoplifting Activity Recognition System. Cybernetics and Information Technologies 2022. researchgate.net
- Muneer, I.; Saddique, M.; Habib, Z.; Mohamed, H.G. Shoplifting Detection Using Hybrid Neural Network CNN-BiLSMT and Development of Benchmark Dataset. Applied Sciences 2023, 13(14), 8341.
- Fawaz, H.I.; Lucas, B.; Forestier, G.; Pelletier, C.; Schmidt, D.F.; Weber, J.; Webb, G.I.; Idoumghar, L.; Muller, P.-A.; Petitjean, F. InceptionTime: Finding alexnet for Time Series classification. Data Min. Knowl. Discov. 2020, 34, 1936–1962.
