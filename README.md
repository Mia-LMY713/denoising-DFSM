# DFSMD-Net: Dynamic Feature Self-Aggregation Multi-scale Denoising Network for Medical Images

This repository contains the official implementation of **DFSMD-Net**, a deep learning framework designed for accelerated deblurring, denoising, and reconstruction of medical images (e.g., dental CT scans) using dynamic feature self-aggregation and controllable gated residual learning.

## 🌟 Key Features & Contributions

* **Controllable Gate Spatial Gating Algorithm:** Introduces a spatial gating algorithm that refines the learning process of the target residual depth network, significantly enhancing detailed feature extraction at critical anatomical locations.
* **Multi-Scale Feature Aggregation Deep Network:** Integrates multi-scale structural and semantic information to accelerate the deblurring and denoising of low-quality medical images.

---

## 🏗️ Network Architecture

The DFSMD-Net framework consists of three primary components: the **Feature Extractor**, the **Dynamic Feature Self-Aggregation Attention Module**, and the **Decoder**.

![Overall Architecture](./image/Overall%20architecture%20of%20deraining%20network.png)

![Gated Residual Block and Attention Module](./image/Structure%20of%20the%20Gated%20Fine-tuning%20Residual%20Convolutional%20Block%20and%20Detailed%20diagram%20of%20the%20Dynamic%20Feature%20Self-Aggregation%20Attention%20Module.png)

---

## 📊 Dataset

A representative, anonymized sample from our private clinical dataset (`Teeth Dataset`), used for evaluating the network's performance.

![Teeth Dataset](./image/teeth%20dataset.png)
---

## 💻 Prerequisites & Environment

Ensure your environment satisfies the following requirements:

* Python 3.8+
* CUDA 10.1 + CuDNN
* pip
* Virtual environment (optional but recommended)

---

## 🚀 Getting Started

### 1. Inference & Testing
Run inference with pretrained weights using default settings:
```bash
python main.py -i

*Note: Sample testing images are provided under `testingImages/sampleImages/`. The denoised outputs will be saved to `modelOutput/sampleImages/`.*

**Custom Inference Settings:**

```bash
python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=15,25,50

```

*Arguments:*

* `-ns`: Specifies the standard deviation of Gaussian noise to test on (e.g., `-ns=15,25,50`).
* `-s`: Path to the source input directory (e.g., `testingImages/`).
* `-d`: Path to the destination directory for output images (e.g., `modelOutput/`).

### 2. Model Training

**Train from scratch with a custom dataset:**

```bash
python main.py -ts -e 50 -b 16

```

*Note: Update the `"trainingImagePath"` field inside `mainModule/config.json` to specify your dataset folder. Use the `-e` flag to specify training epochs and `-b` to set the batch size.*

**Resume Training / Transfer Learning:**

```bash
python main.py -tr -e 100 -b 16

```

### 3. Utility Commands

* **Inspect Model Configuration & Architecture Summary:**
```bash
python main.py -ms

```


* **Create a New `config.json` File Interactively:**
```bash
python main.py -c

```


* **Manually Update Existing Config Fields:**
```bash
python main.py -u

```


* **Run Overfitting Test (Single Sample Verification):**
```bash
python main.py -to

```



---

## 📝 Author & Maintainer

For questions or potential collaborations regarding deep learning in medical image reconstruction, feel free to visit my personal website:
🔗 **[Mia-LMY713.github.io](https://Mia-LMY713.github.io)**
