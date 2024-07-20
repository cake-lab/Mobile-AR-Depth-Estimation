<div align="center">
 <h1>Mobile-AR-Depth-Estimation</h1>

[**Ashkan Ganj**](https://ashkanganj.me/)<sup>1</sup> · [**Yiqin Zhao**](https://yiqinzhao.me/)<sup>1</sup> . [**Hang Su**](https://suhangpro.github.io/)<sup>2</sup> · [**Tian Guo**](https://tianguo.info/)<sup>1</sup>

<sup>1</sup>Worcester Polytechnic Institute
&emsp;&emsp;&emsp;<sup>2</sup>Nvidia Research

<a href="https://arxiv.org/pdf/2310.14437.pdf"><img src='https://img.shields.io/badge/arXiv-Extended Version-red' alt='arXiv (Extended Version)'></a>
<a href='https://dl.acm.org/doi/10.1145/3638550.3641122'><img src='https://img.shields.io/badge/ACM-Proceeding-blue' alt='Proceeding'></a>
<a href='https://www.gnu.org/licenses/gpl-3.0'><img src='https://img.shields.io/badge/License-GPLv3-green.svg'></a>

For any questions or concerns, please feel free to reach out to [Ashkan Ganj](mailto:aganj@wpi.edu)

![teaser](MobileARWorkFlow.png)
</div>

Welcome to the official repository for our HotMobile24 paper. This work presents the challenges and opportunities of achieving accurate metric depth estimation in mobile AR. We tested four different state-of-the-art monocular depth estimation models performance in AR scenarios and identified three types of challenges: hardware, data, and model related challenges. Furthermore, our
research provides promising future directions to explore and solve those challenges.


## Repository Structure

This repository is structured into two main directories, `Analysis` and `Models`, each serving a distinct purpose in the context of our research.

```plaintext
├── Analysis/
│   ├── notebooks/
│   │   ├── ARKitScenes/
│   │   ├── vidar/
│   │   └── ZoeDepth/
│   └── results/
│       ├── ARkitScenes/
│       ├── DistDepth/
│       ├── ZeroDepth(Vidar)/
│       └── ZoeDepth/
├── models/
│   ├── DistDepth/
│   │   ├── ...
│   ├── vidar/
│   │   ├── ...
│   └── ZoeDepth/
│       ├── ...
├── LICENSE
└── README.md
```

### Analysis Directory

The `Analysis` directory is the central hub for all codes, results (including CSV files, figures, and images), and analytical notebooks associated with our paper. By running the notebooks located within this directory for each model, users can replicate our analytical process and generate all the outputs—ranging from CSV data files to the exact figures published in our paper. This directory aims to provide a transparent and replicable pathway for understanding our research findings and methodology.

### Models Directory

In the `Models` directory, you'll find modified versions of various depth estimation models, specifically adapted to work with the ARKitScenes dataset. Each model within this directory comes with its own `README.md`, containing detailed instructions on setup, usage, and evaluation. This ensures that users can easily navigate and utilize the modified models for their depth estimation projects.

## ARKitScenes Dataset

For all evaluations, we utilized the ARKitScenes dataset.

## Download and preprocess

To download and preprocess the ARKitScenes dataset, please use the following notebook: [ARKitScenes Notebook](/Analysis/notebooks/ARKitScenes/download_preprocess.ipynb). This notebook provides step-by-step instructions for obtaining and preparing the dataset for use with the models.

### Additional Analysis Notebooks

Alongside the preprocessing notebook, we have developed several other notebooks aimed at analyzing the dataset in depth. These notebooks explore various aspects of the data, helping to better understand its characteristics and how it impacts the performance of our depth estimation models.

The additional notebooks, found within the same directory, cover topics such as:

- `analysis_per_obj.ipynb`: Examines the performance of models across different object types, identifying any meaningful differences.
- `confidence_eval.ipynb`: Analyzes missing points in the depth map based on confidence levels, including a visualization of the distribution and frame-by-frame analysis.
- `main.ipynb`: Contains comprehensive analysis for the number of missing points in both ARKit and ground truth (GT) depths, offering code for analysis based on different thresholds and visualizing missing points in the depth map.

To access these analysis notebooks, navigate to the following directory in our repository: `/Analysis/notebooks/ARKitScenes`.

## ZoeDepth

For comprehensive details about ZoeDepth, including its methodologies and achievements, we encourage you to visit the [ZoeDepth GitHub repository](https://github.com/isl-org/ZoeDepth) and consult the [ZoeDepth paper](https://arxiv.org/abs/2302.12288).

### Important Note

It's crucial to note that ZoeDepth's implementation might encounter compatibility issues with newer versions of PyTorch. To ensure optimal functionality, we strongly recommend using the exact version of PyTorch specified in the ZoeDepth repository instructions (any version <= 2.1.0 is acceptable).

### Getting Started with ZoeDepth

To begin working with ZoeDepth, please adhere to the instructions provided in the [ZoeDepth repository](https://github.com/isl-org/ZoeDepth). These steps will guide you through cloning the repository, setting up the required environment, and downloading the necessary pre-trained weights.

### Evaluation Steps on ARKitScenes

To evaluate ZoeDepth on the ARKitScenes dataset:

1. Open the `Analysis/notebooks/ZoeDepth/evaluateOnARkitScenes.ipynb` notebook found within our repository.
2. Ensure the dataset path is correctly set to where your ARKitScenes dataset is stored.
3. Adjust the notebook to point to the pre-trained weights, located in the `models/ZoeDepth/` directory. For ARKitScenes, use the specific pre-trained weights provided ([midas_train](https://drive.google.com/file/d/1k6kLfmNWyfGA19x6ae1rIbSUcuYROs4A/view?usp=drive_link), [midas_freeze](https://drive.google.com/file/d/1sZ3jOnp56ExuW8cusuiT6qFzxRDRpTc9/view?usp=sharing)).
4. Execute the notebook to start the evaluation process.
5. After the evaluation, results will be summarized and saved in a `.csv` file, ready for any further analysis or visualization.

#### Evaluation Steps on NYUv2 (Cropping Effect)

To assess the impact of cropping on model performance using the NYUv2 dataset:

1. Go to the `Analysis/notebooks/ZoeDepth/cropped_effect.ipynb` notebook within our repository.
2. Update the dataset path variable to direct to your NYUv2 dataset location.
3. Run the notebook to perform the evaluation.
4. The notebook will present a series of average depth errors associated with different cropping percentages, offering insights into the cropping effect on depth estimation accuracy.

### Train ZoeDepth on ARKitScenes

- `models/ZoeDepth/zoedepth/utils/config.py`: This file contains general configuration for dataset. Please update the file according to your requirements before initiating the training process(dataset path, splits path).
- `models/ZoeDepth/train_mono.py`: This script is the entry point for initiating the training process. It reads from `config.py` and applies the specified configurations during training. Depending on your requirements, you might want to modify this script to alter the training workflow, add custom logging, or implement additional features.

#### Initiating Training

Once you have made the necessary adjustments to the configuration files, you can start the training process by running the following command from your terminal:

```bash
python train_mono.py
```

## DistDepth (Toward Practical Monocular Indoor Depth Estimation)

For more details, please refer to the DistDepth's <a href="https://github.com/facebookresearch/DistDepth">Github repository</a> and <a href="https://arxiv.org/abs/2112.02306">paper</a>.

### Getting Started with DistDepth

To utilize DistDepth, follow the steps outlined below. Detailed instructions and additional information are available in the DistDepth [Readme.md](models/DistDepth/README.md).

1. **Download Pre-trained Weights:** Access the pre-trained weights via the link provided in the DistDepth [Readme.md](models/DistDepth/README.md). These weights are essential for evaluating the model's performance.

2. **Prepare Your Environment:** Ensure your setup meets the prerequisites listed in the `README.md`, including necessary libraries and dependencies.

3. **Running Evaluation:**
   - Place the pre-trained weights in the same directory as the `eval.sh` script.
   - Update the `eval.sh` script with the path to the ARKitScenes dataset.
   - Execute the following command to evaluate the DistDepth model on the ARKitScenes dataset:

     ```bash
     sh eval.sh
     ```

   - The script outputs the evaluation results and save them in .csv file which can be later be used for visualization.

## ZeroDepth (ViDepth)

For a comprehensive understanding of the underlying methodology and insights into the model's development, we direct readers to the official resources:

- For detailed information and the latest updates on ViDepth, visit [Vidar's GitHub repository](https://github.com/TRI-ML/vidar).
- To dive deeper into the research and technical details, the [ViDepth paper](https://arxiv.org/abs/2306.17253) provides a thorough explanation of the technology and its applications.

### Getting Started with ZeroDepth

To begin working with ZeroDepth, you should first set up your environment and acquire the necessary pre-trained weights by following the instructions in the [ZeroDepth Readme.md](https://github.com/cake-lab/Mobile-AR-Depth-Estimation/blob/main/models/vidar/README.md).

#### Evaluation Steps

1. Navigate to the `Analysis/notebooks/vidar/vidar-inference.ipynb` notebook within our repository.
2. Update the dataset path variables:
   - **ARKitScenes Dataset Path:** Ensure the path points to where you've stored the ARKitScenes dataset.
   - **NyuV2 Dataset Path:** Similarly, update this to the location of your NyuV2 dataset.
3. Execute the notebook, following the provided instructions to initiate the evaluation of the ZeroDepth model on the ARKitScenes dataset.
4. Upon completion, the notebook will present the evaluation results and automatically save them to a `.csv` file. This file can be utilized for further analysis or visualization purposes.

These steps are designed to facilitate a smooth experience in assessing the performance of ZeroDepth with the ARKitScenes dataset, enabling users to effectively leverage this model in their depth estimation projects.

## Citation

If our work assists you in your research, please cite it as follows:

```bibtex
@inproceedings{10.1145/3638550.3641122,
author = {Ganj, Ashkan and Zhao, Yiqin and Su, Hang and Guo, Tian},
title = {Mobile AR Depth Estimation: Challenges \& Prospects},
year = {2024},
isbn = {9798400704970},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3638550.3641122},
doi = {10.1145/3638550.3641122},
abstract = {Accurate metric depth can help achieve more realistic user interactions such as object placement and occlusion detection in mobile augmented reality (AR). However, it can be challenging to obtain metricly accurate depth estimation in practice. We tested four different state-of-the-art (SOTA) monocular depth estimation models on a newly introduced dataset (ARKitScenes) and observed obvious performance gaps on this real-world mobile dataset. We categorize the challenges to hardware, data, and model-related challenges and propose promising future directions, including (i) using more hardware-related information from the mobile device's camera and other available sensors, (ii) capturing high-quality data to reflect real-world AR scenarios, and (iii) designing a model architecture to utilize the new information.},
booktitle = {Proceedings of the 25th International Workshop on Mobile Computing Systems and Applications},
pages = {21–26},
numpages = {6},
location = {<conf-loc>, <city>San Diego</city>, <state>CA</state>, <country>USA</country>, </conf-loc>},
series = {HOTMOBILE '24}
}
```

## Acknowledgements

This work was supported in part by NSF Grants #2105564 and #2236987, a VMware grant, the Worcester Polytechnic Institute’s Computer Science Department. Most results presented in this work were obtained using <a href="https://www.cloudbank.org/">CloudBank</a>, which is supported by the National Science Foundation under award #1925001.
