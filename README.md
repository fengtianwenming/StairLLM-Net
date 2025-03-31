# StairLLM-Net
This repository contains the official implementation of the paper titled "StairLLM-Net: A Large Language Model based Approach for Photovoltaic Power Forecasting". The code provided here is intended to facilitate reproducibility and further research in the field of Photovoltaic Power Forecasting.

# Paper Status
The paper has been accepted for publication in the CEEPE 2025.

# Abstract
With the expansion of large-scale grid-connected distributed photovoltaic (PV) systems, accurate photovoltaic power forecasting has become increasingly crucial. Current methods often require extensive hyperparameter tuning and demonstrate limited zero-shot capability. This study introduces StairLLM-Net, an innovative Large Language Model (LLM) based approach that utilizes LLM as a feature extractor and implements feature decoding through segmented blocks. The proposed model effectively integrates time-series data with linguistic embeddings using a patch-based methodology and incorporates a specifically designed input format to leverage the inherent reasoning capabilities of LLMs, thereby enhancing predictive accuracy at individual PV stations. The StairLLM-Net exhibits remarkable robustness and can be directly applied to previously unseen PV stations without requiring fine-tuning. Extensive experiments on real-world station datasets demonstrate the model's exceptional cross-dataset generalization capability.

# Installation

1. Clone the repository to your local machine.
2. Install the required packages using pip. If you use Excel reader of pandas, you also need to install openpyxl. 
3. Get a LLM weights from huggingface, and install it in the PretrainedModels folder. Its structure should be like this: PretrainedModels/*ModelName*/*Model Weight*
4. Prepare your data in the Dataset folder.

# Usage

1. If you have not used Lightning, maybe need to learn how to use it. Please refer to the official website https://lightning.ai/docs/pytorch/stable/.
2. revise the Models/DataModule.py to fit your own data format.
3. revise the train.py to set model parameters and data set index.
4. run train.py, and it will save a checkpoint file in the Checkpoints folder.
5. run test.py to test the model.

# License
This project is licensed under the MPL2.0 License - see the LICENSE file for details.

# Contact
For any questions or concerns, please feel free to issue a request or contact us at 213212555@seu.edu.cn. 
