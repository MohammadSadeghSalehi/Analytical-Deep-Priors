# ADP_modified
Comparison between Analytical Deep Image Priors and the modified version

# Analytical Deep Priors

This repository contains the code implementation for the analytical deep priors method presented in the paper "Analytical Deep Priors for Image Reconstruction" (Salehi, M.S. et al.). This method leverages the analytical properties of deep neural networks to improve image reconstruction tasks.

## Paper Reference

[Analytical Deep Priors for Image Reconstruction](https://arxiv.org/pdf/2502.09758)

## Description

The code implements the analytical deep priors approach, which derives analytical expressions for the gradients of a neural network with respect to its inputs. This analytical approach offers advantages in terms of computational efficiency and potentially improved reconstruction quality compared to traditional iterative optimization methods.

The core idea is to bypass the standard backpropagation and instead calculate gradients symbolically, enabling faster and more accurate image reconstruction.

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/MohammadSadeghSalehi/Analytical-Deep-Priors.git](https://www.google.com/search?q=https://github.com/MohammadSadeghSalehi/Analytical-Deep-Priors.git)
    cd Analytical-Deep-Priors
    ```

2.  Install the necessary dependencies:

    ```bash
    pip install numpy matplotlib scikit-image
    ```
    * Note: Ensure you have Python 3.6 or higher installed.

## Usage

1.  **Running the main script:**

    ```bash
    python main.py
    ```

2.  **Modifying parameters:**

    * The `main.py` script contains parameters that can be adjusted to control the reconstruction process. These include:
        * `noise_level`: The level of noise in the input image.
        * `network_depth`: The depth of the neural network.
        * `learning_rate`: The learning rate used in the reconstruction process.
        * `iterations`: The number of iterations for the reconstruction.
    * Adjust these parameters within the `main.py` file to experiment with different reconstruction settings.

3.  **Input Images:**
    * The `input_image.png` file is used as the default input. Replace this file with your desired input image.
4.  **Output images:**
    * The reconstructed images are saved in the same directory as the script.

## Explanation of Variables

The following table explains the key variables used in the code, and their relation to the paper:

| Variable Name   | Description
