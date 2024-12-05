# Deep Learning Algorithms Implementation

This project implements a wide range of deep learning algorithms from scratch. Each algorithm is accompanied by both a Python implementation file and an example notebook demonstrating its usage.

## Table of Contents
1. [Neural Network Architectures](#neural-network-architectures)
2. [Convolutional Neural Networks](#convolutional-neural-networks)
3. [Recurrent Neural Networks](#recurrent-neural-networks)
4. [Attention Mechanisms](#attention-mechanisms)
5. [Generative Models](#generative-models)
6. [Object Detection](#object-detection)
7. [Natural Language Processing](#natural-language-processing)
8. [Model Evaluation and Visualization](#model-evaluation-and-visualization)


## Neural Network Architectures

### Multilayer Perceptron (MLP) [[notebook]](mlp_example.ipynb)
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/1200px-Colored_neural_network.svg.png" width="400">

Each neuron computes: $y = f(\sum_{i=1}^n w_i x_i + b)$

### Deep Neural Network (DNN) [[notebook]](dnn_example.ipynb)
<img src="https://miro.medium.com/max/1400/1*3fA77_mLNiJTSgZFhYnU0Q.png" width="400">

Layer output: $h^{(l)} = f(W^{(l)}h^{(l-1)} + b^{(l)})$

## Convolutional Neural Networks

### Basic CNN [[notebook]](cnn_cifar10_example.ipynb)
<img src="https://miro.medium.com/max/2000/1*vkQ0hXDaQv57sALXAJquxA.jpeg" width="500">

Convolution: $S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i+m,j+n)K(m,n)$

## Recurrent Neural Networks

### Long Short-Term Memory (LSTM) [[notebook]](lstm_example.ipynb)
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" width="400">

Key equations:
- Forget gate: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- Input gate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- Cell state: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
- Output: $h_t = o_t * \tanh(C_t)$

## Attention Mechanisms

### Self-Attention [[notebook]](visualize_attention.ipynb)
<img src="https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png" width="400">

$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

## Generative Models

### Variational Autoencoder (VAE) [[notebook]](vae_example.ipynb)
<img src="https://www.tensorflow.org/tutorials/generative/images/vae_diagram.png" width="400">

Objective: $\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$

### Generative Adversarial Network (GAN) [[notebook]](gan_example.ipynb)
<img src="https://www.tensorflow.org/tutorials/generative/images/gan1.png" width="400">

Objective: $\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$

## Object Detection

### YOLO (You Only Look Once) [[notebook]](yolo.ipynb)
<img src="https://www.mdpi.com/electronics/electronics-10-03150/article_deploy/html/images/electronics-10-03150-g001.png" width="400">

Each cell predicts:
- Bounding box coordinates (x, y, w, h)
- Object confidence
- Class probabilities

## Natural Language Processing

### Sequence-to-Sequence Model [[notebook]](seq2seq.ipynb)
<img src="https://miro.medium.com/max/1400/1*sO-SP58T4brE9EHazHSeGA.png" width="400">

Objective: $P(y_1, ..., y_{T'} | x_1, ..., x_T) = \prod_{t=1}^{T'} P(y_t | y_1, ..., y_{t-1}, x_1, ..., x_T)$

## Model Evaluation and Visualization

### Model Evaluation [[notebook]](evaluate_model.ipynb)
<img src="https://miro.medium.com/max/712/1*Z54JgbS4DUwWSknhDCvNTQ.png" width="350">

### Attention Visualization [[notebook]](visualize_attention.ipynb)
<img src="https://distill.pub/2016/augmented-rnns/assets/show-attend-tell.png" width="400">


## Usage

Each algorithm is implemented in its own Python file (e.g., `lstm.py`) and has a corresponding example notebook (e.g., `lstm_example.ipynb`) demonstrating its usage. To use an algorithm:

1. Import the necessary class from the corresponding Python file.
2. Create an instance of the class with desired parameters.
3. Prepare your data (and use appropriate data loaders if necessary).
4. Train the model on your data.
5. Use the model to make predictions or generate outputs as needed.

Refer to the example notebooks (linked next to each algorithm title) for detailed usage instructions for each algorithm.

## Conclusion

This project provides implementations and examples of a wide range of deep learning algorithms. These algorithms represent the cutting edge of machine learning and are crucial for understanding modern AI applications. By studying and experimenting with these implementations, you can gain a deeper understanding of how these algorithms work, their strengths and limitations, and when to apply them to different types of problems.

Remember that while these implementations are educational, for production use, it's often better to use well-optimized libraries like PyTorch, TensorFlow, or Keras, which have been extensively tested and optimized for performance and scalability.

## Contributing

Contributions to this project are welcome! If you find a bug, have a suggestion for improvement, or want to add a new algorithm implementation, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.