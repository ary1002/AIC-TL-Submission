# Implementation of the Neural Graph Collaborative Filtering (NGCF) Algorithm

Here is an implementation of the Neural Graph Collaborative Filtering (NGCF) algorithm applied to a recommendation system. NGCF is a state-of-the-art recommendation algorithm that utilizes the collaborative filtering approach with graph convolutional networks to model user-item interactions effectively.

## Methodology

We implemented the NGCF algorithm using PyTorch, leveraging sparse matrix operations for efficiency. The implementation consists of two main components: data preprocessing and model training.

### Data Preprocessing

- We parse the input data files containing user-item interactions to construct the interaction matrices `R_train` and `R_test`.
- We create the adjacency matrix `adj_mtx` representing the user-item graph. The adjacency matrix is then transformed into an NGCF-specific adjacency matrix using normalization techniques.
- We preprocess the data to generate negative item pools for users, ensuring that the model can sample negative items during training.

### Model Training

- We define the NGCF model architecture as a PyTorch module. The model comprises graph convolutional layers with user and item embeddings.
- We train the NGCF model using stochastic gradient descent (SGD) with the Adam optimizer.
- We monitor the training process for convergence using early stopping based on validation metrics such as Recall@k and NDCG@k.
- We evaluate the trained model on test data to measure its performance in terms of recommendation accuracy metrics.

## NGCF Model Overview

1. **Objective Function**:
    - \( L_{BPR} = - \log \sigma(\hat{y}_{uij}) \)
2. **Propagation Layers**:
    - \( H^{(k)} = \text{ReLU}(\tilde{A} \cdot H^{(k-1)} \cdot W^{(k)} + b^{(k)}) \)
3. **Dropout Regularization**:
    - Node dropout: randomly drops nodes from the graph
    - Message dropout: randomly drops messages during message passing
4. **User and Item Embeddings**:
    - Final embeddings obtained after \( K \) propagation layers
5. **Optimization**:
    - Adam optimizer used to minimize the BPR loss function

## Results

We conducted experiments on the MovieLens 100k dataset with the following hyperparameters:

- Batch size: 512
- Embedding dimension: 64
- Number of layers: [64, 64]
- Learning rate: 0.0005
- Regularization parameter: \( 1 \times 10^{-5} \)
- Node dropout: 0.0
- Message dropout: 0.0
- Top-k recommendation: 20
- Number of epochs: 63 (early stopping applied)

The training process converged within 63 epochs, with early stopping triggered after 5 consecutive epochs without improvement in validation metrics. The final performance metrics on the test set were as follows:

- Recall@20: 0.3403
- NDCG@20: 0.6729

## Conclusion

The NGCF algorithm demonstrates promising results in the recommendation task, achieving competitive performance on the MovieLens 100k dataset. By leveraging graph convolutional networks and collaborative filtering techniques, NGCF effectively captures user-item interactions and provides accurate recommendations. Further experimentation and optimization may enhance the algorithm’s performance on larger datasets and in real-world applications.
