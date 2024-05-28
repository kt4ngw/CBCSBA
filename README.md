
## Joint Class-Balanced Client Selection and Bandwidth Allocation for Cost-Efficient Federated Learning in Mobile Edge Computing Networks

The paper is currently under review (in IEEE Trans. Mob. Comput.).

**Title:** Joint Class-Balanced Client Selection and Bandwidth Allocation for Cost-Efficient Federated Learning in Mobile Edge Computing Networks

**Author:**  Jian Tang, Xiuhua Li, Hui Li, Penghua Li, Xiaofei Wang, Victor C. M. Leung



### 1. Background and problem

Given the system and data heterogeneity of MCs, client selection and bandwidth allocation are critical to achieving cost-effective FL in bandwidth-constrained MEC networks. Therefore, we investigate the joint client selection and bandwidth allocation problem to reduce the costs (i.e., latency and energy consumption).

We have formulated the problem and decomposed it into two subproblems. (The process is in the paper.)

![image-20240527225439407](./assets/image-20240527225439407.png)

Figure 1. Federated Learning Model in a MEC Network. 



### 2. Proposed CBCSBA Framework

The CBCSBA framework consists of solving the holistic subproblem and partial subproblem.
To solve the holistic subproblem, we aim to minimize the number of rounds required for the global model to converge. To solve the partial subproblem, we strive to reduce the costs of each round.

![image-20240527230143950](./assets/image-20240527230143950.png)

Figure 2. Framework for joint class-balanced client selection and bandwidth allocation.

### 3. Experiments

You can run through the experiment with the following code

```
python main.py --server proposed
```





