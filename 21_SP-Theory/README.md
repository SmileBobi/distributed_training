# 1. Tensor parallel

如下图所示，张量并行化了 Transformer 层中在训练期间占用大部分时间的部件，因此它在计算上是高效的。 但是，它保留了注意力和 MLP 模块之后的**layernorm以及dropout**，因此它们在张量并行组中被复制。 这些元素不需要大量的计算，但`需要大量的激活内存`**（缺点：Activition很大的话，在layernorm和dropout算子中会保留下来的，TP是解决不了的）**, 因为张量并行对他们无效。<br>

![images](./images/tensor-parallel.png)

# 2. Sequece parallel

我们注意到在 Transformer 层的**非张量并行区域（layernorm以及dropout）**中，操作**在序列维度上是独立的**。 这种特性允许我们在序列维度上对这些区域进行划分。 沿着序列维度进行划分减少了激活所需的内存。 这种额外的并行级别在TP前后通讯外引入了新的通信集合，它们将充当`序列和张量并行区域之间的转换器`。 这些额外的通信引入了开销，并会减慢训练速度。<br>

**一个nlp领域中`[B, Seq, Emb]`是三个维度, layernorm是在Embedding维度上去做取均值和方差, 与Sequece维度是相互解耦的，可以沿Sequece维度上做切分，分别进行layernorm以及dropout，最后再组合起来**

![images](./images/sequence-parallel.png)

# 3. mlp TP and SP special

**SP拆Sequece（在Sequece维度上做切分），单独做layernorm，做完后加一个 g（All gather） 通信算子，此时两个 Y 是一样的（这个时候已经进入到TP中了），这个时候按列切分（ $Y A_1^c$ , $Y A_2^c$ ）,分别进行Gelu得到绿色块的 $Z_1^h$ ， $Z_2^h$ ，蓝色的 $Z_1^h B_1^r$ 和 $Z_2^h B_2^r$ 是做行切分的，做完之后会有一个通信 $\bar{g}$ （Reduce Scatter）`(纯TP的情况下此时做的是All Reduce通信)`, 在Sequece维度上做Scatter，给它拆分开，再做dropout**

- **注意：All Reduce 相当于 All gather + Reduce Scatter**
- **这种组合方案与纯TP的通信量是一样的，但可以放在两张卡上进行计算，就节约了显存**
- **工程上一般都是用SP和TP的组合方案，SP不会单独使用，TP可能会单独使用**

![images](./images/mlp-tensor-sequence-parallel.png)


# 4 通讯开销

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;张量并行在单个正向和反向传播中需要四个全归约，而张量与序列并行在单个正向和反向传播中需要四个全聚合和四个归约散射。 乍一看，似乎张量与序列并行相比张量并行需要更多的通信。 然而，我们注意到环形全归约包含两个步骤：归约散射后跟全聚合。 因此，张量并行和张量与序列并行使用的通信带宽相同。 因此，序列并行不会引入任何通信开销。<br>

# 5 参考连接
- [论文连接-EN](https://arxiv.org/pdf/2205.05198)
- [论文连接-EN](https://yiyibooks.cn/arxiv/2205.05198v1/index.html)
