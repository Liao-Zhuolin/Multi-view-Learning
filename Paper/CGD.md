![在这里插入图片描述](https://img-blog.csdnimg.cn/63f33123b45f41dba45308832bf06800.png)

# Summary
 - 现有的多视图处理模型都是先进行表征学习，通过学习得到的表征得到统一的图，再利用该图进行谱聚类。本文考虑将特征通过kNN构图得到每个视图的图，再通过多视图融合迭代公式进行融合扩散。这样，信息是在多个视图中进行扩散的，因此可以学习得到多个视图之间的互补和公共信息。
# Problem Statement
 - 现有的基于表征学习的多视图处理方法存在诸多挑战，例如：模型庞大、计算量高、需要调参等。
 - 图扩散仅仅在单个视图上有应用，现在还没有人将图扩散应用到多个视图中。
# Method
 - 正则化图扩散流程，可以写成如下优化形式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701205055406.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dhbmRlYmVhdXRpZnVs,size_16,color_FFFFFF,t_70#pic_center)
 - 该问题的闭式解为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701205211324.png#pic_center)
 - 由上述闭式解推导出单个视图的扩散公式为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701205312427.png#pic_center)
 - 将单个视图的扩散公式扩展到多个视图上（这是作者所做的工作）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701205411633.png#pic_center)
 - 论文中的一个与众不用的地方，就是使用了新的构图方法，原始的图行归一化可能会导致优化不稳定，所以改用新的图处理方式。将自己与自己的相似度看作是0.5，然后与其他样本之间的相似度总和为0.5：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701205520805.png#pic_center)
 - 计算自适应的可变参数，用于权衡原始结构和扩散结构，这个主要是针对上述的多视图融合参数$\alpha$，这个参数权衡了当前视图和其他视图的比例，当$\alpha$较小时融合视图中当前视图的比重较大。论文中为了构造自适应的参数，使用了矩阵量连续点乘的方法，连乘的结果如下。根据连乘结果中非零元素的个数$\widetilde{N}$来定$\alpha$的大小。$\alpha=1-\widetilde{N}/N^2$：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021070120565624.png#pic_center)
 - 最终将扩散后的多个视图图结构直接求和作为最终的图结构，模型图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701205850410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dhbmRlYmVhdXRpZnVs,size_16,color_FFFFFF,t_70#pic_center)
 # Evaluation
 
 - 在七个聚类的指标上进行实验评估，多视图数据集如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210701210019660.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dhbmRlYmVhdXRpZnVs,size_16,color_FFFFFF,t_70#pic_center)
 # Conclusion
 - 论文将单个图的扩散公式扩展到多个视图上，取得了显著的效果。
# Notes
 - Graph based multi-view clustering has been paid great attention by exploring the neighborhood relationship among data points from multiple views.
 -  Extensive experiments on several benchmark datasets are conducted to demonstrate the effectiveness of the proposed method in terms of seven clustering evaluation metrics.
 - It is not uncommon that an object is usually described by multi-view features.
 - Multi-view clustering, which partitions these multi-view data into different groups by using the complementary information of multi-view feature sets to ensure that highly similar instances are divided into the same group, is an important branch of multi-view learning.
 - In general, most of previous multi-view clustering methods employ graph-based models since the similarity graph can characterize the data structure effectively.
 -  In biomedical research, both the chemical structure and chemical response in different cells can be used to represent a certain drug, while the sequence and gene expression values can represent a certain protein in different aspects.
 -  In general, most of previous multi-view clustering methods employ graph-based models since the similarity graph can characterize the data structure effectively.
 - 论文使用两个“玩具”数据集将训练过程可视化展示出来了，包括图结构和热力图。图结构能够被展示使得结果更加直观。
# References
 - Regularized diffusion process for visual retrieval.
 - Diffusion processes for retrieval revisited.
 - Mpgraph: multi-view penalised graph clustering for predicting drug-target interactions.
 - Late fusion incomplete multi-view clustering.
 - Parameter-free auto-weighted multiple graph learning: a framework for multi-view clustering and semi-supervised classification.
 -  Learning a joint affinity graph for multiview subspace clustering.
 -  Gmc: graph-based multi-view clustering.
 - Graph learning for multiview clustering.
 - Multiview consensus graph clustering.
