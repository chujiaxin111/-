本周工作内容：

1. 通读Angel文档，留下初步印象
   （https://github.com/Angel-ML/angel/blob/master/README_CN.md）
2. 了解参数服务器理念 
   观看相关论文讲解 Scaling Distributed Machine Learning with the Parameter Server, 李沐
   
   哔哩哔哩：https://www.bilibili.com/video/BV1YA4y197G8/?vd_source=68ad24c7a1fb094d5e218766fe2d8c85
   
   论文：https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf


收获：

* 我们提出一个参数服务器框架，用来针对分布式的机器学习任务，数据和任务都分布在一些任务节点或者工作节点上面。有一些服务器节点来维护全局共享的参数，这些参数通常会表示成一个稠密的或者稀疏的向量或者矩阵。这个框架去管理异步的数据通讯，支持一些灵活的一致性模型，弹性的可扩展性和持续的容灾。（李沐的论文摘要）

* 在工业界真实应用场景，容灾非常重要。

* 系统架构设计是一门艺术，不是一门科学也不是工程。



下周工作内容：

接下来的一至两周，计划深入了解机器学习、图计算的细致知识，重点放在模型相关知识，有一定了解后再阅读Angel文档，想必能比这周更加深刻。
