import numpy as np

class SumTree(object):
    # SumTree由两个数组构成，分别为data数组以及tree数组；data数组存储每组经验，tree存储对应的td误差构成的树
    # 初始化data数组指针
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # SumTree的容量
        # 初始化SumTree数组（叶节点数比非叶节点数多1）
        self.tree = np.zeros(2 * capacity - 1)
        # 初始化data数组
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):  # 向树中添加新的经验
        tree_idx = self.data_pointer + self.capacity - 1  # 获得与data数组下标对应的Tree数组的下标
        self.data[self.data_pointer] = data  # 将新经验插入data数组
        self.update(tree_idx, p)  # 更新tree数组

        self.data_pointer += 1  # data数组下标后移
        if self.data_pointer >= self.capacity:  # 超出capacity的时候重置下标，自动处理溢出问题（将以前的替换掉）
            self.data_pointer = 0

    def update(self, tree_idx, p):  # 更新Tree数组存储的p值
        change = p - self.tree[tree_idx]  # 计算叶节点p值变化量
        self.tree[tree_idx] = p  # 更新叶节点存储值
        # 将误差变化量更新到整棵树上
        while tree_idx != 0:
            # //是整数除法，自动省略小数点后的内容
            tree_idx = (tree_idx - 1) // 2  # 循环获取叶节点对应的父节点下标
            self.tree[tree_idx] += change  # 更新父节点存储的值

    def get_leaf(self, v):  # 根据搜索值返回对应叶节点以及data
        parent_idx = 0  # 初始化父节点为根节点
        while True:
            cl_idx = 2 * parent_idx + 1  # 获得左子节点的下标
            cr_idx = cl_idx + 1  # 获得右子节点的下标
            if cl_idx >= len(self.tree):  # 如果左子节点下标都超过存储Tree数组长度
                leaf_idx = parent_idx  # 证明父节点已经是最底部的叶节点，此时终止搜索
                break
            else:       # 如果左子节点存在
                if v <= self.tree[cl_idx]:  # 如果搜索的值小于左子节点
                    parent_idx = cl_idx  # 向左下移动，将左子节点设置为父节点
                else:
                    v -= self.tree[cl_idx]  # 否则将搜索值减去左子节点的值，向右下移动，将右子节点设置为父节点
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1  # 根据找到的叶节点位置找到对应data位置
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):  # 返回p值总和
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # 设置ε防止出现0优先度值
    alpha = 0.6  # [0~1] 用来调整td误差绝对值与p值之间的相关性
    beta = 0.4  # [0~1] 参与计算经验样本在反向传播训练时的权重的计算
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # p值最大值

    def __init__(self, capacity):
        self.tree = SumTree(capacity)  # 初始化SumTree

    def store(self, transition):  # 将经验样本加入SumTree
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  # max_p存储叶节点中的最大的p值
        if max_p == 0:
            max_p = self.abs_err_upper  # 如果此时最大的p值为0（就是此时树中还没有存储节点）  默认设置p值为1
        self.tree.add(max_p, transition)  # 默认为新加入的经验设置最大的p值（提高新加入的经验的重要性权重）

    def sample(self, n):  # 根据给定的minibatch的length，从SumTree中抽取length个节点
        # b_idx用来存储这些节点在Tree数组的下标，b_memory存储这些节点代表的经验，ISWeights存储这些节点在梯度反向传播时对应的权重
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [], np.empty((n))
        pri_seg = self.tree.total_p / n       # 根据minibatch需要的节点数量，把total_p分成n份
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # ISWeights计算时使用的β值 随着使用次数增多逐渐增大直至1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # 叶节点中找到最小的P值 除以总P值得到最小概率
        if min_prob == 0:
            min_prob = 0.00001  # 如果最小P值为0 设置底限防止出现0
        for i in range(n):  # 对于total_p被分割的每个区间
            a, b = pri_seg * i, pri_seg * (i + 1)  # 获得当前区间的上下限
            v = np.random.uniform(a, b)  # 区间中取一个实数
            idx, p, data = self.tree.get_leaf(v)  # 根据实数取得一个叶节点
            prob = p / self.tree.total_p  # 根据叶节点p值计算概率
            ISWeights[i] = np.power(prob/min_prob, -self.beta)  # 根据概率计算权重  公式来源于论文  np.power()用来乘方
            # b_idx[i], b_memory[i, :] = idx, data  # 将叶节点对应下标以及经验存储
            b_idx[i] = idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):  # 根据Tree下标以及td误差绝对值更新SumTree
        abs_errors += self.epsilon  # 防止td误差绝对值为0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)  # 使用上限限制后的误差绝对值
        ps = np.power(clipped_errors, self.alpha)  # 根据clip后的误差绝对值计算p值 原论文alpha为1
        for ti, p in zip(tree_idx, ps):  # 更新SumTree
            self.tree.update(ti, p)