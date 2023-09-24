# 导入了TVM的relax模块中的一些函数和类，以及TVM的script模块中的relax别名。
from tvm.relax.dpl import PatternContext, is_const, is_op, rewrite_call, wildcard
from tvm.script import relax as R

# 定义了一个名为rewrite_attention的函数，接收一个参数f。
def rewrite_attention(f):
    # 使用wildcard()创建了三个通配符，分别赋值给Q、K和V。
    Q = wildcard()
    K = wildcard()
    V = wildcard()

    # 使用is_op()函数创建了三个操作模式，分别对应Q、K和V的维度重排操作，并将结果分别赋值给Q_BNSH、K_BNSH和V_BNSH。
    Q_BNSH = is_op("relax.permute_dims")(Q)
    K_BNSH = is_op("relax.permute_dims")(K)
    V_BNSH = is_op("relax.permute_dims")(V)

    # 使用is_op()函数创建了一个操作模式，对应K_BNSH的维度重排操作，并将结果赋值给K_BNSH_T。
    K_BNSH_T = is_op("relax.permute_dims")(K_BNSH)

    # 使用is_op()函数创建了一系列操作模式，对应矩阵乘法、除法、最大值、最小值、softmax以及另一个矩阵乘法操作。
    # 这些操作模式（Attention）根据之前定义的通配符和常数匹配不同的计算图节点。
    matmul1 = is_op("relax.matmul")(Q_BNSH, K_BNSH_T)
    divide = is_op("relax.divide")(matmul1, is_const())
    max = is_op("relax.maximum")(divide, is_const())
    min = is_op("relax.minimum")(max, wildcard())
    softmax = is_op("relax.nn.softmax")(is_op("relax.astype")(min))
    matmul2 = is_op("relax.matmul")(is_op("relax.astype")(softmax), V_BNSH)

    # 使用is_op()函数创建了一个操作模式，对应matmul2的维度重排操作，并将结果赋值给pattern。
    pattern = is_op("relax.permute_dims")(matmul2)

    # 定义了一个名为callback的回调函数，接收两个参数_和matchings。
    # 该回调函数使用R.nn.attention函数构建一个新的计算图节点，并使用matchings字典中的匹配结果来填充该节点的参数。
    def callback(_, matchings):
        return R.nn.attention(
            matchings[Q], matchings[K], matchings[V], causal_mask="BottomRight"
        )

    # 使用rewrite_call函数将pattern、callback和输入的计算图f传递给它，以便在计算图中应用模式匹配和重写。
    # 最后，将重写后的计算图返回。
    return rewrite_call(pattern, callback, f)
