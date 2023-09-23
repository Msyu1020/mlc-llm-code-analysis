# pylint: disable=missing-docstring,invalid-name
from dataclasses import dataclass
from typing import List, Literal, Tuple

from tvm import relax, te, tir
from tvm.relax import Expr, op
from tvm.relax.testing import nn
from tvm.script import relax as R
from tvm.script import tir as T

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import ModuleList, Linear
from .param_manager import ParamManager

# Reference: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model_run.py

# @dataclass：这个装饰器用于指示RWKVConfig类是一个数据类。用于存储RWKVModel的配置信息。
@dataclass
class RWKVConfig:
    """The configuration class to store the configuration of a `RWKVModel`."""

    num_hidden_layers: int # 类中的一个属性，用于存储隐藏层的数量，类型为整数。
    vocab_size: int # 类中的一个属性，用于存储词汇表的大小，类型为整数。
    hidden_size: int # 类中的一个属性，用于存储隐藏层的大小，类型为整数。
    intermediate_size: int # 类中的一个属性，用于存储中间层的大小，类型为整数。
    rescale_every: int = 0 # 类中的一个属性，默认值为0，用于存储重新缩放的频率，类型为整数。
    layer_norm_epsilon: float = 1e-5 # 类中的一个属性，默认值为1e-5，用于存储层归一化的epsilon值，类型为浮点数。
    max_sequence_length: int = 1024 # 类中的一个属性，默认值为1024，用于存储最大序列长度，类型为整数。
    dtype: str = "float32" # 类中的一个属性，默认值为"float32"，用于存储数据类型，类型为字符串。

    def __init__(
        self,
        num_hidden_layers: int,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        rescale_every: int = 0,
        layer_norm_epsilon: float = 1e-5,
        context_length: int = 1024,
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rescale_every = rescale_every
        self.layer_norm_epsilon = layer_norm_epsilon
        self.max_sequence_length = context_length
        self.dtype = dtype
        self.kwargs = kwargs

# 用来索引RWKV的Attention和FFN部分分别存储的Tensor，同时这里索引的Tensor也是RWKV线性推理的状态或者叫Cache。
# python代码可以参考： https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/model.py#L858-L867
class State:
    ATT_X = 0
    ATT_A = 1
    ATT_B = 2
    ATT_P = 3
    FFN_X = 4

# 义了一个名为_load_state的函数，它接受一个名为state的参数，类型为Expr，一个名为hidden_size的参数，类型为整数，
# 一个名为dtype的参数，类型为字符串。函数的返回类型为Expr。
def _load_state(state: Expr, hidden_size: int, dtype: str) -> Expr:
    # Reuse `attention_kv_cache_view`
    # 将外部函数vm.builtin.attention_kv_cache_view赋值给变量f_load_cache。relax.extern是一个外部函数调用的语法，
    # 它指示编译器在编译时将该函数调用转换为相应的外部函数调用。
    f_load_cache = relax.extern("vm.builtin.attention_kv_cache_view")
    # 使用nn.emit方法生成一个表达式对象，该表达式表示对外部函数f_load_cache的调用。
    # 调用的参数是一个列表，包含state和R.shape([1, hidden_size])，以及sinfo_args参数指定的一个R.Tensor对象。
    cache = nn.emit(
        relax.Call(
            f_load_cache,
            [state, R.shape([1, hidden_size])],
            sinfo_args=[R.Tensor((1, hidden_size), dtype)],
        )
    )
    return cache

# 定义了一个名为_store_state的函数，它接受一个名为state的参数，类型为Expr，一个名为value的参数，类型为Expr。
def _store_state(state: Expr, value: Expr):
    # Reuse `attention_kv_cache_update`
    # 将外部函数vm.builtin.attention_kv_cache_update赋值给变量f_store_cache。
    # relax.extern是一个外部函数调用的语法，它指示编译器在编译时将该函数调用转换为相应的外部函数调用。
    f_store_cache = relax.extern("vm.builtin.attention_kv_cache_update")

    # 使用nn.emit方法生成一个表达式对象，该表达式表示对外部函数f_store_cache的调用。
    # 调用的参数是一个列表，包含state和value，以及sinfo_args参数指定的一个R.Object()对象。
    return nn.emit(
        relax.Call(
            f_store_cache,
            [state, value],
            sinfo_args=[R.Object()],
        )
    )


def is_one(x: tir.PrimExpr) -> bool:
    # 使用isinstance函数判断x是否为tir.IntImm类型，并且判断x.value是否等于1。
    return isinstance(x, tir.IntImm) and x.value == 1

# 定义了一个名为create_wkv_func的函数，它接受一个名为hidden_size的参数，
# 类型为整数，一个名为dtype的参数，类型为字符串，一个名为out_dtype的参数，类型为字符串。
def create_wkv_func(hidden_size: int, dtype: str, out_dtype: str):
    @T.prim_func
    def wkv_func(
        k: T.handle,
        v: T.handle,
        time_decay: T.handle,
        time_first: T.handle,
        saved_a: T.handle,
        saved_b: T.handle,
        saved_p: T.handle,
        wkv: T.handle,
        out_a: T.handle,
        out_b: T.handle,
        out_p: T.handle,
    ):
        # 设置TIR函数的属性。这里设置了三个属性，包括op_pattern、tir.noalias和tir.is_scheduled。
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        # 声明一个名为context_length的变量，类型为T.int64()，用于存储上下文长度。
        context_length = T.int64()
        # 创建一个名为K的匹配缓冲区，通过T.match_buffer方法匹配参数k的形状和数据类型。
        # K的形状在原始的ChatRWKV中为B，T，C，只不过这里B=1
        # 这里的k就是上面cuda kernel的_k
        K = T.match_buffer(k, (context_length, hidden_size), dtype=dtype)
        # 创建一个名为V的匹配缓冲区，通过T.match_buffer方法匹配参数v的形状和数据类型。
        # 这里的v就是上面cuda kernel的_v
        V = T.match_buffer(v, (context_length, hidden_size), dtype=dtype)
        # 创建一个名为TimeDecay的匹配缓冲区，通过T.match_buffer方法匹配参数time_decay的形状和数据类型。
        # 这里的TimeDecay就是上面的w
        TimeDecay = T.match_buffer(time_decay, (hidden_size,), dtype=dtype)
        # 创建一个名为TimeFirst的匹配缓冲区，通过T.match_buffer方法匹配参数time_first的形状和数据类型。
        # 这里的TimeFirst对应上面的u
        TimeFirst = T.match_buffer(time_first, (hidden_size,), dtype=dtype)
        # 对应kernel里面的_aa的上一个token的状态
        SavedA = T.match_buffer(saved_a, (1, hidden_size), dtype=dtype)
        # 对应kernel里面的_bb的上一个token的状态
        SavedB = T.match_buffer(saved_b, (1, hidden_size), dtype=dtype)
        # 对应kernel里面的_pp的上一个token的状态
        SavedP = T.match_buffer(saved_p, (1, hidden_size), dtype=dtype)
        # 对应_aa的当前token状态
        OutA = T.match_buffer(out_a, (1, hidden_size), dtype=dtype)
        # 对应_bb的当前token状态
        OutB = T.match_buffer(out_b, (1, hidden_size), dtype=dtype)
        # 对应_pp的当前token状态
        OutP = T.match_buffer(out_p, (1, hidden_size), dtype=dtype)

        # 对应kernel里面的p
        P = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        # 对应kernel里面的e1
        E1 = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        # 对应kernel里面的e2
        E2 = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        # 对应kernel里面的aa
        A_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        # 对应kernel里面的bb
        B_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")
        # 对应kernel里面的cc
        P_local = T.alloc_buffer((hidden_size,), dtype=dtype, scope="local")

        # 迭代hidden_size // 32次，使用T.thread_binding方法进行线程绑定，其中hidden_size // 32是块索引的范围。
        # 这里的线程块划分和rwkv kernel里面保持一致：即每个block 32个线程，一共((B=1)*C)/32个blcok
        for bx in T.thread_binding(hidden_size // 32, thread="blockIdx.x"):
            # 迭代32次，使用T.thread_binding方法进行线程绑定，其中32是线程索引的范围。
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                # 创建一个名为"init"的块，用于初始化局部变量。
                with T.block("init"):
                    # 对应 const int _state_offset = _b * C + _c;
                    vi = T.axis.S(hidden_size, bx * 32 + tx)
                    # 对应 float aa = _aa[_state_offset];
                    A_local[vi] = SavedA[0, vi]
                    # 对应 float bb = _bb[_state_offset];
                    B_local[vi] = SavedB[0, vi]
                    # 对应 float pp = _pp[_state_offset];
                    P_local[vi] = SavedP[0, vi]
                for j in range(context_length): # 对应 for (int i = 0; i < T; i++)
                    with T.block("main"):
                        # 对应 const int _state_offset = _b * C + _c;
                        vi = T.axis.S(hidden_size, bx * 32 + tx)
                        # vj 对应 _b * T; [vj, vi] = _b * T * C + _b * C + _c
                        # _b * T * C + _c = _offset
                        vj = T.axis.opaque(context_length, j)
                        # 对应 float p = max(pp, ww); float ww = u + kk; 
                        # const float kk = float(k[ii]); const int ii = i * C;
                        # const F *__restrict__ const k = _k + _offset;
                        P[vi] = T.max(P_local[vi], K[vj, vi] + TimeFirst[vi])
                        # 对应 float e1 = exp(pp - p);
                        E1[vi] = T.exp(P_local[vi] - P[vi])
                        # 对应 float e2 = exp(ww - p);
                        E2[vi] = T.exp(K[vj, vi] + TimeFirst[vi] - P[vi])

                        P[vi] = T.max(P_local[vi] + TimeDecay[vi], K[vj, vi])
                        E1[vi] = T.exp(P_local[vi] + TimeDecay[vi] - P[vi])
                        E2[vi] = T.exp(K[vj, vi] - P[vi])
                        A_local[vi] = E1[vi] * A_local[vi] + E2[vi] * V[vj, vi]
                        B_local[vi] = E1[vi] * B_local[vi] + E2[vi]
                        P_local[vi] = P[vi]

                with T.block("write_back"):
                    vi = T.axis.S(hidden_size, bx * 32 + tx) # 对应 
                    OutA[0, vi] = A_local[vi] # 对应 _aa[_state_offset] = aa;
                    OutB[0, vi] = B_local[vi] # 对应 _bb[_state_offset] = bb;
                    OutP[0, vi] = P_local[vi] # 对应 _pp[_state_offset] = pp;

    return wkv_func

# 定义了一个名为_te_concat_saved_x的函数，它接受两个参数saved_x和x，都是te.Tensor类型的张量。
# 使用TVM的te.compute函数计算一个新的张量，该张量的形状与x相同，元素根据条件判断进行选择。如果i等于0，
# 则选择saved_x[0, j]作为元素值，否则选择x[i - 1, j]作为元素值。其中i和j是迭代变量。
def _te_concat_saved_x(saved_x: te.Tensor, x: te.Tensor):
    return te.compute(
        x.shape,
        lambda i, j: tir.if_then_else(i == 0, saved_x[0, j], x[i - 1, j]),
    )

# 定义了一个名为_te_get_last_x的函数，它接受一个参数x，是一个te.Tensor类型的张量。
# a. seq_len, hidden_size = x.shape：获取x张量的形状，其中seq_len表示序列长度，hidden_size表示隐藏大小。
# b. return te.compute(...)：使用TVM的te.compute函数计算一个新的张量，该张量的形状为(1, hidden_size)，
# 元素值为x[seq_len - 1, j]，其中j是迭代变量。
def _te_get_last_x(x: te.Tensor):
    seq_len, hidden_size = x.shape
    return te.compute((1, hidden_size), lambda _, j: x[seq_len - 1, j])

# 定义了一个名为RWKV_Embedding的PyTorch模块。
class RWKV_Embedding(nn.Module):
    # 定义了RWKV_Embedding类的构造函数，接受三个参数num_embeddings、embedding_dim和dtype。
    def __init__(self, num_embeddings, embedding_dim, dtype):
        self.num_embeddings = num_embeddings # 将num_embeddings赋值给类成员变量self.num_embeddings。
        self.embedding_dim = embedding_dim # 将embedding_dim赋值给类成员变量self.embedding_dim。
        # 创建一个名为weight的Parameter，形状为(num_embeddings, embedding_dim)，
        # 数据类型为dtype，并将其赋值给类成员变量self.weight。
        self.weight = nn.Parameter(
            (num_embeddings, embedding_dim), dtype=dtype, name="weight"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        # 调用op.reshape函数将输入张量x进行reshape，将其展平为一维张量，并将结果重新赋值给x。
        # nn.emit是将一个relax.Expr表达式转化为relax.Var变量，并保存该变量。
        x = nn.emit(op.reshape(x, shape=[-1]))
        # 使用op.take操作从self.weight中按照索引x提取对应的嵌入向量，并返回结果。这里的axis=0表示在第一个维度上进行索引操作。
        return nn.emit(op.take(self.weight, x, axis=0))

# 这段代码定义了一个名为RWKV_LayerNorm的PyTorch模块，它实现了一个Layer Normalization层。
class RWKV_LayerNorm(nn.Module):
    # 定义了RWKV_LayerNorm类的构造函数，接受四个参数intermediate_size、dtype、eps和name_prefix。
    def __init__(self, intermediate_size, dtype, eps=1e-5, name_prefix=""):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            (intermediate_size,), dtype=dtype, name=f"{name_prefix}_ln_weight"
        )
        self.bias = nn.Parameter(
            (intermediate_size,), dtype=dtype, name=f"{name_prefix}_ln_bias"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        # 使用op.nn.layer_norm操作对输入张量x进行Layer Normalization，其中使用Parameter self.weight作为缩放参数（gamma），
        # 使用可学习参数self.bias作为偏移参数（beta），在最后一个维度（axes=-1）上进行标准化操作，
        # 并设置小数值修正项为self.eps。将标准化后的结果重新赋值给x。
        x = nn.emit(
            op.nn.layer_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                axes=-1,
                epsilon=self.eps,
            )
        )
        return x


# 这段代码定义了一个名为RWKV_FFN的PyTorch模块，它实现了Feed-Forward Network（FFN）。
class RWKV_FFN(nn.Module):
    # 定义了RWKV_FFN类的构造函数，接受两个参数RWKVConfig和index。
    def __init__(self, config: RWKVConfig, index: int) -> None:
        super().__init__()
        # 将config.hidden_size赋值给类成员变量self.hidden_size，表示隐藏大小。
        self.hidden_size = config.hidden_size
        # 将config.dtype赋值给类成员变量self.dtype，表示数据类型。
        self.dtype = config.dtype
        # 将index赋值给类成员变
        self.index = index
        # 建一个名为time_mix_key的可学习参数，形状为(self.hidden_size,)，
        # 数据类型为config.dtype，命名为"ffn_{index}_time_mix_k"，并将其赋值给类成员变量self.time_mix_key。
        self.time_mix_key = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"ffn_{index}_time_mix_k"
        )
        # 创建一个名为time_mix_receptance的可学习参数，形状为(self.hidden_size,)，数据类型为config.dtype，
        # 命名为"ffn_{index}_time_mix_r"，并将其赋值给类成员变量self.time_mix_receptance。
        self.time_mix_receptance = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"ffn_{index}_time_mix_r"
        )
        # 创建一个线性层，输入大小为self.hidden_size，输出大小为config.intermediate_size，
        # 数据类型为config.dtype，没有偏置项，并将其赋值给类成员变量self.key。
        self.key = Linear(
            self.hidden_size, config.intermediate_size, dtype=config.dtype, bias=False
        )
        # 创建一个线性层，输入大小为self.hidden_size，输出大小为self.hidden_size，数据类型为config.dtype，
        # 没有偏置项，并将其赋值给类成员变量self.receptance。
        self.receptance = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.value = Linear(
            config.intermediate_size, self.hidden_size, dtype=config.dtype, bias=False
        )

    def forward(self, x: Expr, state: Expr) -> Expr:
        # 计算偏移量，用于在state中获取对应的保存状态。
        offset = self.index * 5 + State.FFN_X
        # 获取x的shape[0]表示上下文长度。
        context_length = x.struct_info.shape[0]
        # 获取隐藏层大小。
        hidden_size = self.hidden_size

        # 调用_load_state函数从state中加载保存的状态state[offset]，并将结果赋值给saved_x。
        saved_x = _load_state(state[offset], hidden_size, self.dtype)
        # 如果上下文长度不为1，则执行下面的操作。
        if not is_one(context_length):
            # 调用nn.emit_te函数，将saved_x和x作为参数传递给
            # _te_concat_saved_x函数进行计算，并将结果重新赋值给saved_x。
            # 类似于transformer 里面的KV Cache的，但是这里的concat是纬度不变的
            # 对应 sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) 这行代码
            saved_x = nn.emit_te(_te_concat_saved_x, saved_x, x)
        # 创建一个全为1的张量，形状为(hidden_size,)，数据类型为self.dtype，并将其赋值给ones。
        ones = nn.emit(relax.op.ones((hidden_size,), self.dtype))
        # 计算xk，根据时间混合参数self.time_mix_key和保存的状态saved_x，使用加权求和的方式得到。
        # 其中，x和saved_x分别乘以self.time_mix_key和(ones - self.time_mix_key)，然后相加。将计算结果赋值给xk。
        # 对应 kx = xx * k_mix + sx * (1 - k_mix) 这行代码
        xk = nn.emit(x * self.time_mix_key + saved_x * (ones - self.time_mix_key))
        # 计算xr，根据时间混合参数self.time_mix_receptance和保存的状态saved_x，使用加权求和的方式得到。
        # 其中，x和saved_x分别乘以self.time_mix_receptance和(ones - self.time_mix_receptance)，然后相加。
        # 将计算结果赋值给xr。
        # 对应 rx = xx * r_mix + sx * (1 - r_mix)
        xr = nn.emit(
            x * self.time_mix_receptance + saved_x * (ones - self.time_mix_receptance)
        )
        # # 如果上下文长度不为1，则执行下面的操作。
        if not is_one(context_length):
            # 调用nn.emit_te函数，使用_te_get_last_x函数从x中获取最后一个token对应的tensor，并将结果重新赋值给x。
            # 对应 xx[-1,:]
            x = nn.emit_te(_te_get_last_x, x)
        # 断言x的结构信息（shape）的第一个维度为1。
        assert is_one(x.struct_info.shape[0])
        # 调用_store_state函数，将x保存到state[offset]中，并将结果重新赋值给saved_x。
        # 对应：https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/model.py#L921
        saved_x = _store_state(state[offset], x)

        # 将xr作为输入，经过sigmoid激活函数计算得到r。对应：r = torch.sigmoid(gemm(rx, rw))
        r = nn.emit(op.sigmoid(self.receptance(xr)))
        # 对应 vx = torch.square(torch.relu(gemm(kx, kw)))
        xv = nn.emit(op.square(op.nn.relu(self.key(xk))))

        return nn.emit(r * self.value(xv)), [saved_x]

# 实现RWKV Attention，对应 https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/model.py#L479
class RWKV_Attention(nn.Module):
    # 初始化函数，接受一个config对象和一个整数index作为参数。其中config是一个RWKVConfig类型的对象，index表示当前层的索引。
    def __init__(self, config: RWKVConfig, index: int) -> None:
        super().__init__()
        self.index = index
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        # 创建一些可学习的参数，如time_decay、time_first、time_mix_key等，这些参数会在模型的前向传播中使用。
        self.time_decay = nn.Parameter(
            (self.hidden_size,), dtype="float32", name=f"att_{index}_time_decay"
        )
        self.time_first = nn.Parameter(
            (self.hidden_size,), dtype="float32", name=f"att_{index}_time_first"
        )
        self.time_mix_key = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"att_{index}_time_mix_k"
        )
        self.time_mix_value = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"att_{index}_time_mix_v"
        )
        self.time_mix_receptance = nn.Parameter(
            (self.hidden_size,), dtype=config.dtype, name=f"att_{index}_time_mix_r"
        )
        # 前向传播用到的线性层
        self.key = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.value = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.receptance = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )
        self.output = Linear(
            self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False
        )

    # 前向传播函数，接受输入张量x和状态张量state作为参数，并返回输出张量
    def forward(self, x: Expr, state: Expr) -> Expr:
        # Load current state
        # 定义了一些局部变量，如ones、index、hidden_size、context_length等。
        ones = nn.emit(relax.op.ones((self.hidden_size,), self.dtype))
        index = self.index
        hidden_size = self.hidden_size
        context_length = x.struct_info.shape[0]
        bb = relax.BlockBuilder.current()

        # _load_state函数从state中加载保存的状态，赋值给saved_a、saved_b、saved_p和saved_x。
        saved_a = _load_state(state[index * 5 + State.ATT_A], hidden_size, "float32")
        saved_b = _load_state(state[index * 5 + State.ATT_B], hidden_size, "float32")
        saved_p = _load_state(state[index * 5 + State.ATT_P], hidden_size, "float32")
        saved_x = _load_state(state[index * 5 + State.ATT_X], hidden_size, self.dtype)
        
        # 调用nn.emit_te函数，将saved_x和x作为参数传递给
        # _te_concat_saved_x函数进行计算，并将结果重新赋值给saved_x。
        # 对应 sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        if not is_one(context_length):
            saved_x = nn.emit_te(_te_concat_saved_x, saved_x, x)

        # 对应 kx = xx * k_mix + sx * (1 - k_mix)
        xk = nn.emit(x * self.time_mix_key + saved_x * (ones - self.time_mix_key))
        # 对应 vx = xx * v_mix + sx * (1 - v_mix)
        xv = nn.emit(x * self.time_mix_value + saved_x * (ones - self.time_mix_value))
        # 对应 rx = xx * r_mix + sx * (1 - r_mix)
        xr = nn.emit(
            x * self.time_mix_receptance + saved_x * (ones - self.time_mix_receptance)
        )

        # 对应 r = torch.sigmoid(gemm(rx, rw))
        r = nn.emit(op.sigmoid(self.receptance(xr)))
        # 对应 k = gemm(kx, kw, output_dtype=torch.float32)
        k = nn.emit(op.astype(self.key(xk), "float32"))
        # 对应 v = gemm(vx, vw, output_dtype=torch.float32)
        v = nn.emit(op.astype(self.value(xv), "float32"))

        # 这部分对应 y, aa, bb, pp = cuda_wkv(T, aa.shape[0], t_decay, t_first, k, v, aa, bb, pp)
        # 这里的 create_wkv_func 在上面已经解析了
        gv = bb.add_func(create_wkv_func(hidden_size, "float32", self.dtype), "wkv")
        ret = nn.emit(
            relax.call_tir(
                gv,
                [k, v, self.time_decay, self.time_first, saved_a, saved_b, saved_p],
                [
                    R.Tensor((context_length, hidden_size), self.dtype), # 对应wkv
                    R.Tensor((1, hidden_size), "float32"), # 对应out_a
                    R.Tensor((1, hidden_size), "float32"), # 对应out_b
                    R.Tensor((1, hidden_size), "float32"), # 对应out_p
                ],
            )
        )
        if not is_one(context_length):
            # 对应 xx[-1,:]
            x = nn.emit_te(_te_get_last_x, x)

        assert is_one(x.struct_info.shape[0])
        saved_x = _store_state(state[self.index * 5 + State.ATT_X], x)
        saved_a = _store_state(state[self.index * 5 + State.ATT_A], ret[1])
        saved_b = _store_state(state[self.index * 5 + State.ATT_B], ret[2])
        saved_p = _store_state(state[self.index * 5 + State.ATT_P], ret[3])

        # 需要注意一下，python代码里面的 return x + out, xx[-1,:], aa, bb, pp
        # 这里的 x + out被放在attention外面做了，因为这里的x已经是被修改之后好的结果而不是原始的x
        return nn.emit(self.output(r * ret[0])), [
            saved_x,
            saved_a,
            saved_b,
            saved_p,
        ]


class RWKVLayer(nn.Module):
    # 初始化函数，接受一个config对象和一个整数index作为参数。其中config是一个RWKVConfig类型的对象，index表示层的索引。
    def __init__(self, config: RWKVConfig, index: int) -> None:
        super().__init__()
        # 如果index为0，创建一个RWKV_LayerNorm对象pre_ln，用于对输入进行Layer Normalization操作。
        if index == 0:
            self.pre_ln = RWKV_LayerNorm(
                config.hidden_size,
                config.dtype,
                eps=config.layer_norm_epsilon,
                name_prefix="pre_ln",
            )
        # 创建两个RWKV_LayerNorm对象，分别命名为ln1和ln2，
        # 用于对注意力机制和前馈神经网络的输出进行Layer Normalization操作。
        self.ln1 = RWKV_LayerNorm(
            config.hidden_size,
            config.dtype,
            eps=config.layer_norm_epsilon,
            name_prefix=f"att_{index}",
        )
        self.ln2 = RWKV_LayerNorm(
            config.hidden_size,
            config.dtype,
            eps=config.layer_norm_epsilon,
            name_prefix=f"ffn_{index}",
        )
        # 创建一个RWKV_Attention对象attention，用于实现注意力机制。
        self.attention = RWKV_Attention(config, index)
        # 创建一个RWKV_FFN对象feed_forward，用于实现前馈神经网络。
        self.feed_forward = RWKV_FFN(config, index)
        self.rescale_every = config.rescale_every
        self.dtype = config.dtype
        self.index = index

    # 前向传播函数，接受输入张量x和状态张量state作为参数，并返回输出张量和更新后的状态列表。
    def forward(self, x: Expr, state: Expr) -> Tuple[Expr, List[Expr]]:
        # 如果index为0，则将输入张量x传入pre_ln进行Layer Normalization操作。
        if self.index == 0:
            x = self.pre_ln(x)
        # 将经过ln1的输入张量x和状态张量state传入attention进行计算，得到注意力机制的输出att和更新后的状态列表att_state。
        att, att_state = self.attention(self.ln1(x), state)
        # 将输入张量x和注意力机制的输出att相加，并将结果赋值给x。
        x = nn.emit(x + att)
        # 将经过ln2的输入张量x和状态张量state传入feed_forward进行计算，得到前馈神经网络的输出ffn和更新后的状态列表ffn_state。
        ffn, ffn_state = self.feed_forward(self.ln2(x), state)
        # 将输入张量x和前馈神经网络的输出ffn相加，并将结果赋值给x。
        x = nn.emit(x + ffn)
        # 如果满足self.rescale_every > 0且(self.index + 1) % self.rescale_every == 0，则对输入张量x进行缩放操作。
        if self.rescale_every > 0 and (self.index + 1) % self.rescale_every == 0:
            x = nn.emit(x / relax.const(2, dtype=self.dtype))
        # 返回输出张量x和注意力机制和前馈神经网络的更新后的状态列表的拼接。
        return x, att_state + ffn_state

# 该代码是一个自定义的PyTorch模型类RWKVModel，继承自nn.Module
class RWKVModel(nn.Module):
    # 初始化函数，接受一个config对象作为参数。其中config是一个RWKVConfig类型的对象。
    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        # 创建一个RWKV_Embedding对象embeddings，用于实现输入的嵌入操作。
        self.embeddings = RWKV_Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )
        # 创建一个ModuleList对象blocks，其中包含了config.num_hidden_layers个RWKVLayer对象，
        # 每个对象的索引从0到config.num_hidden_layers-1。
        self.blocks = ModuleList(
            [RWKVLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # 创建一个RWKV_LayerNorm对象ln_out，用于对输出进行Layer Normalization操作。
        self.ln_out = RWKV_LayerNorm(
            config.hidden_size,
            config.dtype,
            eps=config.layer_norm_epsilon,
            name_prefix="out_ln",
        )
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype

    # 前向传播函数，接受输入张量input_ids和状态张量state作为参数，并返回输出张量和更新后的状态列表。
    def forward(self, input_ids: Expr, state: Expr) -> Tuple[Expr, List[Expr]]:
        # 将输入张量input_ids传入embeddings进行嵌入操作，得到隐藏状态张量hidden_states。
        hidden_states = self.embeddings(input_ids)
        # 创建一个空列表states，用于存储每个RWKVLayer对象的更新后的状态列表。
        states = []
        # 遍历blocks中的每个RWKVLayer对象，将隐藏状态张量hidden_states和状态张量state传入
        # 每个RWKVLayer对象的前向传播函数进行计算，得到更新后的隐藏状态张量和更新后的状态列表，
        # 并将更新后的状态列表添加到states中。
        for _, layer in enumerate(self.blocks):
            hidden_states, layer_states = layer(hidden_states, state)
            states += layer_states
        # 获取隐藏状态张量的上下文长度context_length。
        context_length = hidden_states.struct_info.shape[0]
        # 如果context_length不为1，则调用_te_get_last_x函数获取最后一个token对应的张量。
        if not is_one(context_length):
            hidden_states = nn.emit_te(_te_get_last_x, hidden_states)
        # 将隐藏状态张量传入ln_out进行Layer Normalization操作。
        hidden_states = self.ln_out(hidden_states)
        # 返回输出隐藏状态张量和所有RWKVLayer对象的更新后的状态列表。
        return hidden_states, states

# 该代码是一个自定义的PyTorch模型类RWKVForCausalLM，继承自nn.Module。
class RWKVForCausalLM(nn.Module):
    # 初始化函数，接受一个config对象作为参数。其中config是一个RWKVConfig类型的对象。
    def __init__(self, config: RWKVConfig):
        # 创建一个RWKVModel对象rwkv，用于实现序列模型的计算。
        self.rwkv = RWKVModel(config)
        # 创建一个Linear对象head，用于将隐藏状态映射到词汇表大小的输出空间。
        self.head = Linear(
            config.hidden_size, config.vocab_size, dtype=config.dtype, bias=False
        )
        self.vocab_size = config.vocab_size
        ############ End ############

    # 前向传播函数，接受输入张量input_ids和状态张量state作为参数，并返回预测的logits和更新后的kv cache。
    def forward(
        self,
        input_ids: relax.Expr,
        state: relax.Expr,
    ):
        # 将输入张量input_ids和状态张量state传入rwkv对象的前向传播函数进行计算，
        # 得到更新后的隐藏状态张量hidden_states和key-value缓存key_value_cache。
        hidden_states, key_value_cache = self.rwkv(input_ids, state)
        # 将隐藏状态张量hidden_states传入head进行线性映射操作，得到logits。
        logits = nn.emit(self.head(hidden_states))
        # 对logits进行形状重塑，将其reshape为形状为(1, 1, self.vocab_size)的张量。
        logits = nn.emit(op.reshape(logits, (1, 1, self.vocab_size)))
        # 如果logits的数据类型不是float32，则将其转换为float32类型。
        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, key_value_cache

# 该代码定义了一个函数get_param_quant_kind，用于根据参数名称和参数信息确定参数的量化类型。
def get_param_quant_kind(
    name: str, param_info: relax.TensorStructInfo
) -> ParamQuantKind:
    # 如果参数名称以"embeddings.weight"结尾，返回ParamQuantKind.embedding_table表示该参数是嵌入表的权重。
    if name.endswith("embeddings.weight"):
        return ParamQuantKind.embedding_table
    # 如果参数名称为"head.weight"，返回ParamQuantKind.final_fc_weight表示该参数是最后一个全连接层的权重。
    elif name == "head.weight":
        return ParamQuantKind.final_fc_weight
    # 如果参数的维度为2且名称以".weight"结尾，返回ParamQuantKind.linear_weight表示该参数是线性层的权重。
    elif param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others

# 函数接受一个relax.BlockBuilder对象bb、一个ParamManager对象param_manager、一个RWKVConfig对象config、
# 一个QuantizationScheme对象quant_scheme和一个字符串类型的函数名称func_name（默认为"prefill"或"decode"）
def create_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: RWKVConfig,
    quant_scheme: QuantizationScheme,
    func_name=Literal["prefill", "decode"],
):
    # 如果函数名称不是"prefill"或"decode"，则抛出ValueError异常。
    if func_name not in ["prefill", "decode"]:
        raise ValueError(f"func_name must be 'prefill' or 'decode', got {func_name}")
    # 根据函数名称确定序列的长度seq_len，如果函数名称为"decode"，则将序列长度设为1，否则设为tir.Var("n", "int64")。
    seq_len = 1 if func_name == "decode" else tir.Var("n", "int64")

    # 在BlockBuilder的function上下文中创建函数func_name。
    with bb.function(func_name):
        # 创建一个RWKVForCausalLM模型对象model。
        model = RWKVForCausalLM(config)
        # 调用param_manager的register_params方法注册模型参数。
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        # 创建一个输入占位符input_ids，形状为(1, seq_len)，数据类型为"int32"。
        input_ids = nn.Placeholder((1, seq_len), dtype="int32", name="input_ids")
        # Placeholder for compatibility to LLAMA
        # 创建一个占位符all_seq_len_shape，用于兼容LLAMA。
        all_seq_len_shape = relax.Var("place_holder", R.Object())
        # 创建一个变量state，其值为包含多个R.Object()的元组，长度为config.num_hidden_layers * 5。
        state = relax.Var("state", R.Tuple([R.Object()] * config.num_hidden_layers * 5))
        with bb.dataflow():
            # 调用model的前向传播函数，将input_ids和state作为输入，得到输出logits和状态列表states。
            logits, states = model(input_ids, state)
            # 将input_ids、all_seq_len_shape、state和模型的参数列表作为参数列表params。
            params = [
                input_ids,
                all_seq_len_shape,
                state,
            ] + model.parameters()

            # 使用bb.emit_output将(logits, relax.Tuple(states))作为输出。
            gv = bb.emit_output((logits, relax.Tuple(states)))
        # 使用bb.emit_func_output将输出和参数列表params作为函数的输出。
        bb.emit_func_output(gv, params)

    # 获取构建好的模块mod和global function变量gv。
    mod = bb.get()
    gv = mod.get_global_var(func_name)
    # 根据函数名称更新函数的属性，包括输入数量和tir_var_upper_bound（如果函数名称为"prefill"）。
    f = mod[gv].with_attr("num_input", 3)
    if func_name == "prefill":
        f = f.with_attr("tir_var_upper_bound", {"n": config.max_sequence_length})
    bb.update_func(gv, f)


def create_kv_cache_func(bb: relax.BlockBuilder, config: RWKVConfig) -> None:
    """NOTE: It's not typical kv-cache, but try to reuse the logic for the quick hack."""
    init_shape = relax.ShapeExpr((1, config.hidden_size))
    with bb.function("create_kv_cache", []):
        with bb.dataflow():
            input_dtype_zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            fp32_zeros = bb.emit(relax.op.zeros(init_shape, "float32"))
            fp32_neg_inf = bb.emit(fp32_zeros - relax.const(1e30, "float32"))
            caches = []
            f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
            conf = [
                ("att_x", input_dtype_zeros),
                ("att_a", fp32_zeros),
                ("att_b", fp32_zeros),
                ("att_p", fp32_neg_inf),
                ("ffn_x", input_dtype_zeros),
            ]
            for i in range(config.num_hidden_layers):
                for name, init_value in conf:
                    caches.append(
                        bb.emit(
                            relax.Call(
                                f_kv_cache_create,
                                [init_value, init_shape, relax.PrimValue(1)],
                                sinfo_args=[R.Object()],
                            ),
                            name_hint=f"{name}_state_{i}",
                        )
                    )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def create_kv_cache_reset_func(bb: relax.BlockBuilder, config: RWKVConfig) -> None:
    state = relax.Var("state", R.Tuple([R.Object()] * config.num_hidden_layers * 5))
    init_shape = relax.ShapeExpr((1, config.hidden_size))
    with bb.function("reset_kv_cache", [state]):
        with bb.dataflow():
            input_dtype_zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            fp32_zeros = bb.emit(relax.op.zeros(init_shape, "float32"))
            fp32_neg_inf = bb.emit(fp32_zeros - relax.const(1e30, "float32"))
            caches = []
            for i in range(config.num_hidden_layers):
                caches.append(
                    _store_state(state[i * 5 + State.ATT_X], input_dtype_zeros)
                )
                caches.append(_store_state(state[i * 5 + State.ATT_B], fp32_zeros))
                caches.append(_store_state(state[i * 5 + State.ATT_A], fp32_zeros))
                caches.append(_store_state(state[i * 5 + State.ATT_P], fp32_neg_inf))
                caches.append(
                    _store_state(state[i * 5 + State.FFN_X], input_dtype_zeros)
                )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def create_softmax_func(bb: relax.BlockBuilder, config: RWKVConfig) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder(
            (1, 1, config.vocab_size), dtype="float32", name="logits"
        )
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])

# 定义了一个名为get_model的函数，接受两个参数：args和hf_config。
def get_model(args, hf_config):
    # 从args中获取模型名称、最大序列长度和模型数据类型。
    model_name = args.model
    max_seq_len = args.max_seq_len
    dtype = args.quantization.model_dtype

    # 检查模型名称是否以"rwkv-"开头，如果不是，则抛出ValueError异常。
    if not model_name.lower().startswith("rwkv-"):
        raise ValueError(f"Unsupported model name: {model_name}")

    # 使用hf_config和dtype创建一个RWKVConfig配置对象config。
    config = RWKVConfig(**hf_config, dtype=dtype)
    # 如果指定了最大序列长度max_seq_len，则将config的max_sequence_length属性设置为max_seq_len。
    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len

    # 创建一个ParamManager对象param_manager用于管理模型参数，以及一个relax.BlockBuilder对象bb用于构建计算图。
    param_manager = ParamManager()
    bb = relax.BlockBuilder()
    # 调用一系列create_func函数，向bb中添加计算图的构建指令，
    # 包括"prefill"和"decode"两个函数、KV Cache函数、softmax函数和元数据函数。
    create_func(bb, param_manager, config, args.quantization, "prefill")
    create_func(bb, param_manager, config, args.quantization, "decode")
    create_kv_cache_func(bb, config)
    create_softmax_func(bb, config)
    create_metadata_func(
        bb,
        model_name=model_name,
        # RNN model do not have window size limit
        max_window_size=-1,
        stop_tokens=[0],
        add_prefix_space=False,
    )
    create_kv_cache_reset_func(bb, config)
    # 通过调用bb的get方法获取构建好的模块mod。
    mod = bb.get()

    # 如果args.build_model_only为True，则直接返回模块mod、参数管理器param_manager、None和配置config。
    if args.build_model_only:
        return mod, param_manager, None, config

    # 定义一个名为f_convert_pname_fwd的函数，用于转换前向参数名称。
    def f_convert_pname_fwd(pname: str) -> List[str]:
        if (
            "key_weight" in pname
            or "value_weight" in pname
            or "receptance_weight" in pname
            or "output_weight" in pname
            or "head_weight" in pname
        ):
            return [pname.replace("_weight", ".weight")]
        else:
            return [pname]
    
    # 定义一个名为f_convert_param_bkwd的函数，用于转换反向参数。
    def f_convert_param_bkwd(torch_pname: str, torch_param):
        # torch_param: numpy.ndarray
        import numpy as np  # pylint: disable=import-outside-toplevel

        # rescale_every
        if config.rescale_every > 0 and "blocks." in torch_pname:
            # based-on the assumption that the layer id is the second element in torch_pname
            layer_id = int(torch_pname.split(".")[2])
            if (
                "attention.output.weight" in torch_pname
                or "feed_forward.value.weight" in torch_pname
            ):
                torch_param = torch_param / (2 ** (layer_id // config.rescale_every))

        # reshape
        if "time_" in torch_pname:
            torch_param = torch_param.squeeze()

        # convert dtype
        if "time_decay" in torch_pname:  # need fp32 for this
            return [(torch_pname, -np.exp(torch_param.astype("float32")))]
        elif "time_first" in torch_pname:
            return [(torch_pname, torch_param.astype("float32"))]
        else:
            return [(torch_pname, torch_param.astype(config.dtype))]

    # 调用param_manager的set_param_loading_func方法，设置参数加载函数。
    param_manager.set_param_loading_func(
        args.model_path, args.use_safetensors, f_convert_pname_fwd, f_convert_param_bkwd
    )
    # 返回模块mod、参数管理器param_manager、长度为参数数量的None列表和配置config。
    return mod, param_manager, [None] * len(param_manager.param_names), config
