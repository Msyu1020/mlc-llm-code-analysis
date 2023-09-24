import tvm
from tvm import IRModule, relax, te, tir
from tvm.relax.dpl.pattern import is_op, wildcard


# 这段代码定义了一个名为TransposeMatmulCodeGenerator的类，该类继承自relax.PyExprMutator。
# 通过@relax.expr_functor.mutator装饰器将该类声明为一个表达式重写器。
@relax.expr_functor.mutator
class TransposeMatmulCodeGenerator(relax.PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)

    @staticmethod
    def pattern():
        # 定义了静态方法pattern()，该方法返回一个描述模式的元组。
        # 通过使用通配符(wildcard())和操作模式(is_op())来匹配计算图中的特定模式。
        # 在这个例子中，模式匹配了一个矩阵乘法操作中矩阵w的维度重排操作，并将匹配的结果保存在字典annotations中。
        w = wildcard()
        x = wildcard()
        wT = is_op("relax.permute_dims")(w)
        o = is_op("relax.matmul")(x, wT)
        annotations = {"o": o, "w": w, "x": x, "wT": wT}

        # 定义了内部函数_check()，用于检查模式匹配的结果是否满足特定的条件。
        # 在这个例子中，检查了维度重排操作的维度数和轴的顺序是否正确。
        def _check(context: relax.transform.PatternCheckContext) -> bool:
            transpose_call = context.annotated_expr["wT"]
            ndim = transpose_call.args[0].struct_info.ndim
            if ndim == -1:
                return False
            if ndim == 2 and transpose_call.attrs.axes is None:
                return True
            axes = list(range(ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return list(transpose_call.attrs.axes) == axes

        # 将匹配的计算图节点、注解和检查函数作为元组返回。
        return o, annotations, _check

    # 重写了父类的visit_call_()方法，用于处理特定类型的计算图节点。
    def visit_call_(self, call: relax.Call) -> relax.Expr:
        # 定义了一个变量out_dtype，用于保存输出的数据类型。
        out_dtype = None

        # 定义了一个内部函数te_transposed_matmul()，该函数实现了矩阵乘法的计算逻辑。
        def te_transposed_matmul(a: te.Tensor, b: te.Tensor) -> te.Tensor:
            nonlocal out_dtype
            # 将输入张量 a 和 b 的形状转换为列表形式，分别保存在变量 a_shape 和 b_shape 中。
            a_shape = list(a.shape)
            b_shape = list(b.shape)
            # 定义了两个布尔变量 a_prepended 和 b_appended，用于标记是否在相应的形状的前面或后面添加了维度。
            a_prepended = False
            b_appended = False
            # 如果输入张量 a 的形状为一维，则在其前面添加一个维度，将其形状修改为 (1, original_shape)。
            # 同样地，如果输入张量 b 的形状为一维，则在其后面添加一个维度，将其形状修改为 (original_shape, 1)。
            if len(a_shape) == 1:
                a_prepended = True
                a_shape.insert(0, 1)
            if len(b_shape) == 1:
                b_appended = True
                b_shape.append(1)

            # 比较 a_shape 和 b_shape 的长度，将结果保存在布尔变量 is_a_larger 中。
            # offset 表示两个形状长度之差，用于后续处理。
            is_a_larger = len(a_shape) > len(b_shape)
            offset = (
                len(a_shape) - len(b_shape)
                if is_a_larger
                else len(b_shape) - len(a_shape)
            )

            # 创建两个 relax.Var 对象 a_relax 和 bT_relax，用于表示张量 a 和转置后的张量 bT 的结构信息。
            # a_relax 的形状和 a 的形状相同，bT_relax 的形状是 b 的形状经过维度互换后的结果。
            a_relax = relax.Var("a", relax.TensorStructInfo(a.shape))
            bT_shape = list(b.shape)
            bT_shape[-1], bT_shape[-2] = bT_shape[-2], bT_shape[-1]
            bT_relax = relax.Var("b", relax.TensorStructInfo(bT_shape))
            # 使用 relax.op.matmul() 方法对 a_relax 和 bT_relax 进行矩阵乘法运算。
            # 然后，通过 self.builder_.normalize() 方法对结果进行归一化处理，并获取最终的输出形状。
            output_shape = self.builder_.normalize(
                relax.op.matmul(a_relax, bT_relax)
            ).struct_info.shape

            # 该函数接受可变数量的空间索引参数 idx_spatial，
            def matmul_compute(*idx_spatial):
                # 并定义了一个名为 k 的规约轴（reduce axis），其范围为 0 到 a_shape[-1]。
                k = te.reduce_axis((0, a_shape[-1]), name="k")

                # 定义了一个名为 multiply_compute 的内部函数，用于计算乘法操作时的索引。
                def multiply_compute(idx_reduce):
                    a_indices = []
                    b_indices = []

                    # 根据 is_a_larger 的值，将 idx_spatial 中的索引分配给 a_indices 或 b_indices，用于处理形状长度差异的维度。
                    for i in range(offset):
                        if is_a_larger:
                            a_indices.append(idx_spatial[i])
                        else:
                            b_indices.append(idx_spatial[i])
                    for i in range(
                        offset, len(output_shape) - (2 - a_prepended - b_appended)
                    ):
                        # 根据维度的相等性，将适当的索引添加到 a_indices 和 b_indices 中。
                        # 如果维度不相等或无法确定是否相等，则将索引设为 0 或保持不变。
                        a_dim = a_shape[i if is_a_larger else i - offset]
                        b_dim = b_shape[i if not is_a_larger else i - offset]
                        dim_equal = a_dim == b_dim
                        if not isinstance(dim_equal, tir.IntImm) or dim_equal == 0:
                            a_dim_is_one = isinstance(a_dim, tir.IntImm) and a_dim == 1
                            b_dim_is_one = isinstance(b_dim, tir.IntImm) and b_dim == 1
                            a_indices.append(0 if a_dim_is_one else idx_spatial[i])
                            b_indices.append(0 if b_dim_is_one else idx_spatial[i])
                        else:
                            a_indices.append(idx_spatial[i])
                            b_indices.append(idx_spatial[i])

                    # 在乘法操作的索引中添加规约轴 idx_reduce，并根据 a_prepended 和 b_appended 的值，
                    # 将适当的索引添加到 a_indices 和 b_indices 中。
                    if not a_prepended:
                        a_indices.append(idx_spatial[-2 + b_appended])
                    a_indices.append(idx_reduce)
                    if not b_appended:
                        b_indices.append(idx_spatial[-1])
                    b_indices.append(idx_reduce)

                    # 根据 out_dtype 的值，选择是否进行数据类型转换，并返回乘法操作的结果。
                    dtype = out_dtype
                    if dtype != "":
                        return a(*a_indices).astype(dtype) * b(*b_indices).astype(dtype)
                    return a(*a_indices) * b(*b_indices)

                # 在缩减轴 k 上对 multiply_compute 的结果进行求和操作。
                return te.sum(multiply_compute(k), axis=k)

            # 使用 te.compute() 函数计算最终的输出，其中使用一个 lambda 函数将输入索引传递给 matmul_compute 函数，
            # 并将结果命名为 "NT_matmul"。整个计算过程将根据 output_shape 进行执行。
            return te.compute(
                output_shape,
                lambda *idx: matmul_compute(*idx),  # pylint: disable=unnecessary-lambda
                name="NT_matmul",
            )

        # 首先，检查函数调用的操作符 call.op 是否是 relax.GlobalVar 类型。如果是，获取与该操作符对应的函数对象，
        # 并检查函数的属性中是否包含键 "Composite"，且其值为 "transpose_matmul_fuse"。
        if isinstance(call.op, relax.GlobalVar):
            function = self.builder_.get()[call.op]
            if (
                "Composite" in function.attrs
                and function.attrs["Composite"] == "transpose_matmul_fuse"
            ):
                # 将函数的返回类型 function.ret_struct_info.dtype 赋值给变量 out_dtype
                out_dtype = function.ret_struct_info.dtype
                # 然后调用 self.builder_.call_te() 方法，传递 te_transposed_matmul 函数作为参数，
                # 以及调用的参数 call.args[1] 和 call.args[0]，并指定 primfunc_name_hint 为 "NT_matmul"。
                return self.builder_.call_te(
                    te_transposed_matmul,
                    call.args[1],
                    call.args[0],
                    primfunc_name_hint="NT_matmul",
                )

        return super().visit_call_(call)

# 使用 @tvm.transform.module_pass 装饰器定义了一个名为 FuseTransposeMatmul 的类，
# 并指定了优化级别 opt_level=0 和 pass 的名称为 "FuseTransposeMatmul"。
@tvm.transform.module_pass(opt_level=0, name="FuseTransposeMatmul")
class FuseTransposeMatmul:
    # 定义了 transform_module 方法，接受一个名为 mod 的 IRModule 对象和
    # tvm.transform.PassContext 对象作为参数，并返回一个 IRModule 对象。
    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        # 通过调用 relax.transform.FuseOpsByPattern 并传递一个包含单个模式元组的列表，
        # 对模块 mod 进行融合的转置矩阵乘法操作。
        mod = relax.transform.FuseOpsByPattern(
            [("transpose_matmul_fuse", *TransposeMatmulCodeGenerator.pattern())]
        )(mod)

        # 创建一个名为 transpose_matmul_codegen 的 TransposeMatmulCodeGenerator 对象，
        # 并对模块中的每个函数进行遍历。如果函数是 relax.Function 类型，则调用 transpose_matmul_codegen.visit_expr 
        # 方法对函数进行转置矩阵乘法代码生成，并通过 transpose_matmul_codegen.builder_.update_func 方法更新函数。
        transpose_matmul_codegen = TransposeMatmulCodeGenerator(mod)
        for gv in mod.functions:
            func = mod[gv]
            if not isinstance(func, relax.Function):
                continue
            func = transpose_matmul_codegen.visit_expr(func)
            transpose_matmul_codegen.builder_.update_func(gv, func)

        # 返回转置矩阵乘法代码生成器的 builder 对象中的模块。
        return transpose_matmul_codegen.builder_.get()
