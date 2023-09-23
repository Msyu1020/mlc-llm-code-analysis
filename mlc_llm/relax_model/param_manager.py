import json
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import tvm
from torch import Tensor as torchTensor
from tvm import relax, tir
from tvm._ffi.runtime_ctypes import Device
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr import Expr, Function, Var
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.testing import nn

from .. import quantization
from .modules import named_parameters


# 这段代码定义了一个默认的 f_compute_relax_param 方法,用于映射 torch 参数到 relax 参数的转换。
# relax_pname是一个string,表示要转换的relax参数名。torch_params是一个list,包含了对应的torch参数
# 它返回 Any 类型,主要是映射后的relax参数。如果不重写这个方法,直接抛出NotImplementedError
# 根据函数名和参数,可以推断这个方法的目的是:
# 从torch神经网络训练得到的放在torch_params中的参数
# 转换为relax框架可以使用和理解的relax参数
# 放入relax_pname指定的名称下
# 这样就完成了torch到relax的参数迁移,方便在relax框架下继续推理
# 默认实现是直接抛异常,需要根据实际需求重写这个方法来完成真实的参数映射转换
def f_default_compute_relax_param(relax_pname: str, torch_params: List[Any]) -> Any:
    """The defualt `f_compute_relax_param` for ParamManager.
    See ParamManager for more details.
    """
    raise NotImplementedError()

# 定义了Parameter类,用来表示模型中的参数(权重)
# 本类用来抽象表示模型参数相关信息,包括名称、形状、量化规则等属性,同时支持注册不同函数下的参数变化。
class Parameter:
    """The abstraction of weight tensors (e.g., linear layer weight, embedding
    table, etc.) in a model.

    Attributes
    ----------
    name : str
        The name of the parameter.
        The name of a weight is got by `named_parameters()` method, similar to
        PyTorch's `named_parameters()` function.
        An example name is `model.layers.11.self_attn.k_proj.weight`.
        In a model, the name is the **unique** identifier of a parameter.

    param_info_dict : Dict[str, relax.TensorStructInfo]
        The shape and dtype of the parameter in each function.
        The shape can be accessed by `param_info_dict[func_name].shape`, which is
        a relax.ShapeExpr instance.
        And the dtype can be accessed by `param_info_dict[func_name].dtype`,
        which is a Python string.

    quant_spec : quantization.QuantizationSpec
        The quantization specification of this parameter.
        It specifies the algorithm to quantize and dequantize this parameter (or
        this parameter does not need quantization).

    shard_dim : Optional[int]
        The dimension to be sharded.
    """

    name: str # 参数名称
    param_info_dict: Dict[str, relax.TensorStructInfo] # 记录不同函数下参数信息的变化
    quant_spec: quantization.QuantizationSpec # 表示量化规则
    shard_dim: Optional[int] # 表示分布式推理下的参数分片维度

    def __init__(
        self,
        name: str,
        quant_spec: quantization.QuantizationSpec,
        shard_dim: Optional[int],
    ) -> None:
        self.name = name
        self.param_info_dict = dict()
        self.quant_spec = quant_spec
        self.shard_dim = shard_dim

    # register_func方法注册函数名和形状信息
    def register_func(self, func_name: str, param_info: relax.TensorStructInfo):
        self.param_info_dict[func_name] = param_info

    # param_info property 迭代器方式返回下一个函数下的形状和类型信息
    @property
    def param_info(self):
        """Return the shape and dtype of the parameter (in some arbitrary function)."""
        return next(iter(self.param_info_dict.values()))


class ParamManager:
    """这是一个关于模型权重信息的模型级数据结构，负责在整个模型级别对参数应用量化和反量化。.

    Attributes
    ----------
    params : Dict[str, Parameter]
        参数名称到参数对象的映射。

    param_names : List[str]
        所有参数的名称列表。为了确保参数的唯一顺序和确定性，参数名称以列表形式保存，并且参数顺序由参数名称列表唯一确定。

    func_raw_param_map : Dict[relax.Var, Tuple[str, Parameter]]
       将表示权重参数的 relax.Var 映射到变量所在函数的名称（例如 "prefill" 或 "decode"）和对应的参数对象的字典。
       该映射用于对模型中的 Relax 函数（例如 "prefill"、"decode" 等）应用量化变换。

    param2qrange : Dict[Parameter, range]
        将每个参数映射到其量化张量在所有参数的量化张量列表中的范围。每个参数会被量化为多个张量。例如，假设有参数 p0、p1、p2：
        For example, assume we have parameters `p0`, `p1`, `p2`.
        - p0 被量化为 t0_0、t0_1
        - p1 被量化为 t1_0
        - p2 被量化为 t2_0、t2_1 和 t2_2
        那么所有量化张量的列表为 [t0_0, t0_1, t1_0, t2_0, t2_1, t2_2]，param2qrange 字典为 {p0: range(0, 2), p1: range(2, 3), p2: range(3, 6)}

    f_convert_pname_fwd : Callable[[str], List[str]]
        将 Relax 参数名称（我们的名称）转换为 torch 参数名称的函数，提示“要加载此 Relax 参数，需要哪些 torch 参数”。
        通常，该函数将一个名称映射到其自身。例如，在 LLaMA 中，我们将 lm_head.weight 映射为其本身，因为该参数在 Relax 和 torch 两端具有相同的名称。
        在某些情况下，我们将一个名称映射到多个名称。例如，如果我们支持将 torch 的 QKV 分开计算，而 Relax 端只有一个 QKV 权重，
        那么我们将一个名称映射到三个名称。在某些情况下，我们将一个名称映射到与其本身不同的单个名称。
        这可能发生在 Relax nn.Module 的参数名称与 torch 实现不同，因此我们需要进行名称映射，或者当 Relax 参数是从 torch 参数计算出来时。
        例如，如果 torch 实现支持合并的 QKV，而 Relax 实现不支持，我们需要从 torch 的参数计算出 Relax 参数。
        在这种情况下，我们将 Relax 参数名称映射为 torch 的参数名称。

    f_convert_param_bkwd : Callable[[str, Any], Optional[List[Tuple[str, Any]]]]
        该函数将torch参数和参数名称转换回Relax参数及其名称。这里的Any表示numpy.ndarray。
        - 通常情况下，该函数只返回输入的torch参数和对应的Relax参数名称。
        - 在某些情况下，我们会返回多个Relax参数。例如，如果torch实现支持合并的QKV，而Relax实现不支持，
        该函数会接收torch的合并QKV权重，并返回分离的Q、K、V权重及其对应的名称。
        - 在某些情况下，我们会返回None。这发生在输入的torch参数本身无法确定任何Relax参数的情况下。
        例如，如果我们支持在torch端将合并的QKV分开计算，对于单个Q、K、V权重，在这里返回None，
        因为仅有Q（或K、V）权重无法计算合并的QKV权重。

    f_compute_relax_param : Callable[[str, List[Any]], Any]
        该函数用于从一组torch参数计算Relax参数。这里的Any表示numpy.ndarray。
        在一个Relax参数由多个torch参数计算得出的情况下，使用该函数。
        例如，如果我们支持在torch端将合并的QKV分开计算，我们将使用该函数将torch的Q、K、V权重合并为一个权重。
        通常情况下，不需要使用该函数，并且默认情况下，它通过引发NotImplementedError来实现（参见f_default_compute_relax_param）。

    model_path : str
        磁盘上Huggingface模型路径

    use_safetensors: bool
        是否使用.safetensors而不是.bin来加载模型。

    safetensors_load_func: Callable[[Union[str, os.PathLike], str], Dict[str, torch.Tensor]]
        对从`safetensors.torch`导入的函数`load_file`的引用。目标是避免在TVM注册的函数中重复导入。

    pidx2pname : Dict[int, str]
        解析每个Relax参数在param_names中的索引与Relax参数名称之间的字典。

    torch_pname2binname : Dict[str, str]
       将每个torch参数的名称与保存该torch参数的二进制片段的名称之间的字典。
    """

    params: Dict[str, Parameter] # 一个字典，包含参数名称作为键，对应的Parameter对象作为值。这个字典用于存储relay参数
    param_names: List[str] # 一个列表，包含了参数的名称。这个列表按顺序存储了参数的名称，与params字典中的键相对应。
    # 一个字典，将relax.Var作为键，与一个元组(str, Parameter)相关联。该元组的第一个元素是Relax参数的名称，第二个元素是对应的Relax参数。
    func_raw_param_map: Dict[relax.Var, Tuple[str, Parameter]]
    param2qrange: Dict[Parameter, range] # 一个字典，将Parameter对象作为键，与一个range对象相关联。这个字典用于存储每个Relay参数的量化范围。

    qspec_updater_classes: List[quantization.QuantSpecUpdater] # 一个列表，包含了量化规范更新器的类。这些类用于更新量化规范。

    nparam_to_load: int # 一个整数，表示要加载的参数数量。
    # 一个可调用对象，接受一个或者一系列参数名作为输入，并返回一个包含转换后的参数名列表的结果。
    # 该函数用于将 Relax 参数名称（我们的名称）转换为 torch 参数名称的函数。
    f_convert_pname_fwd: Callable[[str], List[str]]
    # 一个可调用对象，接受参数名和对应的值作为输入，并返回一个包含转换后的参数名和值的元组列表。
    # 该函数将torch参数和参数名称转换回Relax参数及其名称
    f_convert_param_bkwd: Callable[[str, Any], Optional[List[Tuple[str, Any]]]]
    # 一个可调用对象，接受参数名和参数值列表作为输入，并返回计算得到的Relax参数。该函数用于计算Relax参数。
    f_compute_relax_param: Callable[[str, List[Any]], Any]
    # 一个可调用对象或None，接受参数名作为输入，并返回预量化操作后的参数名。如果存在，它将在预量化之前运行一些操作。
    f_run_prequantize: Optional[Callable[[str], str]]

    model_path: str # 一个字符串，表示模型的路径。
    use_safetensors: bool # 一个布尔值，指示是否使用safetensors而不是.bin文件来加载模型。
    # 一个可调用对象，接受模型路径和模型文件后缀作为输入，并返回一个包含torch张量的字典。这个函数用于加载模型文件。
    safetensors_load_func: Callable[[Union[str, os.PathLike], str], Dict[str, torchTensor]]
    pidx2pname: Dict[int, str] # 一个字典，将参数索引作为键，与对应的参数名称相关联。用于从参数索引获取参数名称。
    torch_pname2binname: Dict[str, str] # 一个字典，将torch参数名称作为键，与对应的二进制片段名称相关联。用于从torch参数名称获取对应的二进制片段名称。

    def __init__(self) -> None:
        self.params = {}
        self.param_names = []
        self.params_in_func = {}

        self.func_raw_param_map = {}
        self.param2qrange = None

        self.nparam_to_load = None
        self.f_convert_pname_fwd = None
        self.f_convert_param_bkwd = None
        self.f_compute_relax_param = None
        self.f_run_prequantize = None

        self.qspec_updater_classes = []

    def register_params(
        self,
        model: nn.Module,
        func_name: str,
        quantization_scheme: quantization.QuantizationScheme,
        f_get_param_quant_kind: Callable[
            [str, relax.TensorStructInfo], quantization.ParamQuantKind
        ],
    ) -> None:
        """在参数管理器中注册输入模型的参数（在输入函数的上下文中）

        Parameters
        ----------
        model : nn.Module
            要注册参数的输入模型。

        func_name : str
            输入模型所在函数的名称。例如，"prefill"函数或"decode"函数。

        quantization_scheme : quantization.QuantizationScheme
            输入模型的量化方案，描述如何对模型进行量化。

        f_get_param_quant_kind: Callable[[str, relax.TensorStructInfo], quantization.ParamQuantKind]
            一个函数，接受参数的名称和TensorStructInfo（实际上是形状和数据类型），并返回该参数使用的量化类型。这用于对参数应用量化。
        """
        # 如果量化方案中的qspec_updater_class不为None。
        if quantization_scheme.qspec_updater_class is not None:
            # 将quantization_scheme.qspec_updater_class添加到self.qspec_updater_classes列表中。
            self.qspec_updater_classes.append(quantization_scheme.qspec_updater_class)
        # 如果量化方案中的f_convert_param_bkwd不为None。
        if quantization_scheme.f_convert_param_bkwd is not None:
            # 将quantization_scheme.f_convert_param_bkwd赋值给self.f_convert_param_bkwd。
            self.f_convert_param_bkwd = quantization_scheme.f_convert_param_bkwd
        # 如果量化方案中的f_compute_relax_param不为None。
        if quantization_scheme.f_compute_relax_param is not None:
            # 将quantization_scheme.f_compute_relax_param赋值给self.f_compute_relax_param。
            self.f_compute_relax_param = quantization_scheme.f_compute_relax_param
        # 如果量化方案中的f_run_prequantize不为None。
        if quantization_scheme.f_run_prequantize is not None:
            # 将quantization_scheme.f_run_prequantize赋值给self.f_run_prequantize。
            self.f_run_prequantize = quantization_scheme.f_run_prequantize

        # 在self.params_in_func字典中，以函数名称func_name为键，创建一个空列表作为值。
        self.params_in_func[func_name] = []
        # For each parameter in the input model, get its quantization kind and
        # register the parameter with its name and quantization kind.
        # 对于输入模型中的每个参数，获取其量化类型并注册带有名称和量化类型的参数。
        # 迭代输入模型的命名参数，其中name是参数名称，relax_param是对应的Relax参数对象。
        for name, relax_param in named_parameters(model).items():
            # 使用f_get_param_quant_kind函数获取参数的量化类型。
            quant_kind = f_get_param_quant_kind(name, relax_param.struct_info)
            # 使用_register_param方法注册参数，并将相关信息传递给它。
            param = self._register_param(
                name,
                relax_param,
                getattr(quantization_scheme, quant_kind.name),
                func_name,
                getattr(relax_param, "shard_dim", None),
            )

            # 将注册的参数param添加到self.params_in_func[func_name]列表中。
            self.params_in_func[func_name].append(param)

    def set_param_loading_func(
        self,
        model_path: str,
        use_safetensors: bool,
        f_convert_pname_fwd: Callable[[str], List[str]] = lambda pname: [pname],
        f_convert_param_bkwd: Callable[
            [str, Any], Optional[List[Tuple[str, Any]]]
        ] = lambda pname, torch_param: [(pname, torch_param)],
        f_compute_relax_param: Callable[[str, List[Any]], Any] = f_default_compute_relax_param,
        *,
        no_lazy_param_loading: bool = False,
    ) -> None:
        """Set the parameter loading functions.

        Parameters
        ----------
        model_path : str
            磁盘上Hugging Face模型的路径。

        use_safetensors : bool
            是否使用.safetensors而不是.bin来加载模型。

        f_convert_pname_fwd : Callable[[str], List[str]]
            将Relax参数名称（我们的命名）转换为torch的参数名称的函数。有关更多详细信息，请参阅ParamManager的文档。

        f_convert_param_bkwd : Callable[[str, Any], Optional[List[Tuple[str, Any]]]]
            将torch参数和参数名称反向转换为带有名称的Relax参数的函数。这里的Any代表numpy.ndarray。有关更多详细信息，请参阅ParamManager的文档。

        f_compute_relax_param : Callable[[str, List[Any]], Any]
            从一系列torch参数计算Relax参数的函数。这里的Any代表numpy.ndarray。有关更多详细信息，请参阅ParamManager的文档。

        no_lazy_param_loading : bool
            指示是否不需要从torch进行延迟参数加载。当在构建模型时所有模型权重都已加载时，需要将其设置为True。
        """
        self.f_convert_pname_fwd = f_convert_pname_fwd
        if self.f_convert_param_bkwd is None:
            self.f_convert_param_bkwd = f_convert_param_bkwd
        if self.f_compute_relax_param is None:
            self.f_compute_relax_param = f_compute_relax_param

        self.model_path = model_path
        self.use_safetensors = use_safetensors
        if self.use_safetensors:
            # Use a pointer here to prevent repeated import in tvm registered function
            # 导入safetensors.torch模块中的load_file函数。
            from safetensors.torch import (
                load_file,  # pylint: disable=import-outside-toplevel
            )

            # 将load_file函数赋值给self.safetensors_load_func
            self.safetensors_load_func = load_file

        pnames_to_load = [] # 创建一个空列表pnames_to_load。
        for param_name in self.param_names:
            param = self.params[param_name] # 获取参数对象param。
            # 使用param.quant_spec.get_loaded_tensor_info方法获取加载的张量信息，返回加载的名称列表loaded_names和其他信息。
            loaded_names, _ = param.quant_spec.get_loaded_tensor_info(param_name, param.param_info)
            # 将loaded_names添加到pnames_to_load列表中。
            pnames_to_load += loaded_names

        self.nparam_to_load = len(pnames_to_load) # 将pnames_to_load列表的长度赋值给self.nparam_to_load，表示要加载的参数数量。
        if not no_lazy_param_loading:
            # 创建一个字典，其中键是参数索引pidx，值是参数名称pname，用于将参数索引映射到参数名称。
            self.pidx2pname = {pidx: pname for pidx, pname in enumerate(pnames_to_load)}
        else:
            # 如果no_lazy_param_loading为True，则将self.pidx2pname设置为一个空字典。
            self.pidx2pname = dict()

    def transform_dequantize(self, mod: tvm.IRModule) -> tvm.IRModule:
        """对输入的IRModule应用去量化

        Parameters
        ----------
        mod : tvm.IRModule
            要应用去量化的输入IRModule。
            IRModule包含所有构建的Relax函数（例如，"prefill" / "decode"函数），并且预期所有参数都在ParamManager中注册。

        Returns
        -------
        updated_mod : tvm.IRModule
            使用去量化计算更新后的IRModule。
        """

        # For each Relax function in the input IRModule (e.g., "prefill"),
        # we create its input relax.Var of all the quantized data, and
        # store the mapping from function name to the var.
        # 创建一个空字典func2param_var，用于存储从函数名称到relax.Var的映射关系。
        func2param_var: Dict[str, relax.Var] = {}
        # 对于输入的IRModule中的每个函数（例如，"prefill"函数）：
        for gv, func in mod.functions.items():
            # 如果函数不是relax.Function类型，则跳过。
            if not isinstance(func, relax.Function):
                continue
            # 如果函数的属性为None或者不包含"num_input"属性，则跳过。
            if func.attrs is None or not "num_input" in func.attrs:
                continue
            # 将函数名称作为键，创建一个relax.Var作为值，该relax.Var包含所有量化数据的参数信息。
            func2param_var[gv.name_hint] = relax.Var(
                "params", self.get_quantized_param_info(gv.name_hint)
            )

        # Cache mapping to avoid duplicate dequantization.
        # 创建一个空字典dequantized_cache，用于缓存已进行去量化的relax.Var。
        dequantized_cache: Dict[relax.Var, relax.Var] = {}

        # Define a var replacement function for applying dequantization.
        # 定义一个变量替换函数f_replace，用于应用去量化。
        # 如果变量var已存在于dequantized_cache中，则直接返回缓存的去量化结果。
        def f_replace(var: relax.Var, bb: relax.BlockBuilder, func_name: str) -> relax.Var:
            if var in dequantized_cache:
                return dequantized_cache[var]
            # 确保var存在于self.func_raw_param_map中。
            assert var in self.func_raw_param_map
            # 从self.func_raw_param_map中获取对应的函数名称和参数。
            func_name, param = self.func_raw_param_map[var]
            # 调用self._dequantize方法对参数进行去量化，传入相关的relax.Var和函数名称。           
            dequantized = self._dequantize(param, func2param_var[func_name], bb, func_name)
            # 将去量化的结果存储到dequantized_cache中，并返回去量化的结果。
            dequantized_cache[var] = dequantized
            return dequantized

        # Create the function mutator for applying dequantization.
        # 创建一个ParamReplacer对象replacer，传入IRModule、func2param_var和f_replace，用于进行参数替换。
        replacer = ParamReplacer(mod, func2param_var, f_replace)
        # Update the input IRModule with dequantization.
        # 使用replacer.transform()方法更新输入的IRModule，应用去量化。
        mod = replacer.transform()

        return mod

    def get_quantized_param_info(self, func_name: str) -> List[relax.TensorStructInfo]:
        # 创建一个relax.BlockBuilder对象，用于构建IR的基本块。
        bb = relax.BlockBuilder()

        self.param2qrange = dict() # 创建一个空字典param2qrange，用于存储参数和其对应的量化范围。
        # 创建一个空列表quantized_param_info，用于存储参数的量化信息。
        quantized_param_info: List[relax.TensorStructInfo] = []
        # 对于self.param_names中的每个参数名称：
        for name in self.param_names:
            param = self.params[name] # 获取参数对象param。
            param_info = None # 初始化param_info为None。
            # 如果func_name在param.param_info_dict中存在，则将param_info设置为param.param_info_dict[func_name]，
            # 否则创建一个新的relax.TensorStructInfo对象，该对象包含参数的形状和数据类型。
            if func_name in param.param_info_dict:
                param_info = param.param_info_dict[func_name]
            else:
                param_info = relax.TensorStructInfo(
                    tvm.ir.load_json(tvm.ir.save_json(param.param_info.shape)),
                    param.param_info.dtype,
                )

            # 使用param.quant_spec.get_loaded_tensor_info方法获取加载的张量信息。
            _, loaded_tensor_info = param.quant_spec.get_loaded_tensor_info(name, param_info)

            provided_tensor_vars: List[relax.Var] = [] # 创建一个空列表provided_tensor_vars
            # 对于每个加载的张量信息provided_info，创建一个relax.Var对象，并将其添加到provided_tensor_vars列表中。
            for provided_info in loaded_tensor_info:
                provided_tensor_vars.append(relax.Var("var", provided_info))

            # Get the quantization function of this parameter.
            # 获取该参数的量化函数f_quantize。
            f_quantize = param.quant_spec.get_quantize_func(param_info)
            # 如果f_quantize为None，表示该参数不需要进行量化或已经预先量化。
            if f_quantize is None:
                # If the parameter does not have a quantization function, either it
                # does not need quantization or it is pre-quantized.
                # 将该参数与其对应的量化范围（从len(quantized_param_info)到len(quantized_param_info) + 
                # len(provided_tensor_vars)）存储到self.param2qrange中。
                self.param2qrange[param] = range(
                    len(quantized_param_info),
                    len(quantized_param_info) + len(provided_tensor_vars),
                )
                # 将provided_tensor_vars中每个var的struct_info添加到quantized_param_info列表中。
                quantized_param_info += [var.struct_info for var in provided_tensor_vars]
            else:
                # If the parameter has a quantization function, it is not expected
                # to be pre-quantized.
                # 否则，表示该参数需要进行量化且不应该预先量化
                # 确保provided_tensor_vars的长度为1，因为有量化函数的参数不应该预先量化。
                assert len(provided_tensor_vars) == 1, (
                    "A parameter with quantization function is not expected " "to be pre-quantized."
                )

                # Apply the quantization function.
                # 应用量化函数f_quantize到provided_tensor_vars，并使用bb.normalize对结果进行规范化。
                quantized_data = bb.normalize(f_quantize(bb, provided_tensor_vars))
                # 如果量化后的数据的struct_info是relax.TupleStructInfo类型：
                if isinstance(quantized_data.struct_info, relax.TupleStructInfo):
                    n_tensor = len(quantized_data.struct_info.fields)
                    assert n_tensor > 1
                    # Record the range of quantized tensors of this parameter.
                    # 记录该参数的量化张量范围，从len(quantized_param_info)到len(quantized_param_info) + n_tensor。
                    self.param2qrange[param] = range(
                        len(quantized_param_info),
                        len(quantized_param_info) + n_tensor,
                    )
                    # Collect the quantized tensors to return.
                    # 遍历每个量化后的张量，将其struct_info添加到quantized_param_info列表中。
                    for i in range(n_tensor):
                        quantized_param_info.append(
                            relax.TupleGetItem(quantized_data, i).struct_info
                        )
                else:
                    # 否则，如果量化后的数据的struct_info是relax.TensorStructInfo类型：
                    assert isinstance(quantized_data.struct_info, relax.TensorStructInfo)
                    # 记录该参数的量化张量范围，从len(quantized_param_info)到len(quantized_param_info) + 1。
                    self.param2qrange[param] = range(
                        len(quantized_param_info), len(quantized_param_info) + 1
                    )
                    # 将量化后的张量的struct_info添加到quantized_param_info列表中。
                    quantized_param_info.append(quantized_data.struct_info)

        # 返回一个relax.TupleStructInfo对象，其中包含了quantized_param_info列表中的量化参数信息。
        return relax.TupleStructInfo(quantized_param_info)

    def get_param_loading_functions(
        self,
        model_params: List[Optional[tvm.nd.NDArray]],
        loaded_params: List[tvm.nd.NDArray],
        loaded_idx_set: Set[int],
        loaded_torch_bins: Set[str],
        cached_relax_params: Dict[int, tvm.nd.NDArray],
        cached_torch_params: Dict[str, Any],
        device: Device,
        device_cpu: Device,
    ) -> Tuple[Callable, Callable]:
        """A wrapper function which returns the `get_item` and `set_item`
        functions for parameter lazy loading.

        Parameters
        ----------
        model_params : List[Optional[tvm.nd.NDArray]]
            The pre-loaded model parameters, for which we skip lazy loading.

        loaded_params : List[tvm.nd.NDArray]
            The parameter loading result, storing all the loaded parameters.

        loaded_idx_set : Set[int]
            The set of indices of loaded parameters, serving for robustness
            guarantee to avoid one parameter being loaded for multiple times.

        loaded_torch_bins : Set[str]
            The set of torch binary filenames, serving for robustness guarantee
            to avoid one torch binary file being loaded for multiple times.

        cached_relax_params : Dict[int, tvm.nd.NDArray]
            The set of cached Relax parameters.

        cached_torch_params: Dict[str, Any]
            The set of cached torch parameters. `Any` here stands for numpy.ndarray.

        device : Device
            The device which we load the parameters to.

        device_cpu : Device
            The CPU device.
        """
        import torch  # pylint: disable=import-outside-toplevel

        assert self.f_convert_pname_fwd is not None
        assert self.f_convert_param_bkwd is not None
        assert self.f_compute_relax_param is not None
        pname2pidx: Dict[str, int] = {pname: pidx for pidx, pname in self.pidx2pname.items()}

        def fetch_torch_param(torch_param):
            if str(torch_param.dtype) == "torch.bfloat16":
                # Convert to float32 first.
                return torch_param.detach().cpu().float().numpy()
            else:
                return torch_param.detach().cpu().numpy()

        def load_torch_params_from_bin(torch_binname: str):
            torch_binpath = os.path.join(self.model_path, torch_binname)
            torch_params = None
            if self.use_safetensors:
                torch_params = self.safetensors_load_func(torch_binpath)
            else:
                torch_params = torch.load(
                    torch_binpath,
                    map_location=torch.device("cpu"),
                )
            torch_param_names = list(torch_params.keys())
            for torch_param_name in torch_param_names:
                torch_param = fetch_torch_param(torch_params[torch_param_name])
                del torch_params[torch_param_name]

                relax_params = self.f_convert_param_bkwd(torch_param_name, torch_param)
                if relax_params is not None:
                    for param_name, param in relax_params:
                        if param_name not in pname2pidx.keys():
                            continue
                        pidx = pname2pidx[param_name]
                        assert pidx not in cached_relax_params
                        cached_relax_params[pidx] = tvm.nd.array(param, device_cpu)
                else:
                    assert torch_param_name not in cached_torch_params
                    cached_torch_params[torch_param_name] = torch_param
                del torch_param

        def get_item(i):
            # If the weight is already provided by `model_params`, directly use it
            # and no need to load from binary file.
            if model_params[i] is not None:
                assert i not in cached_relax_params
                return tvm.nd.array(model_params[i], device=device)

            # Otherwise, we load the weight from its corresponding binary file.
            assert i in self.pidx2pname
            relax_pname = self.pidx2pname[i]
            torch_pnames = self.f_convert_pname_fwd(relax_pname)

            if i not in cached_relax_params:
                for torch_binname in [
                    self.torch_pname2binname[torch_pname] for torch_pname in torch_pnames
                ]:
                    if torch_binname in loaded_torch_bins:
                        continue
                    load_torch_params_from_bin(torch_binname)
                    loaded_torch_bins.add(torch_binname)

            if i not in cached_relax_params:
                assert len(torch_pnames) > 1
                assert all([torch_pname in cached_torch_params] for torch_pname in torch_pnames)
                cached_relax_params[i] = self.f_compute_relax_param(
                    relax_pname,
                    [cached_torch_params[torch_pname] for torch_pname in torch_pnames],
                )
                for torch_pname in torch_pnames:
                    del cached_torch_params[torch_pname]

            assert i in cached_relax_params
            assert i not in loaded_idx_set
            param_on_device = tvm.nd.array(cached_relax_params[i], device=device)
            loaded_idx_set.add(i)
            del cached_relax_params[i]
            return param_on_device

        def set_item(i, computed_param):
            if len(loaded_params) <= i:
                loaded_params.extend([None for _ in range(i - len(loaded_params) + 1)])
            loaded_params[i] = tvm.nd.array(computed_param, device=device_cpu)

        return get_item, set_item

    #################### Below are internally called methods ####################

    def _register_param(
        self,
        name: str,
        var: relax.Var,
        quant_spec: quantization.QuantizationSpec,
        func_name: str,
        shard_dim: Optional[int],
    ) -> Parameter:
        """Register a single parameter in the parameter manager.
        In most cases, this method is not directly used outside this class:
        it is called by `register_params` above.

        Parameters
        ----------
        name : str
            The name of the parameter to register.
            Name serves as the unique identifier of the parameter.

        var : relax.Var
            The parameter relax.Var on the nn.Module side.

        quant_spec : quantization.QuantizationSpec
            The quantization specification of the parameter

        func_name : str
            The name of the function the input var is in.
            For example, the "prefill" function or the "decode" function.

        shard_dim : int
            The dimension along which the parameter is sharded.

        Returns
        -------
        param : Parameter
            The registered Parameter.
        """
        assert (
            var not in self.func_raw_param_map
        ), "The input var is not supposed to be already registered."
        assert isinstance(
            var.struct_info.shape, relax.ShapeExpr
        ), "The parameter to register is expected to have shape as a tuple"

        if name in self.params:
            # When the input name appears in `self.params`, it means the input
            # parameter has been previously registered in some other function.
            # Thus, we check if the dtype, shape and the quantization specification
            # of both sides are consistent.
            param = self.params[name]
            assert (
                param.quant_spec == quant_spec
            ), "One parameter is expected to be quantized by single specification in all functions."
            assert (
                param.param_info.dtype == var.struct_info.dtype
            ), "Dtype mismatch of one parameter in two functions."
            assert (
                param.param_info.ndim == var.struct_info.ndim
            ), "Shape mismatch of one parameter in two functions."
            for len0, len1 in zip(param.param_info.shape.values, var.struct_info.shape.values):
                if isinstance(len0, tir.IntImm) and isinstance(len1, tir.IntImm):
                    assert (
                        len0.value == len1.value
                    ), "Shape mismatch of one parameter in two functions."
        else:
            # Otherwise, the parameter is registered for the first time.
            param = Parameter(name, quant_spec, shard_dim)
            self.params[name] = param
            self.param_names.append(name)

        param.register_func(func_name, var.struct_info)
        # Record the mapping from the input relax.Var to the function name and
        # the parameter in the manager.
        self.func_raw_param_map[var] = (func_name, param)
        return param

    def _dequantize(
        self,
        param: Parameter,
        quantized_tuple: relax.Var,
        bb: relax.BlockBuilder,
        func_name: str,
        qparams: List[relax.Var] = None,
    ) -> relax.Var:
        """Applying dequantization to the input parameter.
        This method is called by `transform_module` below, and is not
        directly invoked outside the class.

        Parameters
        ----------
        param : Parameter
            The parameter whose quantized tensors are to be dequantized.

        quantized_tuple : relax.Var
            The relax.Var of the quantized tensors of all parameters in the model.

        bb : relax.BlockBuilder
            The Relax BlockBuilder used for inserting the dequantization computations.

        func_name : str
            The name of the  function which dequantization is applied to.

        qparams : List[relax.Var]
            The quantized parts of the parameter.
            By default it is `None`, in which case we will get the quantized parts
            from `quantized_tuple`.

        Returns
        -------
        The dequantized parameter, in the form of a relax.Var.
        """
        if not qparams:
            # Get the corresponding Relax vars of the quantized tensors of this parameter.
            qparams: List[relax.Var] = []
            for qparam_idx in self.param2qrange[param]:
                qparams.append(bb.emit(relax.TupleGetItem(quantized_tuple, qparam_idx)))

        # Get the dequantization function of this parameter.
        f_dequantize = param.quant_spec.get_dequantize_func(
            param_info=param.param_info_dict[func_name],
            qparam_info=[qparam.struct_info for qparam in qparams],
        )
        if f_dequantize is None:
            # If the parameter does not have a dequantization function, its "quantized
            # data" is expected to have only one element.
            assert len(qparams) == 1, (
                "A parameter without dequantization function is expected not to have "
                'more than one "quantized data".'
            )
            return qparams[0]
        else:
            # Apply the dequantization function.
            return bb.emit(f_dequantize(bb, qparams))


@mutator
class ParamReplacer(PyExprMutator):
    """The function mutator that updates the model with dequantization.

    Attributes
    ----------
    mod : tvm.IRModule
        The IRModule of the model to be updated.

    func2param_var : Dict[str, relax.Var]
        The mapping from each function name to its input var of quantized data tuple.

    f_replace : Callable[[relax.Var, relax.BlockBuilder], relax.Var]
        The function for updating a previous parameter in functions with dequantization.

    param_set : Set[relax.Var]
        The set of previous parameters (before applying quantization and dequantization)
        in the relax functions.
    """

    mod: tvm.IRModule
    func2param_var: Dict[str, relax.Var]
    f_replace: Callable[[relax.Var, relax.BlockBuilder], relax.Var]
    param_set: Set[relax.Var]

    cur_func_name: str

    def __init__(
        self,
        mod: tvm.IRModule,
        func2param_var: Dict[str, relax.Var],
        f_replace: Callable[[relax.Var, relax.BlockBuilder], relax.Var],
    ):
        super().__init__(mod)
        self.mod = mod
        self.func2param_var = func2param_var
        self.f_replace = f_replace
        self.cur_func_name = ""

    def transform(self) -> tvm.IRModule:
        for gv, func in self.mod.functions.items():
            if not isinstance(func, relax.Function):
                continue
            if func.attrs is None or not "num_input" in func.attrs:
                continue

            assert (
                gv.name_hint in self.func2param_var
            ), f"{gv.name_hint} not in {self.func2param_var}"
            self.cur_func_name = gv.name_hint
            updated_func = self.rewrite_func(func, self.func2param_var[gv.name_hint])
            updated_func = remove_all_unused(updated_func)
            self.builder_.update_func(gv, updated_func)
        return self.builder_.get()

    def rewrite_func(self, func: Function, param_var: relax.Var) -> relax.Function:
        num_input = int(func.attrs["num_input"])
        self.param_set = set(func.params[num_input:])

        body = self.visit_expr(func.body)
        return relax.Function(
            params=func.params[:num_input] + [param_var],
            body=body,
            ret_struct_info=func.ret_struct_info,
            is_pure=func.is_pure,
            attrs=func.attrs,
        ).without_attr("num_input")

    def visit_var_(self, var: Var) -> Expr:
        if var not in self.param_set:
            return super().visit_var_(var)
        return self.f_replace(var, self.builder_, self.cur_func_name)


##################################################################


def load_torch_pname2binname_map(
    model_path: str,
    use_safetensors: bool,
    relax_pnames: Set[str],
    f_convert_pname_fwd: Callable[[str], List[str]] = lambda pname: [pname],
) -> Dict[str, str]:
    """Constructing the dictionary from each torch parameter's name to
    the name of the binary shard where the torch parameter is saved.

    Parameters
    ----------
    model_path : str
        The path of the Hugging Face model on disk.

    use_safetensors: bool
        Whether to use ``.safetensors`` instead of ``.bin`` to load model.

    relax_pnames: Set[str]
        The name of the Relax parameters.

    f_convert_pname_fwd: Callable[[str], List[str]]
        The function which converts Relax parameter name to torch's
        parameter names. See ParamManager for more details.
    """
    bin_idx_path = None
    single_shard_file_name = None
    if use_safetensors:
        bin_idx_path = os.path.join(model_path, "model.safetensors.index.json")
        single_shard_file_name = "model.safetensors"
    else:
        bin_idx_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        single_shard_file_name = "pytorch_model.bin"
    single_shard_path = os.path.join(model_path, single_shard_file_name)

    if os.path.isfile(bin_idx_path):
        # Multiple weight shards.
        with open(bin_idx_path, "r") as f_torch_json:
            torch_bin_json = json.load(f_torch_json)
            torch_pname2binname = torch_bin_json["weight_map"]
    elif os.path.isfile(single_shard_path):
        # Single weight shard.
        torch_pname2binname = {
            torch_pname: single_shard_file_name
            for relax_pname in relax_pnames
            for torch_pname in f_convert_pname_fwd(relax_pname)
        }
    else:
        suffix = ".safetensors" if use_safetensors else ".bin"
        shard_names = []
        # Collect Scan every single file with the suffix
        for filename in os.listdir(model_path):
            if filename.endswith(suffix):
                shard_names.append(filename)
        if len(shard_names) == 1:
            torch_pname2binname = {
                torch_pname: shard_names[0]
                for relax_pname in relax_pnames
                for torch_pname in f_convert_pname_fwd(relax_pname)
            }
        else:
            raise ValueError("Multiple weight shard files without json map is not supported")
    return torch_pname2binname


def create_quantize_func(param_manager: ParamManager) -> tvm.IRModule:
    """Construct the Relax function which computes quantization.
    This method is called by `transform_module` below, and is not
    directly invoked outside the class.

    Parameters
    ----------
    param_manager : ParamManager
        The parameter manager which has all the parameter information.

    Returns
    -------
    The created function which computes quantization.
    Precisely, an IRModule which contains the main quantization Relax function
    and a series of TIR functions is returned.
    """
    bb = relax.BlockBuilder()
    param2qrange = dict()

    # Construct the input of the function.
    # We need a list of ranges for each
    # parameter to get its corresponding tensors loaded from disk.
    input_tensor_info: List[relax.TensorStructInfo] = []
    loaded_tensor_ranges: List[range] = []
    for name in param_manager.param_names:
        param = param_manager.params[name]
        _, loaded_tensor_info = param.quant_spec.get_loaded_tensor_info(name, param.param_info)
        loaded_tensor_ranges.append(
            range(
                len(input_tensor_info),
                len(input_tensor_info) + len(loaded_tensor_info),
            )
        )
        input_tensor_info += loaded_tensor_info
    raw_param_tuple = relax.Var("params", relax.TupleStructInfo(input_tensor_info))

    with bb.function("transform_params", params=[raw_param_tuple]):
        with bb.dataflow():
            quantized_params: List[relax.Var] = []
            for pidx, name in enumerate(param_manager.param_names):
                param = param_manager.params[name]
                param_vars: List[relax.Var] = []
                # Emit relax.TupleGetItem to get the raw parameters or pre-quantized params.
                for loaded_tensor_idx in loaded_tensor_ranges[pidx]:
                    param_vars.append(
                        bb.emit(relax.TupleGetItem(raw_param_tuple, loaded_tensor_idx))
                    )

                # Get the quantization function of this parameter.
                f_quantize = param.quant_spec.get_quantize_func(param.param_info)
                if f_quantize is None:
                    # If the parameter does not have a quantization function, either it
                    # does not need quantization or it is pre-quantized.
                    param2qrange[param] = range(
                        len(quantized_params),
                        len(quantized_params) + len(param_vars),
                    )
                    quantized_params += param_vars
                else:
                    # If the parameter has a quantization function, it is not expected
                    # to be pre-quantized.
                    assert len(param_vars) == 1, (
                        "A parameter with quantization function is not expected "
                        "to be pre-quantized."
                    )

                    # Apply the quantization function.
                    quantized_data = bb.emit(f_quantize(bb, param_vars))

                    if isinstance(quantized_data.struct_info, relax.TupleStructInfo):
                        n_tensor = len(quantized_data.struct_info.fields)
                        assert n_tensor > 1
                        # Record the range of quantized tensors of this parameter.
                        param2qrange[param] = range(
                            len(quantized_params), len(quantized_params) + n_tensor
                        )
                        # Collect the quantized tensors to return.
                        for i in range(n_tensor):
                            quantized_params.append(bb.emit(relax.TupleGetItem(quantized_data, i)))
                    else:
                        assert isinstance(quantized_data.struct_info, relax.TensorStructInfo)
                        param2qrange[param] = range(
                            len(quantized_params), len(quantized_params) + 1
                        )
                        quantized_params.append(quantized_data)

            output = bb.emit_output(relax.Tuple(quantized_params))
        bb.emit_func_output(output)

    mod = bb.get()
    param_manager.param2qrange = param2qrange
    # Return the created IRModule.
    return bb.get()
