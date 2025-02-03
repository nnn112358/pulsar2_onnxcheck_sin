# pulsar2_onnxcheck_sin


In pulsar2 version 3.3, when target_hardware is AX620E, SIN operator could not be converted.
It proceeds to generate quant_axmodel.onnx, but fails to generate axmodel.

According to “2. NPU Operators support list(AX620E)”, Sin is Supported to Unlimited.
https://pulsar2-docs.readthedocs.io/en/latest/appendix/op_support_list_ax620e.html

When target_hardware was AX650,converting the SIN operator passed.
I hope that target_hardware is AX620E can also convert SIN operators.



```
# pulsar2 version
version: 3.3
commit: 3cdead5e

# python sin_export_model.py
Sin Operation:
Input shape: (1, 255)
Output shape: (1, 255)
Output range: [-1.0000, 0.9999]
Match: True

Tar file creation complete: input_data.tar
Temporary files deleted


root#  pulsar2 build --config sin_model_AX620E.json
<frozen quant.ppq.quantization.analyse.graphwise>:110: FutureWarning: Decorating classes is deprecated and will be disabled in future versions. You should only decorate functions or methods. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.
2025-02-04 00:59:27.455 | WARNING  | yamain.command.build:fill_default:224 - apply default input processor configuration to ['x']
2025-02-04 00:59:27.456 | WARNING  | yamain.command.build:fill_default:260 - apply default output processor configuration to ['z']
2025-02-04 00:59:27.457 | WARNING  | yamain.command.build:fill_default:330 - ignore x csc config because of src_format is AutoColorSpace or src_format and tensor_format are the same
2025-02-04 00:59:27.469 | INFO     | yamain.common.util:extract_archive:217 - extract [input_data.tar] to [sin_model_AX620E/quant/dataset/x]...
10 File(s) Loaded.
2025-02-04 00:59:28.208 | INFO     | frontend.parsers.onnx_parser:parse_onnx_model:82 - onnxsim...
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃            ┃ Original Model ┃ Simplified Model ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Constant   │ 0              │ 0                │
│ Sin        │ 1              │ 1                │
│ Model Size │ 75.0B          │ 75.0B            │
└────────────┴────────────────┴──────────────────┘
Building onnx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
2025-02-04 00:59:28.430 | INFO     | frontend.parsers.onnx_parser:parse_onnx_model:109 - [ort sess] model check pass
2025-02-04 00:59:28.530 | INFO     | yamain.command.load_model:optimize_onnx_model:935 - [ort sess] model check pass after transformations
2025-02-04 00:59:28.542 | INFO     | yamain.command.build:quant:723 - save optimized onnx to [sin_model_AX620E/frontend/optimized.onnx]
                                        Quant Config Table
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━┓
┃ Input ┃ Shape    ┃ Dataset Directory                ┃ Data Format ┃ Tensor Format ┃ Mean ┃ Std ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━┩
│ x     │ [1, 255] │ sin_model_AX620E/quant/dataset/x │ Numpy       │               │ []   │ []  │
└───────┴──────────┴──────────────────────────────────┴─────────────┴───────────────┴──────┴─────┘
                     Layer Config Table
┏━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Op Type ┃ Layer name ┃ Precision ┃ Op Calibration Method ┃
┡━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Sin     │ /Sin       │ U8        │ /                     │
└─────────┴────────────┴───────────┴───────────────────────┘
Transformer optimize level: 0
10 File(s) Loaded.
Stastic Inf tensor: 100%|████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1079.61it/s]
[00:59:29] AX Set Float Op Table Pass Running ...
[00:59:29] AX Set MixPrecision Pass Running ...
[00:59:29] AX Set LN Quant dtype Quant Pass Running ...
[00:59:29] AX Reset Mul Config Pass Running ...
[00:59:29] AX Refine Operation Config Pass Running ...
[00:59:29] AX Tanh Operation Format Pass Running ...
[00:59:29] AX Confused Op Refine Pass Running ...
[00:59:30] AX Quantization Fusion Pass Running ...
[00:59:30] AX Quantization Simplify Pass Running ...
[00:59:30] AX Parameter Quantization Pass Running ...
[00:59:30] AX Runtime Calibration Pass Running ...
Calibration Progress(Phase 1): 100%|███████████████████████████████████████████████████| 10/10 [00:00<00:00, 2532.64it/s]
[00:59:30] AX Quantization Alignment Pass Running ...
[00:59:30] AX Refine Int Parameter Pass Running ...
[00:59:30] AX Refine Scale Pass Running ...
[00:59:30] AX Passive Parameter Quantization Running ...
[00:59:30] AX Parameter Baking Pass Running ...
--------- Network Snapshot ---------
Num of Op:                    [1]
Num of Quantized Op:          [1]
Num of Variable:              [2]
Num of Quantized Var:         [2]
------- Quantization Snapshot ------
Num of Quant Config:          [2]
ACTIVATED:                    [2]
Network Quantization Finished.
[Warning]File sin_model_AX620E/quant/quant_axmodel.onnx has already exist, quant exporter will overwrite it.
[Warning]File sin_model_AX620E/quant/quant_axmodel.json has already exist, quant exporter will overwrite it.
Do quant optimization
quant.axmodel export success:
        /data/Sin_test/sin_model_AX620E/quant/quant_axmodel.onnx
        /data/Sin_test/sin_model_AX620E/quant/quant_axmodel.data
===>export io data to folder: sin_model_AX620E/quant/debug/io
Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
2025-02-04 00:59:31.819 | INFO     | yamain.command.build:compile_ptq_model:980 - group 0 compiler transformation
2025-02-04 00:59:31.862 | WARNING  | yamain.command.load_model:pre_process:616 - preprocess tensor [x]
2025-02-04 00:59:31.862 | WARNING  | yamain.command.load_model:post_process:638 - postprocess tensor [z]
2025-02-04 00:59:31.863 | INFO     | yamain.command.load_model:ir_compiler_transformation:824 - use quant data as gt input: x, float32, (1, 255)
2025-02-04 00:59:31.891 | INFO     | yamain.command.build:compile_ptq_model:999 - group 0 QuantAxModel macs: 0
2025-02-04 00:59:31.892 | INFO     | yamain.command.build:compile_ptq_model:1139 - subgraph [0], group: 0, type: GraphType.NPU
2025-02-04 00:59:31.892 | INFO     | yamain.command.npu_backend_compiler:compile:157 - compile npu subgraph [0]
tiling op... ⠋ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/3 0:00:00
Traceback (most recent call last):
  File "<frozen yasched.graph_proc.graph_tiling>", line 109, in run
  File "<frozen backend>", line 158, in f
  File "<frozen backend.ax620e.backend_impl>", line 121, in build
AssertionError: AxQuantizedSin dont support

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<frozen yamain.common.error>", line 59, in guard_context
  File "<frozen yamain.command.build>", line 1169, in compile_ptq_model
  File "<frozen yamain.command.npu_backend_compiler>", line 199, in compile
  File "<frozen yasched.test_onepass>", line 3259, in test_onepass_ir
  File "<frozen yasched.test_onepass>", line 3243, in build
  File "<frozen yasched.test_onepass>", line 3383, in graph2models
  File "<frozen yasched.test_onepass>", line 670, in graph_sched
  File "<frozen yasched.graph_proc.graph_tiling>", line 134, in run
yasched.exceptions.TileFailException: AxQuantizedSin, AxQuantizedSin dont support
    op: /Sin
    attrs: {'const_inputs': {'x_scale': array(0.99607843, dtype=float32), 'r_scale': array(0.00784275, dtype=float32), 'x_zeropoint': array(0, dtype=int32), 'r_zeropoint': array(128, dtype=int32)}, 'output_dtype': 'U8', 'quant_method': 0}
    inputs: {'x': Tensor(U8, name=x_QuantizeLinear_out_link_var, shape=(1, 255), offset=0, bit_strides=(2048, 8))}
    mem_limit: MemLimit(workspace=524288, max_mem_size=720640)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "</opt/pulsar2/yamain/main.py>", line 4, in <module>
  File "<frozen yamain.main>", line 329, in <module>
  File "<frozen yamain.main>", line 257, in pulsar2
  File "<frozen yamain.main>", line 152, in wrapper
  File "<frozen yamain.common.error>", line 23, in wrapper
  File "<frozen yamain.command.build>", line 105, in build_error
  File "<frozen yamain.common.error>", line 14, in wrapper
  File "<frozen yamain.command.build>", line 507, in build
  File "<frozen yamain.command.build>", line 1172, in compile_ptq_model
  File "/usr/local/lib/python3.9/contextlib.py", line 137, in __exit__
    self.gen.throw(typ, value, traceback)
  File "<frozen yamain.common.error>", line 61, in guard_context
  File "<frozen yamain.common.error>", line 73, in error_func
yamain.common.error.CodeException: (<ErrorCode.NPUBackendError: 8>, TileFailException("AxQuantizedSin, AxQuantizedSin dont support\n    op: /Sin\n    attrs: {'const_inputs': {'x_scale': array(0.99607843, dtype=float32), 'r_scale': array(0.00784275, dtype=float32), 'x_zeropoint': array(0, dtype=int32), 'r_zeropoint': array(128, dtype=int32)}, 'output_dtype': 'U8', 'quant_method': 0}\n    inputs: {'x': Tensor(U8, name=x_QuantizeLinear_out_link_var, shape=(1, 255), offset=0, bit_strides=(2048, 8))}\n    mem_limit: MemLimit(workspace=524288, max_mem_size=720640)"))
```





