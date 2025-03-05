# pulsar2_onnxcheck_sin

https://github.com/AXERA-TECH/pulsar2-docs/issues/9


## Issue Overview :
~~In pulsar2 version 3.3, when target_hardware is AX620E, SIN operator could not be converted.
It proceeds to generate quant_axmodel.onnx, but fails to generate axmodel.~~

According to “2. NPU Operators support list(AX620E)”, Sin is Supported to Unlimited.
https://pulsar2-docs.readthedocs.io/en/latest/appendix/op_support_list_ax620e.html

When target_hardware was AX650,converting the SIN operator passed.
~~I hope that target_hardware is AX620E can also convert SIN operators.~~

This issue has been resolved in pulsar2 3.4.
https://github.com/AXERA-TECH/pulsar2-docs/issues/9

## reproduction procedure:

```
user$ sudo docker run -it --net host -v $PWD:/data pulsar2:3.4
root# pulsar2 build --config sin_model_AX620E.json
2025-03-06 05:40:58.486 | WARNING  | yamain.command.build:fill_default:228 - apply default input processor configuration to ['x']
2025-03-06 05:40:58.486 | WARNING  | yamain.command.build:fill_default:265 - apply default output processor configuration to ['z']
2025-03-06 05:40:58.487 | WARNING  | yamain.command.build:fill_default:340 - ignore x csc config because of src_format is AutoColorSpace or src_format and tensor_format are the same
2025-03-06 05:40:58.497 | INFO     | yamain.common.util:extract_archive:217 - extract [input_data.tar] to [sin_model_AX620E/quant/dataset/x]...
10 File(s) Loaded.
2025-03-06 05:40:58.993 | INFO     | frontend.parsers.onnx_parser:parse_onnx_model:72 - onnxsim...
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃            ┃ Original Model ┃ Simplified Model ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Constant   │ 2              │ 2                │
│ Div        │ 1              │ 1                │
│ Mul        │ 1              │ 1                │
│ Sin        │ 1              │ 1                │
│ Model Size │ 337.0B         │ 333.0B           │
└────────────┴────────────────┴──────────────────┘
Building onnx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
2025-03-06 05:40:59.094 | INFO     | frontend.parsers.onnx_parser:parse_onnx_model:99 - [ort sess] model check pass
2025-03-06 05:40:59.218 | INFO     | yamain.command.load_model:optimize_onnx_model:935 - [ort sess] model check pass after transformations
2025-03-06 05:40:59.222 | INFO     | yamain.command.build:quant:748 - save optimized onnx to [sin_model_AX620E/frontend/optimized.onnx]
                                        Quant Config Table
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━┓
┃ Input ┃ Shape    ┃ Dataset Directory                ┃ Data Format ┃ Tensor Format ┃ Mean ┃ Std ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━┩
│ x     │ [1, 255] │ sin_model_AX620E/quant/dataset/x │ Numpy       │               │ []   │ []  │
└───────┴──────────┴──────────────────────────────────┴─────────────┴───────────────┴──────┴─────┘
                      Layer Config Table
┏━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Op Type ┃ Layer name    ┃ Precision ┃ Op Calibration Method ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Mul     │ op_2:onnx.Mul │ U8        │ /                     │
├─────────┼───────────────┼───────────┼───────────────────────┤
│ Sin     │ /Sin          │ U8        │ /                     │
└─────────┴───────────────┴───────────┴───────────────────────┘
Transformer optimize level: 0
10 File(s) Loaded.
Stastic Inf tensor: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1051.99it/s]
[05:41:00] AX Set Float Op Table Pass Running ...
[05:41:00] AX Set MixPrecision Pass Running ...
[05:41:00] AX Set LN Quant dtype Quant Pass Running ...
[05:41:00] AX Reset Mul Config Pass Running ...
[05:41:00] AX Refine Operation Config Pass Running ...
[05:41:00] AX Tanh Operation Format Pass Running ...
[05:41:00] AX Confused Op Refine Pass Running ...
[05:41:00] AX Quantization Fusion Pass Running ...
[05:41:00] AX Quantization Simplify Pass Running ...
[05:41:00] AX Parameter Quantization Pass Running ...
[05:41:00] AX Runtime Calibration Pass Running ...
Calibration Progress(Phase 1): 100%|██████████████████████████████████████████████████| 10/10 [00:00<00:00, 1873.96it/s]
[05:41:01] AX Quantization Alignment Pass Running ...
[05:41:01] AX Refine Int Parameter Pass Running ...
[05:41:01] AX Refine Scale Pass Running ...
[05:41:01] AX Passive Parameter Quantization Running ...
[05:41:01] AX Parameter Baking Pass Running ...
--------- Network Snapshot ---------
Num of Op:                    [2]
Num of Quantized Op:          [2]
Num of Variable:              [4]
Num of Quantized Var:         [4]
------- Quantization Snapshot ------
Num of Quant Config:          [5]
BAKED:                        [1]
OVERLAPPED:                   [1]
ACTIVATED:                    [3]
Network Quantization Finished.
[Warning]File sin_model_AX620E/quant/quant_axmodel.onnx has already exist, quant exporter will overwrite it.
[Warning]File sin_model_AX620E/quant/quant_axmodel.json has already exist, quant exporter will overwrite it.
Do quant optimization
quant.axmodel export success:
        /data/sin_model_AX620E/quant/quant_axmodel.onnx
        /data/sin_model_AX620E/quant/quant_axmodel.data
===>export io data to folder: sin_model_AX620E/quant/debug/io
Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
2025-03-06 05:41:02.444 | INFO     | yamain.command.build:compile_ptq_model:1029 - group 0 compiler transformation
2025-03-06 05:41:02.452 | WARNING  | yamain.command.load_model:pre_process:616 - preprocess tensor [x]
2025-03-06 05:41:02.452 | WARNING  | yamain.command.load_model:post_process:638 - postprocess tensor [z]
2025-03-06 05:41:02.454 | INFO     | yamain.command.load_model:ir_compiler_transformation:824 - use quant data as gt input: x, float32, (1, 255)
2025-03-06 05:41:02.488 | INFO     | yamain.command.build:compile_ptq_model:1052 - group 0 QuantAxModel macs: 0
2025-03-06 05:41:02.490 | INFO     | yamain.command.build:compile_ptq_model:1182 - subgraph [0], group: 0, type: GraphType.NPU
2025-03-06 05:41:02.490 | INFO     | yamain.command.npu_backend_compiler:compile:174 - compile npu subgraph [0]
tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3/3 0:00:00
new_ddr_tensor = []
build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4/4 0:00:00
build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4/4 0:00:00
add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7/7 0:00:00
calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:00
calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:00
assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:00
assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:00
assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:00
2025-03-06 05:41:02.570 | INFO     | yasched.test_onepass:results2model:2682 - clear job deps
2025-03-06 05:41:02.571 | INFO     | yasched.test_onepass:results2model:2683 - max_cycle = 860
build jobs   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:00
2025-03-06 05:41:02.576 | INFO     | yamain.command.npu_backend_compiler:compile:235 - assemble model [0] [subgraph_npu_0] b1
2025-03-06 05:41:02.582 | DEBUG    | backend.ax620e.linker:link_with_dispatcher:1606 - eu_chunk time_limit: True
2025-03-06 05:41:02.582 | DEBUG    | backend.ax620e.linker:link_with_dispatcher:1607 - eu_chunk only_bysize: False
2025-03-06 05:41:02.598 | INFO     | backend.ax620e.linker:link_with_dispatcher:1663 - DispatcherQueueType.IO: Generate 2 EU chunks, 2 Dispatcher Chunk
2025-03-06 05:41:02.598 | INFO     | backend.ax620e.linker:link_with_dispatcher:1663 - DispatcherQueueType.Compute: Generate 1 EU chunks, 2 Dispatcher Chunk
2025-03-06 05:41:02.598 | INFO     | backend.ax620e.linker:link_with_dispatcher:1664 - EU mcode size: 1 KiB
2025-03-06 05:41:02.598 | INFO     | backend.ax620e.linker:link_with_dispatcher:1665 - Dispatcher mcode size: 1 KiB
2025-03-06 05:41:02.598 | INFO     | backend.ax620e.linker:link_with_dispatcher:1666 - Total mcode size: 1 KiB
2025-03-06 05:41:02.614 | INFO     | yamain.command.build:compile_ptq_model:1221 - fuse 1 subgraph(s)

```


```
user$ sudo docker run -it --net host -v $PWD:/data pulsar2:3.3
root# pulsar2 version
version: 3.3
commit: 3cdead5e

root# python sin_export_model.py
root# pulsar2 build --config sin_model_AX620E.json
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





