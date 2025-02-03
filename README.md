# pulsar2_onnxcheck_sin


In pulsar2 version 3.3, when target_hardware is AX620E, SIN operator could not be converted.
It proceeds to generate quant_axmodel.onnx, but fails to generate axmodel.

According to “2. NPU Operators support list(AX620E)”, Sin is Supported to Unlimited.
https://pulsar2-docs.readthedocs.io/en/latest/appendix/op_support_list_ax620e.html

When target_hardware was AX650,converting the SIN operator passed.
I hope that target_hardware is AX620E can also convert SIN operators.
