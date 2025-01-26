# Tensor Parallel

# 1 Torchtitan 中 TP 的应用
## 1.1 对 embedding-norm-output进行parallel
```python
"""Apply tensor parallelism."""
# 1. Parallelize the embedding and shard its outputs (which are the first
# transformer block's inputs)
# 2. Parallelize the root norm layer over the sequence dim
# 3. Parallelize the final linear output layer
parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        ),
    },
)
```

## 1.2 对 其他layer进行parallel
```python
for layer_id, transformer_block in model.layers.items():
    layer_plan = {
        "attention_norm": SequenceParallel(),
        "attention": prepare_module_input(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": colwise_parallel(),
        "attention.wk": colwise_parallel(),
        "attention.wv": colwise_parallel(),
        "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": prepare_module_input(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": colwise_parallel(),
        "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
        "feed_forward.w3": colwise_parallel(),
    }

    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
    )
```









