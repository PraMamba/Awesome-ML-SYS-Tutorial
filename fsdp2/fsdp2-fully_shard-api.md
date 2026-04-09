# torch.distributed.fsdp.fully_shard

## PyTorch FSDP2 ( `fully_shard` )

PyTorch FSDP2 (RFC) provides a fully sharded data parallelism (FSDP) implementation targeting performant eager-mode while using per-parameter sharding for improved usability

- See the Getting Started with FSDP2 tutorial for more information.
- If you are currently using FSDP1, consider migrating to FSDP2 using our migration guide .

The user contract for `fully_shard(model)` is as follows

- For model initialization, fully_shard converts model.parameters() from plain torch.Tensor to DTensor in-place. The parameters are moved to the appropriate device according to the device mesh.
- Before forward and backward passes, pre-forward/backward hooks are responsible for all-gathering the parameters and converting model.parameters() from DTensor to plain torch.Tensor.
- After forward and backward passes, post-forward/backward hooks free the unsharded parameters (no communication needed) and convert model.parameters() from plain torch.Tensor back to DTensor.
- For the optimizer, it must be initialized with the DTensor model.parameters(), and the optimizer step should be performed on DTensor parameters.
- Call `model(input)` instead of `model.forward(input)` to trigger pre-forward hooks to all-gather parameters. To make model.forward(input) work, users must either call `model.unshard()` explicitly or use `register_fsdp_forward_method(model, "forward")` to register the forward method for hooking.
- fully_shard groups parameters together for a single all-gather. User should apply fully_shard in a bottom-up manner. For example, in a Transformer model, fully_shard should be applied to each layer before applying it to the root model. When applied to the root model, fully_shard excludes model.parameters() from each layer and groups the remaining parameters (e.g., embeddings, output projection) into a single all-gather group.
- `type(model)` is “unioned” with `FSDPModule` in-place. For example, if model is originally of type nn.Linear, then fully_shard changes `type(model)` from nn.Linear to `FSDPLinear` in-place. `FSDPLinear` is an instance of both nn.Linear and `FSDPModule` . It retains all methods of nn.Linear while also exposing FSDP2-specific APIs under FSDPModule, such as `reshard()` and `unshard()` .
- Fully Qualified Names (FQNs) for parameters remain unchanged. If we call `model.state_dict()` , the FQNs are the same before and after applying fully_shard. This is because fully_shard does not wrap the module but only registers hooks to the original module.

Compared to PyTorch FSDP1 ( `FullyShardedDataParallel` ):

- FSDP2 uses `DTensor` -based dim-0 per-parameter sharding for a simpler sharding representation compared to FSDP1’s flat-parameter sharding, while preserving similar throughput performance. More specifically, FSDP2 chunks each parameter on dim-0 across the data parallel workers (using `torch.chunk(dim=0)` ), whereas FSDP1 flattens, concatenates, and chunks a group of tensors together, making reasoning about what data is present on each worker and resharding to different parallelisms complex. Per-parameter sharding provides a more intuitive user experience, relaxes constraints around frozen parameters, and allows for communication-free (sharded) state dicts, which otherwise require all-gathers in FSDP1.
- FSDP2 implements a different memory management approach to handle the multi-stream usages that avoids `torch.Tensor.record_stream` . This ensures deterministic and expected memory usage and does not require blocking the CPU like in FSDP1’s `limit_all_gathers=True` .
- FSDP2 exposes APIs for manual control over prefetching and collective scheduling, allowing power users more customization. See the methods on `FSDPModule` below for details.
- FSDP2 simplifies some of the API surface: e.g. FSDP2 does not directly support full state dicts. Instead, users can reshard the sharded state dicts containing `DTensor` s to full state dicts themselves using `DTensor` APIs like `DTensor.full_tensor()` or by using higher-level APIs like PyTorch Distributed Checkpoint ‘s distributed state dict APIs. Also, some other args have been removed; see here for details.

The frontend API is `fully_shard` that can be called on a `module` :

torch.distributed.fsdp. fully_shard ( module , * , mesh = None , reshard_after_forward = None , shard_placement_fn = None , mp_policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=True) , offload_policy = OffloadPolicy() , ignored_params = None ) [source]
Apply fully sharded data parallelism (FSDP) to `module` , where FSDP shards module parameters, gradients, and optimizer states across data parallel workers to save memory at the cost of communication.
At initialization, FSDP shards the module’s parameters across the data parallel workers given by `mesh` . Before forward, FSDP all-gathers the sharded parameters across the data-parallel workers to get the unsharded parameters for forward computation. If `reshard_after_forward` is `True` , then FSDP frees the unsharded parameters after forward and re-all-gathers them in backward before gradient computation. After gradient computation, FSDP frees the unsharded parameters and reduce-scatters the unsharded gradients across data-parallel workers.
This implementation represents the sharded parameters as `DTensor` s sharded on dim-0, while the unsharded parameters will be like the original parameters on `module` (e.g. `torch.Tensor` if originally `torch.Tensor` ). A module forward pre-hook on `module` all-gathers the parameters, and a module forward hook on `module` frees them (if needed). Similar backward hooks all-gather parameters and later free parameters and reduce-scatter gradients.
Since grouping multiple tensors together for one collective is critical for communication efficiency, this implementation makes this grouping first class. Calling `fully_shard()` on `module` constructs one group that includes the parameters in `module.parameters()` except those already assigned to a group from an earlier call on a submodule. This means that `fully_shard()` should be called bottom-up on your model. Each group’s parameters are all-gathered in one collective, and its gradients are reduce-scattered in one collective. Partitioning the model into multiple groups (“layer by layer”) allows for peak memory savings and communication/computation overlap. Users generally should not call `fully_shard()` only on the topmost root module.
Parameters :

- module ( Union [ nn.Module , List [ nn.Module ] ) – The module or modules to shard with FSDP and group together for communication.
- mesh ( Optional [ DeviceMesh ] ) – This data parallel mesh defines the sharding and device. If 1D, then parameters are fully sharded across the 1D mesh (FSDP) with `(Shard(0),)` placement. If 2D, then parameters are sharded across the 1st dim and replicated across the 0th dim (HSDP) with `(Replicate(), Shard(0))` placement. The mesh’s device type gives the device type used for communication; if a CUDA or CUDA-like device type, then we use the current device.
- reshard_after_forward ( Optional [ Union [ bool , int ] ] ) –
This controls the parameter behavior after forward and can trade off memory and communication:

  - If `True` , then this reshards parameters after forward and re-all-gathers in backward.
  - If `False` , then this keeps the unsharded parameters in memory after forward and avoids the all-gather in backward. For best performance, we usually set `False` for the root module, because the root module is typically required immediately when the backward pass begins.
  - If `None` , it is set to `True` for non-root modules and `False` for root modules.
  - If an `int` , then this represents the world size to reshard to after forward. It should be a non-trivial divisor of the `mesh` shard dim size (i.e. excluding 1 and the dim size itself). A choice may be the intra-node size (e.g. `torch.cuda.device_count()` ). This allows the all-gather in backward to be over a smaller world size at the cost of higher memory usage than setting to `True` .
  - After forward, the parameters registered to the module depend on to this: The registered parameters are the sharded parameters if `True` ; unsharded parameters if `False` ; and the parameters resharded to the smaller mesh otherwise. To modify the parameters between forward and backward, the registered parameters must be the sharded parameters. For `False` or an `int` , this can be done by manually resharding via `reshard()` .
- shard_placement_fn ( Optional [ Callable [ [ nn.Parameter ] , Optional [ Shard ] ] ] ) – This callable can be used to override the sharding placement for a parameter to shard a parameter on a dimension other than dim-0. If this callable returns a `Shard` placement (not `None` ), then FSDP will shard according to that placement (e.g. `Shard(1)` ). If sharding on a nonzero dim, we currently require even sharding, i.e. the tensor dim size on that dim must be divisible by the FSDP shard mesh size.
- mp_policy ( MixedPrecisionPolicy ) – This controls the mixed precision policy, which offers parameter/reduction mixed precision for this module. See `MixedPrecisionPolicy` for details.
- offload_policy ( OffloadPolicy ) – This controls the offloading policy, which offers parameter/gradient/optimizer state offloading. See `OffloadPolicy` and its subclasses for details.
- ignored_params ( Optional [ set [ nn.Parameter ] ] ) – Optional(Set[nn.Parameter]): The set of parameters to be ignored by FSDP. They will not be sharded, nor moved to the device during init, nor have their gradients reduced in backward.
Returns :
The module with FSDP applied (in-place).
Return type :
FSDPModule

class torch.distributed.fsdp. FSDPModule ( * args , ** kwargs )

unshard ( async_op = False ) [source]
Unshards the module’s parameters by allocating memory and all-gathering the parameters. This method is not recursive. The unshard follows the `MixedPrecisionPolicy` , so it will all-gather following `param_dtype` if set.
Note
If `async_op=True` , then FSDP will wait on the pending unshard in the module’s pre-forward for the user. The user only needs to call `wait()` explicitly if the wait should happen before pre-forward.
Parameters :
async_op ( bool ) – If `True` , then returns a `UnshardHandle` that has a `wait()` method to wait on the unshard op. If `False` , then returns `None` and waits on the handle inside this function.
Return type :
UnshardHandle | None

class torch.distributed.fsdp. UnshardHandle
A handle to wait on a `FSDPModule.unshard()` op.
wait ( ) [source] #
Waits on the unshard op. This ensures that the current stream can use the unsharded parameters, which are now registered to the module.

torch.distributed.fsdp. register_fsdp_forward_method ( module , method_name ) [source] #
Registers a method on `module` to be considered a forward method for FSDP.
FSDP all-gathers parameters pre-forward and optionally frees parameters post-forward (depending on `reshard_after_forward` ). FSDP only knows to do this for `nn.Module.forward()` by default. This function patches a user-specified method to run the pre/post-forward hooks before/after the method, respectively. If `module` is not an `FSDPModule` , then this is a no-op.
Parameters :

- module ( nn.Module ) – Module to register the forward method on.
- method_name ( str ) – Name of the forward method.

class torch.distributed.fsdp. MixedPrecisionPolicy ( param_dtype = None , reduce_dtype = None , output_dtype = None , cast_forward_inputs = True ) #
This configures FSDP’s mixed precision. Unlike autocast, this applies mixed precision at the module level, not op level, which means low-precision activations are saved for backward and high-to-low-precision casts are incurred only at module boundaries.
FSDP works well with module-level mixed precision since it keeps the high-precision sharded parameters in memory anyway. In other words, FSDP does not require any extra memory to keep a high-precision copy of the parameters for the optimizer step.

Variables :

- param_dtype ( Optional [ torch.dtype ] ) – This specifies the dtype for the unsharded parameter and hence the dtype for forward/backward computation and the parameter all-gather. If this is `None` , then the unsharded parameter uses the original dtype. The optimizer step uses the sharded parameter in the original dtype. (Default: `None` )
- reduce_dtype ( Optional [ torch.dtype ] ) – This specifies the dtype for gradient reduction (i.e. reduce-scatter or all-reduce). If this is `None` but `param_dtype` is not `None` , then the reduction uses the compute dtype. This can be used to run gradient reduction in full precision while using low precision for compute. If also gradient reduction is disabled via `set_requires_gradient_sync()` , then FSDP will accumulate gradients using `reduce_dtype` . (Default: `None` )
- output_dtype ( Optional [ torch.dtype ] ) – This specifies the dtype for casting floating-point forward outputs. This can be used to help implement cases where different modules have different mixed precision policies. (Default: `None` )
- cast_forward_inputs ( bool ) – This specifies whether FSDP should cast the forward’s floating-point input tensors to `param_dtype` or not.

class torch.distributed.fsdp. OffloadPolicy
This base class represents the policy of no offloading and is only used as the default value for the `offload_policy` arg.

class torch.distributed.fsdp. CPUOffloadPolicy ( pin_memory = True )
This offload policy offloads parameters, gradients, and optimizer states to CPU. Sharded parameters are copied host-to-device before all-gather. The all-gathered parameters are freed according to `reshard_after_forward` . Sharded gradients are copied device-to-host in backward, and the optimizer step runs on CPU with CPU optimizer states.
Variables :
pin_memory ( bool ) – Whether to pin sharded parameter and gradient memory. Pinning memory allows both more efficient H2D/D2H copies and for the copies to overlap with compute. However, the pinned memory cannot be used by other processes. Set this to `False` if you have insufficient CPU memory. (Default: `True` )

torch.distributed.fsdp. share_comm_ctx ( modules ) [source]
Share cuda streams for multiple FSDPModules
For Pipeline Parallelism (PP), each model chunk is a FSDP root. We want to share cuda streams for all-gather, reduce-scatter, and all-reduce. This avoids allocating inter-stream memory framgmentation

Parameters :
modules ( List [ FSDPModule ] ) – modules to share cuda streams
