将GMLake与PyTorch解耦，使其可以作为一个单独模块被编译为动态链接库，供开发者选择性使用。基于 [torch.cuda.memory.CUDAPluggableAllocator](https://pytorch.org/docs/2.0/generated/torch.cuda.CUDAPluggableAllocator.html#torch.cuda.CUDAPluggableAllocator)实现，要求PyTorch版本2.0及以上。

## 启动容器执行环境

按照如下命令拉取并启动docker镜像

```
# workdir path/to/glake/GMLake
sudo docker run -td  -v .:/GMlake --net=host --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all easydl/glake:v1
```

进入镜像中

```
sudo docker exec -it container-id bash
```

## 编译GMLake

```shell
# Build GMlake 
cd /GMlake
mkdir build
cd build
PYTORCH_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")
# Debug mode
cmake .. -DPYTORCH_DIR=$PYTORCH_PATH -DDEBUG=ON
# Release mode
cmake .. -DPYTORCH_DIR=$PYTORCH_PATH
make
```

如果编译过程正常，build/lib目录下会输出libgmlake.so

## GMLake库使用

进入到模型的测试代码中

```
cd /GMLake-test/DeepSpeed-Chat/training/step1_supervised_finetuning/
```

### 修改main.py

修改当前目录下的`main.py`，通过动态库使用GMLake。可以参考`path/to/glake/GMLake/test/main.py`。

```python
# 1. 添加头文件
import pathlib
import ctypes

# 2. GMlakeAllocator定义
class GMlakeAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str, 
                    init_fn_name: str, reset_fn_name: str, set_memory_fraction_fn_name: str,
                    base_alloc_fn_name: str, record_stream_fn_name: str):
        super().__init__(path_to_so_file, alloc_fn_name, free_fn_name)
        allocator = ctypes.CDLL(path_to_so_file)

        init_fn = ctypes.cast(getattr(allocator, init_fn_name), ctypes.c_void_p).value
        assert init_fn is not None
        self._allocator.set_init_fn(init_fn)

        reset_fn = ctypes.cast(getattr(allocator, reset_fn_name), ctypes.c_void_p).value
        assert reset_fn is not None
        self._allocator.set_reset_fn(reset_fn)

        set_memory_fraction_fn = ctypes.cast(getattr(allocator, set_memory_fraction_fn_name), ctypes.c_void_p).value
        assert set_memory_fraction_fn is not None
        self._allocator.set_memory_fraction_fn(set_memory_fraction_fn)

        base_alloc_fn = ctypes.cast(getattr(allocator, base_alloc_fn_name), ctypes.c_void_p).value
        assert base_alloc_fn is not None
        self._allocator.set_base_alloc_fn(base_alloc_fn)

        record_stream_fn = ctypes.cast(getattr(allocator, record_stream_fn_name), ctypes.c_void_p).value
        assert record_stream_fn is not None
        self._allocator.set_record_stream_fn(record_stream_fn)
        
def main():
    # 3. GMLake加载
    sofile = pathlib.Path("/GMlake/build/lib/libgmlake.so")

    if not sofile.exists():
        raise FileNotFoundError(f"{sofile} not found")

    # Load the GMlake allocator
    gmlake_allocator = GMlakeAllocator(
        str(sofile.absolute()),
        alloc_fn_name="gmlake_malloc",
        free_fn_name="gmlake_free",
        init_fn_name="gmlake_init",
        reset_fn_name="gmlake_reset",
        set_memory_fraction_fn_name="gmlake_memory_fraction",
        base_alloc_fn_name="gmlake_base_alloc",
        record_stream_fn_name="gmlake_record_stream"
    )

    # Swap the current allocator
    torch.cuda.memory.change_current_allocator(gmlake_allocator)

```

### 注释部分代码

CUDAPluggableAllocator目前不支持替换部分接口：[getDeviceStats](https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/CUDAPluggableAllocator.cpp#L194)、[resetPeakStats](https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/CUDAPluggableAllocator.cpp#L209)等，导致程序出现运行时异常，需要通过注释相应代码解决。

1. `/opt/conda/lib/python3.8/site-packages/deepspeed/runtime/utils.py` L786

   ```python
   def see_memory_usage(message, force=False):
       if not force:
           return
       if dist.is_initialized() and not dist.get_rank() == 0:
           return
   
       # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
       gc.collect()
   
       # Print message except when distributed but not rank 0
       logger.info(message)
       # logger.info(f"MA {round(get_accelerator().memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
       #    Max_MA {round(get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
       #    CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
       #    Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")
   
       vm_stats = psutil.virtual_memory()
       used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
       logger.info(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')
   
       # get the peak memory to report correct data, so reset the counter for the next call
       # get_accelerator().reset_peak_memory_stats()
   ```

2. `/opt/conda/lib/python3.8/site-packages/deepspeed/runtime/zero/stage3.py` L1783

   ```python
           # warn user about caching allocator flushes
           '''memory_stats = get_accelerator().memory_stats()
           alloc_retries = memory_stats["num_alloc_retries"] if memory_stats != None else 0
           if alloc_retries > self.__n_caching_allocator_flushes:
               if dist.get_rank() == 0:
                   logger.warning(
                       "%d pytorch allocator cache flushes since last step. this happens "
                       "when there is high memory pressure and is detrimental to "
                       "performance. if this is happening frequently consider adjusting "
                       "settings to reduce memory consumption. If you are unable to "
                       "make the cache flushes go away consider adding "
                       "get_accelerator().empty_cache() calls in your training loop to ensure "
                       "that all ranks flush their caches at the same time",
                       alloc_retries - self.__n_caching_allocator_flushes)
               self.__n_caching_allocator_flushes = alloc_retries'''
   ```

3. `/opt/conda/lib/python3.8/site-packages/deepspeed/utils/timer.py` L207

   ```python
   self.logging(
     "epoch={}/micro_step={}/global_step={}, RunningAvgSamplesPerSec={}, CurrSamplesPerSec={}, "
     "MemAllocated={}GB, MaxMemAllocated={}GB".format(
       self.epoch_count,
       self.micro_step_count,
       self.global_step_count,
       self.avg_samples_per_sec(),
       self.batch_size / self.step_elapsed_time,
       210.5, # 用常量替换memory_allocated()和max_memory_allocated()的调用
       210.5,
       # round(get_accelerator().memory_allocated() / 1024**3, 2),
       # round(get_accelerator().max_memory_allocated() / 1024**3, 2),
     ))
   ```

4. `main.py`

   ```python
   # 注释调用torch.cuda.memory_summary()的代码
   # print_rank_0(torch.cuda.memory_summary())
   # print_rank_0(torch.cuda.memory_summary(), args.global_rank)
   ```

   执行脚本进行训练测试

