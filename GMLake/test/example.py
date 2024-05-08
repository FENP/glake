import ctypes
import torch
import pathlib


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

if __name__=='__main__':
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

    # test
    x = torch.rand(5, 3, dtype=torch.float32, device=torch.device('cuda:0'))
    y = x.sum()

    torch.cuda.empty_cache()