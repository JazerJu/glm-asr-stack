#include <cuda.h>

typedef void (*init_fn)(CUlibrary *);
typedef int  (*load_dev_fn)(CUlibrary *, int);

/* Generic cuda_dialect_init_library_once override.
 * The TVM FFI already handles cuda_load internally.
 * We just need to call the provided init and load_to_device callbacks. */
int cuda_dialect_init_library_once(void *state, init_fn do_init, load_dev_fn do_load_dev, void *err_handler) {
    (void)err_handler;
    CUlibrary lib = NULL;
    do_init(&lib);
    if (lib) {
        int r = do_load_dev(&lib, 0);
        (void)r;
        if (state) *(CUlibrary *)state = lib;
    }
    return 0;
}
