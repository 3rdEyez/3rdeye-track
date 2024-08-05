#ifndef __RGA_BUFFER_HELPER_H__
#define __RGA_BUFFER_HELPER_H__
#include <cstdio>
#include <cstring>
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "dma_alloc.h"

class rga_buffer_helper
{
public:
    rga_buffer_helper(int width, int height, int format);
    ~rga_buffer_helper();
    int load_from(void *buf, int size);
    rga_buffer_t &get_rga_buf();
    void const *get_buf();
private:
    rga_buffer_t rga_buf;
    char *buf = nullptr;
    rga_buffer_handle_t handle;
    int fd = 0;
    int buf_size;
};

#endif //__RGA_BUFFER_HELPER_H__