#include "rga_buffer_helper.h"

rga_buffer_helper::rga_buffer_helper(int width, int height, int format)
{
    int ret;
    memset(&rga_buf, 0, sizeof(rga_buffer_t));
    buf_size = width * height * get_bpp_from_format(format);
    
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, buf_size, &fd, (void **)&buf);
    if (ret) {
        printf("dma_buf_alloc failed\n");
        goto out;
    }
    handle = importbuffer_fd(fd, buf_size);
    if (!handle) {
        printf("importbuffer_fd failed\n");
        goto out;
    }
    rga_buf = wrapbuffer_handle(handle, width, height, format);
    return;
out:
    if (buf) {
        dma_buf_free(buf_size, &fd, buf);
        buf = nullptr;
    }
}

rga_buffer_helper::~rga_buffer_helper()
{
    releasebuffer_handle(handle);
    dma_buf_free(buf_size, &fd, buf);
}


int rga_buffer_helper::load_from(void *buf, int size)
{
    if (size != buf_size) {
        printf("size not match\n");
        return - 1;
    }
    memcpy(this->buf, buf, size);
    return 0;
}

rga_buffer_t& rga_buffer_helper::get_rga_buf()
{
    return rga_buf;
}

void const *rga_buffer_helper::get_buf()
{
    return buf;
}