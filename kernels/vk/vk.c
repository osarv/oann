/* vk.c - the one place C's types meet olang's for the Vulkan backend (docs/vulkan.md), as kernels/gpu/cuda.c is for
 * CUDA's.
 *
 * Imported by kernels/vk/device.olang as a C module (olang X17, X20). Vulkan's interface is structs of pointers to
 * structs, which olang's C imports hold only as pointers of no type (X18), so the calls are made here and everything
 * crosses as plain numbers: a device, a buffer and a kernel are uint64_t handles, host arrays cross as arrays with an
 * offset in bytes.
 *
 * No link. The Vulkan loader (libvulkan.so.1, vulkan-1.dll, Android's libvulkan.so) is opened when a device is asked
 * for, and each function found in it by name: a program built with the backend runs on a machine with no Vulkan at
 * all (oann_vk_open says so), and a build needs the headers only.
 *
 * The order of things. Everything a device is asked to do - an upload, a kernel's dispatch, a download - is recorded
 * into one command buffer in program order, each behind a barrier on what came before, and runs at the next
 * oann_vk_sync (or sooner, where the staging memory or the descriptor pool is full). So nothing races by construction,
 * as on CUDA's one compute stream. An upload's bytes are copied into staging memory before the call returns; a
 * download's bytes reach the host array at the sync, so the host reads them after it.
 *
 * Errors. Setup calls return Vulkan's VkResult (or one of OANN_VK_*). The recording calls return nothing and note the
 * first failure in the device's record; oann_vk_sync gives it back and clears it.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#ifdef _WIN32
/* kernel32's three, declared here: <windows.h> is far more than a C import needs translated */
__declspec(dllimport) void *__stdcall LoadLibraryA(const char *name);
__declspec(dllimport) void *__stdcall GetProcAddress(void *module, const char *name);
__declspec(dllimport) int __stdcall FreeLibrary(void *module);
static void *lib_open(void) { return LoadLibraryA("vulkan-1.dll"); }
static void *lib_sym(void *lib, const char *name) { return GetProcAddress(lib, name); }
static void lib_close(void *lib) { FreeLibrary(lib); }
#else
#include <dlfcn.h>
static void *lib_open(void) {
#ifdef __ANDROID__
    return dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
#else
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    return lib ? lib : dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
#endif
}
static void *lib_sym(void *lib, const char *name) { return dlsym(lib, name); }
static void lib_close(void *lib) { dlclose(lib); }
#endif

enum { OANN_VK_NO_LOADER = -2000, OANN_VK_NO_DEVICE = -2001, OANN_VK_NO_QUEUE = -2002, OANN_VK_NO_MEMORY_TYPE = -2003 };

/* the functions used, each found in the loader by its name */
#define OANN_VK_FUNCS(X) \
    X(CreateInstance) X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
    X(GetPhysicalDeviceMemoryProperties) X(GetPhysicalDeviceQueueFamilyProperties) X(CreateDevice) X(DestroyDevice) \
    X(GetDeviceQueue) X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) \
    X(EndCommandBuffer) X(ResetCommandBuffer) X(QueueSubmit) X(CreateFence) X(DestroyFence) X(WaitForFences) \
    X(ResetFences) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) \
    X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(CreateShaderModule) X(DestroyShaderModule) \
    X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreatePipelineLayout) X(DestroyPipelineLayout) \
    X(CreateComputePipelines) X(DestroyPipeline) X(CreateDescriptorPool) X(DestroyDescriptorPool) \
    X(ResetDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CmdBindPipeline) \
    X(CmdBindDescriptorSets) X(CmdPushConstants) X(CmdDispatch) X(CmdCopyBuffer) X(CmdFillBuffer) \
    X(CmdPipelineBarrier) X(DeviceWaitIdle)

enum { MAX_SETS = 1024, MAX_BINDINGS = 8, MAX_DOWNLOADS = 256 };

typedef struct {
    void *host;        /* where the bytes go at the sync */
    uint64_t at;       /* where they are in the staging memory */
    uint64_t bytes;
} download;

typedef struct {
    void *lib;
#define X(name) PFN_vk##name name;
    OANN_VK_FUNCS(X)
#undef X
    VkInstance instance;
    VkPhysicalDevice phys;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem;
    VkDevice dev;
    VkQueue queue;
    uint32_t family;
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkDescriptorPool sets;
    uint32_t nsets;        /* descriptor sets given out since the pool was last reset */
    int recording;         /* the command buffer is begun */
    int dirty;             /* something recorded before must be seen by what is recorded next: a barrier is owed */
    VkBuffer stage;        /* host-visible staging memory, for uploads and downloads */
    VkDeviceMemory stagemem;
    uint8_t *stagemap;
    uint64_t stagesize;
    uint64_t stageat;      /* its first free byte */
    download downs[MAX_DOWNLOADS];
    uint32_t ndowns;
    uint32_t refs;         /* the buffers and kernels of this device still alive */
    int closed;            /* its holder let go of it: it is destroyed with the last of them */
    int32_t code;          /* the first failure since the last sync, and the call it was */
    const char *what;
} vkdev;

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint64_t bytes;
} vkbuf;

typedef struct {
    VkShaderModule module;
    VkDescriptorSetLayout setlayout;
    VkPipelineLayout layout;
    VkPipeline pipe;
    uint32_t nbufs;
    uint32_t pushbytes;
} vkkern;

#define D(h) ((vkdev *)(uintptr_t)(h))

static void note(vkdev *d, int32_t code, const char *what) {
    if (code != 0 && d->code == 0) {
        d->code = code;
        d->what = what;
    }
}

/* ---- the device ---- */

static void destroy(vkdev *d) {
    if (d->dev) {
        d->DeviceWaitIdle(d->dev);
        if (d->stagemap) d->UnmapMemory(d->dev, d->stagemem);
        if (d->stage) d->DestroyBuffer(d->dev, d->stage, 0);
        if (d->stagemem) d->FreeMemory(d->dev, d->stagemem, 0);
        if (d->sets) d->DestroyDescriptorPool(d->dev, d->sets, 0);
        if (d->fence) d->DestroyFence(d->dev, d->fence, 0);
        if (d->pool) d->DestroyCommandPool(d->dev, d->pool, 0);
        d->DestroyDevice(d->dev, 0);
    }
    if (d->instance) d->DestroyInstance(d->instance, 0);
    if (d->lib) lib_close(d->lib);
    free(d);
}

/* how much a kind of device is wanted where the caller names none: a discrete GPU, then an integrated one, then a
 * virtual one, last a CPU's (software Vulkan) */
static int rank(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
    default: return 0;
    }
}

/* the memory type of these properties among those bits allows, or -1 */
static int32_t memtype(vkdev *d, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < d->mem.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (d->mem.memoryTypes[i].propertyFlags & want) == want) return (int32_t)i;
    return -1;
}

static int32_t makebuffer(vkdev *d, uint64_t bytes, int host, VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = d->CreateBuffer(d->dev, &bi, 0, buf);
    if (r) return r;
    VkMemoryRequirements req;
    d->GetBufferMemoryRequirements(d->dev, *buf, &req);
    int32_t type;
    if (host) {
        type = memtype(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    } else {
        type = memtype(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0) type = memtype(d, req.memoryTypeBits, 0);
    }
    if (type < 0) {
        d->DestroyBuffer(d->dev, *buf, 0);
        *buf = 0;
        return OANN_VK_NO_MEMORY_TYPE;
    }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = (uint32_t)type;
    r = d->AllocateMemory(d->dev, &ai, 0, mem);
    if (!r) r = d->BindBufferMemory(d->dev, *buf, *mem, 0);
    if (r) {
        d->DestroyBuffer(d->dev, *buf, 0);
        if (*mem) d->FreeMemory(d->dev, *mem, 0);
        *buf = 0;
        *mem = 0;
    }
    return r;
}

/* the number of Vulkan devices here and, into names (cap bytes each, NUL-ended, up to max of them), what each is
 * called; 0 where there is no loader or no driver */
int32_t oann_vk_devices(uint8_t *names, int32_t cap, int32_t max) {
    void *lib = lib_open();
    if (!lib) return 0;
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)lib_sym(lib, "vkCreateInstance");
    PFN_vkDestroyInstance destroyi = (PFN_vkDestroyInstance)lib_sym(lib, "vkDestroyInstance");
    PFN_vkEnumeratePhysicalDevices enumerate = (PFN_vkEnumeratePhysicalDevices)lib_sym(lib, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties getprops = (PFN_vkGetPhysicalDeviceProperties)lib_sym(lib, "vkGetPhysicalDeviceProperties");
    uint32_t n = 0;
    VkInstance inst = 0;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "oann";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    if (create && destroyi && enumerate && getprops && create(&ci, 0, &inst) == VK_SUCCESS) {
        VkPhysicalDevice list[16];
        n = 16;
        VkResult r = enumerate(inst, &n, list);
        if (r != VK_SUCCESS && r != VK_INCOMPLETE) n = 0;
        for (uint32_t i = 0; i < n && (int32_t)i < max && names; i++) {
            VkPhysicalDeviceProperties p;
            getprops(list[i], &p);
            strncpy((char *)names + (size_t)i * (size_t)cap, p.deviceName, (size_t)cap - 1);
            names[(size_t)i * (size_t)cap + (size_t)cap - 1] = 0;
        }
        destroyi(inst, 0);
    }
    lib_close(lib);
    return (int32_t)n;
}

/* opens device number want (as oann_vk_devices lists them), or the most wanted kind for want < 0, with stagebytes of
 * staging memory: the handle into out[0], and 0, else a VkResult or OANN_VK_* (nothing is left open) */
int32_t oann_vk_open(uint64_t *out, int32_t want, uint64_t stagebytes) {
    out[0] = 0;
    void *lib = lib_open();
    if (!lib) return OANN_VK_NO_LOADER;
    vkdev *d = (vkdev *)calloc(1, sizeof(vkdev));
    d->lib = lib;
#define X(name) \
    d->name = (PFN_vk##name)lib_sym(lib, "vk" #name); \
    if (!d->name) { destroy(d); return OANN_VK_NO_LOADER; }
    OANN_VK_FUNCS(X)
#undef X
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "oann";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VkResult r = d->CreateInstance(&ci, 0, &d->instance);
    if (r) { destroy(d); return r == VK_ERROR_INCOMPATIBLE_DRIVER ? OANN_VK_NO_DEVICE : r; }
    VkPhysicalDevice list[16];
    uint32_t n = 16;
    r = d->EnumeratePhysicalDevices(d->instance, &n, list);
    if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || n == 0 || want >= (int32_t)n) { destroy(d); return OANN_VK_NO_DEVICE; }
    uint32_t pick = want < 0 ? 0 : (uint32_t)want;
    if (want < 0) {
        int best = -1;
        for (uint32_t i = 0; i < n; i++) {
            VkPhysicalDeviceProperties p;
            d->GetPhysicalDeviceProperties(list[i], &p);
            if (rank(p.deviceType) > best) { best = rank(p.deviceType); pick = i; }
        }
    }
    d->phys = list[pick];
    d->GetPhysicalDeviceProperties(d->phys, &d->props);
    d->GetPhysicalDeviceMemoryProperties(d->phys, &d->mem);
    VkQueueFamilyProperties fams[32];
    uint32_t nf = 32;
    d->GetPhysicalDeviceQueueFamilyProperties(d->phys, &nf, fams);
    d->family = nf;
    for (uint32_t i = 0; i < nf; i++)
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { d->family = i; break; }
    if (d->family == nf) { destroy(d); return OANN_VK_NO_QUEUE; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = d->family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    r = d->CreateDevice(d->phys, &di, 0, &d->dev);
    if (r) { destroy(d); return r; }
    d->GetDeviceQueue(d->dev, d->family, 0, &d->queue);
    VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = d->family;
    r = d->CreateCommandPool(d->dev, &pi, 0, &d->pool);
    if (r) { destroy(d); return r; }
    VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = d->pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    r = d->AllocateCommandBuffers(d->dev, &ca, &d->cmd);
    if (r) { destroy(d); return r; }
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    r = d->CreateFence(d->dev, &fi, 0, &d->fence);
    if (r) { destroy(d); return r; }
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_SETS * MAX_BINDINGS};
    VkDescriptorPoolCreateInfo dp = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = MAX_SETS;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    r = d->CreateDescriptorPool(d->dev, &dp, 0, &d->sets);
    if (r) { destroy(d); return r; }
    d->stagesize = stagebytes;
    r = makebuffer(d, stagebytes, 1, &d->stage, &d->stagemem);
    if (r) { destroy(d); return r; }
    void *map = 0;
    r = d->MapMemory(d->dev, d->stagemem, 0, VK_WHOLE_SIZE, 0, &map);
    if (r) { destroy(d); return r; }
    d->stagemap = (uint8_t *)map;
    out[0] = (uint64_t)(uintptr_t)d;
    return 0;
}

/* lets go of the device: destroyed now, or where buffers or kernels of it are still alive (a holder's destructors run
 * in an order the shim does not choose), with the last of them */
void oann_vk_close(uint64_t h) {
    vkdev *d = D(h);
    if (d->refs > 0) {
        d->closed = 1;
        return;
    }
    destroy(d);
}

/* a buffer or a kernel of the device is gone */
static void unref(vkdev *d) {
    d->refs--;
    if (d->closed && d->refs == 0) destroy(d);
}

/* the device's name into name (cap bytes, NUL-ended) and into info: its kind (VkPhysicalDeviceType), Vulkan version
 * (major, minor), PCI vendor, the MiB of its largest device-local heap, the most elements of F32 one buffer binding
 * holds in MiB, the most workgroups a dispatch takes along x, the most invocations a workgroup holds and the bytes of
 * its shared memory */
void oann_vk_info(uint64_t h, uint8_t *name, int32_t cap, int32_t *info) {
    vkdev *d = D(h);
    strncpy((char *)name, d->props.deviceName, (size_t)cap - 1);
    name[cap - 1] = 0;
    uint64_t heap = 0;
    for (uint32_t i = 0; i < d->mem.memoryHeapCount; i++)
        if ((d->mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && d->mem.memoryHeaps[i].size > heap)
            heap = d->mem.memoryHeaps[i].size;
    info[0] = (int32_t)d->props.deviceType;
    info[1] = (int32_t)VK_VERSION_MAJOR(d->props.apiVersion);
    info[2] = (int32_t)VK_VERSION_MINOR(d->props.apiVersion);
    info[3] = (int32_t)d->props.vendorID;
    info[4] = (int32_t)(heap >> 20);
    info[5] = (int32_t)(d->props.limits.maxStorageBufferRange >> 20);
    info[6] = (int32_t)(d->props.limits.maxComputeWorkGroupCount[0] > 0x7fffffffu ? 0x7fffffffu : d->props.limits.maxComputeWorkGroupCount[0]);
    info[7] = (int32_t)(d->props.limits.maxComputeWorkGroupInvocations > 0x7fffffffu ? 0x7fffffffu : d->props.limits.maxComputeWorkGroupInvocations);
    info[8] = (int32_t)(d->props.limits.maxComputeSharedMemorySize > 0x7fffffffu ? 0x7fffffffu : d->props.limits.maxComputeSharedMemorySize);
}

/* ---- recording and running ---- */

static void begin(vkdev *d) {
    if (d->recording) return;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    note(d, d->BeginCommandBuffer(d->cmd, &bi), "vkBeginCommandBuffer");
    d->recording = 1;
    d->dirty = 0;
}

/* what is recorded next sees everything recorded before it: writes by a kernel or a copy are made visible to reads and
 * writes by the next kernel or copy */
static void barrier(vkdev *d) {
    begin(d);
    if (!d->dirty) {
        d->dirty = 1;
        return;
    }
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    mb.dstAccessMask = mb.srcAccessMask;
    VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    d->CmdPipelineBarrier(d->cmd, stages, stages, 0, 1, &mb, 0, 0, 0, 0);
}

/* runs what is recorded and waits for it; the downloads' bytes go to their host arrays; staging memory and the
 * descriptor sets are free again */
static void flush(vkdev *d) {
    if (d->recording) {
        note(d, d->EndCommandBuffer(d->cmd), "vkEndCommandBuffer");
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cmd;
        VkResult r = d->QueueSubmit(d->queue, 1, &si, d->fence);
        note(d, r, "vkQueueSubmit");
        if (!r) note(d, d->WaitForFences(d->dev, 1, &d->fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        d->ResetFences(d->dev, 1, &d->fence);
        d->ResetCommandBuffer(d->cmd, 0);
        d->recording = 0;
    }
    if (d->code == 0)
        for (uint32_t i = 0; i < d->ndowns; i++) memcpy(d->downs[i].host, d->stagemap + d->downs[i].at, d->downs[i].bytes);
    d->ndowns = 0;
    d->stageat = 0;
    if (d->nsets) {
        d->ResetDescriptorPool(d->dev, d->sets, 0);
        d->nsets = 0;
    }
}

/* everything recorded has run when this returns: 0, or the first failure since the last sync (its call's name from
 * oann_vk_what until the next recording call), the record cleared */
int32_t oann_vk_sync(uint64_t h) {
    vkdev *d = D(h);
    flush(d);
    int32_t code = d->code;
    d->code = 0;
    return code;
}

/* the call the last sync's failure was */
const char *oann_vk_what(uint64_t h) { return D(h)->what ? D(h)->what : ""; }

/* ---- device memory ---- */

/* bytes of device memory: the buffer's handle into out[0], and 0, else Vulkan's status */
int32_t oann_vk_buf_new(uint64_t h, uint64_t bytes, uint64_t *out) {
    vkdev *d = D(h);
    out[0] = 0;
    vkbuf *b = (vkbuf *)calloc(1, sizeof(vkbuf));
    b->bytes = bytes;
    int32_t r = makebuffer(d, bytes, 0, &b->buf, &b->mem);
    if (r) {
        free(b);
        return r;
    }
    d->refs++;
    out[0] = (uint64_t)(uintptr_t)b;
    return 0;
}

/* frees a buffer, after what is recorded (which may name it) has run */
void oann_vk_buf_free(uint64_t h, uint64_t buf) {
    vkdev *d = D(h);
    vkbuf *b = (vkbuf *)(uintptr_t)buf;
    flush(d);
    d->DestroyBuffer(d->dev, b->buf, 0);
    d->FreeMemory(d->dev, b->mem, 0);
    free(b);
    unref(d);
}

/* bytes of host from byte hostoff to the buffer from byte at, in order; host may be written again on return */
void oann_vk_upload(uint64_t h, uint64_t buf, uint64_t at, const void *host, uint64_t hostoff, uint64_t bytes) {
    vkdev *d = D(h);
    vkbuf *b = (vkbuf *)(uintptr_t)buf;
    const uint8_t *src = (const uint8_t *)host + hostoff;
    while (bytes > 0) {
        if (d->stageat == d->stagesize) flush(d);
        uint64_t n = d->stagesize - d->stageat;
        if (n > bytes) n = bytes;
        memcpy(d->stagemap + d->stageat, src, n);
        barrier(d);
        VkBufferCopy c = {d->stageat, at, n};
        d->CmdCopyBuffer(d->cmd, d->stage, b->buf, 1, &c);
        d->stageat += (n + 15) & ~(uint64_t)15;
        if (d->stageat > d->stagesize) d->stageat = d->stagesize;
        src += n;
        at += n;
        bytes -= n;
    }
}

/* bytes of the buffer from byte at to host from byte hostoff, in order; the host reads them after the next sync */
void oann_vk_download(uint64_t h, void *host, uint64_t hostoff, uint64_t buf, uint64_t at, uint64_t bytes) {
    vkdev *d = D(h);
    vkbuf *b = (vkbuf *)(uintptr_t)buf;
    uint8_t *dst = (uint8_t *)host + hostoff;
    while (bytes > 0) {
        if (d->stageat == d->stagesize || d->ndowns == MAX_DOWNLOADS) flush(d);
        uint64_t n = d->stagesize - d->stageat;
        if (n > bytes) n = bytes;
        barrier(d);
        VkBufferCopy c = {at, d->stageat, n};
        d->CmdCopyBuffer(d->cmd, b->buf, d->stage, 1, &c);
        download w = {dst, d->stageat, n};
        d->downs[d->ndowns++] = w;
        d->stageat += (n + 15) & ~(uint64_t)15;
        if (d->stageat > d->stagesize) d->stageat = d->stagesize;
        dst += n;
        at += n;
        bytes -= n;
    }
}

/* bytes copied from one buffer to another (or within one, where the ranges do not overlap), in order */
void oann_vk_copy(uint64_t h, uint64_t to, uint64_t toat, uint64_t from, uint64_t fromat, uint64_t bytes) {
    vkdev *d = D(h);
    barrier(d);
    VkBufferCopy c = {fromat, toat, bytes};
    d->CmdCopyBuffer(d->cmd, ((vkbuf *)(uintptr_t)from)->buf, ((vkbuf *)(uintptr_t)to)->buf, 1, &c);
}

/* bytes of a buffer from byte at (both multiples of 4) set to the 32-bit word, in order */
void oann_vk_fill(uint64_t h, uint64_t buf, uint64_t at, uint64_t bytes, uint32_t word) {
    vkdev *d = D(h);
    barrier(d);
    d->CmdFillBuffer(d->cmd, ((vkbuf *)(uintptr_t)buf)->buf, at, bytes, word);
}

/* ---- kernels ---- */

/* a compute pipeline of a SPIR-V module (nwords words, its entry point "main") reading nbufs storage buffers at
 * bindings 0 .. nbufs - 1 of set 0 and pushbytes of push constants: its handle into out[0], and 0, else Vulkan's
 * status (nothing is left made) */
int32_t oann_vk_kernel_new(uint64_t h, const uint32_t *words, int64_t nwords, int32_t nbufs, int32_t pushbytes, uint64_t *out) {
    vkdev *d = D(h);
    out[0] = 0;
    if (nbufs > MAX_BINDINGS) return VK_ERROR_TOO_MANY_OBJECTS;
    vkkern *k = (vkkern *)calloc(1, sizeof(vkkern));
    k->nbufs = (uint32_t)nbufs;
    k->pushbytes = (uint32_t)pushbytes;
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = (size_t)nwords * 4;
    mi.pCode = words;
    VkResult r = d->CreateShaderModule(d->dev, &mi, 0, &k->module);
    VkDescriptorSetLayoutBinding binds[MAX_BINDINGS];
    memset(binds, 0, sizeof(binds));
    for (int32_t i = 0; i < nbufs; i++) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    if (!r) {
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)nbufs;
        li.pBindings = binds;
        r = d->CreateDescriptorSetLayout(d->dev, &li, 0, &k->setlayout);
    }
    if (!r) {
        VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pushbytes};
        VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &k->setlayout;
        pl.pushConstantRangeCount = pushbytes > 0 ? 1 : 0;
        pl.pPushConstantRanges = &pr;
        r = d->CreatePipelineLayout(d->dev, &pl, 0, &k->layout);
    }
    if (!r) {
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = k->module;
        ci.stage.pName = "main";
        ci.layout = k->layout;
        r = d->CreateComputePipelines(d->dev, 0, 1, &ci, 0, &k->pipe);
    }
    if (r) {
        if (k->layout) d->DestroyPipelineLayout(d->dev, k->layout, 0);
        if (k->setlayout) d->DestroyDescriptorSetLayout(d->dev, k->setlayout, 0);
        if (k->module) d->DestroyShaderModule(d->dev, k->module, 0);
        free(k);
        return r;
    }
    d->refs++;
    out[0] = (uint64_t)(uintptr_t)k;
    return 0;
}

/* frees a kernel, after what is recorded (which may name it) has run */
void oann_vk_kernel_free(uint64_t h, uint64_t kern) {
    vkdev *d = D(h);
    vkkern *k = (vkkern *)(uintptr_t)kern;
    flush(d);
    d->DestroyPipeline(d->dev, k->pipe, 0);
    d->DestroyPipelineLayout(d->dev, k->layout, 0);
    d->DestroyDescriptorSetLayout(d->dev, k->setlayout, 0);
    d->DestroyShaderModule(d->dev, k->module, 0);
    free(k);
    unref(d);
}

/* the kernel over x by y by z workgroups, its bindings these buffers (whole) and its push constants these words, in
 * order */
void oann_vk_dispatch(uint64_t h, uint64_t kern, uint32_t x, uint32_t y, uint32_t z, const uint64_t *bufs, const uint32_t *push) {
    vkdev *d = D(h);
    vkkern *k = (vkkern *)(uintptr_t)kern;
    if (d->nsets == MAX_SETS) flush(d);
    barrier(d);
    VkDescriptorSet set = 0;
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = d->sets;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &k->setlayout;
    VkResult r = d->AllocateDescriptorSets(d->dev, &ai, &set);
    if (r) {
        note(d, r, "vkAllocateDescriptorSets");
        return;
    }
    d->nsets++;
    VkDescriptorBufferInfo infos[MAX_BINDINGS];
    VkWriteDescriptorSet writes[MAX_BINDINGS];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < k->nbufs; i++) {
        infos[i].buffer = ((vkbuf *)(uintptr_t)bufs[i])->buf;
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    d->UpdateDescriptorSets(d->dev, k->nbufs, writes, 0, 0);
    d->CmdBindPipeline(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipe);
    d->CmdBindDescriptorSets(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &set, 0, 0);
    if (k->pushbytes) d->CmdPushConstants(d->cmd, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, k->pushbytes, push);
    d->CmdDispatch(d->cmd, x, y, z);
}
