import ctypes
from dataclasses import dataclass
import os

import torch


class _DLDevice(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int), ('device_id', ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('device', _DLDevice), ('ndim', ctypes.c_int),
                ('dtype', _DLDataType), ('shape', ctypes.POINTER(ctypes.c_int64)),
                ('strides', ctypes.POINTER(ctypes.c_int64)), ('byte_offset', ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    pass


_Deleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [('dl_tensor', _DLTensor), ('manager_ctx', ctypes.c_void_p), ('deleter', _Deleter)]
_live = {}


@_Deleter
def _release(managed):
    _live.pop(ctypes.addressof(managed.contents), None)


_capsule = ctypes.pythonapi.PyCapsule_New
_capsule.restype = ctypes.py_object
_capsule.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)


def _device_bytes(address, size, device_id):
    kind, index = torch.empty(0, device=f'cuda:{device_id}').__dlpack_device__()
    shape = (ctypes.c_int64 * 1)(size)
    tensor = _DLTensor(address, _DLDevice(kind, index), 1, _DLDataType(1, 8, 1), shape, None, 0)
    managed = _DLManagedTensor(tensor, None, _release)
    pointer = ctypes.addressof(managed)
    _live[pointer] = (managed, shape)
    try:
        return torch.utils.dlpack.from_dlpack(_capsule(pointer, b'dltensor', None))
    except BaseException:
        _live.pop(pointer, None)
        raise


def stloader_open_file(filename):
    return [os.open(filename, os.O_RDONLY)]


def stloader_close_file(handles):
    for handle in handles:
        os.close(handle)


def _read(handles, offset, size):
    if not handles or offset < 0 or size < 0:
        raise ValueError('ファイル読み込みの引数が不正です')
    data = os.pread(handles[0], size, offset)
    if len(data) != size:
        raise EOFError('ファイルの読み込みが不足しています')
    return bytearray(data)


def stloader_read(handles, offset, size, target):
    if not target.is_contiguous() or size > target.numel() * target.element_size():
        raise ValueError('読み込み先のサイズまたは連続性が不正です')
    data = _read(handles, offset, size)
    if size:
        target.view(-1).view(torch.uint8)[:size].copy_(torch.frombuffer(data, dtype=torch.uint8))


@dataclass
class TensorLoadJob:
    handles: list
    file_offset: int
    bytesize: int
    destination: int
    dest_size: int
    bf16_to_fp16: bool
    fp32_to_fp16: bool
    cuda: bool
    device_id: int


def _deferred(jobs, cuda):
    for job in jobs:
        if (job.bytesize < 0 or job.bytesize > job.dest_size or (job.bytesize and not job.destination)
                or (job.bf16_to_fp16 and job.bytesize % 2) or job.cuda != cuda):
            raise ValueError('遅延読み込みの引数が不正です')
    for job in jobs:
        data = _read(job.handles, job.file_offset, job.bytesize)
        if not data:
            continue
        source = torch.frombuffer(data, dtype=torch.uint8)
        if job.bf16_to_fp16:
            source = source.view(torch.bfloat16).to(torch.float16).view(torch.uint8)
        if cuda:
            target = _device_bytes(job.destination, job.bytesize, job.device_id)
            target.copy_(source)
            torch.cuda.synchronize(job.device_id)
            del target
        else:
            ctypes.memmove(job.destination, source.data_ptr(), job.bytesize)


def stloader_deferred_cpu(jobs):
    _deferred(jobs, False)


def stloader_deferred_cuda(jobs, max_chunk_size):
    if max_chunk_size <= 0 or max_chunk_size > 4 * 1024 ** 2 or max_chunk_size % 2:
        raise ValueError('遅延読み込みのチャンクサイズが不正です')
    _deferred(jobs, True)
