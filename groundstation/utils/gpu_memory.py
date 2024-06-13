import torch
import gc
import tabulate
import sys
import os


def get_memsize_of_tensor(a: torch.Tensor):
    return a.element_size() * a.nelement()


def tensor_mem_report():
    print("### BEGIN - Memory Report ###")
    print()

    data = []
    mem_sizes = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                mem_size = get_memsize_of_tensor(obj)
                device = str(obj.device)
                data.append([obj.device, obj.size(), mem_size, type(obj), obj.dtype])

                if str(obj.device) not in mem_sizes.keys():
                    mem_sizes[device] = 0

                mem_sizes[device] += mem_size
            else:
                print(obj.device)
        except:
            pass  # some django stuff throws exceptions

    print(tabulate.tabulate(data, headers=['Device', 'Size', 'MemSize', 'Type', 'DataType']))

    print()

    for device in mem_sizes:
        print("%s memory usage: %i byte" % (device.upper(), mem_sizes[device]))
    print("### END - Memory Report ###")


def print_cuda_allocated_memory():
    print(f"CUDA allocated memory: max {torch.cuda.memory_stats('cuda:0')['active_bytes.all.peak']};"
          f" current {torch.cuda.memory_stats('cuda:0')['active_bytes.all.current']}")

def group_mem_report():
    print("### BEGIN - Memory Report ###")
    print()

    data = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                device = str(obj.device)
                obj_type = str(type(obj))
                if device not in data:
                    data[device] = {}

                mem_size = get_memsize_of_tensor(obj)
                if obj_type not in data[device]:
                    data[device][obj_type] = 0

                data[device][obj_type] += mem_size
        except:
            print(obj)
            pass  # some django stuff throws exceptions

    print(tabulate.tabulate(
        [
            (
                d,
                t,
                data[d][t]
            ) for d in data.keys() for t in data[d].keys()],
        headers=['Device', 'Type', 'Size']))

    print("### END - Memory Report ###")


def get_largest_tensor():
    list_of_largest_elements = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                mem_size = get_memsize_of_tensor(obj)
                list_of_largest_elements.append((mem_size, obj))
        except:
            pass  # some django stuff throws exceptions

    list_of_largest_elements.sort(key=lambda x: x[0], reverse=True)

    return list_of_largest_elements


TRACE_CALL_MUTED = False


def trace_calls(frame, event, arg):
    if event != 'call':
        return

    if TRACE_CALL_MUTED:
        return

    return tensor_mem_report()


def register_mem_report_on_each_call():
    sys.settrace(trace_calls)


if __name__ == '__main__':
    a = torch.randn((3, 3))
    b = torch.randn((3, 100)).cuda()

    register_mem_report_on_each_call()


    def test():
        print('a')


    # tensor_mem_report()
    test()
