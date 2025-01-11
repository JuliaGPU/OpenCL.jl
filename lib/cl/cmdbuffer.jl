# list

export ZeCommandList, execute!

mutable struct ZeCommandList
    handle::ze_command_list_handle_t

    context::ZeContext
    device::ZeDevice

    function ZeCommandList(ctx::ZeContext, dev::ZeDevice, ordinal=1; flags=0)
        desc_ref = Ref(ze_command_list_desc_t(;
            commandQueueGroupOrdinal=ordinal-1, flags,
        ))
        handle_ref = Ref{ze_command_list_handle_t}()
        zeCommandListCreate(ctx, dev, desc_ref, handle_ref)
        obj = new(handle_ref[], ctx, dev)
        finalizer(obj) do obj
            zeCommandListDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_command_list_handle_t}, list::ZeCommandList) = list.handle

Base.:(==)(a::ZeCommandList, b::ZeCommandList) = a.handle == b.handle
Base.hash(e::ZeCommandList, h::UInt) = hash(e.handle, h)

Base.close(list::ZeCommandList) = zeCommandListClose(list)

Base.reset(list::ZeCommandList) = zeCommandListReset(list)

"""
    ZeCommandList(dev::ZeDevice, ...) do list
        append_...!(list)
    end

Create a command list for device `dev`, passing in a do block that appends operations.
The list is then closed and can be used immediately, e.g. for execution.

"""
function ZeCommandList(f::Base.Callable, args...; kwargs...)
    list = ZeCommandList(args...; kwargs...)
    f(list)
    close(list)
    return list
end

execute!(queue::ZeCommandQueue, lists::Vector{ZeCommandList}, fence=nothing) =
    zeCommandQueueExecuteCommandLists(queue, length(lists), lists, something(fence, C_NULL))

"""
    execute!(queue::ZeCommandQueue, ...) do list
        append_...!(list)
    end

Create a command list for the device that owns `queue`, passing in a do block that appends
operations. The list is then closed and executed on the queue.
"""
function execute!(f::Base.Callable, queue::ZeCommandQueue, fence=nothing; kwargs...)
    list = ZeCommandList(f, queue.context, queue.device, queue.ordinal; kwargs...)
    execute!(queue, [list], fence)
end

