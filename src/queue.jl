# low level OpenCL queue

immutable Queue
    id::CL_command_queue
end 

@ocl_function(clReleaseCommandQueue, (CL_command_queue,))

#TODO: Better implementation
function free(q::Queue)
    if q.id != C_NULL
        clReleaseCOmmandQueue(q.id)
    end
    q = nothing
end


