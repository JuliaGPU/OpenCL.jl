; ModuleID = 'start'
source_filename = "start"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-unknown-unknown-unknown"

@0 = private unnamed_addr addrspace(2) constant [20 x i8] c"length(y[1]) = %ld\0A\00", align 1
@1 = private unnamed_addr addrspace(2) constant [20 x i8] c"length(x) > 0 = %d\0A\00", align 1
@2 = private unnamed_addr addrspace(2) constant [5 x i8] c"Bar\0A\00", align 1
@3 = private unnamed_addr addrspace(2) constant [36 x i8] c"ERROR: Out-of-bounds array access.\0A\00", align 1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #1

; Function Attrs: nobuiltin
declare i32 @printf(i8 addrspace(2)*, ...) local_unnamed_addr #2

; Function Attrs: noinline noreturn
define internal fastcc void @julia__throw_boundserror_22948() unnamed_addr #3 {
top:
  %0 = call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([36 x i8], [36 x i8] addrspace(2)* @3, i64 0, i64 0))
  call fastcc void @gpu_report_exception()
  call fastcc void @gpu_signal_exception()
  call void @llvm.trap()
  unreachable
}

define internal fastcc void @julia_DimensionMismatch_22938() unnamed_addr {
top:
  %0 = insertvalue [1 x {}*] zeroinitializer, {}* poison, 0
  ret void
}

define internal fastcc void @gpu_report_exception() unnamed_addr {
top:
  ret void
}

define internal fastcc void @gpu_signal_exception() unnamed_addr {
top:
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #4

define spir_kernel void @_Z6kernel5OneToI5Int64E11BroadcastedI12CLArrayStyleILi1E19UnifiedDeviceMemoryEv5tuple5TupleI13CLDeviceArrayI7Float32Li1ELi1EE9EachIndexIS0_Li1ES1_EEE({ [1 x i64] }* byval({ [1 x i64] }) %0, { { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }* byval({ { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }) %1) local_unnamed_addr {
conversion:
  %2 = getelementptr inbounds { [1 x i64] }, { [1 x i64] }* %0, i32 0, i32 0
  %3 = getelementptr inbounds { { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }, { { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }* %1, i32 0, i32 0
  br label %top

top:                                              ; preds = %conversion
  %y = alloca [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }], align 8
  %4 = alloca [1 x i64], align 8
  %.sroa.035.0..sroa_cast = bitcast { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }* %3 to i8*
  %y127128 = bitcast [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }]* %y to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(48) %y127128, i8* noundef nonnull align 8 dereferenceable(48) %.sroa.035.0..sroa_cast, i64 48, i1 false)
  %5 = getelementptr inbounds [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }], [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }]* %y, i64 0, i64 0, i32 0, i32 0, i32 3
  %6 = getelementptr inbounds [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }], [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }]* %y, i64 0, i64 0, i32 0, i32 1, i64 0, i64 0
  %7 = load i64, i64* %6, align 8
  %.fr135 = freeze i64 %7
  %8 = load i64, i64* %5, align 8
  %.fr134 = freeze i64 %8
  %9 = icmp eq i64 %.fr135, %.fr134
  %10 = icmp eq i64 %.fr134, 1
  %or.cond = or i1 %9, %10
  br i1 %or.cond, label %L37, label %L23

L23:                                              ; preds = %top
  %11 = icmp eq i64 %.fr135, 1
  br i1 %11, label %L37, label %L26

L26:                                              ; preds = %L23
  call fastcc void @julia_DimensionMismatch_22938()
  call fastcc void @gpu_report_exception()
  call fastcc void @gpu_signal_exception()
  call void @llvm.trap()
  unreachable

L37:                                              ; preds = %L23, %top
  %value_phi1 = phi i64 [ %.fr135, %top ], [ %.fr134, %L23 ]
  %12 = call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([20 x i8], [20 x i8] addrspace(2)* @0, i64 0, i64 0), i64 %value_phi1)
  %13 = getelementptr inbounds [1 x i64], [1 x i64]* %2, i64 0, i64 0
  %14 = load i64, i64* %13, align 8
  %15 = icmp sgt i64 %14, 0
  %16 = zext i1 %15 to i32
  %17 = call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([20 x i8], [20 x i8] addrspace(2)* @1, i64 0, i64 0), i32 %16)
  br i1 %15, label %L46, label %L204

L46:                                              ; preds = %L37
  %18 = call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([5 x i8], [5 x i8] addrspace(2)* @2, i64 0, i64 0))
  %19 = bitcast [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }]* %y to float addrspace(1)**
  %20 = load float addrspace(1)*, float addrspace(1)** %19, align 8
  %.not38 = icmp eq i64 %.fr135, 1
  %21 = getelementptr inbounds [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }], [1 x { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } }]* %y, i64 0, i64 0, i32 0, i32 1, i64 1
  %22 = getelementptr inbounds [1 x i64], [1 x i64]* %21, i64 0, i64 0
  %23 = load i64, i64* %22, align 8
  br i1 %10, label %L46.split.us, label %L46.split

L46.split.us:                                     ; preds = %L46
  %24 = load float, float addrspace(1)* %20, align 4
  %.fr137 = freeze float %24
  %25 = bitcast float %.fr137 to i32
  %26 = xor i32 %25, 2147483647
  %27 = icmp slt i32 %25, 0
  %28 = select i1 %27, i32 %26, i32 %25
  br i1 %.not38, label %L46.split.us.split.us, label %L46.split.us.split

L46.split.us.split.us:                            ; preds = %L46.split.us
  %.not39.us.us.not = icmp eq i64 %23, 0
  br i1 %.not39.us.us.not, label %L165, label %L204

L46.split.us.split:                               ; preds = %L46.split.us
  %29 = fcmp ord float %.fr137, 0.000000e+00
  %30 = add i64 %23, -9223372036854775807
  %smax = call i64 @llvm.smax.i64(i64 %30, i64 -1)
  %31 = sub i64 %23, %smax
  %isnotneg.inv140 = icmp slt i64 %23, 0
  %32 = call i64 @llvm.smin.i64(i64 %31, i64 3)
  %33 = call i64 @llvm.smax.i64(i64 %32, i64 1)
  %exit.mainloop.at = select i1 %isnotneg.inv140, i64 1, i64 %33
  %34 = icmp ugt i64 %exit.mainloop.at, 1
  br i1 %29, label %L46.split.us.split.split.us, label %L46.split.us.split.split

L46.split.us.split.split.us:                      ; preds = %L46.split.us.split
  br i1 %34, label %L176.us.us75, label %main.pseudo.exit

L176.us.us75:                                     ; preds = %L195.us.us79.thread, %L46.split.us.split.split.us
  %value_phi343.us.us73 = phi float [ %41, %L195.us.us79.thread ], [ 0xFFF0000000000000, %L46.split.us.split.split.us ]
  %value_phi242.us.us74 = phi i64 [ %42, %L195.us.us79.thread ], [ 1, %L46.split.us.split.split.us ]
  %35 = fcmp ord float %value_phi343.us.us73, 0.000000e+00
  br i1 %35, label %L195.us.us79, label %L195.us.us79.thread

L195.us.us79:                                     ; preds = %L176.us.us75
  %36 = bitcast float %value_phi343.us.us73 to i32
  %37 = xor i32 %36, 2147483647
  %38 = icmp slt i32 %36, 0
  %39 = select i1 %38, i32 %37, i32 %36
  %.fr162 = freeze i32 %39
  %40 = icmp slt i32 %.fr162, %28
  %spec.select = select i1 %40, float %.fr137, float %value_phi343.us.us73
  br label %L195.us.us79.thread

L195.us.us79.thread:                              ; preds = %L195.us.us79, %L176.us.us75
  %41 = phi float [ %value_phi343.us.us73, %L176.us.us75 ], [ %spec.select, %L195.us.us79 ]
  %42 = add nuw nsw i64 %value_phi242.us.us74, 1
  %.not141 = icmp ult i64 %42, %exit.mainloop.at
  br i1 %.not141, label %L176.us.us75, label %main.exit.selector

main.exit.selector:                               ; preds = %L195.us.us79.thread
  %43 = icmp ult i64 %value_phi242.us.us74, 2
  br i1 %43, label %main.pseudo.exit, label %L204

main.pseudo.exit:                                 ; preds = %main.exit.selector, %L46.split.us.split.split.us
  %value_phi343.us.us73.copy = phi float [ 0xFFF0000000000000, %L46.split.us.split.split.us ], [ %41, %main.exit.selector ]
  %value_phi242.us.us74.copy = phi i64 [ 1, %L46.split.us.split.split.us ], [ %42, %main.exit.selector ]
  br label %L52.us.us72.postloop

L46.split.us.split.split:                         ; preds = %L46.split.us.split
  br i1 %34, label %L176.us, label %main.pseudo.exit107

L176.us:                                          ; preds = %L176.us, %L46.split.us.split.split
  %value_phi242.us = phi i64 [ %44, %L176.us ], [ 1, %L46.split.us.split.split ]
  %44 = add nuw nsw i64 %value_phi242.us, 1
  %.not139 = icmp ult i64 %44, %exit.mainloop.at
  br i1 %.not139, label %L176.us, label %main.exit.selector106

main.exit.selector106:                            ; preds = %L176.us
  %45 = icmp ult i64 %value_phi242.us, 2
  br i1 %45, label %main.pseudo.exit107, label %L204

main.pseudo.exit107:                              ; preds = %main.exit.selector106, %L46.split.us.split.split
  %value_phi242.us.copy = phi i64 [ 1, %L46.split.us.split.split ], [ %44, %main.exit.selector106 ]
  br label %L52.us.postloop

L46.split:                                        ; preds = %L46
  br i1 %.not38, label %L46.split.split.us, label %L46.split.split

L46.split.split.us:                               ; preds = %L46.split
  %.not39.us60.not = icmp eq i64 %23, 0
  br i1 %.not39.us60.not, label %L165, label %L204

L46.split.split:                                  ; preds = %L46.split
  %46 = add i64 %23, -9223372036854775807
  %smax113 = call i64 @llvm.smax.i64(i64 %46, i64 -1)
  %47 = sub i64 %23, %smax113
  %isnotneg.inv = icmp slt i64 %23, 0
  %48 = call i64 @llvm.smin.i64(i64 %47, i64 3)
  %49 = call i64 @llvm.smax.i64(i64 %48, i64 1)
  %exit.mainloop.at118 = select i1 %isnotneg.inv, i64 1, i64 %49
  %50 = icmp ugt i64 %exit.mainloop.at118, 1
  br i1 %50, label %L176, label %main.pseudo.exit121

L165:                                             ; preds = %L52.postloop, %L52.us.postloop, %L52.us.us72.postloop, %L46.split.split.us, %L46.split.us.split.us
  %.us-phi46 = phi i64 [ 1, %L46.split.us.split.us ], [ 1, %L46.split.split.us ], [ %value_phi242.us.us74.postloop, %L52.us.us72.postloop ], [ %value_phi242.us.postloop, %L52.us.postloop ], [ %value_phi242.postloop, %L52.postloop ]
  %51 = getelementptr inbounds [1 x i64], [1 x i64]* %4, i64 0, i64 0
  store i64 %.us-phi46, i64* %51, align 8
  call fastcc void @julia__throw_boundserror_22948() #5
  unreachable

L176:                                             ; preds = %L195, %L46.split.split
  %value_phi343 = phi float [ %value_phi3., %L195 ], [ 0xFFF0000000000000, %L46.split.split ]
  %value_phi242 = phi i64 [ %65, %L195 ], [ 1, %L46.split.split ]
  %value_phi2.op = add nsw i64 %value_phi242, -1
  %52 = getelementptr inbounds float, float addrspace(1)* %20, i64 %value_phi2.op
  %53 = load float, float addrspace(1)* %52, align 4
  %54 = fcmp ord float %value_phi343, 0.000000e+00
  %55 = fcmp ord float %53, 0.000000e+00
  %or.cond147 = select i1 %54, i1 %55, i1 false
  br i1 %or.cond147, label %L182, label %L195

L182:                                             ; preds = %L176
  %56 = bitcast float %value_phi343 to i32
  %57 = xor i32 %56, 2147483647
  %58 = icmp slt i32 %56, 0
  %59 = select i1 %58, i32 %57, i32 %56
  %60 = bitcast float %53 to i32
  %61 = xor i32 %60, 2147483647
  %62 = icmp slt i32 %60, 0
  %63 = select i1 %62, i32 %61, i32 %60
  %64 = icmp slt i32 %59, %63
  br label %L195

L195:                                             ; preds = %L182, %L176
  %value_phi6.in = phi i1 [ %64, %L182 ], [ %54, %L176 ]
  %value_phi3. = select i1 %value_phi6.in, float %53, float %value_phi343
  %65 = add nuw nsw i64 %value_phi242, 1
  %.not136 = icmp ult i64 %65, %exit.mainloop.at118
  br i1 %.not136, label %L176, label %main.exit.selector120

main.exit.selector120:                            ; preds = %L195
  %66 = icmp ult i64 %value_phi242, 2
  br i1 %66, label %main.pseudo.exit121, label %L204

main.pseudo.exit121:                              ; preds = %main.exit.selector120, %L46.split.split
  %value_phi343.copy = phi float [ 0xFFF0000000000000, %L46.split.split ], [ %value_phi3., %main.exit.selector120 ]
  %value_phi242.copy = phi i64 [ 1, %L46.split.split ], [ %65, %main.exit.selector120 ]
  br label %L52.postloop

L204:                                             ; preds = %L195.postloop, %L176.us.postloop, %L195.us.us79.postloop.thread, %main.exit.selector120, %L46.split.split.us, %main.exit.selector106, %main.exit.selector, %L46.split.us.split.us, %L37
  ret void

L52.us.us72.postloop:                             ; preds = %L195.us.us79.postloop.thread, %main.pseudo.exit
  %value_phi343.us.us73.postloop = phi float [ %value_phi343.us.us73.copy, %main.pseudo.exit ], [ %74, %L195.us.us79.postloop.thread ]
  %value_phi242.us.us74.postloop = phi i64 [ %value_phi242.us.us74.copy, %main.pseudo.exit ], [ %75, %L195.us.us79.postloop.thread ]
  %67 = add nsw i64 %value_phi242.us.us74.postloop, -1
  %.not39.us.us82.postloop = icmp ult i64 %67, %23
  br i1 %.not39.us.us82.postloop, label %L176.us.us75.postloop, label %L165

L176.us.us75.postloop:                            ; preds = %L52.us.us72.postloop
  %68 = fcmp ord float %value_phi343.us.us73.postloop, 0.000000e+00
  br i1 %68, label %L195.us.us79.postloop, label %L195.us.us79.postloop.thread

L195.us.us79.postloop:                            ; preds = %L176.us.us75.postloop
  %69 = bitcast float %value_phi343.us.us73.postloop to i32
  %70 = xor i32 %69, 2147483647
  %71 = icmp slt i32 %69, 0
  %72 = select i1 %71, i32 %70, i32 %69
  %.fr = freeze i32 %72
  %73 = icmp slt i32 %.fr, %28
  %spec.select148 = select i1 %73, float %.fr137, float %value_phi343.us.us73.postloop
  br label %L195.us.us79.postloop.thread

L195.us.us79.postloop.thread:                     ; preds = %L195.us.us79.postloop, %L176.us.us75.postloop
  %74 = phi float [ %value_phi343.us.us73.postloop, %L176.us.us75.postloop ], [ %spec.select148, %L195.us.us79.postloop ]
  %75 = add nuw nsw i64 %value_phi242.us.us74.postloop, 1
  %76 = icmp sgt i64 %value_phi242.us.us74.postloop, 1
  br i1 %76, label %L204, label %L52.us.us72.postloop

L52.us.postloop:                                  ; preds = %L176.us.postloop, %main.pseudo.exit107
  %value_phi242.us.postloop = phi i64 [ %value_phi242.us.copy, %main.pseudo.exit107 ], [ %78, %L176.us.postloop ]
  %77 = add nsw i64 %value_phi242.us.postloop, -1
  %.not39.us.postloop = icmp ult i64 %77, %23
  br i1 %.not39.us.postloop, label %L176.us.postloop, label %L165

L176.us.postloop:                                 ; preds = %L52.us.postloop
  %78 = add nuw nsw i64 %value_phi242.us.postloop, 1
  %79 = icmp sgt i64 %value_phi242.us.postloop, 1
  br i1 %79, label %L204, label %L52.us.postloop

L52.postloop:                                     ; preds = %L195.postloop, %main.pseudo.exit121
  %value_phi343.postloop = phi float [ %value_phi343.copy, %main.pseudo.exit121 ], [ %value_phi3..postloop, %L195.postloop ]
  %value_phi242.postloop = phi i64 [ %value_phi242.copy, %main.pseudo.exit121 ], [ %93, %L195.postloop ]
  %value_phi2.op.postloop = add nsw i64 %value_phi242.postloop, -1
  %80 = getelementptr inbounds float, float addrspace(1)* %20, i64 %value_phi2.op.postloop
  %81 = load float, float addrspace(1)* %80, align 4
  %.not39.postloop = icmp ult i64 %value_phi2.op.postloop, %23
  br i1 %.not39.postloop, label %L176.postloop, label %L165

L176.postloop:                                    ; preds = %L52.postloop
  %82 = fcmp ord float %value_phi343.postloop, 0.000000e+00
  %83 = fcmp ord float %81, 0.000000e+00
  %or.cond149 = select i1 %82, i1 %83, i1 false
  br i1 %or.cond149, label %L182.postloop, label %L195.postloop

L182.postloop:                                    ; preds = %L176.postloop
  %84 = bitcast float %value_phi343.postloop to i32
  %85 = xor i32 %84, 2147483647
  %86 = icmp slt i32 %84, 0
  %87 = select i1 %86, i32 %85, i32 %84
  %88 = bitcast float %81 to i32
  %89 = xor i32 %88, 2147483647
  %90 = icmp slt i32 %88, 0
  %91 = select i1 %90, i32 %89, i32 %88
  %92 = icmp slt i32 %87, %91
  br label %L195.postloop

L195.postloop:                                    ; preds = %L182.postloop, %L176.postloop
  %value_phi6.in.postloop = phi i1 [ %92, %L182.postloop ], [ %82, %L176.postloop ]
  %value_phi3..postloop = select i1 %value_phi6.in.postloop, float %81, float %value_phi343.postloop
  %93 = add nuw nsw i64 %value_phi242.postloop, 1
  %94 = icmp sgt i64 %value_phi242.postloop, 1
  br i1 %94, label %L204, label %L52.postloop
}

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { cold noreturn nounwind }
attributes #2 = { nobuiltin }
attributes #3 = { noinline noreturn }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { noreturn }

!llvm.module.flags = !{!0, !1}
!julia.kernel = !{!2}
!opencl.ocl.version = !{!3}
!opencl.spirv.version = !{!4}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{void ({ [1 x i64] }*, { { { { i8 addrspace(1)*, i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }*)* @_Z6kernel5OneToI5Int64E11BroadcastedI12CLArrayStyleILi1E19UnifiedDeviceMemoryEv5tuple5TupleI13CLDeviceArrayI7Float32Li1ELi1EE9EachIndexIS0_Li1ES1_EEE}
!3 = !{i32 2, i32 0}
!4 = !{i32 1, i32 5}
