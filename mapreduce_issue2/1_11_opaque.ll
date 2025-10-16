; ModuleID = 'start'
source_filename = "start"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-unknown-unknown-unknown"

@0 = private unnamed_addr addrspace(2) constant [20 x i8] c"length(y[1]) = %ld\0A\00", align 1
@1 = private unnamed_addr addrspace(2) constant [20 x i8] c"length(x) > 0 = %d\0A\00", align 1
@2 = private unnamed_addr addrspace(2) constant [5 x i8] c"Bar\0A\00", align 1
@3 = private unnamed_addr addrspace(2) constant [36 x i8] c"ERROR: Out-of-bounds array access.\0A\00", align 1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #1

; Function Attrs: nobuiltin
declare i32 @printf(ptr addrspace(2), ...) local_unnamed_addr #2

; Function Attrs: noinline noreturn
define internal fastcc void @julia__throw_boundserror_18148() unnamed_addr #3 {
top:
  %0 = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @3)
  call fastcc void @gpu_report_exception()
  call fastcc void @gpu_signal_exception()
  call void @llvm.trap()
  unreachable
}

define internal fastcc void @julia_DimensionMismatch_18138() unnamed_addr {
top:
  %0 = insertvalue [1 x ptr] zeroinitializer, ptr poison, 0
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

define spir_kernel void @_Z6kernel5OneToI5Int64E11BroadcastedI12CLArrayStyleILi1E19UnifiedDeviceMemoryEv5tuple5TupleI13CLDeviceArrayI7Float32Li1ELi1EE9EachIndexIS0_Li1ES1_EEE(ptr byval({ [1 x i64] }) %0, ptr byval({ { { { ptr addrspace(1), i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }) %1) local_unnamed_addr {
conversion:
  %2 = getelementptr inbounds { [1 x i64] }, ptr %0, i32 0, i32 0
  %3 = getelementptr inbounds { { { { ptr addrspace(1), i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }, ptr %1, i32 0, i32 0
  br label %top

top:                                              ; preds = %conversion
  %y = alloca [1 x { { { ptr addrspace(1), i64, [1 x i64], i64 }, [2 x [1 x i64]] } }], align 8
  %4 = alloca [1 x i64], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) %y, ptr noundef nonnull align 8 dereferenceable(48) %3, i64 48, i1 false)
  %5 = getelementptr inbounds { { ptr addrspace(1), i64, [1 x i64], i64 }, [2 x [1 x i64]] }, ptr %y, i64 0, i32 1
  %6 = getelementptr inbounds { ptr addrspace(1), i64, [1 x i64], i64 }, ptr %y, i64 0, i32 3
  %7 = load i64, ptr %5, align 8
  %.fr131 = freeze i64 %7
  %8 = load i64, ptr %6, align 8
  %.fr130 = freeze i64 %8
  %9 = icmp eq i64 %.fr131, %.fr130
  %10 = icmp eq i64 %.fr130, 1
  %or.cond = or i1 %9, %10
  br i1 %or.cond, label %L37, label %L23

L23:                                              ; preds = %top
  %11 = icmp eq i64 %.fr131, 1
  br i1 %11, label %L37, label %L26

L26:                                              ; preds = %L23
  call fastcc void @julia_DimensionMismatch_18138()
  call fastcc void @gpu_report_exception()
  call fastcc void @gpu_signal_exception()
  call void @llvm.trap()
  unreachable

L37:                                              ; preds = %L23, %top
  %value_phi1 = phi i64 [ %.fr131, %top ], [ %.fr130, %L23 ]
  %12 = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @0, i64 %value_phi1)
  %13 = load i64, ptr %2, align 8
  %14 = icmp sgt i64 %13, 0
  %15 = zext i1 %14 to i32
  %16 = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @1, i32 %15)
  br i1 %14, label %L46, label %L204

L46:                                              ; preds = %L37
  %17 = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @2)
  %18 = load ptr addrspace(1), ptr %y, align 8
  %.not36 = icmp eq i64 %.fr131, 1
  %19 = getelementptr inbounds { { ptr addrspace(1), i64, [1 x i64], i64 }, [2 x [1 x i64]] }, ptr %y, i64 0, i32 1, i64 1
  %20 = load i64, ptr %19, align 8
  br i1 %10, label %L46.split.us, label %L46.split

L46.split.us:                                     ; preds = %L46
  %21 = load float, ptr addrspace(1) %18, align 4
  %.fr133 = freeze float %21
  %22 = bitcast float %.fr133 to i32
  %23 = xor i32 %22, 2147483647
  %24 = icmp slt i32 %22, 0
  %25 = select i1 %24, i32 %23, i32 %22
  br i1 %.not36, label %L46.split.us.split.us, label %L46.split.us.split

L46.split.us.split.us:                            ; preds = %L46.split.us
  %.not37.us.us.not = icmp eq i64 %20, 0
  br i1 %.not37.us.us.not, label %L165, label %L204

L46.split.us.split:                               ; preds = %L46.split.us
  %26 = fcmp ord float %.fr133, 0.000000e+00
  %27 = add i64 %20, -9223372036854775807
  %smax = call i64 @llvm.smax.i64(i64 %27, i64 -1)
  %28 = sub i64 %20, %smax
  %isnotneg.inv136 = icmp slt i64 %20, 0
  %29 = call i64 @llvm.smin.i64(i64 %28, i64 3)
  %30 = call i64 @llvm.smax.i64(i64 %29, i64 1)
  %exit.mainloop.at = select i1 %isnotneg.inv136, i64 1, i64 %30
  %31 = icmp ugt i64 %exit.mainloop.at, 1
  br i1 %26, label %L46.split.us.split.split.us, label %L46.split.us.split.split

L46.split.us.split.split.us:                      ; preds = %L46.split.us.split
  br i1 %31, label %L176.us.us73, label %main.pseudo.exit

L176.us.us73:                                     ; preds = %L195.us.us77.thread, %L46.split.us.split.split.us
  %value_phi341.us.us71 = phi float [ %38, %L195.us.us77.thread ], [ 0xFFF0000000000000, %L46.split.us.split.split.us ]
  %value_phi240.us.us72 = phi i64 [ %39, %L195.us.us77.thread ], [ 1, %L46.split.us.split.split.us ]
  %32 = fcmp ord float %value_phi341.us.us71, 0.000000e+00
  br i1 %32, label %L195.us.us77, label %L195.us.us77.thread

L195.us.us77:                                     ; preds = %L176.us.us73
  %33 = bitcast float %value_phi341.us.us71 to i32
  %34 = xor i32 %33, 2147483647
  %35 = icmp slt i32 %33, 0
  %36 = select i1 %35, i32 %34, i32 %33
  %.fr158 = freeze i32 %36
  %37 = icmp slt i32 %.fr158, %25
  %spec.select = select i1 %37, float %.fr133, float %value_phi341.us.us71
  br label %L195.us.us77.thread

L195.us.us77.thread:                              ; preds = %L195.us.us77, %L176.us.us73
  %38 = phi float [ %value_phi341.us.us71, %L176.us.us73 ], [ %spec.select, %L195.us.us77 ]
  %39 = add nuw nsw i64 %value_phi240.us.us72, 1
  %.not137 = icmp ult i64 %39, %exit.mainloop.at
  br i1 %.not137, label %L176.us.us73, label %main.exit.selector

main.exit.selector:                               ; preds = %L195.us.us77.thread
  %40 = icmp ult i64 %value_phi240.us.us72, 2
  br i1 %40, label %main.pseudo.exit, label %L204

main.pseudo.exit:                                 ; preds = %main.exit.selector, %L46.split.us.split.split.us
  %value_phi341.us.us71.copy = phi float [ 0xFFF0000000000000, %L46.split.us.split.split.us ], [ %38, %main.exit.selector ]
  %value_phi240.us.us72.copy = phi i64 [ 1, %L46.split.us.split.split.us ], [ %39, %main.exit.selector ]
  br label %L52.us.us70.postloop

L46.split.us.split.split:                         ; preds = %L46.split.us.split
  br i1 %31, label %L176.us, label %main.pseudo.exit105

L176.us:                                          ; preds = %L176.us, %L46.split.us.split.split
  %value_phi240.us = phi i64 [ %41, %L176.us ], [ 1, %L46.split.us.split.split ]
  %41 = add nuw nsw i64 %value_phi240.us, 1
  %.not135 = icmp ult i64 %41, %exit.mainloop.at
  br i1 %.not135, label %L176.us, label %main.exit.selector104

main.exit.selector104:                            ; preds = %L176.us
  %42 = icmp ult i64 %value_phi240.us, 2
  br i1 %42, label %main.pseudo.exit105, label %L204

main.pseudo.exit105:                              ; preds = %main.exit.selector104, %L46.split.us.split.split
  %value_phi240.us.copy = phi i64 [ 1, %L46.split.us.split.split ], [ %41, %main.exit.selector104 ]
  br label %L52.us.postloop

L46.split:                                        ; preds = %L46
  br i1 %.not36, label %L46.split.split.us, label %L46.split.split

L46.split.split.us:                               ; preds = %L46.split
  %.not37.us58.not = icmp eq i64 %20, 0
  br i1 %.not37.us58.not, label %L165, label %L204

L46.split.split:                                  ; preds = %L46.split
  %43 = add i64 %20, -9223372036854775807
  %smax111 = call i64 @llvm.smax.i64(i64 %43, i64 -1)
  %44 = sub i64 %20, %smax111
  %isnotneg.inv = icmp slt i64 %20, 0
  %45 = call i64 @llvm.smin.i64(i64 %44, i64 3)
  %46 = call i64 @llvm.smax.i64(i64 %45, i64 1)
  %exit.mainloop.at116 = select i1 %isnotneg.inv, i64 1, i64 %46
  %47 = icmp ugt i64 %exit.mainloop.at116, 1
  br i1 %47, label %L176, label %main.pseudo.exit119

L165:                                             ; preds = %L52.postloop, %L52.us.postloop, %L52.us.us70.postloop, %L46.split.split.us, %L46.split.us.split.us
  %.us-phi44 = phi i64 [ 1, %L46.split.us.split.us ], [ 1, %L46.split.split.us ], [ %value_phi240.us.us72.postloop, %L52.us.us70.postloop ], [ %value_phi240.us.postloop, %L52.us.postloop ], [ %value_phi240.postloop, %L52.postloop ]
  store i64 %.us-phi44, ptr %4, align 8
  call fastcc void @julia__throw_boundserror_18148() #5
  unreachable

L176:                                             ; preds = %L195, %L46.split.split
  %value_phi341 = phi float [ %value_phi3., %L195 ], [ 0xFFF0000000000000, %L46.split.split ]
  %value_phi240 = phi i64 [ %61, %L195 ], [ 1, %L46.split.split ]
  %value_phi2.op = add nsw i64 %value_phi240, -1
  %48 = getelementptr inbounds float, ptr addrspace(1) %18, i64 %value_phi2.op
  %49 = load float, ptr addrspace(1) %48, align 4
  %50 = fcmp ord float %value_phi341, 0.000000e+00
  %51 = fcmp ord float %49, 0.000000e+00
  %or.cond143 = select i1 %50, i1 %51, i1 false
  br i1 %or.cond143, label %L182, label %L195

L182:                                             ; preds = %L176
  %52 = bitcast float %value_phi341 to i32
  %53 = xor i32 %52, 2147483647
  %54 = icmp slt i32 %52, 0
  %55 = select i1 %54, i32 %53, i32 %52
  %56 = bitcast float %49 to i32
  %57 = xor i32 %56, 2147483647
  %58 = icmp slt i32 %56, 0
  %59 = select i1 %58, i32 %57, i32 %56
  %60 = icmp slt i32 %55, %59
  br label %L195

L195:                                             ; preds = %L182, %L176
  %value_phi6.in = phi i1 [ %60, %L182 ], [ %50, %L176 ]
  %value_phi3. = select i1 %value_phi6.in, float %49, float %value_phi341
  %61 = add nuw nsw i64 %value_phi240, 1
  %.not132 = icmp ult i64 %61, %exit.mainloop.at116
  br i1 %.not132, label %L176, label %main.exit.selector118

main.exit.selector118:                            ; preds = %L195
  %62 = icmp ult i64 %value_phi240, 2
  br i1 %62, label %main.pseudo.exit119, label %L204

main.pseudo.exit119:                              ; preds = %main.exit.selector118, %L46.split.split
  %value_phi341.copy = phi float [ 0xFFF0000000000000, %L46.split.split ], [ %value_phi3., %main.exit.selector118 ]
  %value_phi240.copy = phi i64 [ 1, %L46.split.split ], [ %61, %main.exit.selector118 ]
  br label %L52.postloop

L204:                                             ; preds = %L195.postloop, %L176.us.postloop, %L195.us.us77.postloop.thread, %main.exit.selector118, %L46.split.split.us, %main.exit.selector104, %main.exit.selector, %L46.split.us.split.us, %L37
  ret void

L52.us.us70.postloop:                             ; preds = %L195.us.us77.postloop.thread, %main.pseudo.exit
  %value_phi341.us.us71.postloop = phi float [ %value_phi341.us.us71.copy, %main.pseudo.exit ], [ %70, %L195.us.us77.postloop.thread ]
  %value_phi240.us.us72.postloop = phi i64 [ %value_phi240.us.us72.copy, %main.pseudo.exit ], [ %71, %L195.us.us77.postloop.thread ]
  %63 = add nsw i64 %value_phi240.us.us72.postloop, -1
  %.not37.us.us80.postloop = icmp ult i64 %63, %20
  br i1 %.not37.us.us80.postloop, label %L176.us.us73.postloop, label %L165

L176.us.us73.postloop:                            ; preds = %L52.us.us70.postloop
  %64 = fcmp ord float %value_phi341.us.us71.postloop, 0.000000e+00
  br i1 %64, label %L195.us.us77.postloop, label %L195.us.us77.postloop.thread

L195.us.us77.postloop:                            ; preds = %L176.us.us73.postloop
  %65 = bitcast float %value_phi341.us.us71.postloop to i32
  %66 = xor i32 %65, 2147483647
  %67 = icmp slt i32 %65, 0
  %68 = select i1 %67, i32 %66, i32 %65
  %.fr = freeze i32 %68
  %69 = icmp slt i32 %.fr, %25
  %spec.select144 = select i1 %69, float %.fr133, float %value_phi341.us.us71.postloop
  br label %L195.us.us77.postloop.thread

L195.us.us77.postloop.thread:                     ; preds = %L195.us.us77.postloop, %L176.us.us73.postloop
  %70 = phi float [ %value_phi341.us.us71.postloop, %L176.us.us73.postloop ], [ %spec.select144, %L195.us.us77.postloop ]
  %71 = add nuw nsw i64 %value_phi240.us.us72.postloop, 1
  %72 = icmp sgt i64 %value_phi240.us.us72.postloop, 1
  br i1 %72, label %L204, label %L52.us.us70.postloop

L52.us.postloop:                                  ; preds = %L176.us.postloop, %main.pseudo.exit105
  %value_phi240.us.postloop = phi i64 [ %value_phi240.us.copy, %main.pseudo.exit105 ], [ %74, %L176.us.postloop ]
  %73 = add nsw i64 %value_phi240.us.postloop, -1
  %.not37.us.postloop = icmp ult i64 %73, %20
  br i1 %.not37.us.postloop, label %L176.us.postloop, label %L165

L176.us.postloop:                                 ; preds = %L52.us.postloop
  %74 = add nuw nsw i64 %value_phi240.us.postloop, 1
  %75 = icmp sgt i64 %value_phi240.us.postloop, 1
  br i1 %75, label %L204, label %L52.us.postloop

L52.postloop:                                     ; preds = %L195.postloop, %main.pseudo.exit119
  %value_phi341.postloop = phi float [ %value_phi341.copy, %main.pseudo.exit119 ], [ %value_phi3..postloop, %L195.postloop ]
  %value_phi240.postloop = phi i64 [ %value_phi240.copy, %main.pseudo.exit119 ], [ %89, %L195.postloop ]
  %value_phi2.op.postloop = add nsw i64 %value_phi240.postloop, -1
  %76 = getelementptr inbounds float, ptr addrspace(1) %18, i64 %value_phi2.op.postloop
  %77 = load float, ptr addrspace(1) %76, align 4
  %.not37.postloop = icmp ult i64 %value_phi2.op.postloop, %20
  br i1 %.not37.postloop, label %L176.postloop, label %L165

L176.postloop:                                    ; preds = %L52.postloop
  %78 = fcmp ord float %value_phi341.postloop, 0.000000e+00
  %79 = fcmp ord float %77, 0.000000e+00
  %or.cond145 = select i1 %78, i1 %79, i1 false
  br i1 %or.cond145, label %L182.postloop, label %L195.postloop

L182.postloop:                                    ; preds = %L176.postloop
  %80 = bitcast float %value_phi341.postloop to i32
  %81 = xor i32 %80, 2147483647
  %82 = icmp slt i32 %80, 0
  %83 = select i1 %82, i32 %81, i32 %80
  %84 = bitcast float %77 to i32
  %85 = xor i32 %84, 2147483647
  %86 = icmp slt i32 %84, 0
  %87 = select i1 %86, i32 %85, i32 %84
  %88 = icmp slt i32 %83, %87
  br label %L195.postloop

L195.postloop:                                    ; preds = %L182.postloop, %L176.postloop
  %value_phi6.in.postloop = phi i1 [ %88, %L182.postloop ], [ %78, %L176.postloop ]
  %value_phi3..postloop = select i1 %value_phi6.in.postloop, float %77, float %value_phi341.postloop
  %89 = add nuw nsw i64 %value_phi240.postloop, 1
  %90 = icmp sgt i64 %value_phi240.postloop, 1
  br i1 %90, label %L204, label %L52.postloop
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
!2 = !{ptr @_Z6kernel5OneToI5Int64E11BroadcastedI12CLArrayStyleILi1E19UnifiedDeviceMemoryEv5tuple5TupleI13CLDeviceArrayI7Float32Li1ELi1EE9EachIndexIS0_Li1ES1_EEE}
!3 = !{i32 2, i32 0}
!4 = !{i32 1, i32 5}
