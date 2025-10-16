; ModuleID = 'start'
source_filename = "start"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-unknown-unknown-unknown"

@0 = private unnamed_addr addrspace(2) constant [20 x i8] c"length(y[1]) = %ld\0A\00", align 1
@1 = private unnamed_addr addrspace(2) constant [20 x i8] c"length(x) > 0 = %d\0A\00", align 1
@2 = private unnamed_addr addrspace(2) constant [5 x i8] c"Bar\0A\00", align 1
@3 = private unnamed_addr addrspace(2) constant [36 x i8] c"ERROR: Out-of-bounds array access.\0A\00", align 1

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #0

; Function Attrs: nobuiltin
declare i32 @printf(ptr addrspace(2), ...) local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: noinline noreturn
define internal fastcc void @julia_throw_boundserror_43602() unnamed_addr #3 {
top:
  %0 = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @3)
  call fastcc void @gpu_report_exception()
  call fastcc void @gpu_signal_exception()
  call void @llvm.trap()
  unreachable
}

define internal fastcc void @julia_DimensionMismatch_43612() unnamed_addr {
top:
  %.unbox.fca.0.insert = insertvalue [1 x ptr] zeroinitializer, ptr poison, 0
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

define spir_kernel void @_Z6kernel5OneToI5Int64E11BroadcastedI12CLArrayStyleILi1E19UnifiedDeviceMemoryEv5tuple5TupleI13CLDeviceArrayI7Float32Li1ELi1EE9EachIndexIS0_Li1ES1_EEE(ptr byval({ [1 x i64] }) %"x::OneTo", ptr byval({ { { { ptr addrspace(1), i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }) %"y[1]::Broadcasted") local_unnamed_addr {
conversion:
  %0 = getelementptr inbounds { [1 x i64] }, ptr %"x::OneTo", i32 0, i32 0
  %1 = getelementptr inbounds { { { { ptr addrspace(1), i64, [1 x i64], i64 }, [2 x [1 x i64]] } } }, ptr %"y[1]::Broadcasted", i32 0, i32 0
  br label %top

top:                                              ; preds = %conversion
  %y = alloca [6 x i64], align 8
  %"new::Tuple28" = alloca [1 x i64], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) %y, ptr noundef nonnull align 8 dereferenceable(48) %1, i64 48, i1 false)
  %"y[2]_ptr" = getelementptr inbounds i8, ptr %y, i64 32
  %y.len_ptr = getelementptr inbounds i8, ptr %y, i64 24
  %"y[2]_ptr.unbox" = load i64, ptr %"y[2]_ptr", align 8
  %"y[2]_ptr.unbox.fr" = freeze i64 %"y[2]_ptr.unbox"
  %y.len_ptr.unbox = load i64, ptr %y.len_ptr, align 8
  %y.len_ptr.unbox.fr = freeze i64 %y.len_ptr.unbox
  %2 = icmp ne i64 %"y[2]_ptr.unbox.fr", %y.len_ptr.unbox.fr
  %value_phi.v141 = icmp ne i64 %y.len_ptr.unbox.fr, 1
  %value_phi.v.not = and i1 %2, %value_phi.v141
  br i1 %value_phi.v.not, label %L18, label %L37

L18:                                              ; preds = %top
  %value_phi62.v142.not = icmp eq i64 %"y[2]_ptr.unbox.fr", 1
  br i1 %value_phi62.v142.not, label %L37, label %L26

L26:                                              ; preds = %L18
  call fastcc void @julia_DimensionMismatch_43612()
  call fastcc void @gpu_report_exception()
  call fastcc void @gpu_signal_exception()
  call void @llvm.trap()
  unreachable

L37:                                              ; preds = %L18, %top
  %value_phi1 = phi i64 [ %"y[2]_ptr.unbox.fr", %top ], [ %y.len_ptr.unbox.fr, %L18 ]
  %3 = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @0, i64 %value_phi1)
  %"x::OneTo.unbox" = load i64, ptr %0, align 8
  %4 = icmp sgt i64 %"x::OneTo.unbox", 0
  %5 = zext i1 %4 to i32
  %6 = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @1, i32 %5)
  br i1 %4, label %L46, label %L276

L46:                                              ; preds = %L37
  %7 = call i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @2)
  %y.unbox = load ptr addrspace(1), ptr %y, align 8
  %.not145 = icmp eq i64 %"y[2]_ptr.unbox.fr", 1
  %"y[2]_ptr24.indices_ptr" = getelementptr inbounds i8, ptr %y, i64 40
  %bitcast29 = load i64, ptr %"y[2]_ptr24.indices_ptr", align 8
  br i1 %value_phi.v.not, label %L46.split.us, label %L46.split

L46.split.us:                                     ; preds = %L46
  br i1 %.not145, label %L46.split.us.split.split.split.us, label %L46.split.us.split.us

L46.split.us.split.us:                            ; preds = %L46.split.us
  call fastcc void @julia_DimensionMismatch_43612()
  call fastcc void @gpu_report_exception()
  call fastcc void @gpu_signal_exception()
  call void @llvm.trap()
  unreachable

L46.split.us.split.split.split.us:                ; preds = %L46.split.us
  %.not146.us.us224.not = icmp eq i64 %bitcast29, 0
  br i1 %.not146.us.us224.not, label %L235, label %L276

L46.split:                                        ; preds = %L46
  %.not = icmp eq i64 %y.len_ptr.unbox.fr, 1
  br i1 %.not, label %L46.split.split.us, label %L46.split.split

L46.split.split.us:                               ; preds = %L46.split
  br i1 %.not145, label %L46.split.split.us.split.us, label %L46.split.split.us.split

L46.split.split.us.split.us:                      ; preds = %L46.split.split.us
  %.not146.us167.us.not = icmp eq i64 %bitcast29, 0
  br i1 %.not146.us167.us.not, label %L235, label %L276

L46.split.split.us.split:                         ; preds = %L46.split.split.us
  %8 = add i64 %bitcast29, -9223372036854775807
  %smax243 = call i64 @llvm.smax.i64(i64 %8, i64 -1)
  %9 = sub i64 %bitcast29, %smax243
  %isnotneg.inv290 = icmp slt i64 %bitcast29, 0
  %10 = call i64 @llvm.smin.i64(i64 %9, i64 3)
  %11 = call i64 @llvm.smax.i64(i64 %10, i64 1)
  %exit.mainloop.at248 = select i1 %isnotneg.inv290, i64 1, i64 %11
  %12 = icmp ugt i64 %exit.mainloop.at248, 1
  %.pre285.pre = load float, ptr addrspace(1) %y.unbox, align 4
  br i1 %12, label %L248.us158, label %main.pseudo.exit251

L248.us158:                                       ; preds = %L267.us164, %L46.split.split.us.split
  %value_phi5148.us155 = phi float [ %value_phi5..us166, %L267.us164 ], [ 0xFFF0000000000000, %L46.split.split.us.split ]
  %value_phi4147.us156 = phi i64 [ %22, %L267.us164 ], [ 1, %L46.split.split.us.split ]
  %13 = fcmp ord float %value_phi5148.us155, 0.000000e+00
  %14 = fcmp ord float %.pre285.pre, 0.000000e+00
  %or.cond.us159 = select i1 %13, i1 %14, i1 false
  br i1 %or.cond.us159, label %L254.us161, label %L267.us164

L254.us161:                                       ; preds = %L248.us158
  %bitcast_coercion.us162 = bitcast float %value_phi5148.us155 to i32
  %15 = xor i32 %bitcast_coercion.us162, 2147483647
  %16 = icmp slt i32 %bitcast_coercion.us162, 0
  %17 = select i1 %16, i32 %15, i32 %bitcast_coercion.us162
  %bitcast_coercion33.us163 = bitcast float %.pre285.pre to i32
  %18 = xor i32 %bitcast_coercion33.us163, 2147483647
  %19 = icmp slt i32 %bitcast_coercion33.us163, 0
  %20 = select i1 %19, i32 %18, i32 %bitcast_coercion33.us163
  %21 = icmp slt i32 %17, %20
  br label %L267.us164

L267.us164:                                       ; preds = %L254.us161, %L248.us158
  %value_phi30.in.us165 = phi i1 [ %21, %L254.us161 ], [ %13, %L248.us158 ]
  %value_phi5..us166 = select i1 %value_phi30.in.us165, float %.pre285.pre, float %value_phi5148.us155
  %22 = add nuw nsw i64 %value_phi4147.us156, 1
  %.not291 = icmp ult i64 %22, %exit.mainloop.at248
  br i1 %.not291, label %L248.us158, label %main.exit.selector250

main.exit.selector250:                            ; preds = %L267.us164
  %23 = icmp ult i64 %value_phi4147.us156, 2
  br i1 %23, label %main.pseudo.exit251, label %L276

main.pseudo.exit251:                              ; preds = %main.exit.selector250, %L46.split.split.us.split
  %value_phi5148.us155.copy = phi float [ 0xFFF0000000000000, %L46.split.split.us.split ], [ %value_phi5..us166, %main.exit.selector250 ]
  %value_phi4147.us156.copy = phi i64 [ 1, %L46.split.split.us.split ], [ %22, %main.exit.selector250 ]
  br label %L52.us154.postloop

L46.split.split:                                  ; preds = %L46.split
  br i1 %.not145, label %L46.split.split.split.us, label %L46.split.split.split

L46.split.split.split.us:                         ; preds = %L46.split.split
  %.not146.us187.not = icmp eq i64 %bitcast29, 0
  br i1 %.not146.us187.not, label %L235, label %L276

L46.split.split.split:                            ; preds = %L46.split.split
  %24 = add i64 %bitcast29, -9223372036854775807
  %smax271 = call i64 @llvm.smax.i64(i64 %24, i64 -1)
  %25 = sub i64 %bitcast29, %smax271
  %isnotneg.inv = icmp slt i64 %bitcast29, 0
  %26 = call i64 @llvm.smin.i64(i64 %25, i64 3)
  %27 = call i64 @llvm.smax.i64(i64 %26, i64 1)
  %exit.mainloop.at276 = select i1 %isnotneg.inv, i64 1, i64 %27
  %28 = icmp ugt i64 %exit.mainloop.at276, 1
  br i1 %28, label %L248, label %main.pseudo.exit279

L235:                                             ; preds = %L52.postloop, %L52.us154.postloop, %L46.split.split.split.us, %L46.split.split.us.split.us, %L46.split.us.split.split.split.us
  %.us-phi151 = phi i64 [ 1, %L46.split.us.split.split.split.us ], [ 1, %L46.split.split.us.split.us ], [ 1, %L46.split.split.split.us ], [ %value_phi4147.us156.postloop, %L52.us154.postloop ], [ %value_phi4147.postloop, %L52.postloop ]
  store i64 %.us-phi151, ptr %"new::Tuple28", align 1
  call fastcc void @julia_throw_boundserror_43602() #5
  unreachable

L248:                                             ; preds = %L267, %L46.split.split.split
  %value_phi5148 = phi float [ %value_phi5., %L267 ], [ 0xFFF0000000000000, %L46.split.split.split ]
  %value_phi4147 = phi i64 [ %41, %L267 ], [ 1, %L46.split.split.split ]
  %29 = getelementptr float, ptr addrspace(1) %y.unbox, i64 %value_phi4147
  %30 = getelementptr float, ptr addrspace(1) %29, i64 -1
  %31 = load float, ptr addrspace(1) %30, align 4
  %32 = fcmp ord float %value_phi5148, 0.000000e+00
  %33 = fcmp ord float %31, 0.000000e+00
  %or.cond = select i1 %32, i1 %33, i1 false
  br i1 %or.cond, label %L254, label %L267

L254:                                             ; preds = %L248
  %bitcast_coercion = bitcast float %value_phi5148 to i32
  %34 = xor i32 %bitcast_coercion, 2147483647
  %35 = icmp slt i32 %bitcast_coercion, 0
  %36 = select i1 %35, i32 %34, i32 %bitcast_coercion
  %bitcast_coercion33 = bitcast float %31 to i32
  %37 = xor i32 %bitcast_coercion33, 2147483647
  %38 = icmp slt i32 %bitcast_coercion33, 0
  %39 = select i1 %38, i32 %37, i32 %bitcast_coercion33
  %40 = icmp slt i32 %36, %39
  br label %L267

L267:                                             ; preds = %L254, %L248
  %value_phi30.in = phi i1 [ %40, %L254 ], [ %32, %L248 ]
  %value_phi5. = select i1 %value_phi30.in, float %31, float %value_phi5148
  %41 = add nuw nsw i64 %value_phi4147, 1
  %.not289 = icmp ult i64 %41, %exit.mainloop.at276
  br i1 %.not289, label %L248, label %main.exit.selector278

main.exit.selector278:                            ; preds = %L267
  %42 = icmp ult i64 %value_phi4147, 2
  br i1 %42, label %main.pseudo.exit279, label %L276

main.pseudo.exit279:                              ; preds = %main.exit.selector278, %L46.split.split.split
  %value_phi5148.copy = phi float [ 0xFFF0000000000000, %L46.split.split.split ], [ %value_phi5., %main.exit.selector278 ]
  %value_phi4147.copy = phi i64 [ 1, %L46.split.split.split ], [ %41, %main.exit.selector278 ]
  br label %L52.postloop

L276:                                             ; preds = %L267.postloop, %L267.us164.postloop, %main.exit.selector278, %L46.split.split.split.us, %main.exit.selector250, %L46.split.split.us.split.us, %L46.split.us.split.split.split.us, %L37
  ret void

L52.us154.postloop:                               ; preds = %L267.us164.postloop, %main.pseudo.exit251
  %value_phi5148.us155.postloop = phi float [ %value_phi5148.us155.copy, %main.pseudo.exit251 ], [ %value_phi5..us166.postloop, %L267.us164.postloop ]
  %value_phi4147.us156.postloop = phi i64 [ %value_phi4147.us156.copy, %main.pseudo.exit251 ], [ %53, %L267.us164.postloop ]
  %43 = add nsw i64 %value_phi4147.us156.postloop, -1
  %.not146.us167.postloop = icmp ult i64 %43, %bitcast29
  br i1 %.not146.us167.postloop, label %L248.us158.postloop, label %L235

L248.us158.postloop:                              ; preds = %L52.us154.postloop
  %44 = fcmp ord float %value_phi5148.us155.postloop, 0.000000e+00
  %45 = fcmp ord float %.pre285.pre, 0.000000e+00
  %or.cond.us159.postloop = select i1 %44, i1 %45, i1 false
  br i1 %or.cond.us159.postloop, label %L254.us161.postloop, label %L267.us164.postloop

L254.us161.postloop:                              ; preds = %L248.us158.postloop
  %bitcast_coercion.us162.postloop = bitcast float %value_phi5148.us155.postloop to i32
  %46 = xor i32 %bitcast_coercion.us162.postloop, 2147483647
  %47 = icmp slt i32 %bitcast_coercion.us162.postloop, 0
  %48 = select i1 %47, i32 %46, i32 %bitcast_coercion.us162.postloop
  %bitcast_coercion33.us163.postloop = bitcast float %.pre285.pre to i32
  %49 = xor i32 %bitcast_coercion33.us163.postloop, 2147483647
  %50 = icmp slt i32 %bitcast_coercion33.us163.postloop, 0
  %51 = select i1 %50, i32 %49, i32 %bitcast_coercion33.us163.postloop
  %52 = icmp slt i32 %48, %51
  br label %L267.us164.postloop

L267.us164.postloop:                              ; preds = %L254.us161.postloop, %L248.us158.postloop
  %value_phi30.in.us165.postloop = phi i1 [ %52, %L254.us161.postloop ], [ %44, %L248.us158.postloop ]
  %value_phi5..us166.postloop = select i1 %value_phi30.in.us165.postloop, float %.pre285.pre, float %value_phi5148.us155.postloop
  %53 = add nuw nsw i64 %value_phi4147.us156.postloop, 1
  %54 = icmp sgt i64 %value_phi4147.us156.postloop, 1
  br i1 %54, label %L276, label %L52.us154.postloop

L52.postloop:                                     ; preds = %L267.postloop, %main.pseudo.exit279
  %value_phi5148.postloop = phi float [ %value_phi5148.copy, %main.pseudo.exit279 ], [ %value_phi5..postloop, %L267.postloop ]
  %value_phi4147.postloop = phi i64 [ %value_phi4147.copy, %main.pseudo.exit279 ], [ %67, %L267.postloop ]
  %55 = add nsw i64 %value_phi4147.postloop, -1
  %56 = getelementptr inbounds float, ptr addrspace(1) %y.unbox, i64 %55
  %57 = load float, ptr addrspace(1) %56, align 4
  %.not146.postloop = icmp ult i64 %55, %bitcast29
  br i1 %.not146.postloop, label %L248.postloop, label %L235

L248.postloop:                                    ; preds = %L52.postloop
  %58 = fcmp ord float %value_phi5148.postloop, 0.000000e+00
  %59 = fcmp ord float %57, 0.000000e+00
  %or.cond.postloop = select i1 %58, i1 %59, i1 false
  br i1 %or.cond.postloop, label %L254.postloop, label %L267.postloop

L254.postloop:                                    ; preds = %L248.postloop
  %bitcast_coercion.postloop = bitcast float %value_phi5148.postloop to i32
  %60 = xor i32 %bitcast_coercion.postloop, 2147483647
  %61 = icmp slt i32 %bitcast_coercion.postloop, 0
  %62 = select i1 %61, i32 %60, i32 %bitcast_coercion.postloop
  %bitcast_coercion33.postloop = bitcast float %57 to i32
  %63 = xor i32 %bitcast_coercion33.postloop, 2147483647
  %64 = icmp slt i32 %bitcast_coercion33.postloop, 0
  %65 = select i1 %64, i32 %63, i32 %bitcast_coercion33.postloop
  %66 = icmp slt i32 %62, %65
  br label %L267.postloop

L267.postloop:                                    ; preds = %L254.postloop, %L248.postloop
  %value_phi30.in.postloop = phi i1 [ %66, %L254.postloop ], [ %58, %L248.postloop ]
  %value_phi5..postloop = select i1 %value_phi30.in.postloop, float %57, float %value_phi5148.postloop
  %67 = add nuw nsw i64 %value_phi4147.postloop, 1
  %68 = icmp sgt i64 %value_phi4147.postloop, 1
  br i1 %68, label %L276, label %L52.postloop
}

attributes #0 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #1 = { nobuiltin }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
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
