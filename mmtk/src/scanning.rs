use crate::slots::JuliaVMSlot;
use crate::{SINGLETON, UPCALLS};
use mmtk::memory_manager;
use mmtk::scheduler::*;
use mmtk::util::opaque_pointer::*;
use mmtk::util::Address;
use mmtk::util::ObjectReference;
use mmtk::vm::slot::Slot;
use mmtk::vm::ObjectTracerContext;
use mmtk::vm::RootsWorkFactory;
use mmtk::vm::Scanning;
use mmtk::vm::SlotVisitor;
use mmtk::vm::VMBinding;
use mmtk::Mutator;
use mmtk::MMTK;

use crate::JuliaVM;

pub struct VMScanning {}

impl Scanning<JuliaVM> for VMScanning {
    fn scan_roots_in_mutator_thread(
        _tls: VMWorkerThread,
        mutator: &'static mut Mutator<JuliaVM>,
        mut factory: impl RootsWorkFactory<JuliaVMSlot>,
    ) {
        // This allows us to reuse mmtk_scan_gcstack which expectes an SlotVisitor
        // Push the nodes as they need to be transitively pinned
        struct SlotBuffer {
            pub buffer: Vec<ObjectReference>,
        }
        impl mmtk::vm::SlotVisitor<JuliaVMSlot> for SlotBuffer {
            fn visit_slot(&mut self, slot: JuliaVMSlot) {
                match slot {
                    JuliaVMSlot::Simple(se) => {
                        if let Some(object) = se.load() {
                            self.buffer.push(object);
                        }
                    }
                    JuliaVMSlot::Offset(oe) => {
                        if let Some(object) = oe.load() {
                            self.buffer.push(object);
                        }
                    }
                }
            }
        }

        use crate::julia_scanning::*;
        use crate::julia_types::*;

        let ptls: &mut mmtk__jl_tls_states_t = unsafe { std::mem::transmute(mutator.mutator_tls) };
        let pthread = ptls.system_id;
        let mut tpinning_slot_buffer = SlotBuffer { buffer: vec![] }; // need to be transitively pinned
        let mut pinning_slot_buffer = SlotBuffer { buffer: vec![] }; // roots from the shadow stack that we know that do not need to be transitively pinned
        let mut node_buffer = vec![];

        let mut conservative_buffer = vec![];

        log::info!(
            "Scanning ptls {:?}, pthread {:x}",
            mutator.mutator_tls,
            pthread,
        );

        // Conservatively scan registers saved with the thread
        #[cfg(feature = "conservative")]
        {
            crate::conservative::mmtk_conservative_scan_ptls_registers(ptls, &mut conservative_buffer);
            crate::conservative::mmtk_conservative_scan_ptls_stack(ptls, &mut conservative_buffer);
        }

        // Scan thread local from ptls: See gc_queue_thread_local in gc.c
        let mut root_scan_task = |task: *const mmtk__jl_task_t, task_is_root: bool| {
            if !task.is_null() {
                log::info!(
                    "Scanning task {:?}",
                    task,
                );
                #[cfg(feature = "conservative")]
                crate::conservative::CONSERVATIVE_SCANNED_TASK.lock().unwrap().insert(Address::from_ptr(task));
                // Scan shadow stack
                unsafe {
                    // process gc preserve stack
                    mmtk_scan_gcpreserve_stack(task, &mut tpinning_slot_buffer);

                    // process gc stack
                    mmtk_scan_gcstack(
                        task,
                        &mut tpinning_slot_buffer,
                        Some(&mut pinning_slot_buffer),
                    );
                }
                // Conservatively scan native stacks to make sure we won't move objects that the runtime is using.
                // Conservative scan stack and registers
                #[cfg(feature = "conservative")]
                {
                    crate::conservative::mmtk_conservative_scan_task_registers(task, &mut conservative_buffer);
                    crate::conservative::mmtk_conservative_scan_task_stack(task, &mut conservative_buffer);
                }

                if task_is_root {
                    // captures wrong root nodes before creating the work
                    debug_assert!(
                        Address::from_ptr(task).as_usize() % 16 == 0
                            || Address::from_ptr(task).as_usize() % 8 == 0,
                        "root node {:?} is not aligned to 8 or 16",
                        Address::from_ptr(task)
                    );

                    // unsafe: We checked `!task.is_null()` before.
                    let objref = unsafe {
                        ObjectReference::from_raw_address_unchecked(Address::from_ptr(task))
                    };
                    node_buffer.push(objref);
                }
            }
        };
        root_scan_task(ptls.root_task, true);

        // need to iterate over live tasks as well to process their shadow stacks
        // we should not set the task themselves as roots as we will know which ones are still alive after GC
        let mut i = 0;
        while i < ptls.heap.live_tasks.len {
            let mut task_address = Address::from_ptr(ptls.heap.live_tasks.items);
            task_address = task_address.shift::<Address>(i as isize);
            let task = unsafe { task_address.load::<*const mmtk_jl_task_t>() };
            root_scan_task(task, false);
            i += 1;
        }

        root_scan_task(ptls.current_task as *mut mmtk__jl_task_t, true);
        root_scan_task(ptls.next_task, true);
        root_scan_task(ptls.previous_task, true);
        if !ptls.previous_exception.is_null() {
            node_buffer.push(unsafe {
                // unsafe: We have just checked `ptls.previous_exception` is not null.
                ObjectReference::from_raw_address_unchecked(Address::from_mut_ptr(
                    ptls.previous_exception,
                ))
            });
        }

        // Scan backtrace buffer: See gc_queue_bt_buf in gc.c
        let mut i = 0;
        while i < ptls.bt_size {
            let bt_entry = unsafe { ptls.bt_data.add(i) };
            let bt_entry_size = mmtk_jl_bt_entry_size(bt_entry);
            if mmtk_jl_bt_is_native(bt_entry) {
                i += bt_entry_size;
                continue;
            }
            let njlvals = mmtk_jl_bt_num_jlvals(bt_entry);
            for j in 0..njlvals {
                let bt_entry_value = mmtk_jl_bt_entry_jlvalue(bt_entry, j);

                // captures wrong root nodes before creating the work
                debug_assert!(
                    bt_entry_value.to_raw_address().as_usize() % 16 == 0
                        || bt_entry_value.to_raw_address().as_usize() % 8 == 0,
                    "root node {:?} is not aligned to 8 or 16",
                    bt_entry_value
                );

                node_buffer.push(bt_entry_value);
            }
            i += bt_entry_size;
        }

        // We do not need gc_queue_remset from gc.c (we are not using remset in the thread)

        // Push work
        const CAPACITY_PER_PACKET: usize = 4096;
        for tpinning_roots in tpinning_slot_buffer
            .buffer
            .chunks(CAPACITY_PER_PACKET)
            .map(|c| c.to_vec())
        {
            factory.create_process_tpinning_roots_work(tpinning_roots);
        }
        for pinning_roots in pinning_slot_buffer
            .buffer
            .chunks(CAPACITY_PER_PACKET)
            .map(|c| c.to_vec())
        {
            factory.create_process_pinning_roots_work(pinning_roots);
        }
        for nodes in node_buffer.chunks(CAPACITY_PER_PACKET).map(|c| c.to_vec()) {
            factory.create_process_pinning_roots_work(nodes);
        }
        // Conservatively found objs
        let tpin_types: [&str; 0] = [
            // "buffer",
            // "jl_symbol",
            // "jl_simplevector",
            // "jl_string",
            // "jl_weakref",
            // "jl_array",
            // "jl_module",
            // "jl_task",
            // "jl_datatype",

            // "#106#107",
            // "#382#383",
            // "#433#435",
            // "#441#444",
            // "#469#470",
            // "#487#492",
            // "#489#494",
            // "#dst#524",
            // "AbstractIterationResult",
            // "ArgInfo",
            // "Argument",
            // "BBIdxIter",
            // "BackedgePair",
            // "BasicBlock",
            // "BasicStmtChange",
            // "BestguessInfo",
            // "BitArray",
            // "BitSet",
            // "BlockLiveness",
            // "Box",
            // "CFG",
            // "CFGInliningState",
            // "CachedMethodTable",
            // "CachedResult",
            // "CallMeta",
            // "DataType",

            // "CodeInfo",
            // "CodeInstance",
            // "Expr", // <-----
            // "Method",
            // "MethodInstance",
            // "MethodTable",

            // "ConcreteResult",
            // "Conditional",
            // "Const",
            // "ConstCallInfo",
            // "ConstCallResults",
            // "ConstantCase",
            // "EdgeCallResult",
            // "EdgeTracker",
            // "Effects",
            // "Enumerate",

            // "Float64",
            // "Generator",
            // "GenericDomTree",
            // "GlobalRef",
            // "GotoIfNot",
            // "GotoNode",
            // "IRCode",
            // "IdDict",
            // "IdSet",
            // "IncrementalCompact",
            // "InfStackUnwind",
            // "InferenceLoopState",
            // "InferenceParams",
            // "InferenceResult",
            // "InferenceState",
            // "InliningCase",
            // "InliningEdgeTracker",
            // "InliningState",
            // "InliningTodo",
            // "InsertBefore",
            // "InsertHere",
            // "Instruction",
            // "InstructionStream",
            // "Int64",
            // "InterConditional",
            // "IntrinsicFunction",
            // "InvokeCase",
            // "JLOptions",
            // "LazyGenericDomtree",
            // "LiftedValue",
            // "LineInfoNode",
            // "LineNumberNode",
            // "LinearIndices",
            // "MethodCallResult",
            // "MethodLookupResult",
            // "MethodMatch",
            // "MethodMatchInfo",
            // "MethodMatchKey",
            // "MethodMatchResult",
            // "MethodMatches",
            // "NamedTuple",
            // "NativeInterpreter",
            // "NewInstruction",
            // "NewNodeInfo",
            // "NewNodeStream",
            // "NewSSAValue",
            // "OldSSAValue",
            // "OneTo",
            // "OptimizationParams",
            // "OptimizationState",
            // "Pair",
            // "Pairs",
            // "PartialStruct",
            // "PhiNode",
            // "PiNode",
            // "Ptr",
            // "QuoteNode",
            // "RTEffects",
            // "RefValue",
            // "ReturnNode",
            // "Reverse",
            // "SSADefUse",
            // "SSAValue",
            // "SemiConcreteResult",
            // "Signature",
            // "SlotInfo",
            // "SlotNumber",
            // "StateUpdate",
            // "StmtInfo",
            // "StmtRange",
            // "Tuple",
            // "TwoPhaseDefUseMap",
            // "TypeMapEntry",
            // "TypeMapLevel",
            // "TypeName",
            // "TypeVar",
            // "TypedSlot",
            // "TypeofVararg",
            // "TypesView",
            // "UInt32",
            // "UInt64",
            // "Union",
            // "UnionAll",
            // "UnionSplitApplyCallInfo",
            // "UnionSplitInfo",
            // "UnionSplitMethodMatches",
            // "UnitRange",
            // "UseRef",
            // "UseRefIterator",
            // "VarState",
            // "WorldRange",
            // "WorldView",
            // "Zip",
        ];
        let mut conservative_tpin = vec![];
        for obj in conservative_buffer {
            let type_name = crate::julia_scanning::get_julia_object_type_string(&obj);
            if tpin_types.iter().any(|&s| s == type_name) {
                // Tpin them
                conservative_tpin.push(obj);
            } else {
                // Those will be pinned.
                crate::conservative::CONSERVATIVE_ROOTS.lock().unwrap().insert(obj);
            }
        }
        for tpinning_roots in conservative_tpin.chunks(CAPACITY_PER_PACKET).map(|c| c.to_vec()) {
            factory.create_process_tpinning_roots_work(tpinning_roots);
        }
    }

    fn scan_vm_specific_roots(
        _tls: VMWorkerThread,
        mut factory: impl RootsWorkFactory<JuliaVMSlot>,
    ) {
        use crate::slots::RootsWorkClosure;
        let mut roots_closure = RootsWorkClosure::from_roots_work_factory(&mut factory);
        unsafe {
            ((*UPCALLS).scan_vm_specific_roots)(&mut roots_closure as _);
        }
    }

    fn scan_object<SV: SlotVisitor<JuliaVMSlot>>(
        _tls: VMWorkerThread,
        object: ObjectReference,
        slot_visitor: &mut SV,
    ) {
        crate::collection::OBJECTS_SCANNED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        process_object(object, slot_visitor);
    }

    fn notify_initial_thread_scan_complete(_partial_scan: bool, _tls: VMWorkerThread) {
        // pin concservative roots from stack scanning
        #[cfg(feature = "conservative")]
        crate::conservative::pin_conservative_roots();

        let sweep_vm_specific_work = SweepVMSpecific::new();
        memory_manager::add_work_packet(
            &SINGLETON,
            WorkBucketStage::Compact,
            sweep_vm_specific_work,
        );
    }

    fn supports_return_barrier() -> bool {
        unimplemented!()
    }

    fn prepare_for_roots_re_scanning() {
        unimplemented!()
    }

    fn process_weak_refs(
        _worker: &mut GCWorker<JuliaVM>,
        tracer_context: impl ObjectTracerContext<JuliaVM>,
    ) -> bool {
        let single_thread_process_finalizer = ScanFinalizersSingleThreaded { tracer_context };
        memory_manager::add_work_packet(
            &SINGLETON,
            WorkBucketStage::VMRefClosure,
            single_thread_process_finalizer,
        );

        // We have pushed work. No need to repeat this method.
        false
    }
}

pub fn process_object<EV: SlotVisitor<JuliaVMSlot>>(object: ObjectReference, closure: &mut EV) {
    let addr = object.to_raw_address();
    unsafe {
        crate::julia_scanning::scan_julia_object(addr, closure);
    }
}

// Sweep malloced arrays work
pub struct SweepVMSpecific {
    swept: bool,
}

impl SweepVMSpecific {
    pub fn new() -> Self {
        Self { swept: false }
    }
}

impl<VM: VMBinding> GCWork<VM> for SweepVMSpecific {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        // call sweep malloced arrays and sweep stack pools from UPCALLS
        unsafe { ((*UPCALLS).mmtk_sweep_malloced_array)() }
        unsafe { ((*UPCALLS).mmtk_sweep_stack_pools)() }
        unsafe { ((*UPCALLS).mmtk_sweep_weak_refs)() }
        self.swept = true;
    }
}

pub struct ScanFinalizersSingleThreaded<C: ObjectTracerContext<JuliaVM>> {
    tracer_context: C,
}

impl<C: ObjectTracerContext<JuliaVM>> GCWork<JuliaVM> for ScanFinalizersSingleThreaded<C> {
    fn do_work(&mut self, worker: &mut GCWorker<JuliaVM>, _mmtk: &'static MMTK<JuliaVM>) {
        unsafe { ((*UPCALLS).mmtk_clear_weak_refs)() }
        self.tracer_context.with_tracer(worker, |tracer| {
            crate::julia_finalizer::scan_finalizers_in_rust(tracer);
        });
    }
}
