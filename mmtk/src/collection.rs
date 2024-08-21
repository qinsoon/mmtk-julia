use crate::JuliaVM;
use crate::{SINGLETON, UPCALLS};
use log::{info, trace};
use mmtk::util::alloc::AllocationError;
use mmtk::util::heap::GCTriggerPolicy;
use mmtk::util::opaque_pointer::*;
use mmtk::vm::ActivePlan;
use mmtk::vm::{Collection, GCThreadContext};
use mmtk::Mutator;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use crate::{BLOCK_FOR_GC, STW_COND, WORLD_HAS_STOPPED};

static GC_START: AtomicU64 = AtomicU64::new(0);

extern "C" {
    pub static jl_gc_disable_counter: AtomicU32;
}

pub static OBJECTS_SCANNED: AtomicU64 = AtomicU64::new(0);
pub static OBJECTS_COPIED: AtomicU64 = AtomicU64::new(0);
pub static GC_COUNT: AtomicU64 = AtomicU64::new(0);

#[no_mangle]
fn gc_start(count: u64) {
    info!("GC #{} starts", count);
}
#[no_mangle]
fn gc_end(count: u64) {
    info!("GC #{} ends", count);
}

pub struct VMCollection {}

impl Collection<JuliaVM> for VMCollection {
    fn stop_all_mutators<F>(_tls: VMWorkerThread, mut mutator_visitor: F)
    where
        F: FnMut(&'static mut Mutator<JuliaVM>),
    {
        // Wait for all mutators to stop and all finalizers to run
        while !AtomicBool::load(&WORLD_HAS_STOPPED, Ordering::SeqCst) {
            // Stay here while the world has not stopped
            // FIXME add wait var
        }

        trace!("Stopped the world!");

        // Tell MMTk the stacks are ready.
        {
            let mut c_mutators: Vec<*mut Mutator<JuliaVM>> = vec![];
            let mut c_tls: Vec<VMMutatorThread> = vec![];

            // Check if we really know every mutators
            let mut mutators: *mut *mut Mutator<JuliaVM> = std::ptr::null_mut();
            let mut threads: *mut *mut crate::julia_types::mmtk__jl_tls_states_t = std::ptr::null_mut();
            let mut n_mutators: i32 = 0;
            unsafe { ((*UPCALLS).mmtk_jl_get_all_mutators)(&mut mutators as *mut *mut *mut Mutator<JuliaVM>, &mut threads as *mut *mut *mut crate::julia_types::mmtk__jl_tls_states_t, &mut n_mutators as *mut i32) };

            let mut i = 0;
            while i < n_mutators {
                let thread: *mut crate::julia_types::mmtk__jl_tls_states_t = unsafe { *(threads.add(i as usize)) };
                let mutator: *mut Mutator<JuliaVM> = unsafe { *(mutators.add(i as usize)) };
                let mutator_tls = unsafe { &*mutator }.mutator_tls;
                info!("stop_all_mutators: visiting tls={:?}, mutator={:?} (from jl_all_tls_states)", mutator_tls, mutator);

                assert_eq!(unsafe { std::mem::transmute::<_, *mut crate::julia_types::mmtk__jl_tls_states_t>(mutator_tls) }, thread);

                c_mutators.push(mutator);
                c_tls.push(mutator_tls);

                i += 1;
            }
            assert_eq!(crate::active_plan::VMActivePlan::number_of_mutators(), i as usize);

            for mutator in crate::active_plan::VMActivePlan::mutators() {
                info!("stop_all_mutators: visiting tls={:?}, mutator={:?} ", mutator.mutator_tls, mutator as *const _);

                assert!(c_mutators.iter().find(|m| **m == mutator as *mut Mutator<JuliaVM>).is_some());
                assert!(c_tls.iter().find(|t| **t == mutator.mutator_tls).is_some());

                mutator_visitor(mutator);
            }
        }

        // Record the start time of the GC
        let now = unsafe { ((*UPCALLS).jl_hrtime)() };
        trace!("gc_start = {}", now);
        GC_START.store(now, Ordering::Relaxed);

        OBJECTS_SCANNED.store(0, Ordering::SeqCst);
        OBJECTS_COPIED.store(0, Ordering::SeqCst);

        #[cfg(feature = "conservative")]
        crate::conservative::CONSERVATIVE_SCANNED_TASK.lock().unwrap().clear();

        gc_start(GC_COUNT.load(Ordering::SeqCst));
    }

    fn resume_mutators(_tls: VMWorkerThread) {
        // unpin conservative roots
        #[cfg(feature = "conservative")]
        crate::conservative::unpin_conservative_roots();

        // {
        //     let mut names: Vec<String> = crate::julia_scanning::DATATYPE_NAMES.lock().unwrap().iter().map(|ref_s| ref_s.clone()).collect();
        //     names.sort();
        //     println!("Datatype names:");
        //     for n in names {
        //         println!("\"{}\",", n);
        //     }
        // }

        gc_end(GC_COUNT.load(Ordering::SeqCst));
        GC_COUNT.fetch_add(1, Ordering::SeqCst);

        // Get the end time of the GC
        let end = unsafe { ((*UPCALLS).jl_hrtime)() };
        trace!("gc_end = {}", end);
        let gc_time = end - GC_START.load(Ordering::Relaxed);
        unsafe {
            ((*UPCALLS).update_gc_stats)(
                gc_time,
                crate::api::mmtk_used_bytes(),
                is_current_gc_nursery(),
            )
        }

        AtomicBool::store(&BLOCK_FOR_GC, false, Ordering::SeqCst);
        AtomicBool::store(&WORLD_HAS_STOPPED, false, Ordering::SeqCst);

        let &(_, ref cvar) = &*STW_COND.clone();
        cvar.notify_all();

        info!(
            "Live bytes = {}, total bytes = {}",
            crate::api::mmtk_used_bytes(),
            crate::api::mmtk_total_bytes()
        );
        info!("Objects: scanned = {}, copied = {}", OBJECTS_SCANNED.load(Ordering::SeqCst), OBJECTS_COPIED.load(Ordering::SeqCst));

        trace!("Resuming mutators.");
    }

    fn block_for_gc(_tls: VMMutatorThread) {
        info!("Triggered GC!");

        unsafe { ((*UPCALLS).prepare_to_collect)() };

        info!("Finished blocking mutator for GC!");
    }

    fn spawn_gc_thread(_tls: VMThread, ctx: GCThreadContext<JuliaVM>) {
        // Just drop the join handle. The thread will run until the process quits.
        let _ = std::thread::Builder::new()
            .name("MMTk Worker".to_string())
            .spawn(move || {
                use mmtk::util::opaque_pointer::*;
                use mmtk::util::Address;
                let worker_tls = VMWorkerThread(VMThread(OpaquePointer::from_address(unsafe {
                    Address::from_usize(thread_id::get())
                })));
                match ctx {
                    GCThreadContext::Worker(w) => {
                        mmtk::memory_manager::start_worker(&SINGLETON, worker_tls, w)
                    }
                }
            });
    }

    fn schedule_finalization(_tls: VMWorkerThread) {}

    fn out_of_memory(_tls: VMThread, _err_kind: AllocationError) {
        println!("Out of Memory!");
        unsafe { ((*UPCALLS).jl_throw_out_of_memory_error)() };
    }

    fn vm_live_bytes() -> usize {
        crate::api::JULIA_MALLOC_BYTES.load(Ordering::SeqCst)
    }

    fn is_collection_enabled() -> bool {
        unsafe { AtomicU32::load(&jl_gc_disable_counter, Ordering::SeqCst) <= 0 }
    }

    fn create_gc_trigger() -> Box<dyn GCTriggerPolicy<JuliaVM>> {
        use crate::gc_trigger::*;
        use std::convert::TryInto;
        let total_mem = unsafe { ((*UPCALLS).mmtk_get_total_memory)() }
            .try_into()
            .unwrap();
        let constrained_mem = unsafe { ((*UPCALLS).mmtk_get_constrained_memory)() }
            .try_into()
            .unwrap();
        let size_hint = unsafe { ((*UPCALLS).mmtk_get_heap_size_hint)() }
            .try_into()
            .unwrap();
        Box::new(JuliaGCTrigger::new(total_mem, constrained_mem, size_hint))
    }
}

pub fn is_current_gc_nursery() -> bool {
    match crate::SINGLETON.get_plan().generational() {
        Some(gen) => gen.is_current_gc_nursery(),
        None => false,
    }
}

#[no_mangle]
pub extern "C" fn mmtk_block_thread_for_gc(gc_n_threads: u16) {
    AtomicBool::store(&BLOCK_FOR_GC, true, Ordering::SeqCst);

    let &(ref lock, ref cvar) = &*STW_COND.clone();
    let mut count = lock.lock().unwrap();

    info!("Blocking for GC!");

    debug_assert!(
        gc_n_threads as usize == crate::active_plan::VMActivePlan::number_of_mutators(),
        "gc_nthreads = {} != number_of_mutators = {}",
        gc_n_threads,
        crate::active_plan::VMActivePlan::number_of_mutators()
    );

    AtomicBool::store(&WORLD_HAS_STOPPED, true, Ordering::SeqCst);

    while AtomicBool::load(&BLOCK_FOR_GC, Ordering::SeqCst) {
        count = cvar.wait(count).unwrap();
    }
}
