#include "mmtk_julia.h"
#include "mmtk.h"
#include "mmtk_julia_types.h"
#include <stdbool.h>
#include <stddef.h>
#include "gc.h"
#include "julia_gcext.h"

extern int64_t perm_scanned_bytes;
extern void run_finalizer(jl_task_t *ct, void *o, void *ff);
extern int gc_n_threads;
extern jl_ptls_t* gc_all_tls_states;
extern jl_value_t *cmpswap_names JL_GLOBALLY_ROOTED;
extern jl_array_t *jl_global_roots_table JL_GLOBALLY_ROOTED;
extern jl_typename_t *jl_array_typename JL_GLOBALLY_ROOTED;
extern long BI_METADATA_START_ALIGNED_DOWN;
extern long BI_METADATA_END_ALIGNED_UP;
extern void gc_premark(jl_ptls_t ptls2);
extern uint64_t finalizer_rngState[4];
extern const unsigned pool_sizes[];
extern void mmtk_store_obj_size_c(void* obj, size_t size);
extern void jl_gc_free_array(jl_array_t *a);
extern size_t mmtk_get_obj_size(void* obj);
extern void jl_rng_split(uint64_t to[4], uint64_t from[4]);
extern void _jl_free_stack(jl_ptls_t ptls, void *stkbuf, size_t bufsz);
extern void free_stack(void *stkbuf, size_t bufsz);
extern jl_mutex_t finalizers_lock;
extern void jl_gc_wait_for_the_world(jl_ptls_t* gc_all_tls_states, int gc_n_threads);
extern void mmtk_block_thread_for_gc(int gc_n_threads);
extern void combine_thread_gc_counts(jl_gc_num_t *dest);
extern void reset_thread_gc_counts(void);
extern void _jl_free_stack(jl_ptls_t ptls, void *stkbuf, size_t bufsz);
extern void free_stack(void *stkbuf, size_t bufsz);
extern void clear_weak_refs(void);
extern void sweep_weak_refs(void);
extern int64_t live_bytes;

extern void* new_mutator_iterator(void);
extern jl_ptls_t get_next_mutator_tls(void*);
extern void* close_mutator_iterator(void*);

JL_DLLEXPORT void (jl_mmtk_harness_begin)(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    mmtk_harness_begin(ptls);
}

JL_DLLEXPORT void (jl_mmtk_harness_end)(void)
{
    mmtk_harness_end();
}

JL_DLLEXPORT jl_value_t *jl_mmtk_gc_alloc_default(jl_ptls_t ptls, int osize, size_t align, void *ty)
{
    // safepoint
    jl_gc_safepoint_(ptls);

    jl_value_t *v;
    if ((uintptr_t)ty != jl_buff_tag) {
        // v needs to be 16 byte aligned, therefore v_tagged needs to be offset accordingly to consider the size of header
        jl_taggedvalue_t *v_tagged = (jl_taggedvalue_t *)mmtk_immix_alloc_fast(&ptls->mmtk_mutator, LLT_ALIGN(osize, align), align, sizeof(jl_taggedvalue_t));
        v = jl_valueof(v_tagged);
        mmtk_immix_post_alloc_fast(&ptls->mmtk_mutator, v, LLT_ALIGN(osize, align));
    } else {
        // allocating an extra word to store the size of buffer objects
        jl_taggedvalue_t *v_tagged = (jl_taggedvalue_t *)mmtk_immix_alloc_fast(&ptls->mmtk_mutator, LLT_ALIGN(osize+sizeof(jl_taggedvalue_t), align), align, 0);
        jl_value_t* v_tagged_aligned = ((jl_value_t*)((char*)(v_tagged) + sizeof(jl_taggedvalue_t)));
        v = jl_valueof(v_tagged_aligned);
        mmtk_store_obj_size_c(v, LLT_ALIGN(osize+sizeof(jl_taggedvalue_t), align));
        mmtk_immix_post_alloc_fast(&ptls->mmtk_mutator, v, LLT_ALIGN(osize+sizeof(jl_taggedvalue_t), align));
    }
    
    ptls->gc_num.allocd += osize;
    ptls->gc_num.poolalloc++;

    return v;
}

JL_DLLEXPORT jl_value_t *jl_mmtk_gc_alloc_big(jl_ptls_t ptls, size_t sz)
{
    // safepoint
    jl_gc_safepoint_(ptls);

    size_t offs = offsetof(bigval_t, header);
    assert(sz >= sizeof(jl_taggedvalue_t) && "sz must include tag");
    static_assert(offsetof(bigval_t, header) >= sizeof(void*), "Empty bigval header?");
    static_assert(sizeof(bigval_t) % JL_HEAP_ALIGNMENT == 0, "");
    size_t allocsz = LLT_ALIGN(sz + offs, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz) { // overflow in adding offs, size was "negative"
        assert(0 && "Error when allocating big object");
        jl_throw(jl_memory_exception);
    }

    bigval_t *v = (bigval_t*)mmtk_alloc_large(&ptls->mmtk_mutator, allocsz, JL_CACHE_BYTE_ALIGNMENT, 0, 2);

    if (v == NULL) {
        assert(0 && "Allocation failed");
        jl_throw(jl_memory_exception);
    }
    v->sz = allocsz;

    ptls->gc_num.allocd += allocsz;
    ptls->gc_num.bigalloc++;

    jl_value_t *result = jl_valueof(&v->header);
    mmtk_post_alloc(&ptls->mmtk_mutator, result, allocsz, 2);

    return result;
}

static void mmtk_sweep_malloced_arrays(void) JL_NOTSAFEPOINT
{
    void* iter = new_mutator_iterator();
    jl_ptls_t ptls2 = get_next_mutator_tls(iter);
    while(ptls2 != NULL) {
        mallocarray_t *ma = ptls2->heap.mallocarrays;
        mallocarray_t **pma = &ptls2->heap.mallocarrays;
        while (ma != NULL) {
            mallocarray_t *nxt = ma->next;
            if (!mmtk_object_is_managed_by_mmtk(ma->a)) {
                pma = &ma->next;
                ma = nxt;
                continue;
            }
            if (mmtk_is_live_object(ma->a)) {
                // if the array has been forwarded, the reference needs to be updated
                jl_array_t *maybe_forwarded = (jl_array_t*)mmtk_get_possibly_forwared(ma->a);
                ma->a = maybe_forwarded;
                pma = &ma->next;
            }
            else {
                *pma = nxt;
                assert(ma->a->flags.how == 2);
                jl_gc_free_array(ma->a);
                ma->next = ptls2->heap.mafreelist;
                ptls2->heap.mafreelist = ma;
            }
            ma = nxt;
        }
        ptls2 = get_next_mutator_tls(iter);
    }
    gc_sweep_sysimg();
    close_mutator_iterator(iter);
}

void mmtk_wait_in_a_safepoint(void) {
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_gc_safepoint_(ptls);
}

void mmtk_exit_from_safepoint(int8_t old_state) {
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_gc_state_set(ptls, old_state, JL_GC_STATE_WAITING);
}

extern void run_finalizers(jl_task_t *ct);

// based on jl_gc_collect from gc.c
JL_DLLEXPORT void jl_gc_prepare_to_collect(void)
{
    // FIXME: set to JL_GC_AUTO since we're calling it from mmtk
    // maybe just remove this?
    JL_PROBE_GC_BEGIN(JL_GC_AUTO);

    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    if (jl_atomic_load_acquire(&jl_gc_disable_counter)) {
        size_t localbytes = jl_atomic_load_relaxed(&ptls->gc_num.allocd) + gc_num.interval;
        jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);
        static_assert(sizeof(_Atomic(uint64_t)) == sizeof(gc_num.deferred_alloc), "");
        jl_atomic_fetch_add((_Atomic(uint64_t)*)&gc_num.deferred_alloc, localbytes);
        return;
    }

    int8_t old_state = jl_atomic_load_relaxed(&ptls->gc_state);
    jl_atomic_store_release(&ptls->gc_state, JL_GC_STATE_WAITING);
    // `jl_safepoint_start_gc()` makes sure only one thread can run the GC.
    uint64_t t0 = jl_hrtime();
    if (!jl_safepoint_start_gc()) {
        // either another thread is running GC, or the GC got disabled just now.
        jl_gc_state_set(ptls, old_state, JL_GC_STATE_WAITING);
        return;
    }
    JL_TIMING(GC);

    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    // Now we are ready to wait for other threads to hit the safepoint,
    // we can do a few things that doesn't require synchronization.
    //
    // We must sync here with the tls_lock operations, so that we have a
    // seq-cst order between these events now we know that either the new
    // thread must run into our safepoint flag or we must observe the
    // existence of the thread in the jl_n_threads count.
    //
    // TODO: concurrently queue objects
    jl_fence();
    gc_n_threads = jl_atomic_load_acquire(&jl_n_threads);
    gc_all_tls_states = jl_atomic_load_relaxed(&jl_all_tls_states);
    jl_gc_wait_for_the_world(gc_all_tls_states, gc_n_threads);
    JL_PROBE_GC_STOP_THE_WORLD();

    uint64_t t1 = jl_hrtime();
    uint64_t duration = t1 - t0;
    if (duration > gc_num.max_time_to_safepoint)
        gc_num.max_time_to_safepoint = duration;
    gc_num.time_to_safepoint = duration;
    gc_num.total_time_to_safepoint += duration;

    if (!jl_atomic_load_acquire(&jl_gc_disable_counter)) {
        jl_save_context_for_conservative_scanning(ptls, NULL);

        JL_LOCK_NOGC(&finalizers_lock); // all the other threads are stopped, so this does not make sense, right? otherwise, failing that, this seems like plausibly a deadlock
        combine_thread_gc_counts(&gc_num);
        mmtk_block_thread_for_gc(gc_n_threads);
        reset_thread_gc_counts();
        JL_UNLOCK_NOGC(&finalizers_lock);
    }

    gc_n_threads = 0;
    gc_all_tls_states = NULL;
    jl_safepoint_end_gc();
    jl_gc_state_set(ptls, old_state, JL_GC_STATE_WAITING);
    JL_PROBE_GC_END();

    // Only disable finalizers on current thread
    // Doing this on all threads is racy (it's impossible to check
    // or wait for finalizers on other threads without dead lock).
    if (!ptls->finalizers_inhibited && ptls->locks.len == 0) {
        run_finalizers(ct);
    }
    JL_PROBE_GC_FINALIZER();

#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
}

// Called after GC to run finalizers
void mmtk_jl_run_finalizers(void* ptls_raw) {
    jl_ptls_t ptls = (jl_ptls_t) ptls_raw;
    if (!ptls->finalizers_inhibited && ptls->locks.len == 0) {
        run_finalizers(jl_current_task);
    }
}

// We implement finalization in the binding side. These functions
// returns some pointers so MMTk can manipulate finalizer lists.

extern jl_mutex_t finalizers_lock;
extern arraylist_t to_finalize;
extern arraylist_t finalizer_list_marked;

void* get_thread_finalizer_list(void* ptls_raw) {
    jl_ptls_t ptls = (jl_ptls_t) ptls_raw;
    return (void*)&ptls->finalizers;
}

void* get_to_finalize_list(void) {
    return (void*)&to_finalize;
}

void* get_marked_finalizers_list(void) {
    return (void*)&finalizer_list_marked;
}

int* get_jl_gc_have_pending_finalizers(void) {
    return (int*)&jl_gc_have_pending_finalizers;
}

static void add_node_to_roots_buffer(RootsWorkClosure* closure, RootsWorkBuffer* buf, size_t* buf_len, void* root) {
    if (root == NULL)
        return;

    buf->ptr[*buf_len] = root;
    *buf_len += 1;
    if (*buf_len >= buf->cap) {
        RootsWorkBuffer new_buf = (closure->report_nodes_func)(buf->ptr, *buf_len, buf->cap, closure->data, true);
        *buf = new_buf;
        *buf_len = 0;
    }
}

static void add_node_to_tpinned_roots_buffer(RootsWorkClosure* closure, RootsWorkBuffer* buf, size_t* buf_len, void* root) {
    if (root == NULL)
        return;

    buf->ptr[*buf_len] = root;
    *buf_len += 1;
    if (*buf_len >= buf->cap) {
        RootsWorkBuffer new_buf = (closure->report_tpinned_nodes_func)(buf->ptr, *buf_len, buf->cap, closure->data, true);
        *buf = new_buf;
        *buf_len = 0;
    }
}

// staticdata_utils.c
extern jl_array_t *edges_map;
extern jl_array_t *newly_inferred;
// task.c
extern jl_function_t* task_done_hook_func;

#define ADD_GLOBALLY_ROOTED(roots, r) roots[i] = (void*)r; i++;

// This is a list of global variables that are marked with JL_GLOBALLY_ROOTED. We need to make sure that they
// won't be moved. Otherwise, when we access those objects from the C global variables, we may see moved references.
void mmtk_get_globally_rooted(void** roots, int* n)
{
    int i = 0;
    ADD_GLOBALLY_ROOTED(roots, cmpswap_names);
    ADD_GLOBALLY_ROOTED(roots, jl_typeinf_func);
    ADD_GLOBALLY_ROOTED(roots, _jl_debug_method_invalidation);

    // Max 4096
    for (int j = 0; j < N_CALL_CACHE; j++) {
        ADD_GLOBALLY_ROOTED(roots, call_cache[j]);
    }

    // julia_internal.h
    ADD_GLOBALLY_ROOTED(roots, jl_type_type_mt);
    ADD_GLOBALLY_ROOTED(roots, jl_nonfunction_mt);
    ADD_GLOBALLY_ROOTED(roots, jl_kwcall_mt);
    ADD_GLOBALLY_ROOTED(roots, jl_all_methods);
    ADD_GLOBALLY_ROOTED(roots, jl_module_init_order);
    // ADD_GLOBALLY_ROOTED(roots, jl_current_modules); -- we cannot trace a htable_t. But all the modules are pinned. So we are fine.
    ADD_GLOBALLY_ROOTED(roots, jl_precompile_toplevel_module);
    ADD_GLOBALLY_ROOTED(roots, jl_global_roots_table);

    // julia.h
    ADD_GLOBALLY_ROOTED(roots, jl_typeofbottom_type);
    ADD_GLOBALLY_ROOTED(roots, jl_datatype_type);
    ADD_GLOBALLY_ROOTED(roots, jl_uniontype_type);
    ADD_GLOBALLY_ROOTED(roots, jl_unionall_type);
    ADD_GLOBALLY_ROOTED(roots, jl_tvar_type);

    ADD_GLOBALLY_ROOTED(roots, jl_any_type);
    ADD_GLOBALLY_ROOTED(roots, jl_type_type);
    ADD_GLOBALLY_ROOTED(roots, jl_typename_type);
    ADD_GLOBALLY_ROOTED(roots, jl_type_typename);
    ADD_GLOBALLY_ROOTED(roots, jl_symbol_type);
    ADD_GLOBALLY_ROOTED(roots, jl_ssavalue_type);
    ADD_GLOBALLY_ROOTED(roots, jl_abstractslot_type);
    ADD_GLOBALLY_ROOTED(roots, jl_slotnumber_type);
    ADD_GLOBALLY_ROOTED(roots, jl_typedslot_type);
    ADD_GLOBALLY_ROOTED(roots, jl_argument_type);
    ADD_GLOBALLY_ROOTED(roots, jl_const_type);
    ADD_GLOBALLY_ROOTED(roots, jl_partial_struct_type);
    ADD_GLOBALLY_ROOTED(roots, jl_partial_opaque_type);
    ADD_GLOBALLY_ROOTED(roots, jl_interconditional_type);
    ADD_GLOBALLY_ROOTED(roots, jl_method_match_type);
    ADD_GLOBALLY_ROOTED(roots, jl_simplevector_type);
    ADD_GLOBALLY_ROOTED(roots, jl_tuple_typename);
    ADD_GLOBALLY_ROOTED(roots, jl_vecelement_typename);
    ADD_GLOBALLY_ROOTED(roots, jl_anytuple_type);
    ADD_GLOBALLY_ROOTED(roots, jl_emptytuple_type);
    ADD_GLOBALLY_ROOTED(roots, jl_anytuple_type_type);
    ADD_GLOBALLY_ROOTED(roots, jl_vararg_type);
    ADD_GLOBALLY_ROOTED(roots, jl_function_type);
    ADD_GLOBALLY_ROOTED(roots, jl_builtin_type);
    ADD_GLOBALLY_ROOTED(roots, jl_opaque_closure_type);
    ADD_GLOBALLY_ROOTED(roots, jl_opaque_closure_typename);

    ADD_GLOBALLY_ROOTED(roots, jl_bottom_type);
    ADD_GLOBALLY_ROOTED(roots, jl_method_instance_type);
    ADD_GLOBALLY_ROOTED(roots, jl_code_instance_type);
    ADD_GLOBALLY_ROOTED(roots, jl_code_info_type);
    ADD_GLOBALLY_ROOTED(roots, jl_method_type);
    ADD_GLOBALLY_ROOTED(roots, jl_module_type);
    ADD_GLOBALLY_ROOTED(roots, jl_abstractarray_type);
    ADD_GLOBALLY_ROOTED(roots, jl_densearray_type);
    ADD_GLOBALLY_ROOTED(roots, jl_array_type);
    ADD_GLOBALLY_ROOTED(roots, jl_array_typename);
    ADD_GLOBALLY_ROOTED(roots, jl_weakref_type);
    ADD_GLOBALLY_ROOTED(roots, jl_abstractstring_type);
    ADD_GLOBALLY_ROOTED(roots, jl_string_type);
    ADD_GLOBALLY_ROOTED(roots, jl_errorexception_type);
    ADD_GLOBALLY_ROOTED(roots, jl_argumenterror_type);
    ADD_GLOBALLY_ROOTED(roots, jl_loaderror_type);
    ADD_GLOBALLY_ROOTED(roots, jl_initerror_type);
    ADD_GLOBALLY_ROOTED(roots, jl_typeerror_type);
    ADD_GLOBALLY_ROOTED(roots, jl_methoderror_type);
    ADD_GLOBALLY_ROOTED(roots, jl_undefvarerror_type);
    ADD_GLOBALLY_ROOTED(roots, jl_atomicerror_type);
    ADD_GLOBALLY_ROOTED(roots, jl_lineinfonode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_stackovf_exception);
    ADD_GLOBALLY_ROOTED(roots, jl_memory_exception);
    ADD_GLOBALLY_ROOTED(roots, jl_readonlymemory_exception);
    ADD_GLOBALLY_ROOTED(roots, jl_diverror_exception);
    ADD_GLOBALLY_ROOTED(roots, jl_undefref_exception);
    ADD_GLOBALLY_ROOTED(roots, jl_interrupt_exception);
    ADD_GLOBALLY_ROOTED(roots, jl_boundserror_type);
    ADD_GLOBALLY_ROOTED(roots, jl_an_empty_vec_any);
    ADD_GLOBALLY_ROOTED(roots, jl_an_empty_string);

    ADD_GLOBALLY_ROOTED(roots, jl_bool_type);
    ADD_GLOBALLY_ROOTED(roots, jl_char_type);
    ADD_GLOBALLY_ROOTED(roots, jl_int8_type);
    ADD_GLOBALLY_ROOTED(roots, jl_uint8_type);
    ADD_GLOBALLY_ROOTED(roots, jl_int16_type);
    ADD_GLOBALLY_ROOTED(roots, jl_uint16_type);
    ADD_GLOBALLY_ROOTED(roots, jl_int32_type);
    ADD_GLOBALLY_ROOTED(roots, jl_uint32_type);
    ADD_GLOBALLY_ROOTED(roots, jl_int64_type);
    ADD_GLOBALLY_ROOTED(roots, jl_uint64_type);
    ADD_GLOBALLY_ROOTED(roots, jl_float16_type);
    ADD_GLOBALLY_ROOTED(roots, jl_float32_type);
    ADD_GLOBALLY_ROOTED(roots, jl_float64_type);
    ADD_GLOBALLY_ROOTED(roots, jl_floatingpoint_type);
    ADD_GLOBALLY_ROOTED(roots, jl_number_type);
    ADD_GLOBALLY_ROOTED(roots, jl_void_type);  // deprecated
    ADD_GLOBALLY_ROOTED(roots, jl_nothing_type);
    ADD_GLOBALLY_ROOTED(roots, jl_signed_type);
    ADD_GLOBALLY_ROOTED(roots, jl_voidpointer_type);
    ADD_GLOBALLY_ROOTED(roots, jl_uint8pointer_type);
    ADD_GLOBALLY_ROOTED(roots, jl_pointer_type);
    ADD_GLOBALLY_ROOTED(roots, jl_llvmpointer_type);
    ADD_GLOBALLY_ROOTED(roots, jl_ref_type);
    ADD_GLOBALLY_ROOTED(roots, jl_pointer_typename);
    ADD_GLOBALLY_ROOTED(roots, jl_llvmpointer_typename);
    ADD_GLOBALLY_ROOTED(roots, jl_namedtuple_typename);
    ADD_GLOBALLY_ROOTED(roots, jl_namedtuple_type);
    ADD_GLOBALLY_ROOTED(roots, jl_task_type);
    ADD_GLOBALLY_ROOTED(roots, jl_pair_type);

    ADD_GLOBALLY_ROOTED(roots, jl_array_uint8_type);
    ADD_GLOBALLY_ROOTED(roots, jl_array_any_type);
    ADD_GLOBALLY_ROOTED(roots, jl_array_symbol_type);
    ADD_GLOBALLY_ROOTED(roots, jl_array_int32_type);
    ADD_GLOBALLY_ROOTED(roots, jl_array_uint64_type);
    ADD_GLOBALLY_ROOTED(roots, jl_expr_type);
    ADD_GLOBALLY_ROOTED(roots, jl_globalref_type);
    ADD_GLOBALLY_ROOTED(roots, jl_linenumbernode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_gotonode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_gotoifnot_type);
    ADD_GLOBALLY_ROOTED(roots, jl_returnnode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_phinode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_pinode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_phicnode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_upsilonnode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_quotenode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_newvarnode_type);
    ADD_GLOBALLY_ROOTED(roots, jl_intrinsic_type);
    ADD_GLOBALLY_ROOTED(roots, jl_methtable_type);
    ADD_GLOBALLY_ROOTED(roots, jl_typemap_level_type);
    ADD_GLOBALLY_ROOTED(roots, jl_typemap_entry_type);

    ADD_GLOBALLY_ROOTED(roots, jl_emptysvec);
    ADD_GLOBALLY_ROOTED(roots, jl_emptytuple);
    ADD_GLOBALLY_ROOTED(roots, jl_true);
    ADD_GLOBALLY_ROOTED(roots, jl_false);
    ADD_GLOBALLY_ROOTED(roots, jl_nothing);
    ADD_GLOBALLY_ROOTED(roots, jl_kwcall_func);

    ADD_GLOBALLY_ROOTED(roots, jl_main_module);
    ADD_GLOBALLY_ROOTED(roots, jl_core_module);
    ADD_GLOBALLY_ROOTED(roots, jl_base_module);
    ADD_GLOBALLY_ROOTED(roots, jl_top_module);

    // staticdata_utils.c
    ADD_GLOBALLY_ROOTED(roots, edges_map);
    ADD_GLOBALLY_ROOTED(roots, newly_inferred);
    // task.c
    ADD_GLOBALLY_ROOTED(roots, task_done_hook_func);
    // threading.c
    ADD_GLOBALLY_ROOTED(roots, jl_all_tls_states);

    *n = i;
}

void scan_vm_specific_roots(RootsWorkClosure* closure)
{
    // Create a new buf
    RootsWorkBuffer buf = (closure->report_nodes_func)((void**)0, 0, 0, closure->data, true);
    size_t len = 0;

    // add module
    add_node_to_roots_buffer(closure, &buf, &len, jl_main_module);

    // buildin values
    add_node_to_roots_buffer(closure, &buf, &len, jl_an_empty_vec_any);
    add_node_to_roots_buffer(closure, &buf, &len, jl_module_init_order);
    for (size_t i = 0; i < jl_current_modules.size; i += 2) {
        if (jl_current_modules.table[i + 1] != HT_NOTFOUND) {
            add_node_to_roots_buffer(closure, &buf, &len, jl_current_modules.table[i]);
        }
    }
    add_node_to_roots_buffer(closure, &buf, &len, jl_anytuple_type_type);
    for (size_t i = 0; i < N_CALL_CACHE; i++) {
         jl_typemap_entry_t *v = jl_atomic_load_relaxed(&call_cache[i]);
        add_node_to_roots_buffer(closure, &buf, &len, v);
    }
    add_node_to_roots_buffer(closure, &buf, &len, jl_all_methods);
    add_node_to_roots_buffer(closure, &buf, &len, _jl_debug_method_invalidation);

    // constants
    add_node_to_roots_buffer(closure, &buf, &len, jl_emptytuple_type);
    add_node_to_roots_buffer(closure, &buf, &len, cmpswap_names);

    // globally rooted
    #define MAX_GLOBALLY_ROOTED 5000
    void** globally_rooted = (void**) malloc(sizeof(void*) * MAX_GLOBALLY_ROOTED);
    int n_roots = 0;
    mmtk_get_globally_rooted(globally_rooted, &n_roots);
    assert(n_roots <= MAX_GLOBALLY_ROOTED);
    for (int i = 0; i < n_roots; i++) {
        add_node_to_roots_buffer(closure, &buf, &len, globally_rooted[i]);
    }
    free(globally_rooted);

    // jl_global_roots_table must be transitively pinned 
    RootsWorkBuffer tpinned_buf = (closure->report_tpinned_nodes_func)((void**)0, 0, 0, closure->data, true);
    size_t tpinned_len = 0;
    add_node_to_tpinned_roots_buffer(closure, &tpinned_buf, &tpinned_len, jl_global_roots_table);

    // Push the result of the work.
    (closure->report_nodes_func)(buf.ptr, len, buf.cap, closure->data, false);
    (closure->report_tpinned_nodes_func)(tpinned_buf.ptr, tpinned_len, tpinned_buf.cap, closure->data, false);
}

JL_DLLEXPORT void scan_julia_exc_obj(void* obj_raw, void* closure, ProcessSlotFn process_slot) {
    jl_task_t *ta = (jl_task_t*)obj_raw;

    if (ta->excstack) { // inlining label `excstack` from mark_loop
        // if it is not managed by MMTk, nothing needs to be done because the object does not need to be scanned
        if (mmtk_object_is_managed_by_mmtk(ta->excstack)) {
            process_slot(closure, &ta->excstack);
        }
        jl_excstack_t *excstack = ta->excstack;
        size_t itr = ta->excstack->top;
        size_t jlval_index = 0;
        while (itr > 0) {
            size_t bt_size = jl_excstack_bt_size(excstack, itr);
            jl_bt_element_t *bt_data = jl_excstack_bt_data(excstack, itr);
            for (size_t bt_index = 0; bt_index < bt_size; bt_index += jl_bt_entry_size(bt_data + bt_index)) {
                jl_bt_element_t *bt_entry = bt_data + bt_index;
                if (jl_bt_is_native(bt_entry))
                    continue;
                // Found an extended backtrace entry: iterate over any
                // GC-managed values inside.
                size_t njlvals = jl_bt_num_jlvals(bt_entry);
                while (jlval_index < njlvals) {
                    jl_value_t** new_obj_slot = &bt_entry[2 + jlval_index].jlvalue;
                    jlval_index += 1;
                    process_slot(closure, new_obj_slot);
                }
                jlval_index = 0;
            }

            jl_bt_element_t *stack_raw = (jl_bt_element_t *)(excstack+1);
            jl_value_t** stack_obj_slot = &stack_raw[itr-1].jlvalue;

            itr = jl_excstack_next(excstack, itr);
            jlval_index = 0;
            process_slot(closure, stack_obj_slot);
        }
    }
}

// number of stacks to always keep available per pool - from gc-stacks.c
#define MIN_STACK_MAPPINGS_PER_POOL 5

// if data is inlined inside the array object --- to->data needs to be updated when copying the array
void update_inlined_array(void* from, void* to) {
    jl_value_t* jl_from = (jl_value_t*) from;
    jl_value_t* jl_to = (jl_value_t*) to;

    uintptr_t tag_to = (uintptr_t)jl_typeof(jl_to);
    jl_datatype_t *vt = (jl_datatype_t*)tag_to;

    if(vt->name == jl_array_typename) {
        jl_array_t *a = (jl_array_t*)jl_from;
        jl_array_t *b = (jl_array_t*)jl_to;
        if (a->flags.how == 0 && mmtk_object_is_managed_by_mmtk(a->data)) { // a is inlined (a->data is an mmtk object)
            size_t offset_of_data = ((size_t)a->data - a->offset*a->elsize) - (size_t)a;
            if (offset_of_data > 0 && offset_of_data <= ARRAY_INLINE_NBYTES) {
                b->data = (void*)((size_t) b + offset_of_data);
            }
        }
    }
}

// modified sweep_stack_pools from gc-stacks.c
void mmtk_sweep_stack_pools(void)
{
    // Stack sweeping algorithm:
    //    // deallocate stacks if we have too many sitting around unused
    //    for (stk in halfof(free_stacks))
    //        free_stack(stk, pool_sz);
    //    // then sweep the task stacks
    //    for (t in live_tasks)
    //        if (!gc-marked(t))
    //            stkbuf = t->stkbuf
    //            bufsz = t->bufsz
    //            if (stkbuf)
    //                push(free_stacks[sz], stkbuf)
    assert(gc_n_threads);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls2 = jl_all_tls_states[i];

        // free half of stacks that remain unused since last sweep
        for (int p = 0; p < JL_N_STACK_POOLS; p++) {
            small_arraylist_t *al = &ptls2->heap.free_stacks[p];
            size_t n_to_free;
            if (al->len > MIN_STACK_MAPPINGS_PER_POOL) {
                n_to_free = al->len / 2;
                if (n_to_free > (al->len - MIN_STACK_MAPPINGS_PER_POOL))
                    n_to_free = al->len - MIN_STACK_MAPPINGS_PER_POOL;
            }
            else {
                n_to_free = 0;
            }
            for (int n = 0; n < n_to_free; n++) {
                void *stk = small_arraylist_pop(al);
                free_stack(stk, pool_sizes[p]);
            }
        }

        small_arraylist_t *live_tasks = &ptls2->heap.live_tasks;
        size_t n = 0;
        size_t ndel = 0;
        size_t l = live_tasks->len;
        void **lst = live_tasks->items;
        if (l == 0)
            continue;
        while (1) {
            jl_task_t *t = (jl_task_t*)lst[n];
            if (mmtk_is_live_object(t)) {
                jl_task_t *maybe_forwarded = (jl_task_t*)mmtk_get_possibly_forwared(t);
                live_tasks->items[n] = maybe_forwarded;
                t = maybe_forwarded;
                assert(jl_is_task(t));
                if (t->stkbuf == NULL)
                    ndel++; // jl_release_task_stack called
                else
                    n++;
            } else {
                ndel++;
                void *stkbuf = t->stkbuf;
                size_t bufsz = t->bufsz;
                if (stkbuf) {
                    t->stkbuf = NULL;
                    _jl_free_stack(ptls2, stkbuf, bufsz);
                }
#ifdef _COMPILER_TSAN_ENABLED_
                if (t->ctx.tsan_state) {
                    __tsan_destroy_fiber(t->ctx.tsan_state);
                    t->ctx.tsan_state = NULL;
                }
#endif
            }
            if (n >= l - ndel)
                break;
            void *tmp = lst[n];
            lst[n] = lst[n + ndel];
            lst[n + ndel] = tmp;
        }
        live_tasks->len -= ndel;
    }
}

#define jl_array_data_owner_addr(a) (((jl_value_t**)((char*)a + jl_array_data_owner_offset(jl_array_ndims(a)))))

JL_DLLEXPORT void* get_stackbase(int16_t tid) {
    assert(tid >= 0);
    jl_ptls_t ptls2 = jl_all_tls_states[tid];
    return ptls2->stackbase;
}

const bool PRINT_OBJ_TYPE = false;

void update_gc_stats(uint64_t inc, size_t mmtk_live_bytes, bool is_nursery_gc) {
    gc_num.total_time += inc;
    gc_num.pause += 1;
    gc_num.full_sweep += !(is_nursery_gc);
    gc_num.total_allocd += gc_num.allocd;
    gc_num.allocd = 0;
    live_bytes = mmtk_live_bytes;
}

bool check_is_collection_disabled(void) {
    return (jl_atomic_load_acquire(&jl_gc_disable_counter) > 0);
}

size_t get_lo_size(void* obj_raw) 
{
    jl_value_t* obj = (jl_value_t*) obj_raw;
    jl_taggedvalue_t *v = jl_astaggedvalue(obj);
    // bigval_header: but we cannot access the function here. So use container_of instead.
    bigval_t* hdr = container_of(v, bigval_t, header);
    return hdr->sz;
}

uint64_t mmtk_get_total_memory(void) {
    return uv_get_total_memory();
}

uint64_t mmtk_get_constrained_memory(void) {
    return uv_get_constrained_memory();
}

uint64_t mmtk_get_heap_size_hint(void) {
    return jl_options.heap_size_hint;
}

#define assert_size(ty_a, ty_b) \
    if(sizeof(ty_a) != sizeof(ty_b)) {\
        printf("%s size = %ld, %s size = %ld. Need to update our type definition.\n", #ty_a, sizeof(ty_a), #ty_b, sizeof(ty_b));\
        exit(1); \
    }

#define PRINT_STRUCT_SIZE false
#define print_sizeof(type) (PRINT_STRUCT_SIZE ? (printf("C " #type " = %zu bytes\n", sizeof(type)), sizeof(type)) : sizeof(type))

uintptr_t get_abi_structs_checksum_c(void) {
    assert_size(struct mmtk__jl_taggedvalue_bits, struct _jl_taggedvalue_bits);
    assert_size(mmtk_jl_taggedvalue_t, jl_taggedvalue_t);
    assert_size(mmtk_jl_array_flags_t, jl_array_flags_t);
    assert_size(mmtk_jl_datatype_layout_t, jl_datatype_layout_t);
    assert_size(mmtk_jl_typename_t, jl_typename_t);
    assert_size(mmtk_jl_svec_t, jl_svec_t);
    assert_size(mmtk_jl_datatype_t, jl_datatype_t);
    assert_size(mmtk_jl_array_t, jl_array_t);
    assert_size(mmtk_jl_sym_t, jl_sym_t);
    assert_size(mmtk_jl_binding_t, jl_binding_t);
    assert_size(mmtk_htable_t, htable_t);
    assert_size(mmtk_arraylist_t, arraylist_t);
    assert_size(mmtk_jl_uuid_t, jl_uuid_t);
    assert_size(mmtk_jl_mutex_t, jl_mutex_t);
    assert_size(mmtk_jl_module_t, jl_module_t);
    assert_size(mmtk_jl_excstack_t, jl_excstack_t);
    assert_size(mmtk_jl_bt_element_t, jl_bt_element_t);
    assert_size(mmtk_jl_stack_context_t, jl_stack_context_t);
    assert_size(mmtk_jl_ucontext_t, jl_ucontext_t);
    assert_size(struct mmtk__jl_gcframe_t, struct _jl_gcframe_t);
    assert_size(mmtk_jl_task_t, jl_task_t);
    assert_size(mmtk_jl_weakref_t, jl_weakref_t);
    assert_size(mmtk_jl_thread_heap_t, jl_thread_heap_t);

    return print_sizeof(MMTkMutatorContext)
        ^ print_sizeof(struct mmtk__jl_taggedvalue_bits)
        ^ print_sizeof(mmtk_jl_taggedvalue_t)
        ^ print_sizeof(mmtk_jl_array_flags_t)
        ^ print_sizeof(mmtk_jl_datatype_layout_t)
        ^ print_sizeof(mmtk_jl_typename_t)
        ^ print_sizeof(mmtk_jl_svec_t)
        ^ print_sizeof(mmtk_jl_datatype_t)
        ^ print_sizeof(mmtk_jl_array_t)
        ^ print_sizeof(mmtk_jl_sym_t)
        ^ print_sizeof(mmtk_jl_binding_t)
        ^ print_sizeof(mmtk_htable_t)
        ^ print_sizeof(mmtk_arraylist_t)
        ^ print_sizeof(mmtk_jl_uuid_t)
        ^ print_sizeof(mmtk_jl_mutex_t)
        ^ print_sizeof(mmtk_jl_module_t)
        ^ print_sizeof(mmtk_jl_excstack_t)
        ^ print_sizeof(mmtk_jl_bt_element_t)
        ^ print_sizeof(mmtk_jl_stack_context_t)
        ^ print_sizeof(mmtk_jl_ucontext_t)
        ^ print_sizeof(struct mmtk__jl_gcframe_t)
        ^ print_sizeof(mmtk_jl_task_t)
        ^ print_sizeof(mmtk_jl_weakref_t)
        ^ print_sizeof(mmtk_jl_tls_states_t)
        ^ print_sizeof(mmtk_jl_thread_heap_t)
        ^ print_sizeof(mmtk_jl_thread_gc_num_t);
}

void mmtk_jl_throw_out_of_memory_error(void)
{
    jl_throw(jl_memory_exception);
}

uint64_t mmtk_jl_hrtime(void) JL_NOTSAFEPOINT
{
    return uv_hrtime();
}

JL_DLLEXPORT void *mmtk_jl_task_stack_buffer(void *task, size_t *size, int *ptid)
{
    return jl_task_stack_buffer((jl_task_t *)task, size, ptid);
}

Julia_Upcalls mmtk_upcalls = (Julia_Upcalls) {
    .scan_julia_exc_obj = scan_julia_exc_obj,
    .get_stackbase = get_stackbase,
    // .run_finalizer_function = run_finalizer_function,
    .mmtk_jl_run_finalizers = mmtk_jl_run_finalizers,
    .mmtk_jl_throw_out_of_memory_error = mmtk_jl_throw_out_of_memory_error,
    .sweep_malloced_array = mmtk_sweep_malloced_arrays,
    .sweep_stack_pools = mmtk_sweep_stack_pools,
    .clear_weak_refs = clear_weak_refs,
    .sweep_weak_refs = sweep_weak_refs,
    .wait_in_a_safepoint = mmtk_wait_in_a_safepoint,
    .exit_from_safepoint = mmtk_exit_from_safepoint,
    .get_lo_size = get_lo_size,
    .mmtk_jl_hrtime = mmtk_jl_hrtime,
    .update_gc_stats = update_gc_stats,
    .get_abi_structs_checksum_c = get_abi_structs_checksum_c,
    .get_thread_finalizer_list = get_thread_finalizer_list,
    .get_to_finalize_list = get_to_finalize_list,
    .get_marked_finalizers_list = get_marked_finalizers_list,
    .arraylist_grow = (void (*)(void*, long unsigned int))arraylist_grow,
    .get_jl_gc_have_pending_finalizers = get_jl_gc_have_pending_finalizers,
    .scan_vm_specific_roots = scan_vm_specific_roots,
    .update_inlined_array = update_inlined_array,
    .prepare_to_collect = jl_gc_prepare_to_collect,
    .check_is_collection_disabled = check_is_collection_disabled,
    .mmtk_get_total_memory = mmtk_get_total_memory,
    .mmtk_get_constrained_memory = mmtk_get_constrained_memory,
    .mmtk_get_heap_size_hint = mmtk_get_heap_size_hint,
    .mmtk_jl_task_stack_buffer = mmtk_jl_task_stack_buffer,
};
