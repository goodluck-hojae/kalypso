import random

from vllm.kalypso.sem_ops import OpBehavior, OpName, ops
from vllm.kalypso.query import Query
from vllm.kalypso.data import data
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.execution.pipeline_execution import PlanExecutor
from vllm.kalypso.controller.map_estimator import MapRatioEstimator
from .stage import Stage, Task, stage_builder

class SemanticPlan:
    ESTIMATED_CTX_SAMPLE_SIZE = 100

    def __init__(self, executor):
        self.executor = executor
        self.plan_executor = PlanExecutor()


    def build(self, ctxs, operators):

        plan = []

        current_segment = []          # [stage, stage, ...]
        stage_ops_chain = []          # tuple-independent ops to fuse
        next_stage_id = 1
        next_priority_offset = 0

        def emit_stage():
            nonlocal stage_ops_chain, current_segment, next_stage_id, next_priority_offset

            if not stage_ops_chain:
                return

            current_stage = stage_builder(
                tuple(stage_ops_chain),
                next_stage_id,
                priority_offset=next_priority_offset,
            )

            current_segment.append(current_stage)

            next_stage_id += 1
            next_priority_offset += len(stage_ops_chain)
            stage_ops_chain = []

        def flush_segment():
            nonlocal current_segment

            if not current_segment:
                return

            # collect stage_ids in this segment
            stage_ids = [
                item.stage_id
                for item in current_segment
                if hasattr(item, "stage_id")
            ]

            kv = KVMemoryManager.get_instance()
            num_stages = len(stage_ids)
            stage_min_caps = {}
            estimated_ctxs = list(ctxs)

            def estimate_fanout_ctxs(fanout_op, input_ctxs):
                if isinstance(fanout_op, ops.IndexedCartesianProduct):
                    cp_limit = fanout_op.top_k if fanout_op.top_k is not None else len(fanout_op.right_table)
                    approx_cp = ops.CartesianProduct(
                        right_table=fanout_op.right_table[:cp_limit],
                        position=fanout_op.position,
                    )
                    next_ctxs = []
                    for ctx in input_ctxs:
                        next_ctxs.extend(approx_cp(ctx) or [])
                    return next_ctxs

                next_ctxs = []
                for ctx in input_ctxs:
                    next_ctxs.extend(fanout_op(ctx) or [])
                return next_ctxs

            for stage in current_segment:
                sampled_ctxs = estimated_ctxs
                if len(estimated_ctxs) > self.ESTIMATED_CTX_SAMPLE_SIZE:
                    sampled_ctxs = random.sample(
                        estimated_ctxs,
                        self.ESTIMATED_CTX_SAMPLE_SIZE,
                    )
                budgets = [
                    stage.estimate_budget(Task(ctx=ctx, stage_index=0))
                    for ctx in sampled_ctxs
                ]
                if budgets:
                    stage_min_caps[stage.stage_id] = max(budgets)

                if stage.fanout_op is not None:
                    estimated_ctxs = estimate_fanout_ctxs(
                        stage.fanout_op,
                        estimated_ctxs,
                    )

            if num_stages == 1:
                kv.register_stage(
                    stage_ids[0],
                    1.0,
                    min_fraction=1.0,
                    max_fraction=1.0,
                )
            # Start early stages at their minimum admission budget and give the
            # last stage the remaining capacity, since it is usually the
            # bottleneck after fanout.
            # elif num_stages == 2:
            #     fixed_fractions = (0.9, 0.1)
            #     for sid, fraction in zip(stage_ids, fixed_fractions):
            #         kv.register_stage(
            #             sid,
            #             fraction,
            #             min_fraction=fraction,
            #             max_fraction=fraction,
            #         )
            elif num_stages > 1:
                total_capacity = kv.capacity()
                assigned_caps = {
                    sid: stage_min_caps.get(sid, 0)
                    for sid in stage_ids[:-1]
                }
                assigned_caps[stage_ids[-1]] = total_capacity - sum(
                    assigned_caps.values()
                )
                

                for idx, sid in enumerate(stage_ids):
                    assigned_cap = max(kv.bytes_per_token, assigned_caps[sid])
                    protected_min_cap = stage_min_caps.get(sid, 0)
                    initial_fraction = min(1.0, assigned_cap / total_capacity)
                    if idx == num_stages - 1:
                        max_fraction = 1.0
                    else:
                        max_fraction = min(1.0, initial_fraction + 0.5)
                    kv.register_stage(
                        sid,
                        initial_fraction,
                        min_fraction=0.0,
                        max_fraction=max_fraction,
                        min_capacity_bytes=protected_min_cap,
                    )

            plan.append(current_segment)
            current_segment = []

        for op in operators:

            if op.behavior == OpBehavior.TUPLE_INDEPENDENT:
                stage_ops_chain.append(op)
                continue

            # boundary encountered
            emit_stage()

            if op.behavior == OpBehavior.JOIN:  # CP
                if not current_segment:
                    flush_segment()
                    plan.append(op)
                else:
                    current_segment[-1].fanout_op = op

            elif op.behavior == OpBehavior.BLOCKING:
                flush_segment()
                plan.append(op)

            else:
                raise ValueError(f"Unknown op behavior: {op.behavior}")

        # tail flush
        emit_stage()
        flush_segment()

        return plan



    def print_plan(self, plan):
        for i, item in enumerate(plan, start=1):
            # PIPELINE: [pipe, cp, pipe, ...]
            if isinstance(item, list):
                print(f"[{i}] PIPELINE")
                for j, stage in enumerate(item, start=1):
                    op_names = [op.__class__.__name__ for op in stage.operators]
                    fanout_name = (
                        stage.fanout_op.__class__.__name__
                        if stage.fanout_op is not None
                        else None
                    )
                    print(
                        f"  ({j}) STAGE stage_id={stage.stage_id} "
                        f"behavior={stage.behavior.name} ops={op_names} fanout={fanout_name}"
                    )
            elif isinstance(item, ops.BaseOp) and item.behavior == OpBehavior.JOIN:
                print(f"[{i}] JOIN {item.__class__.__name__}")
            else:
                # Blocking op
                print(f"[{i}] BLOCKING {item.__class__.__name__}")


    def parse_ops(self, logical_ops):

        def apply_pin_unpin(ops_list):

            # Reset flags
            for op in ops_list:
                op.pin = False
                op.unpin = False

            chain = []

            def finalize_chain(next_behavior):
                nonlocal chain
                if not chain:
                    return

                first = chain[0]

                # Chain feeds CP
                if next_behavior == OpBehavior.JOIN:
                    first.pin = True

                # End of query
                elif next_behavior is None:
                    if len(chain) > 1:
                        last = chain[-1]
                        first.pin = True
                        last.unpin = True

                chain = []

            n = len(ops_list)

            for i, op in enumerate(ops_list):
                next_op = ops_list[i + 1] if i + 1 < n else None
                next_behavior = next_op.behavior if next_op else None

                if op.behavior == OpBehavior.TUPLE_INDEPENDENT:
                    chain.append(op)

                    # boundary detected
                    if next_op is None or next_behavior != OpBehavior.TUPLE_INDEPENDENT:
                        finalize_chain(next_behavior)

                else:
                    chain = []

            for op in ops_list:
                print(f"op: {op}, pin: {op.pin} unpin: {op.unpin}")
                assert not (op.pin and op.unpin), (
                    f"Invalid pin state for {op}: pin and unpin cannot both be True."
                )

            return ops_list

        physical = []

        for idx, node in enumerate(logical_ops):
            name = node["op"]
            args = node.get("args", {})

            if name == OpName.SEM_FILTER:
                if args.get("cascade", False) or args.get("llm_cascade", False):
                    physical.append(
                        ops.CascadeOperator(
                            instruction=args["prompt"],
                            model_name=args.get("cascade_model"),
                            api_base=args.get("cascade_api_base"),
                            api_port=args.get("cascade_port"),
                            max_tokens=args.get("cascade_max_tokens", 8),
                            low_threshold=args.get("cascade_low_threshold", args.get("low_threshold")),
                            high_threshold=args.get("cascade_high_threshold", args.get("high_threshold")),
                            position=idx,
                        )
                    )
                else:
                    physical.append(
                        ops.SemFilter(
                            instruction=args["prompt"],
                            position=idx
                        )
                    )

            elif name == OpName.SEM_MAP:
                physical.append(
                    ops.SemMap(
                        instruction=args["prompt"],
                        position=idx
                    )
                )

            elif name in (OpName.SEM_CLASSIFY):
                physical.append(
                    ops.SemClassify(
                        classes=args['classes'],
                        position=idx
                    )
                )

            elif name == OpName.SEM_AGG:
                physical.append(
                    ops.SemAgg(
                        instruction=args["instruction"],
                        position=idx
                    )
                )

            elif name == OpName.SEM_TOPK:
                physical.append(
                    ops.SemTopK(
                        instruction=args["instruction"],
                        k=args["k"],
                        position=idx
                    )
                )

            elif name == OpName.JOIN:
                icp = args.get("icp", False)
                icp_oracle_fallback = args.get("icp_oracle_fallback", False)
                right_table = (
                    list(data._data_source(None, args["right_table"], None))
                    if args.get("right_table") is not None
                    else []
                )

                if icp is True:
                    physical.append(
                        ops.IndexedCartesianProduct(
                            right_table=right_table,
                            service_address=args.get(
                                "icp_address",
                                "127.0.0.1",
                            ),
                            service_port=args.get("icp_port", 8000),
                            top_k=args.get("icp_top_k", 5),
                            low_threshold=args.get("icp_low_threshold"),
                            high_threshold=args.get("icp_high_threshold"),
                            cp_id=args.get("icp_cp_id"),
                            position=idx,
                        )
                    )
                else:
                    physical.append(
                        ops.CartesianProduct(
                            right_table=right_table,
                            position=idx
                        )
                    )

                if args.get("cascade", False) or args.get("llm_cascade", False):
                    physical.append(
                        ops.CascadeOperator(
                            instruction=args["instruction"],
                            model_name=args.get("cascade_model"),
                            api_base=args.get("cascade_api_base"),
                            api_port=args.get("cascade_port"),
                            max_tokens=args.get("cascade_max_tokens", 8),
                            low_threshold=args.get("cascade_low_threshold", args.get("low_threshold")),
                            high_threshold=args.get("cascade_high_threshold", args.get("high_threshold")),
                            position=idx,
                        )
                    )
                elif (
                    icp is True
                    and args.get("icp_low_threshold") is not None
                    and args.get("icp_high_threshold") is not None
                ):
                    oracle_instruction = (
                        args.get("icp_oracle_instruction")
                        or args.get("instruction")
                    )
                    if not oracle_instruction:
                        raise ValueError(
                            "ICP join filtering requires instruction or "
                            "icp_oracle_instruction."
                        )
                    physical.append(
                        ops.ICPFilter(
                            instruction=oracle_instruction,
                            low_threshold=args.get("icp_low_threshold"),
                            high_threshold=args.get("icp_high_threshold"),
                            max_tokens=args.get("icp_oracle_max_tokens", 8),
                            position=idx,
                        )
                    )
                else:
                    physical.append(
                        ops.SemFilter(
                            instruction=args["instruction"],
                            position=idx
                        )
                    )

            elif name == OpName.CARTESIAN_PRODUCT:
                icp = args.get("icp", False)
                icp_oracle_fallback = args.get("icp_oracle_fallback", False)
                right_table = (
                    list(data._data_source(None, args["right_table"], None))
                    if args.get("right_table") is not None
                    else []
                )

                if icp is True:
                    physical.append(
                        ops.IndexedCartesianProduct(
                            right_table=right_table,
                            service_address=args.get(
                                "icp_address",
                                "127.0.0.1",
                            ),
                            service_port=args.get("icp_port", 8000),
                            top_k=args.get("icp_top_k", 5),
                            low_threshold=args.get("icp_low_threshold"),
                            high_threshold=args.get("icp_high_threshold"),
                            cp_id=args.get("icp_cp_id"),
                            position=idx,
                        )
                    )
                else:
                    physical.append(
                        ops.CartesianProduct(
                            right_table=right_table,
                            position=idx
                        )
                    )

                if icp_oracle_fallback:
                    oracle_instruction = (
                        args.get("icp_oracle_instruction")
                        or args.get("instruction")
                    )
                    if not oracle_instruction:
                        raise ValueError(
                            "icp_oracle_fallback requires icp_oracle_instruction "
                            "or instruction for a follow-up SemFilter."
                        )
                    physical.append(
                        ops.ICPFilter(
                            instruction=oracle_instruction,
                            low_threshold=args.get("icp_low_threshold"),
                            high_threshold=args.get("icp_high_threshold"),
                            max_tokens=args.get("icp_oracle_max_tokens", 8),
                            position=idx,
                        )
                    )

            elif name == OpName.INDEXED_SEARCH:
                right_table = (
                    list(data._data_source(None, args["right_table"], None))
                    if args.get("right_table") is not None
                    else []
                )
                physical.append(
                    ops.IndexedSearch(
                        right_table=right_table,
                        service_address=args.get(
                            "icp_address",
                            "127.0.0.1",
                        ),
                        service_port=args.get("icp_port", 8000),
                        top_k=args.get("icp_top_k", 5),
                        low_threshold=args.get("icp_low_threshold"),
                        high_threshold=args.get("icp_high_threshold"),
                        cp_id=args.get("icp_cp_id"),
                        position=idx,
                    )
                )

            else:
                raise ValueError(f"Unknown op: {name}")

        apply_pin_unpin(physical)

        return tuple(physical)


    
    async def warmup(self, ctxs, plan):
        if not ctxs:
            return []
        out, _ = await self.plan_executor.execute(ctxs, plan)
        return out

        

    async def execute(self, raw_request, query: Query):
        
        ctxs = list(data._data_source(raw_request, query.data_path, self.executor))
        MapRatioEstimator.instance()
        physical_ops = self.parse_ops(query.ops)
        plan = self.build(ctxs, physical_ops)
        self.print_plan(plan)

        ops.ICPFilter.reset_stats()
        ops.CascadeOperator.reset_stats()
        out, _ = await self.plan_executor.execute(ctxs, plan)
        MapRatioEstimator.instance().reset()
        icp_stats = ops.ICPFilter.get_stats()
        if icp_stats["called"] > 0:
            print(f"[query] ICPFilter stats={icp_stats}")
        cascade_stats = ops.CascadeOperator.get_stats()
        if cascade_stats["called"] > 0:
            print(f"[query] CascadeOperator stats={cascade_stats}")
        print(f'len(out){len(out)}')
        return out 
