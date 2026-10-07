from vllm.kalypso import op_usage
from vllm.kalypso.sem_ops import ops
from vllm.kalypso.budget import KVMemoryManager

def pipeline_builder(ops, stage_id):
    def _pipeline(ctx):
        return SemanticPipeline(
            ctx,
            *ops,
            bytes_per_token=KVMemoryManager.get_instance().bytes_per_token,
            stage_id=stage_id,
        )
    _pipeline.ops = ops
    _pipeline.stage_id = stage_id
    return _pipeline

def is_pipeline_builder(obj):
    return (
        callable(obj)
        and hasattr(obj, "ops")
        and hasattr(obj, "stage_id")
    )

class SemanticPipeline:
    def __init__(self, ctx, *ops, bytes_per_token: int, stage_id: str):
        self.ctx = ctx
        self.ops = ops
        self.bytes_per_token = bytes_per_token
        self.stage_id = stage_id
        self.budget = self.estimate_token_budget(ctx)
        

    # budget function should be updated based on operations
    def estimate_token_budget(self, ctx) -> int:
        max_boundary = -1
        for op in self.ops:
            if isinstance(op, ops.CartesianProduct):
                break

            estimated_tokens = op.estimate_tokens(ctx)
            if not hasattr(op, "estimate_tokens"):
                raise AttributeError(
                    f"{op} must define `estimate_tokens`"
                )
            if estimated_tokens > max_boundary:
                max_boundary = estimated_tokens

        self.budget = max_boundary * self.bytes_per_token
        return self.budget


    async def __call__(self):
        # Set stage_id on context for budget tracking during retries
        self.ctx.state.stage_id = self.stage_id
        
        next = True
        for idx, op in enumerate(self.ops):
            if next:
                token = op_usage.current_op.set(op_usage.op_label(op, self.stage_id, idx))
                try:
                    next = await op(self.ctx, priority=-idx)
                finally:
                    op_usage.current_op.reset(token)

                if next is False:
                    return None
        return self.ctx
