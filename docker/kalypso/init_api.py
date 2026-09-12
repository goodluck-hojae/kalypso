async def init_kalypso(engine_client, state, args):
    from vllm.kalypso.execution.vllm_executor import VLLMExecutor

    config = engine_client.vllm_config
    if config.parallel_config.data_parallel_size != 1:
        raise ValueError("Kalypso currently requires data_parallel_size=1")
    if getattr(args, "api_server_count", 1) not in (None, 1):
        raise ValueError("Kalypso currently requires one API server process")
    if config.kv_transfer_config is not None:
        raise ValueError("Kalypso does not yet support KV transfer connectors")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Kalypso currently requires --no-async-scheduling")
    if config.speculative_config is not None:
        raise ValueError("Kalypso speculative decoding has not been validated")
    if not config.cache_config.enable_prefix_caching:
        raise ValueError("Kalypso requires --enable-prefix-caching")
    per_gpu = await engine_client.engine_core.call_utility_async("get_kv_cache_budget")
    if per_gpu <= 0:
        raise ValueError("Kalypso requires a positive GPU KV-cache budget")
    state.query_processor = query_processor.QueryProcessor(
        model_name=config.model_config.model,
        budget=per_gpu * config.parallel_config.tensor_parallel_size,
    )
    # Tokenization uses the model path; API requests use its public alias.
    aliases = config.model_config.served_model_name
    model = aliases[0] if isinstance(aliases, list) else aliases
    state.query_processor.executor = VLLMExecutor(model or config.model_config.model)
    state.query_processor.start_stuck_monitor(engine_client)
