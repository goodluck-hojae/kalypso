"""Kalypso KV ownership for vLLM 0.29.0.

Finished pinned requests keep their existing block references. Releasing one
owner therefore cannot evict a shared prefix still owned by another request.
"""


def unpin_requests(scheduler, request_ids):
    keys = set(request_ids)
    for request in list(scheduler.requests.values()):
        if request.kalypso_pin_key not in keys:
            continue
        request.kalypso_pin_key = None
        if request.is_finished():
            scheduler._free_blocks(request)


def get_pinned_requests(scheduler):
    result = []
    for request in scheduler.requests.values():
        if not request.kalypso_pin_key:
            continue
        blocks = scheduler.kv_cache_manager.coordinator.get_blocks(request.request_id)
        count = sum(not block.is_null for group in blocks for block in group)
        result.append({
            "request_id": request.kalypso_pin_key,
            "pinned_blocks": count,
            "total_blocks": count,
        })
    return result
