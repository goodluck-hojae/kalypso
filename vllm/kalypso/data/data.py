import os
import csv 
from pathlib import Path 
from vllm.kalypso.context import SemContext, SemanticInput, ExecutionState
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.sem_ops.prompt_utils import get_data_prompt


def _data_source(raw_request, data_path, executor):
    path = Path(data_path)

    if path.is_dir():
        for txt_path in sorted(path.glob("*.txt"), key=lambda p: int(p.stem)):
            yield from _message_reader(raw_request, txt_path, executor)

    elif path.suffix.lower() == ".json":
        yield from _message_reader(raw_request, path, executor)

    elif path.suffix.lower() == ".csv":
        for ctx in _csv_reader(raw_request, path, executor):
            yield ctx


def _message_reader(raw_request, path: Path, executor):
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    yield SemContext(
        input=SemanticInput(
            data=get_data_prompt(text),
            token_len=-1,
        ),
        state=ExecutionState(
            raw_request=raw_request,
            pin_req_id=None,
            executor=executor,
            idx=int(path.stem),
        ),
    )


def _csv_reader(raw_request, path: Path, executor):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for idx, row in enumerate(rows):
            if "data" in row:
                text = str(row["data"]).strip()
            elif "Text" in row and "Sentences" in row:
                text = (
                    f"Text ID: {str(row.get('Text ID', '')).strip()}\n\n"
                    f"Clinical note:\n{str(row['Text']).strip()}\n\n"
                    f"Numbered sentences:\n{str(row['Sentences']).strip()}"
                )
            elif "Text" in row:
                text = str(row["Text"]).strip()
            else:
                text = " ".join(str(value).strip() for value in row.values())
            yield SemContext(
                input=SemanticInput(
                        data=text,
                        token_len=KVMemoryManager.get_instance().token_length(text),
                    ),
                state=ExecutionState(
                    raw_request=raw_request,
                    pin_req_id=None,
                    executor=executor,
                    idx=idx,
                )
            )


def research_category_data():
    
    categories = ['ai', 'biology', 'chemistry', 'geology', 'math', 'phyics']
    out = []
    for category in categories:
        
        ctx = SemContext(
            input=SemanticInput(
                data=get_data_prompt(category),
                token_len=-1,
            ),
            state=ExecutionState(
                raw_request=None,
                pin_req_id=None,
                executor=None,
                idx=-1,
            )
        )
        out.append(ctx)
    return out
                

# def _message_reader(raw_request, path: Path, executor):
#     with path.open("r", encoding="utf-8") as f:
#         messages = json.load(f)

#     prompt = KVMemoryManager.get_instance().tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=False,
#     )
#     # messages must be a list[dict] like [{"role":..., "type":..., "content":...}, ...]
#     yield SemContext(
#         input=SemanticInput(
#             data=messages,
#             token_len=KVMemoryManager.get_instance().token_length(prompt),
#         ),
#         state=ExecutionState(
#             raw_request=raw_request,
#             pin_req_id=None,
#             executor=executor,
#             idx=int(path.name.split('.')[0])
#         ),
#     )

