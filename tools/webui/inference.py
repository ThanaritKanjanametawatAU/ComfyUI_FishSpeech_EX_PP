import html
from functools import partial
from typing import Any, Callable

from fish_speech.i18n import i18n
from tools.schema import ServeReferenceAudio, ServeTTSRequest
import torch


def inference_wrapper(
    text,
    normalize,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    """

    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)
    else:
        references = []

    # 如果seed为0，生成一个随机种子
    if not seed:
        seed_value = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        seed_value = int(seed)
    
    # 设置随机种子
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)

    req = ServeTTSRequest(
        text=text,
        normalize=normalize,
        reference_id=reference_id if reference_id else None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=seed_value,
        use_memory_cache=use_memory_cache,
    )

    for result in engine.inference(req):
        match result.code:
            case "final":
                return result.audio, None, f"当前使用的种子：{seed_value}"
            case "error":
                return None, build_html_error_message(i18n(result.error)), ""
            case _:
                pass

    return None, i18n("No audio generated"), ""


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )
