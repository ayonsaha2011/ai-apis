from __future__ import annotations

from collections.abc import Iterator
from threading import Thread
from typing import TYPE_CHECKING, Any

from openai_multi_backend.models.base import BaseModelAdapter, TextGeneration, import_optional

if TYPE_CHECKING:
    from openai_multi_backend.api.schemas import ChatMessage, CompletionRequest


class CausalLMAdapter(BaseModelAdapter):
    tokenizer: Any
    model: Any

    def load(self) -> None:
        transformers = import_optional("transformers")
        torch = import_optional("torch")
        self.device = self.resolve_device()
        torch_dtype = self.resolve_torch_dtype()
        self.dtype = str(torch_dtype).replace("torch.", "")
        hf_kwargs = self.common_hf_kwargs()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **hf_kwargs)
        model_kwargs = dict(hf_kwargs)
        model_kwargs["torch_dtype"] = torch_dtype
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id, **model_kwargs
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.set_grad_enabled(False)

    def generate_chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stop: str | list[str] | None,
        frequency_penalty: float | None,
    ) -> TextGeneration:
        prompt = self._messages_to_prompt(messages, add_generation_prompt=True)
        return self._generate(prompt, temperature, top_p, max_tokens, stop, frequency_penalty)

    def generate_completion(self, request: CompletionRequest) -> TextGeneration:
        return self._generate(
            request.first_prompt(),
            request.temperature,
            request.top_p,
            request.max_tokens,
            request.stop,
            request.frequency_penalty,
        )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stop: str | list[str] | None,
        frequency_penalty: float | None,
    ) -> Iterator[str]:
        prompt = self._messages_to_prompt(messages, add_generation_prompt=True)
        yield from self._stream_prompt(
            prompt, temperature, top_p, max_tokens, stop, frequency_penalty
        )

    def stream_completion(self, request: CompletionRequest) -> Iterator[str]:
        yield from self._stream_prompt(
            request.first_prompt(),
            request.temperature,
            request.top_p,
            request.max_tokens,
            request.stop,
            request.frequency_penalty,
        )

    def _stream_prompt(
        self,
        prompt: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stop: str | list[str] | None,
        frequency_penalty: float | None,
    ) -> Iterator[str]:
        transformers = import_optional("transformers")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._model_device()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        streamer = transformers.TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = self._generation_kwargs(temperature, top_p, max_tokens, frequency_penalty)
        kwargs.update(inputs)
        kwargs["streamer"] = streamer
        thread = Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        stop_strings = self._stop_strings(stop)
        buffer = ""
        for chunk in streamer:
            buffer += chunk
            trimmed, stopped = self._trim_at_stop(buffer, stop_strings)
            if stopped:
                if trimmed:
                    yield trimmed
                break
            yield chunk
            buffer = ""
        thread.join(timeout=1)

    def _generate(
        self,
        prompt: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stop: str | list[str] | None,
        frequency_penalty: float | None,
    ) -> TextGeneration:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._model_device()
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        generation_kwargs = self._generation_kwargs(
            temperature, top_p, max_tokens, frequency_penalty
        )
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        prompt_length = input_ids.shape[-1]
        completion_ids = output_ids[0][prompt_length:]
        raw_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        text, stopped = self._trim_at_stop(raw_text, self._stop_strings(stop))
        prompt_tokens = int(prompt_length)
        completion_tokens = int(completion_ids.shape[-1])
        return TextGeneration(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop" if stopped else "length",
        )

    def _generation_kwargs(
        self,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        frequency_penalty: float | None,
    ) -> dict[str, Any]:
        max_new_tokens = max_tokens or self.settings.text_max_new_tokens_default
        max_new_tokens = min(max_new_tokens, self.settings.text_max_new_tokens_limit)
        do_sample = temperature is not None and temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p or 1.0
        if frequency_penalty and frequency_penalty > 0:
            kwargs["repetition_penalty"] = 1.0 + min(frequency_penalty, 2.0) / 10.0
        return kwargs

    def _messages_to_prompt(self, messages: list[ChatMessage], add_generation_prompt: bool) -> str:
        normalized = [
            {"role": message.role, "content": message.text_content()} for message in messages
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        lines = [f"{message['role'].title()}: {message['content']}" for message in normalized]
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)

    def _model_device(self) -> Any:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    @staticmethod
    def _stop_strings(stop: str | list[str] | None) -> list[str]:
        if stop is None:
            return []
        return [stop] if isinstance(stop, str) else stop

    @staticmethod
    def _trim_at_stop(text: str, stop_strings: list[str]) -> tuple[str, bool]:
        earliest: int | None = None
        for stop in stop_strings:
            index = text.find(stop)
            if index >= 0 and (earliest is None or index < earliest):
                earliest = index
        if earliest is None:
            return text, False
        return text[:earliest], True
