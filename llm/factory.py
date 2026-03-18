from __future__ import annotations
from dataclasses import dataclass
from CognitiveRAG.core.settings import settings
from CognitiveRAG.llm.clients_ollama import OllamaClient
from pydantic import BaseModel
from typing import TypeVar
import json

SchemaT = TypeVar('SchemaT', bound=BaseModel)


@dataclass
class OllamaLLMClient:
    client: OllamaClient
    model: str

    async def ainvoke_text(self, system_prompt: str, user_prompt: str) -> str:
        prompt = system_prompt + "\n\n" + user_prompt
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        resp = self.client.chat(self.model, messages)
        # prefer message.content path
        if isinstance(resp, dict):
            if 'message' in resp and isinstance(resp['message'], dict):
                return resp['message'].get('content', '')
            if 'choices' in resp and resp['choices']:
                return resp['choices'][0].get('message', {}).get('content', '')
        return str(resp)

    async def ainvoke_structured(self, system_prompt: str, user_prompt: str, schema: type[SchemaT]) -> SchemaT:
        # ask Ollama to respond with JSON matching the Pydantic schema
        instruct = (
            system_prompt
            + "\n\nRespond only with a JSON object matching the requested schema."
        )
        messages = [{"role": "system", "content": instruct}, {"role": "user", "content": user_prompt}]
        resp = self.client.chat(self.model, messages)
        content = ''
        if isinstance(resp, dict):
            if 'message' in resp and isinstance(resp['message'], dict):
                content = resp['message'].get('content', '')
            elif 'choices' in resp and resp['choices']:
                content = resp['choices'][0].get('message', {}).get('content', '')
            else:
                content = str(resp)
        else:
            content = str(resp)
        # attempt to parse JSON from content
        try:
            # sometimes content is quoted JSON inside strings; try to find first {..}
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                j = json.loads(content[start:end+1])
                return schema.parse_obj(j)
            return schema.parse_raw(content)
        except Exception:
            # fallback: return empty schema instance (still better to raise in future)
            try:
                return schema()
            except Exception:
                raise


@dataclass
class LLMClients:
    planner: OllamaLLMClient
    synthesizer: OllamaLLMClient
    critic: OllamaLLMClient


def build_llm_clients(settings):
    base = settings.llm.ollama_base_url
    path = settings.llm.ollama_api_path
    api_key = settings.llm.ollama_api_key or None
    # models from settings
    planner_m = settings.llm.planner_model
    synth_m = settings.llm.synthesis_model
    critic_m = settings.llm.reflection_model

    client = OllamaClient(base, api_path=path, api_key=api_key)
    return LLMClients(
        planner=OllamaLLMClient(client=client, model=planner_m),
        synthesizer=OllamaLLMClient(client=client, model=synth_m),
        critic=OllamaLLMClient(client=client, model=critic_m),
    )
