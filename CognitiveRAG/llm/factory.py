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
        except Exception as e:
            # log raw content for debugging
            try:
                from CognitiveRAG.core.logging import logger
                logger.error("AINVOKE_STRUCTURED PARSE FAILED: %s", content)
            except Exception:
                pass
            # For PlannerOutput specifically, return a minimal sensible PlannerOutput
            if schema.__name__ == 'PlannerOutput':
                try:
                    # create a minimal compatible object
                    data = {"objective": user_prompt, "steps": ["Read the retrieved context", "Identify the main topic", "Answer briefly"]}
                    return schema.parse_obj(data)
                except Exception:
                    pass
            # For CriticOutput, try to extract mapped fields from raw JSON then return explicit parse_error if not possible
            if schema.__name__ == 'CriticOutput':
                try:
                    j = json.loads(content)
                    approved = j.get('approved') if isinstance(j, dict) else None
                    issues = j.get('issues') if isinstance(j, dict) else None
                    if approved is None:
                        # try grounded/complete/actionable mapping
                        grounded = j.get('grounded')
                        complete = j.get('complete')
                        actionable = j.get('actionable')
                        issues = issues or []
                        if grounded is not None:
                            approved = bool(grounded)
                            if not bool(grounded):
                                issues.append('not_grounded')
                            if complete is not None and not bool(complete):
                                issues.append('incomplete')
                            if actionable is not None and not bool(actionable):
                                issues.append('not_actionable')
                    if approved is None:
                        # fallback explicit parse error
                        return schema.parse_obj({"approved": False, "issues": ["parse_error"]})
                    return schema.parse_obj({"approved": bool(approved), "issues": issues or []})
                except Exception:
                    try:
                        return schema.parse_obj({"approved": False, "issues": ["parse_error"]})
                    except Exception:
                        pass
            # generic fallback: attempt to populate minimal fields when possible
            try:
                # build minimal dict from schema fields
                fields = getattr(schema, 'model_fields', None) or {}
                data = {}
                for fname, finfo in fields.items():
                    # best-effort defaults: booleans -> False, lists -> [], else -> ''
                    if fname == 'approved':
                        data[fname] = False
                    elif fname.endswith('s'):
                        data[fname] = []
                    else:
                        data[fname] = ''
                # best-effort populate 'objective' if present
                if 'objective' in data:
                    data['objective'] = user_prompt
                return schema.parse_obj(data)
            except Exception:
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
