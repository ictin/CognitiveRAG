import asyncio

def test_synthesizer_includes_session_context(tmp_path):
    from CognitiveRAG.agents.synthesizer import SynthesizerAgent
    class FakeLLM:
        def __init__(self):
            self.last_system = None
            self.last_user = None
        async def ainvoke_text(self, system_prompt=None, user_prompt=None):
            self.last_system = system_prompt
            self.last_user = user_prompt
            return "stub-answer"

    # prepare raw fallback
    import os, json
    sess = 'synth_sess'
    raw = []
    for i in range(10):
        raw.append({'index': i, 'text': f'msg {i}', 'meta': {}})
    data_dir = os.path.join(os.getcwd(), 'data', 'session_memory')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, f'raw_{sess}.json'), 'w', encoding='utf-8') as f:
        json.dump(raw, f)

    fake = FakeLLM()
    synth = SynthesizerAgent(fake)

    class FakeRetrieval:
        def __init__(self):
            self.chunks = []

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ans = loop.run_until_complete(synth.run('query', FakeRetrieval(), session_id=sess))
    loop.close()

    assert fake.last_user is not None
    assert '[session_fresh_tail]' in fake.last_user

    # cleanup
    os.remove(os.path.join(data_dir, f'raw_{sess}.json'))
