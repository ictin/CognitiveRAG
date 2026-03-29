from CognitiveRAG.crag.promotion.speech_act import strip_speech_act


def test_strip_speech_act_removes_conversational_wrapper():
    text = "The user said that we concluded that user prefers practical steps."
    out = strip_speech_act(text)
    assert "the user said" not in out.lower()
    assert "we concluded" not in out.lower()
    assert "user prefers practical steps" in out.lower()

