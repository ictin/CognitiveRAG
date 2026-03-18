from fastapi import Request

from CognitiveRAG.core.lifecycle import Services


def get_services(request: Request) -> Services:
    return request.app.state.services
