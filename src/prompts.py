from ._constants import (
    SYSTEM_MESSAGES_COMBINED,
    SYSTEM_MESSAGES_DEFAULT,
    SYSTEM_MESSAGES_LANGFLOW,
)


def get_sys_messages(strategy):
    if strategy == "default":
        return SYSTEM_MESSAGES_DEFAULT
    elif strategy == "langflow":
        return SYSTEM_MESSAGES_LANGFLOW
    else:
        return SYSTEM_MESSAGES_COMBINED
