from huggingface_hub import login, whoami

try:
    whoami()
except OSError:
    login()
