FROM vllm/vllm-openai:v0.29.0

WORKDIR /workspace

# Only what the entropy_plugin needs to be installed as a package -- not the
# whole project (model weights, logs, etc. are bind-mounted at run time).
COPY pyproject.toml README.md ./
COPY muse_glimmer ./muse_glimmer

RUN pip install --no-deps -e .

ENTRYPOINT ["vllm", "serve"]

