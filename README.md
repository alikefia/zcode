## ZCode

This is a **personal** project with the ambition to build a local coder assistant.

Target features are :
- An LSP server with completion based on a small model running locally
  ([Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/tree/main)
  is a good candidate to start). The idea is already used by Zed editor
  ([zeta](https://huggingface.co/zed-industries/zeta) model)
- A basic agent that will answer basic questions (ie: what is the idiomatic way
  to convert string slice to String in Rust) without the need of web search and
  remote models calls. It should rely on a locally running LLM and few tools to
  navigate and search the local languages and libraries documentations
  (that no one use)
