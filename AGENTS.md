## Agent Workflow

Before running any command, calling a tool, or changing code or files, present a concise research plan and wait for the user's explicit approval; the original task request does not count as approval. After completing the approved research, report the findings and, if changes are needed, present a separate correction plan and wait for explicit approval before implementation. Any action outside an approved plan requires separate approval. Push approved changes directly to `origin/main` without creating a branch or pull request.

Before presenting a change plan, remove every item that is not required by the user's goal or proven repository constraints. Do not preserve backward compatibility speculatively; require evidence of an active caller or an explicit user requirement. Present the smallest internally consistent plan, not a list of possible implementation options.
