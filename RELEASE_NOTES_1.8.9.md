# Tamfis Code 1.8.9

- Allow safe, AST-checked Python heredoc inspection commands in read-only tasks.
- Prevent Python heredocs containing filesystem, process, network, or dynamic-execution primitives from entering the read-only path.
