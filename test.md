# Golem LLM Test Suite

A comprehensive Rust-based test application demonstrating `golem-llm` integration capabilities. Located in `test/components-rust/test-llm/src/lib.rs`, this test suite validates LLM functionality across different scenarios including streaming, tool usage, image processing, and fault tolerance.

## Examples

Take the [test application](test/components-rust/test-llm/src/lib.rs) as an example of using `golem-llm` from Rust. The
implemented test functions are demonstrating the following:

| Function Name | Description                                                                                |
|---------------|--------------------------------------------------------------------------------------------|
| `test1`       | Simple text question and answer, no streaming                                              | 
| `test2`       | Demonstrates using **tools** without streaming                                             |
| `test3`       | Simple text question and answer with streaming                                             |
| `test4`       | Tool usage with streaming                                                                  |
| `test5`       | Using an image in the prompt                                                               |
| `test6`       | Demonstrates that the streaming response is continued in case of a crash (with Golem only) |
| `test7`       | Using a source image by passing byte array as base64 in the prompt                         |

### Running the examples

To run the examples first you need a running Golem instance. This can be Golem Cloud or the single-executable `golem`
binary
started with `golem server run`.

**NOTE**: `golem-llm` requires the latest (unstable) version of Golem currently. It's going to work with the next public
stable release 1.2.2.

Then build and deploy the _test application_. Select one of the following profiles to choose which provider to use:
| Profile Name | Description |
|--------------|-----------------------------------------------------------------------------------------------|
| `anthropic-debug` | Uses the Anthropic LLM implementation and compiles the code in debug profile |
| `anthropic-release` | Uses the Anthropic LLM implementation and compiles the code in release profile |
| `ollama-debug` | Uses the Ollama LLM implementation and compiles the code in debug profile |
| `ollama-release` | Uses the Ollama LLM implementation and compiles the code in release profile |
| `grok-debug` | Uses the Grok LLM implementation and compiles the code in debug profile |
| `grok-release` | Uses the Grok LLM implementation and compiles the code in release profile |
| `openai-debug` | Uses the OpenAI LLM implementation and compiles the code in debug profile |
| `openai-release` | Uses the OpenAI LLM implementation and compiles the code in release profile |
| `openrouter-debug` | Uses the OpenRouter LLM implementation and compiles the code in debug profile |
| `openrouter-release` | Uses the OpenRouter LLM implementation and compiles the code in release profile |

### Test Application

```bash
cd test
golem app build -b openai-debug
golem app deploy -b openai-debug
```

### Testing Worker with API Key

Depending on the provider selected, an environment variable has to be set for the worker to be started, containing the API key for the given provider:

```bash
golem worker new test:llm/debug --env OPENAI_API_KEY=xxx --env GOLEM_LLM_LOG=trace
```

### Invoke Test Functions

Execute individual test functions on the worker:

```bash
golem worker invoke test:llm/debug test1 --stream 
```

## Prerequisites

The `test` directory contains a **Golem application** for testing various features of the LLM components.
Check [the Golem documentation](https://learn.golem.cloud/quickstart) to learn how to install Golem and `golem-cli` to
run these tests.
