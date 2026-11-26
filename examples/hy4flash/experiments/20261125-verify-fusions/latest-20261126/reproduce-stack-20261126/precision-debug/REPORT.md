# HY4 raw completion repetition diagnostic

The user observed repeated text from the raw `/v1/completions` example. The
earlier four successful smoke requests established execution only; they did not
establish answer quality or numerical accuracy.

## Controlled comparison

The same running service, model weights, FP8 KV, DFLY5, target/Draft graphs and
fusion code were retained. No runtime source or service configuration changed.

| Request | Result |
| --- | --- |
| Raw completion, Chinese question asking for three Hangzhou attractions | Repeated text; reached the 128-token limit |
| Chat completion, same question with the official conversation template | Coherent list of West Lake, Lingyin Temple and Xixi Wetland; normal stop |
| Chat completion asking for 17 + 25 | Answered 42; normal stop |

All three requests returned HTTP 200. Exact payloads and responses are in
`cases.json`. The model directory contains `chat_template.jinja`; its bundled
README demonstrates `chat.completions` with
`chat_template_kwargs.reasoning_effort=no_think`.

## Finding and correction

The comparison points to the raw request format as the immediate cause of the
observed continuation/repetition behavior. A raw prompt bypasses the model's
conversation template. Instruction tests should use `/v1/chat/completions`,
`messages`, and the model's supported template arguments. Do not force
`ignore_eos=true` when checking whether an answer terminates naturally.

Corrected request examples are documented in the reproduction README and the
public integration guide. No operator rollback switch or numerical workaround
was introduced. Service and runtime source remain available for further testing.

## Limits

These samples do not prove full numerical equivalence to an unoptimized model.
No baseline tensor/logit comparison, accuracy benchmark or profiling was performed.
If correctly templated chat requests still fail, preserve those cases and compare
Prefill, Decode, graph and Draft paths against a reference before attributing the
failure to a particular operator.
