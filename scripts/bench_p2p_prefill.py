#!/usr/bin/env python3
"""Measure repeatable end-to-end prefill on an already-running fifi server.

The reported rate divides prompt tokens by wall time to the complete reply;
it includes request overhead and the few generated tokens. Supply a scratch
output path; an existing result is never overwritten. Run --self-test without
a server to check the payload and result validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


def make_payload(model, repeats):
    if not model or repeats < 1:
        raise ValueError('a model and positive repeat count are required')
    text = 'The sky is blue. ' * repeats + '\nWhat is two plus two? Reply with one digit.'
    body = {'model': model, 'messages': [{'role': 'user', 'content': text}],
            'max_tokens': 4, 'temperature': 0, 'x-thinking': 'off'}
    return (json.dumps(body) + '\n').encode()


def sample_of(reply, elapsed):
    usage = reply['usage']
    prompt = usage['prompt_tokens']
    answer = reply['choices'][0]['message'].get('content') or ''
    if not isinstance(prompt, int) or prompt < 1 or elapsed <= 0 or answer.strip() != '4':
        raise ValueError(f'unexpected result: prompt={prompt!r}, answer={answer!r}, seconds={elapsed}')
    return {'prompt_tokens': prompt, 'completion_tokens': usage['completion_tokens'],
            'answer': answer, 'seconds': round(elapsed, 6),
            'estimated_prompt_tokens_per_second': round(prompt / elapsed, 3)}


def self_test():
    first = make_payload('fifi-q36', 850)
    assert hashlib.sha256(first).hexdigest() == '83bc090b8c20bbf350317d552c592dd83fd0f8a0e9505c1ed895b914be94e890'
    fake = {'usage': {'prompt_tokens': 4274, 'completion_tokens': 2},
            'choices': [{'message': {'content': '4'}}]}
    assert sample_of(fake, 100)['estimated_prompt_tokens_per_second'] == 42.74
    try:
        sample_of(fake, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('zero duration was accepted')
    print('measurement self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:18792')
    parser.add_argument('--model', default='fifi-q36')
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--expected-context', type=int, default=131072)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--api-key-env', help='name of an environment variable holding the key')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.samples < 1 or args.timeout <= 0 or args.output is None:
        parser.error('--samples and --timeout must be positive, and --output is required')
    payload = make_payload(args.model, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + '.partial')
    if args.output.exists():
        parser.error('the result already exists')
    with partial.open('x') as out:
        json.dump({'state': 'started', 'samples': []}, out)
    headers = {'Content-Type': 'application/json'}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            parser.error('the requested API key environment variable is empty')
        headers['Authorization'] = 'Bearer ' + key
    base = args.url.rstrip('/')
    samples = []
    for index in range(args.samples):
        req = urllib.request.Request(base + '/v1/system', headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            info = json.load(response)
        engine = next((e for e in info['engines'] if e['model'] == args.model), None)
        if engine is None or engine['state'] != 'ready':
            raise RuntimeError('the requested engine is not ready; no sample was started')
        if engine['context_limit'] != args.expected_context:
            raise RuntimeError('the engine context does not match --expected-context')
        request = urllib.request.Request(base + '/v1/chat/completions', payload, headers=headers)
        began = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            reply = json.load(response)
        sample = sample_of(reply, time.monotonic() - began)
        sample['context_limit'] = engine['context_limit']
        sample['request_id'] = reply.get('id')
        samples.append(sample)
        partial.write_text(json.dumps({'state': 'in_progress', 'model': args.model,
            'repeat': args.repeat, 'payload_sha256': hashlib.sha256(payload).hexdigest(),
            'samples': samples}, indent=2) + '\n')
        print(f'sample {index + 1}/{args.samples}: {sample["seconds"]:.3f}s, '
              f'{sample["estimated_prompt_tokens_per_second"]:.2f} prompt tokens/s', flush=True)
    if len({s['prompt_tokens'] for s in samples}) != 1:
        raise RuntimeError('prompt token counts differed between samples')
    result = {'model': args.model, 'repeat': args.repeat,
              'payload_sha256': hashlib.sha256(payload).hexdigest(), 'samples': samples,
              'median_seconds': statistics.median(s['seconds'] for s in samples),
              'median_prompt_tokens_per_second': statistics.median(
                  s['estimated_prompt_tokens_per_second'] for s in samples)}
    partial.write_text(json.dumps(result, indent=2) + '\n')
    os.link(partial, args.output)  # never replaces an existing result
    partial.unlink()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
