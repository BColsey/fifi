#!/usr/bin/env python3
"""Capture deterministic short replies to compare two P2P weight placements."""
import argparse
import json
from pathlib import Path
import urllib.request

PROMPTS = (
    'Compute 19 plus 23. Reply with only the number.',
    'Name the capital of Japan in one word.',
    'What are the three letters immediately after D? Reply with just those letters.',
    'What is the opposite of cold? Reply with one word.',
)


def result_of(reply):
    choice = reply['choices'][0]
    content = choice['message'].get('content') or ''
    usage = reply['usage']
    if not content.strip() or usage['prompt_tokens'] < 1 or usage['completion_tokens'] < 1:
        raise ValueError('the server returned an empty reply or invalid token counts')
    return {'content': content, 'finish_reason': choice['finish_reason'],
            'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': usage['completion_tokens']}


def assert_equal(baseline, current):
    if baseline['model'] != current['model'] or baseline['prompts'] != current['prompts']:
        raise ValueError('the model or prompts do not match the baseline')
    if baseline['results'] != current['results']:
        raise ValueError('the placement changed a reply, finish reason, or token count')


def self_test():
    sample = {'choices': [{'message': {'content': '42'}, 'finish_reason': 'stop'}],
              'usage': {'prompt_tokens': 9, 'completion_tokens': 2}}
    result = result_of(sample)
    first = {'model': 'fifi-q36', 'prompts': list(PROMPTS), 'results': [result]}
    assert_equal(first, first)
    second = {'model': 'fifi-q36', 'prompts': list(PROMPTS), 'results': []}
    try:
        assert_equal(first, second)
    except ValueError:
        pass
    else:
        raise AssertionError('a changed result was accepted')
    print('placement comparison self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:18792')
    parser.add_argument('--model', default='fifi-q36')
    parser.add_argument('--expected-context', type=int, default=131072)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None:
        parser.error('--output is required')
    if args.output.exists():
        parser.error('the output already exists')
    base = args.url.rstrip('/')
    results = []
    for prompt in PROMPTS:
        with urllib.request.urlopen(base + '/v1/system', timeout=5) as response:
            info = json.load(response)
        engine = next((e for e in info['engines'] if e['model'] == args.model), None)
        if engine is None or engine['state'] != 'ready' or engine['context_limit'] != args.expected_context:
            raise RuntimeError('engine is busy, unavailable, or has the wrong context')
        body = json.dumps({'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
                           'max_tokens': 24, 'temperature': 0, 'x-thinking': 'off'}).encode()
        request = urllib.request.Request(base + '/v1/chat/completions', body,
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            results.append(result_of(json.load(response)))
    current = {'model': args.model, 'prompts': list(PROMPTS), 'results': results}
    if args.baseline:
        assert_equal(json.loads(args.baseline.read_text()), current)
    with args.output.open('x') as output:
        json.dump(current, output, indent=2)
        output.write('\n')
    print(f'{len(results)} deterministic replies saved' + (' and matched' if args.baseline else ''))


if __name__ == '__main__':
    main()
