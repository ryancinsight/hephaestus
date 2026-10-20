"""Generate narrow baseline failure classifications from exported arithmetic.

This is a CPU source model, never shader/compiler qualification. Constants and
each explicit operation round to binary32. Exp uses Python's binary64 libm then
rounds; it is neither a correctly-rounded-f32 claim nor an intrinsic envelope.
Both select arms are evaluated. The second profile flushes each subnormal result.
Only exact input bits and actually modelled failure classes become exceptions.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
from verify import double, exports, failure_class, manifest, require, sha256, single


def rounded(value, flush=False):
    try:
        result = struct.unpack('>f', struct.pack('>f', value))[0]
    except OverflowError:
        result = math.copysign(math.inf, value)
    return math.copysign(0.0, result) if flush and 0 < abs(result) < 2.0 ** -126 else result


class Expression:
    def __init__(self, source):
        self.root = self.lower(ast.parse(source, mode='eval').body)

    def lower(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return ('constant', rounded(node.value))
        if isinstance(node, ast.Name) and node.id == 'x':
            return ('x',)
        if isinstance(node, ast.BinOp) and type(node.op) in [ast.Add, ast.Sub, ast.Mult, ast.Div]:
            return (type(node.op).__name__, self.lower(node.left), self.lower(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in [ast.USub, ast.UAdd]:
            return (type(node.op).__name__, self.lower(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in [ast.Lt, ast.LtE, ast.Gt, ast.GtE]:
            return (type(node.ops[0]).__name__, self.lower(node.left), self.lower(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ['select', 'abs', 'exp']:
            require(not node.keywords, 'keyword arguments are unsupported')
            return (node.func.id, *(self.lower(arg) for arg in node.args))
        raise ValueError('unsupported expression syntax: ' + ast.dump(node))

    def evaluate(self, value, flush):
        cache = {}

        def visit(node):
            if node in cache:
                return cache[node]
            kind, *children = node
            if kind == 'constant':
                result = children[0]
            elif kind == 'x':
                result = rounded(value, flush)
            else:
                args = [visit(child) for child in children]
                if kind == 'Add':
                    result = args[0] + args[1]
                elif kind == 'Sub':
                    result = args[0] - args[1]
                elif kind == 'Mult':
                    result = args[0] * args[1]
                elif kind == 'Div':
                    if args[1] == 0.0:
                        result = math.nan if args[0] == 0.0 else math.copysign(math.inf, args[0] * math.copysign(1, args[1]))
                    else:
                        result = args[0] / args[1]
                elif kind == 'USub':
                    result = -args[0]
                elif kind == 'UAdd':
                    result = args[0]
                elif kind == 'abs':
                    result = abs(args[0])
                elif kind == 'exp':
                    try:
                        result = math.exp(args[0])
                    except OverflowError:
                        result = math.inf
                elif kind == 'select':
                    result = args[1] if args[2] else args[0]
                elif kind == 'Lt':
                    result = args[0] < args[1]
                elif kind == 'LtE':
                    result = args[0] <= args[1]
                elif kind == 'Gt':
                    result = args[0] > args[1]
                elif kind == 'GtE':
                    result = args[0] >= args[1]
                else:
                    raise ValueError(kind)
                if not isinstance(result, bool):
                    result = rounded(result, flush)
            cache[node] = result
            return result

        return visit(self.root)


def generate(export_path, manifest_path):
    source = exports(export_path)
    cpu = manifest(manifest_path)
    require(sha256(export_path) == cpu['meta']['expression_sha256'], 'predictor source/binary mismatch')
    predicates = {}
    counts = {}
    for (op, band), rows in cpu['inputs'].items():
        program = Expression(source[op, 'Wgsl'])
        require(cpu['expressions'][op] == source[op, 'Wgsl'], 'predictor expression differs from compiled binary')
        for flush in [False, True]:
            profile = 'per-operation-FTZ' if flush else 'gradual'
            failures = 0
            for _, bits, reference_bits, bound, absolute in rows:
                got = program.evaluate(single(bits), flush)
                classification = failure_class(got, double(reference_bits), bound, absolute)
                if classification:
                    failures += 1
                    predicates.setdefault((op, band, bits, classification), set()).add(profile)
            counts[op + '/' + band + '/' + profile] = failures
    return {'schema': 1, 'generator_sha256': sha256(__file__),
            'expression_sha256': sha256(export_path), 'fixture_sha256': cpu['fixture_sha256'],
            'scope': __doc__, 'profile_failure_counts': counts,
            'predicates': [{'op': op, 'band': band, 'input_bits': bits, 'class': classification,
                            'source_model_profiles': sorted(profiles)}
                           for (op, band, bits, classification), profiles in sorted(predicates.items())]}


def write(path, data):
    header = {key: value for key, value in data.items() if key != 'predicates'}
    lines = json.dumps(header, indent=2).splitlines()[:-1]
    lines[-1] += ','
    lines.append('  "predicates": [')
    lines.extend('    ' + json.dumps(row, separators=(',', ':')) + (',' if i + 1 < len(data['predicates']) else '')
                 for i, row in enumerate(data['predicates']))
    path.write_text('\n'.join(lines + ['  ]', '}']) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    write(options.output, generate(options.exports, options.manifest))
