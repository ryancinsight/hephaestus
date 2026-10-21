"""Generate narrow baseline failure classifications from exported arithmetic.

This is a CPU source model, never shader/compiler qualification. Constants and
each explicit operation round to binary32. Exp uses Python's binary64 libm then
rounds; it is neither a correctly-rounded-f32 claim nor an intrinsic envelope.
Both select arms are evaluated. Each intrinsic profile has gradual and per-operation
FTZ modes. The fixed alternative exp2(RN32(a * RN32(log2(e)))) uses no device
outputs; its active finite arguments must pass a high-precision WGSL exp bound.
Only exact input bits and actually modelled failure classes become exceptions.
"""
import argparse
import ast
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
import functools
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
from verify import double, exports, failure_class, manifest, require, sha256, single

# RNE32(log2(e)); independently derived constant, not a device-output parameter.
LOG2_E = single(0x3fb8aa3b)
PROFILES = [('gradual', False, False), ('per-operation-FTZ', True, False),
            ('exp2-reduction-gradual', False, True), ('exp2-reduction-per-operation-FTZ', True, True)]


def word(value):
    return struct.unpack('>I', struct.pack('>f', value))[0]


def exp_model(argument, reduction):
    try:
        return math.exp2(rounded(argument * LOG2_E)) if reduction else math.exp(argument)
    except OverflowError:
        return math.inf


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

    def evaluate(self, value, flush, reduction=False, active_arguments=None):
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
                    result = exp_model(args[0], reduction)
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

        result = visit(self.root)
        if active_arguments is not None:
            visited = set()
            def trace(node):
                if node in visited:
                    return
                visited.add(node)
                kind, *children = node
                if kind == 'select':
                    trace(children[2])
                    trace(children[1] if cache[children[2]] else children[0])
                else:
                    if kind == 'exp':
                        active_arguments.add(word(cache[children[0]]))
                    for child in children:
                        if isinstance(child, tuple):
                            trace(child)
            trace(self.root)
        return result



def directed(operation, mode):
    with localcontext() as context:
        context.prec, context.rounding = 80, mode
        return operation()



@functools.lru_cache(maxsize=1)
def prove_log2e():
    with localcontext() as context:
        context.prec = 80
        ln2 = Decimal(2).ln()
        lower = directed(lambda: Decimal(1) / ln2.next_plus(context), ROUND_FLOOR)
        upper = directed(lambda: Decimal(1) / ln2.next_minus(context), ROUND_CEILING)
        value = Decimal.from_float(LOG2_E)
        left = (Decimal.from_float(single(0x3fb8aa3a)) + value) / 2
        right = (Decimal.from_float(single(0x3fb8aa3c)) + value) / 2
        require(left < lower <= upper < right, 'log2(e) constant is not nearest binary32')
        return {'bits': '3fb8aa3b', 'real_interval': [str(lower), str(upper)]}


def result_accounting(got):
    classification = ('infinite' if math.isinf(got) else 'zero' if got == 0.0
                      else 'subnormal' if abs(got) < 2.0 ** -126 else 'normal')
    return {'preflush_result_bits': f'{word(got):08x}', 'preflush_result_class': classification,
            'post_ftz_result_bits': f'{word(rounded(got, True)):08x}'}


@functools.lru_cache(maxsize=65536)
def prove_reduction(argument_bits):
    """Validate this fixed approximation at one active argument, not all real inputs.

    Decimal exp/ln are correctly rounded; adjacent values enclose their real
    results. Directed arithmetic bounds the error against WGSL's (3+2*abs(a)) ULP.
    Huge arguments use analytic IEEE overflow/underflow bounds, avoiding exp(1e38).
    The proof is for the pre-flush intrinsic; FTZ remains the existing explicit mode.
    """
    argument = single(argument_bits)
    if not math.isfinite(argument):
        return {'argument_bits': f'{argument_bits:08x}', 'status': 'NONFINITE_MODELED_SEPARATELY'}
    got = rounded(exp_model(argument, True))
    accounting = result_accounting(got)
    q = rounded(argument * LOG2_E)
    with localcontext() as context:
        context.prec = 80
        a, candidate = Decimal.from_float(argument), Decimal.from_float(got)
        ln2 = Decimal(2).ln()
        ln2_lo, ln2_hi = ln2.next_minus(context), ln2.next_plus(context)
        # These comparisons prove the exact-real result rounds to the same extreme.
        if a >= Decimal(89):
            require(a > directed(lambda: Decimal(128) * ln2_hi, ROUND_CEILING)
                    and math.isinf(got) and got > 0 and q >= 128, 'unproved exp overflow reduction')
            return {'argument_bits': f'{argument_bits:08x}', 'status': 'EXACT_RNE_OVERFLOW', **accounting}
        if a <= Decimal(-105):
            require(a < directed(lambda: Decimal(-150) * ln2_hi, ROUND_FLOOR)
                    and got == 0.0 and q <= -150, 'unproved exp underflow reduction')
            return {'argument_bits': f'{argument_bits:08x}', 'status': 'EXACT_RNE_UNDERFLOW', **accounting}
        if argument == 0.0:
            require(got == 1.0 and q == 0.0, 'exp zero identity changed')
            return {'argument_bits': f'{argument_bits:08x}', 'status': 'BOUNDED_FINITE_RESULT', **accounting,
                    'exp_error_upper_ulp': '0', 'exp_allowance_ulp': '3',
                    'exp2_error_upper_ulp': '0', 'exp2_allowance_ulp': '3'}
        exact = a.exp()
        lower, upper = exact.next_minus(context), exact.next_plus(context)
        if math.isinf(got):
            midpoint = Decimal(2) ** 128 - Decimal(2) ** 103
            require(lower > midpoint and q >= 128, 'finite exp result overflowed reduction')
            return {'argument_bits': f'{argument_bits:08x}', 'status': 'EXACT_RNE_OVERFLOW', **accounting}
        # exact result is positive; frexp is used only for an exponent and checked.
        e = math.frexp(float(exact))[1] - 1
        require(Decimal(2) ** e <= lower <= upper < Decimal(2) ** (e + 1), 'ambiguous intrinsic exponent')
        spacing = Decimal(2) ** max(e - 23, -149)
        distances = [directed(lambda end=end: candidate - end if candidate >= end else end - candidate,
                              ROUND_CEILING) for end in [lower, upper]]
        error = directed(lambda: max(distances) / spacing, ROUND_CEILING)
        allowance = Decimal(3) + Decimal(2) * abs(a)
        require(error <= allowance, 'fixed exp2 profile exceeds WGSL exp allowance')
        # Its exp2 stage also respects exp2's own bound at the rounded q argument.
        qd = Decimal.from_float(q)
        if q.is_integer():
            e2 = int(q)
            error2 = abs(Fraction(got) - Fraction(2) ** e2) / Fraction(2) ** max(e2 - 23, -149)
            require(error2 <= Fraction(3) + 2 * abs(Fraction(q)), 'integer exp2 bound failed')
            return {'argument_bits': f'{argument_bits:08x}', 'status': 'BOUNDED_FINITE_RESULT', **accounting,
                    'exp_error_upper_ulp': str(error), 'exp_allowance_ulp': str(allowance),
                    'exp2_error_upper_ulp': str(Decimal(error2.numerator) / Decimal(error2.denominator)),
                    'exp2_allowance_ulp': str(Decimal(3) + Decimal(2) * abs(qd))}
        products_lo = [directed(lambda end=end: qd * end, ROUND_FLOOR) for end in [ln2_lo, ln2_hi]]
        products_hi = [directed(lambda end=end: qd * end, ROUND_CEILING) for end in [ln2_lo, ln2_hi]]
        lo2 = min(products_lo).exp().next_minus(context)
        hi2 = max(products_hi).exp().next_plus(context)
        e2 = math.frexp(float(lo2))[1] - 1
        require(Decimal(2) ** e2 <= lo2 <= hi2 < Decimal(2) ** (e2 + 1), 'ambiguous exp2 exponent')
        s2 = Decimal(2) ** max(e2 - 23, -149)
        d2 = [directed(lambda end=end: candidate - end if candidate >= end else end - candidate,
                       ROUND_CEILING) for end in [lo2, hi2]]
        error2 = directed(lambda: max(d2) / s2, ROUND_CEILING)
        allowance2 = Decimal(3) + Decimal(2) * abs(qd)
        require(error2 <= allowance2, 'fixed reduction exceeds WGSL exp2 allowance')
        return {'argument_bits': f'{argument_bits:08x}', 'status': 'BOUNDED_FINITE_RESULT', **accounting,
                'exp_error_upper_ulp': str(error), 'exp_allowance_ulp': str(allowance),
                'exp2_error_upper_ulp': str(error2), 'exp2_allowance_ulp': str(allowance2)}


def reduction_admissibility(arguments):
    records = [{**prove_reduction(bits), 'profiles': sorted(profiles)}
               for bits, profiles in sorted(arguments.items())]
    counts = {}
    for record in records:
        counts[record['status']] = counts.get(record['status'], 0) + 1
    canonical = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return {'scope': 'Every active finite exp argument in this fixed compiled fixture set only; no global/JIT qualification',
            'formula': 'RN32(exp2(RN32(a * RN32(log2(e)))))', 'log2e_constant_proof': prove_log2e(),
            'wgsl_exp_exp2_allowance': '(3 + 2*abs(argument)) ULP', 'decimal_precision': 80,
            'ftz_scope': 'Check pre-flush intrinsic; existing per-operation FTZ semantics retained',
            'status_counts': counts, 'argument_count': len(records),
            'preflush_result_counts': {kind: sum(r.get('preflush_result_class') == kind for r in records)
                                       for kind in ['normal', 'subnormal', 'zero', 'infinite']},
            'proof_records_sha256': hashlib.sha256(canonical).hexdigest()}, records

def generate(export_path, manifest_path, audit=None):
    source = exports(export_path)
    cpu = manifest(manifest_path)
    require(sha256(export_path) == cpu['meta']['expression_sha256'], 'predictor source/binary mismatch')
    predicates = {}
    counts = {}
    arguments = {}
    for (op, band), rows in cpu['inputs'].items():
        program = Expression(source[op, 'Wgsl'])
        require(cpu['expressions'][op] == source[op, 'Wgsl'], 'predictor expression differs from compiled binary')
        for profile, flush, reduction in PROFILES:
            failures = 0
            for _, bits, reference_bits, bound, absolute in rows:
                active = set() if reduction else None
                got = program.evaluate(single(bits), flush, reduction, active)
                for argument in active or ():
                    arguments.setdefault(argument, set()).add(profile)
                classification = failure_class(got, double(reference_bits), bound, absolute)
                if classification:
                    failures += 1
                    predicates.setdefault((op, band, bits, classification), set()).add(profile)
            counts[op + '/' + band + '/' + profile] = failures
    summary, records = reduction_admissibility(arguments)
    if audit is not None:
        audit.update(summary=summary, arguments=records)
    return {'schema': 1, 'generator_sha256': sha256(__file__),
            'expression_sha256': sha256(export_path), 'fixture_sha256': cpu['fixture_sha256'],
            'scope': __doc__, 'profile_failure_counts': counts,
            'exp2_profile_admissibility': summary,
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
    parser.add_argument('--admissibility-output', type=Path)
    options = parser.parse_args()
    audit = {}
    write(options.output, generate(options.exports, options.manifest, audit))
    if options.admissibility_output:
        options.admissibility_output.write_text(json.dumps(audit, indent=2) + '\n')
