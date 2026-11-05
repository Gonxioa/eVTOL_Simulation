"""Independent V3 entry point; stages are resumable, test remains gated."""
import argparse
import sys
from provenance import runtime_info
from simulink_data import read
from simulink_v3_data import CONFIG, prepare_test_plan, development_registry, new_test_registry, root
from simulink_v3_experiment import develop, freeze, evaluate_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'develop', 'freeze', 'test', 'all', 'report', 'audit'])
    parser.add_argument('--config', default=str(CONFIG))
    args = parser.parse_args(); config = read(args.config)
    logs = root(config) / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    log = (logs / f'{args.stage}.log').open('a', encoding='utf-8', buffering=1)
    class Tee:
        def __init__(self, stream): self.stream = stream
        def write(self, value): self.stream.write(value); log.write(value)
        def flush(self): self.stream.flush(); log.flush()
    sys.stdout = Tee(sys.stdout); sys.stderr = Tee(sys.stderr)
    print('V3_STAGE', args.stage, runtime_info(), flush=True)
    if args.stage == 'plan':
        print(prepare_test_plan(config)); return
    if args.stage in ('develop', 'all'):
        development = development_registry(config)
        develop(config, development)
    if args.stage in ('freeze', 'all'):
        development = development_registry(config)
        registry = new_test_registry(config, development)
        freeze(config, development, registry)
    if args.stage in ('test', 'all'):
        registry = read(root(config) / 'new_test_sources.json')
        evaluate_test(config, registry, read(root(config) / 'freeze.json'))
    if args.stage in ('audit', 'all'):
        from audit_simulink_v3 import audit
        audit(config)
    if args.stage in ('report', 'all'):
        from simulink_v3_report import report
        report(config)


if __name__ == '__main__':
    main()
