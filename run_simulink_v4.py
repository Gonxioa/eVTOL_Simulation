"""Independent V4 evidence protocol. Logs and task status are version-specific."""
import argparse
import sys
from provenance import runtime_info
from simulink_data import read
from simulink_v4_data import CONFIG,root,snapshot,registry,actual_source_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','audit','analyze','batch_a','batch_b','confirm_plan','freeze',
                                         'confirm_test','final_audit','report_a','report'])
    parser.add_argument('--config',default=str(CONFIG)); args=parser.parse_args(); config=read(args.config)
    logdir=root(config)/'logs'; logdir.mkdir(parents=True,exist_ok=True)
    log=(logdir/f'{args.stage}.log').open('a',encoding='utf-8',buffering=1)
    class Tee:
        def __init__(self,stream): self.stream=stream
        def write(self,value): self.stream.write(value); log.write(value)
        def flush(self): self.stream.flush(); log.flush()
    sys.stdout=Tee(sys.stdout); sys.stderr=Tee(sys.stderr)
    print('V4_STAGE',args.stage,runtime_info(),flush=True)
    if args.stage in ('batch_a','batch_b') and (root(config)/'freeze.json').exists():
        raise RuntimeError('V4 already frozen; development selection requires a new registered experiment directory')
    if args.stage=='prepare':
        from maintenance.prepare_simulink_v4 import prepare
        prepare(config)
    elif args.stage=='audit': actual_source_audit(config,registry(config))
    elif args.stage=='analyze':
        from simulink_v4_analysis import analyze
        analyze(config,registry(config))
    elif args.stage=='batch_a':
        from simulink_v4_experiment import run_batch_A
        run_batch_A(config)
    elif args.stage=='batch_b':
        from simulink_v4_augmentation import run_batch_B
        run_batch_B(config)
    elif args.stage=='report_a':
        from simulink_v4_report import report_A
        report_A(config)
    elif args.stage=='confirm_plan':
        from simulink_v4_confirmation import confirmation_plan
        confirmation_plan(config)
    elif args.stage=='freeze':
        from simulink_v4_confirmation import freeze
        freeze(config)
    elif args.stage=='confirm_test':
        from simulink_v4_confirmation import test
        test(config)
    elif args.stage=='final_audit':
        from simulink_v4_confirmation import audit
        audit(config)
    elif args.stage=='report':
        from simulink_v4_report import report_final
        report_final(config)


if __name__=='__main__': main()
