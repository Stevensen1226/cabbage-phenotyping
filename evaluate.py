import json
import logging
import os

from cabbage_pheno.service.evaluation import build_parser, run_evaluation, TRAIT_MAE_FIELDS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Evaluator")


def evaluate(args):
    result = run_evaluation(args)

    # --- Auto-save ---
    output_path = getattr(args, 'output', None)
    if not output_path:
        split = getattr(args, 'split', None)
        if split:
            cfg_name = os.path.splitext(os.path.basename(args.config))[0]
            if getattr(args, 'skip_clustering', False):
                cfg_name = cfg_name + '_nocluster'
            output_path = os.path.join('output', f'{cfg_name}_{split}.json')

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'output',
                    exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        s = result.get('summary', {})
        logger.info(f"Results saved to {output_path}")
        logger.info(f"  Split: {result.get('split', 'N/A')} | Files: {s.get('processed_files', 'N/A')}")
        logger.info(f"  Inst Prec/Rec/F1: {s.get('avg_inst_prec',0):.3f}/{s.get('avg_inst_rec',0):.3f}/{s.get('avg_inst_f1',0):.3f}")
        logger.info(f"  Sem mIoU: {s.get('avg_sem_miou',0):.3f} | Runtime: {s.get('avg_runtime',0):.1f}s")
        logger.info(f"  表型 MAE ({s.get('n_trait_matched_pairs',0)} pairs):")
        for _key, _label, _unit in TRAIT_MAE_FIELDS:
            _m = s.get(f'mae_{_key}')
            _r = s.get(f'rel_{_key}_pct')
            _ms = f'{_m:.3f}{_unit}' if _m is not None else 'N/A'
            _rs = f'{_r:.1f}%' if _r is not None else 'N/A'
            logger.info(f"    {_label:14s} MAE={_ms:>14s}  Rel={_rs:>8s}")

    return result


if __name__ == '__main__':
    args = build_parser().parse_args()
    evaluate(args)
