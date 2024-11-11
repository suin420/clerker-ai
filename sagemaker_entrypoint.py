import os
import argparse
from Keywords.BllossomKeyword_to_md import generate_summary_jsons, generate_report_from_json
from Diagrams.DiagramGeneration import diagram_gen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_chunk_dict', type=str, required=True)
    parser.add_argument('--diagram_summary_json', type=str, required=True)
    parser.add_argument('--report_summary_json', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--font_path', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    os.environ['MPLCONFIGDIR'] = '/opt/ml/processing/matplotlib'
    os.environ['TRANSFORMERS_CACHE'] = '/opt/ml/processing/.cache/huggingface'
    os.environ['NUMBA_CACHE_DIR'] = '/opt/ml/processing/numba_cache'

    os.makedirs('/opt/ml/processing/output', exist_ok=True)
    os.makedirs('/opt/ml/processing/Diagrams/mermaid', exist_ok=True)
    os.makedirs('/opt/ml/processing/matplotlib', exist_ok=True)
    os.makedirs('/opt/ml/processing/.cache/huggingface', exist_ok=True)
    os.makedirs('/opt/ml/processing/numba_cache', exist_ok=True)

    model_id = args.model_id
    model_path = args.model_path

    font_path = args.font_path

    generate_summary_jsons(
        args.input_chunk_dict,
        args.diagram_summary_json,
        args.report_summary_json,
        model_id=model_id,
        model_path=model_path
    )
    print(f"다이어그램 및 보고서용 JSON 파일 생성 완료 : {args.report_summary_json}")

    diagram_gen(
        args.diagram_summary_json,
        model_id=model_id,
        model_path=model_path
    )
    print("다이어그램 생성 완료")

    output_report_md = os.path.join('/opt/ml/processing/output', 'report.md')
    generate_report_from_json(args.report_summary_json, output_report_md, font_path)
    print(f"Report 파일 생성 완료 : {output_report_md}")

if __name__ == "__main__":
    main()
