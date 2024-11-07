from STT.ClovaText import make_stt_txt
from Chunking.EmbeddingChunking import make_chunk
from Keywords.BllossomKeyword_to_md import generate_summary_jsons, generate_report_from_json
from Diagrams.DiagramGeneration import diagram_gen

## Defining Arguments
input_audio_file = "./STT/stt_audio/example.mp3"
output_txt_file = "./STT/stt_text/stt_text.txt"
input_domain = "IT"

input_stt_txt = output_txt_file
output_chunk_dict = "./Chunking/chunking_text.json"

output_report_md = "./Keywords/report.md"
output_summary_json = "./Keywords/summary.json"
diagram_summary_json = "./Diagrams/diagram_summary.json"

input_summary_json = output_summary_json

model_id = 'models/models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7'
model_path = 'models/models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf'

if __name__ == "__main__":

    make_stt_txt(input_domain, input_audio_file, output_txt_file)
    print(f"STT 파일 생성 완료 : {output_txt_file}")

    make_chunk(input_stt_txt,output_chunk_dict)
    print(f"Chunk Dict 파일 생성 완료 : {output_chunk_dict}")

    generate_summary_jsons(output_chunk_dict, diagram_summary_json, output_summary_json, model_id, model_path)
    print(f"다이어그램 및 보고서용 JSON 파일 생성 완료 : {output_summary_json}")

    diagram_gen(diagram_summary_json)
    print("다이어그램 생성 완료")

    generate_report_from_json(output_summary_json, output_report_md)
    print(f"Report 파일 생성 완료 : {output_report_md}")