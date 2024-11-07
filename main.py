from STT.ClovaText import make_stt_txt
from Chunking.EmbeddingChunking import make_chunk
from Keywords.BlossomKeyword_to_md import bllossom_keyword_extraction
from Diagrams.DiagramGeneration import diagram_gen

## Defining Arguments
input_audio_file = "./STT/stt_audio/council_medical_domain.wav"
output_txt_file = "./STT/stt_text/stt_text.txt"
input_domain = "Medical"

input_stt_txt = output_txt_file
output_chunk_dict = "./Chunking/chunking_text.json"

output_report_md = "./Keywords/report.md"
output_summary_json = "./Keywords/summary.json"

input_summary_json = output_summary_json

if __name__ == "__main__":

    make_stt_txt(input_domain, input_audio_file, output_txt_file)
    print(f"STT 파일 생성 완료 : {output_txt_file}")
    make_chunk(input_stt_txt,output_chunk_dict)
    print(f"Chunk Dict 파일 생성 완료 : {output_chunk_dict}")
    bllossom_keyword_extraction(output_chunk_dict,output_report_md,output_summary_json)
    print(f"Report 파일 생성 완료 : {output_report_md}")
    diagram_gen(input_summary_json)
    print('Diagram 생성 완료')
   