import codecs

def convert_log_to_english(input_file, output_file):
    with codecs.open(input_file, 'r', encoding='utf-8') as f_in:
        with codecs.open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                # Replace Chinese text with English
                line = line.replace("项目根目录", "Project Root Directory")
                line = line.replace("日志文件", "Log File")
                line = line.replace("开始压缩评估", "Starting Compression Evaluation")
                line = line.replace("使用模型", "Using Models")
                line = line.replace("开始压缩文件", "Start compressing file")
                line = line.replace("文件大小", "File size")
                line = line.replace("压缩完成，用时", "Compression completed, time used")
                line = line.replace("压缩后大小", "Compressed size")
                line = line.replace("压缩比", "Compression ratio")
                line = line.replace("处理批次", "Processing batch")
                line = line.replace("当前批次大小", "Current batch size")
                line = line.replace("批次", "Batch")
                line = line.replace("完成，用时", "completed, time used")
                line = line.replace("总用时", "total time")
                f_out.write(line)

if __name__ == "__main__":
    convert_log_to_english(
        "FineZip-main/logs/compression_20241030_014303.log",
        "FineZip-main/logs/compression_20241030_014303_en.log"
    ) 