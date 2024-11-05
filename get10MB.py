# 创建extract_data.py在项目根目录
with open('F:\LLMascompressor\FineZip-main\datasets\enwik8', 'rb') as f_in:
    data = f_in.read(10 * 1024 * 1024)  # 读取10MB
    with open('F:\LLMascompressor\FineZip-main\datasets\enwik10mb.txt', 'wb') as f_out:
        f_out.write(data)