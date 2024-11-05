import chardet

def convert_file_encoding(input_file, output_file):
    # 读取原始文件并检测编码
    with open(input_file, 'rb') as f:
        content = f.read()
        detected = chardet.detect(content)
        original_encoding = detected['encoding']
    
    print(f"Detected original encoding: {original_encoding}")
    
    try:
        # 解码原始内容，然后用UTF-8重新编码
        decoded_content = content.decode(original_encoding)
        
        # 写入新文件，使用UTF-8编码
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(decoded_content)
            
        print(f"Successfully converted {input_file} from {original_encoding} to UTF-8")
        print(f"New file saved as: {output_file}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    input_log = "FineZip-main/logs/compression_20241030_014303.log"
    output_log = "FineZip-main/logs/compression_20241030_014303_utf8.log"
    
    convert_file_encoding(input_log, output_log) 