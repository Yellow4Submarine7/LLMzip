from huggingface_hub import login, HfApi
import sys

def test_token(token):
    try:
        # 尝试登录
        login(token)
        api = HfApi()
        user_info = api.whoami()
        
        print("=== Token 验证 ===")
        print(f"✓ Token 有效")
        print(f"✓ 用户ID: {user_info['id']}")
        
        print("\n=== Llama-3-8B 访问权限 ===")
        try:
            api.model_info("meta-llama/Meta-Llama-3-8B")
            print("✓ 已获得Llama-3-8B访问权限")
        except Exception as e:
            print("✗ 需要申请Llama-3-8B访问权限")
            print("➜ 请访问: https://huggingface.co/meta-llama/Meta-Llama-3-8B 申请访问")
            
    except Exception as e:
        print(f"验证失败: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = input("请输入你的HuggingFace token: ")
    
    test_token(token) 