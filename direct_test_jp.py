import grpc
import sys
import os

# 嘗試匯入產生的 proto 代碼
sys.path.append(os.path.join(os.getcwd(), "models"))
try:
    import inference_pb2
    import inference_pb2_grpc
except ImportError:
    print("❌ 找不到 proto 檔案，請執行 ./scripts/generate_proto.sh")
    sys.exit(1)

def run():
    print("正在連線至 gRPC 50051...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = inference_pb2_grpc.InferenceServerStub(channel)
    
    # 建立一個簡單的語音/文字請求
    # 注意：這裡的欄位名稱需對應你的 inference.proto 定義
    try:
        print("發送日文口型生成請求...")
        # 這裡模擬一個簡單的請求，具體參數需根據你的 proto 調整
        # 假設你的 proto 有一個簡單的文本輸入介面
        request = inference_pb2.InferenceRequest(
            text="こんにちは。日本語の口型テストです。よろしくお願いします。",
            avatar_id="flash_head"
        )
        responses = stub.Inference(request)
        print("✅ 請求已送出，請觀察伺服器端日誌！")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    run()
