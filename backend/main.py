import os
import sys  # <-- EKLENDİ
import uvicorn
import aiofiles
from pathlib import Path  # <-- EKLENDİ
TEST_MODE = False
# --- Path (Yol) Düzeltmesi (Daha Güçlü Versiyon) ---
# 'main.py' dosyasının bulunduğu dizini ve onun bir üst dizinini Python yoluna ekle.
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent  # 'main.py'nin olduğu klasör (örn: .../backend)
parent_root = project_root.parent         # Bir üst klasör (örn: .../DI-502)

# Her iki yolu da (eğer zaten yoksa) ekle
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(parent_root) not in sys.path:
    sys.path.append(str(parent_root))
# --- Path Düzeltmesi Sonu ---


from fastapi import (
    FastAPI, 
    Request, 
    Form, 
    File, 
    UploadFile, 
    HTTPException
)
from werkzeug.utils import secure_filename
from typing import Optional

# Assume the rag_service is in a 'src' directory relative to this file

from backend.src.rag_service import query_document, plain_chat, query_online


# --- App Setup ---

app = FastAPI(
    title="RAG API",
    description="Belgelerle, çevrimiçi araştırmayla veya düz sohbetle sohbet edin.",
    version="1.0.0"
)

# Configuration for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Routes ---

@app.get("/")
async def root():
    """
    Kök endpoint. Basit bir karşılama mesajı ve belgelere yönlendirme sağlar.
    """
    return {
        "message": "RAG API'ye hoş geldiniz. API belgelerini görmek için /docs veya /redoc adresini ziyaret edin."
    }


@app.post("/chat")
async def chat(
    request: Request,
    question: str = Form(...),
    use_online_research: bool = Form(False),
    document: Optional[UploadFile] = File(None)
):
    """
    Ana sohbet endpoint'i. 
    ...
    """
    
    if not question:
        raise HTTPException(status_code=400, detail="There is no question provided.")

    file_path = None
    
    try:
        if use_online_research:
            # Mod 1: Çevrimiçi Araştırma RAG
            answer = query_online(question,test=TEST_MODE)
            
        elif document and document.filename:
            # Mod 2: Belge RAG
            # ...
            answer = query_document(question, 
                                    doc_path = r"/home/umut_dundar/repositories/economind/DI-502/apple_test.pdf",
                                    test=TEST_MODE)
            
        else:
            # Mod 3: Düz Sohbet
            answer = plain_chat(question, test=TEST_MODE)
            print(answer) # <-- Terminalde "Test response" yazıyor
        
        # --- ÇÖZÜM BURADA ---
        # Frontend 'ai_response' bekliyor, 'answer' değil.
        
        # return {"answer": answer}  # <--- BU YANLIŞTI
        return {"ai_response": answer} # <--- BUNUNLA DEĞİŞTİRİN

    except Exception as e:
        # Bu kısım doğru, frontend'deki 'catch' bloğu
        # HTTP 500 hatasını yakalayacaktır.
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Yüklenen dosyayı varsa temizle
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Dosya temizlenirken hata oluştu {file_path}: {e}")


# --- Sunucuyu Çalıştırma ---

if __name__ == '__main__':
    # 'main:app' 'main.py' dosyasındaki 'app' nesnesine atıfta bulunur
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)